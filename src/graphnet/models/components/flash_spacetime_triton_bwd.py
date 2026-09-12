"""Triton backward kernels for fused spacetime-bias attention.

Deterministic two-kernel scheme (DESIGN.md §8) with the projection folded
to the host on both sides, mirroring the forward:

- host prologue: `u = (q·scale) @ W`, `w~ = dO @ W`, `dOβ = dO·β` (all
  deterministic GEMMs/reductions);
- row-owned kernel: a first pass over the keys forms the softmax row sum
  `D_i = Σ_j P_ij dP_ij / Σ_j P_ij` from the same P and dP that the second
  pass turns into dS; the second pass writes the dS@k part of dq plus the per-row
  E-accumulators `H = Σ_j dS·E` and `G = Σ_j P·E` and `σ = Σ_j dS`, and
  `D` itself for the column kernel;
- column-owned kernel: recomputes S, P, dS tile-by-tile with that `D` and
  writes dk, dv (exclusive rows, fixed loop order — no atomics);
- host epilogue: `dq = scale·(dqp + [A](H @ W^T + σ⊗β))`,
  `dW = [A] Σ qt⊗H + [V] Σ dO⊗G` (two GEMMs over flattened rows),
  `dβ = [V] Σ_valid dO` (the logit path vanishes analytically).

Bitwise run-to-run determinism is structural: every buffer has exactly one
writer and every reduction is either an in-kernel fixed-order loop or a
cuBLAS call.

`D` is formed in-kernel rather than taken from the forward output through
the identity `D_i = dO_i·O_i`. The identity holds for the exact softmax,
but the forward accumulates with weights rounded to the compute dtype and
the backward recomputes P from the LSE, so a `D` read off the output is
consistent with neither the P it multiplies nor the dP it is subtracted
from, and `Σ_j dS_ij` does not vanish: every rounding on either side leaks
into a per-row term of one sign, which in bf16 training walks the
attention toward one-hot rows. Formed from the very P and dP that form dS,
the row sum of dS is zero identically, whatever the forward stored, and
the backward needs no saved output at all.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from graphnet.models.components.flash_spacetime import (
    SINEMB_CLIP,
    SINEMB_INPUT_SCALE,
    TIME_SCALE,
    sinusoidal_frequencies,
)
from graphnet.models.components.flash_spacetime_triton import (
    _e_chunk,
    _next_pow2,
    _pair_angle,
)


@triton.jit
def _recompute_p_dpr(
    qt0: tl.tensor,
    qt1: tl.tensor,
    qt2: tl.tensor,  # [M, G, CC]
    u0: tl.tensor,
    u1: tl.tensor,
    u2: tl.tensor,
    do0: tl.tensor,
    do1: tl.tensor,
    do2: tl.tensor,
    k0: tl.tensor,
    k1: tl.tensor,
    k2: tl.tensor,  # [N, G, CC]
    v0: tl.tensor,
    v1: tl.tensor,
    v2: tl.tensor,
    e0: tl.tensor,
    e1: tl.tensor,
    e2: tl.tensor,  # [M, N, CC]
    wt0: tl.tensor,
    wt1: tl.tensor,
    wt2: tl.tensor,  # w~ chunks [M, G, CC]
    dob: tl.tensor,  # [M, G] dO·β
    lse: tl.tensor,  # [M, G]
    col_valid: tl.tensor,  # [N]
    C_: tl.constexpr,
    C_CHUNK: tl.constexpr,
    USE_ATTN_BIAS: tl.constexpr,
    USE_ACT_BIAS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
) -> tuple:
    """P and raw dP for one (M, N) tile, in [M, G, N] layout (fp32).

    P is exactly zero at masked columns; dP there is whatever the masked
    (zero) operands give and must not be read on its own.

    Annotated with the builtin rather than `typing.Tuple`: Triton's
    frontend parses the return annotation of a jit'd function and
    rejects `typing` constructs.
    """
    s = tl.dot(
        tl.trans(qt0, 1, 0, 2),
        tl.trans(k0, 1, 2, 0),
        input_precision=INPUT_PRECISION,
    )
    if C_ > C_CHUNK:
        s += tl.dot(
            tl.trans(qt1, 1, 0, 2),
            tl.trans(k1, 1, 2, 0),
            input_precision=INPUT_PRECISION,
        )
    if C_ > 2 * C_CHUNK:
        s += tl.dot(
            tl.trans(qt2, 1, 0, 2),
            tl.trans(k2, 1, 2, 0),
            input_precision=INPUT_PRECISION,
        )  # [G, M, N]
    if USE_ATTN_BIAS:
        sb = tl.dot(u0, e0, input_precision=INPUT_PRECISION)
        if C_ > C_CHUNK:
            sb += tl.dot(u1, e1, input_precision=INPUT_PRECISION)
        if C_ > 2 * C_CHUNK:
            sb += tl.dot(u2, e2, input_precision=INPUT_PRECISION)
        s += tl.trans(sb, 1, 0, 2)
    s = tl.trans(s, 1, 0, 2)  # [M, G, N]

    # P from the saved LSE; exact zeros at masked columns.
    p = tl.exp(s - lse[:, :, None])
    p = tl.where(col_valid[None, None, :], p, 0.0)

    # dPraw = dO·v^T (+ [V] (w~·E + dO·β)).
    dpr_acc = tl.dot(
        tl.trans(do0, 1, 0, 2),
        tl.trans(v0, 1, 2, 0),
        input_precision=INPUT_PRECISION,
    )
    if C_ > C_CHUNK:
        dpr_acc += tl.dot(
            tl.trans(do1, 1, 0, 2),
            tl.trans(v1, 1, 2, 0),
            input_precision=INPUT_PRECISION,
        )
    if C_ > 2 * C_CHUNK:
        dpr_acc += tl.dot(
            tl.trans(do2, 1, 0, 2),
            tl.trans(v2, 1, 2, 0),
            input_precision=INPUT_PRECISION,
        )
    dpr = tl.trans(dpr_acc, 1, 0, 2)  # [M, G, N]
    if USE_ACT_BIAS:
        wb = tl.dot(wt0, e0, input_precision=INPUT_PRECISION)
        if C_ > C_CHUNK:
            wb += tl.dot(wt1, e1, input_precision=INPUT_PRECISION)
        if C_ > 2 * C_CHUNK:
            wb += tl.dot(wt2, e2, input_precision=INPUT_PRECISION)
        dpr += wb + dob[:, :, None]
    return p, dpr


@triton.jit
def _col_tiles(
    k_ptr: tl.tensor,
    v_ptr: tl.tensor,
    freq_ptr: tl.tensor,
    col_off: tl.tensor,  # [N, G, CC] element offsets of the key block
    col_mask: tl.tensor,  # [N, G, 1]
    feats_j: tl.tensor,  # [N] feature-row pointers of the key block
    col_valid: tl.tensor,  # [N]
    pix: tl.tensor,
    piy: tl.tensor,
    piz: tl.tensor,
    pit: tl.tensor,  # [M] query positions
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    C_: tl.constexpr,
    C_CHUNK: tl.constexpr,
    F_: tl.constexpr,
    USE_ATTN_BIAS: tl.constexpr,
    USE_ACT_BIAS: tl.constexpr,
    CDTYPE: tl.constexpr,
    TIME_SCALE_C: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
    CLIP: tl.constexpr,
) -> tuple:
    """K, v and pair-feature chunks of one key block for a row block."""
    k0 = tl.load(k_ptr + col_off + 0 * C_CHUNK, mask=col_mask, other=0.0).to(
        CDTYPE
    )
    if C_ > C_CHUNK:
        k1 = tl.load(
            k_ptr + col_off + 1 * C_CHUNK, mask=col_mask, other=0.0
        ).to(CDTYPE)
    else:
        k1 = k0
    if C_ > 2 * C_CHUNK:
        k2 = tl.load(
            k_ptr + col_off + 2 * C_CHUNK, mask=col_mask, other=0.0
        ).to(CDTYPE)
    else:
        k2 = k0
    v0 = tl.load(v_ptr + col_off + 0 * C_CHUNK, mask=col_mask, other=0.0).to(
        CDTYPE
    )
    if C_ > C_CHUNK:
        v1 = tl.load(
            v_ptr + col_off + 1 * C_CHUNK, mask=col_mask, other=0.0
        ).to(CDTYPE)
    else:
        v1 = v0
    if C_ > 2 * C_CHUNK:
        v2 = tl.load(
            v_ptr + col_off + 2 * C_CHUNK, mask=col_mask, other=0.0
        ).to(CDTYPE)
    else:
        v2 = v0
    if USE_ATTN_BIAS or USE_ACT_BIAS:
        pjx = tl.load(feats_j + 0, mask=col_valid, other=0.0)
        pjy = tl.load(feats_j + 1, mask=col_valid, other=0.0)
        pjz = tl.load(feats_j + 2, mask=col_valid, other=0.0)
        pjt = tl.load(feats_j + 3, mask=col_valid, other=0.0)
        x = _pair_angle(
            pix,
            piy,
            piz,
            pit,
            pjx,
            pjy,
            pjz,
            pjt,
            TIME_SCALE_C,
            INPUT_SCALE,
            CLIP,
        )
        e0 = _e_chunk(x, freq_ptr, 0 * C_CHUNK, F_, C_CHUNK, CDTYPE)
        if C_ > C_CHUNK:
            e1 = _e_chunk(x, freq_ptr, 1 * C_CHUNK, F_, C_CHUNK, CDTYPE)
        else:
            e1 = e0
        if C_ > 2 * C_CHUNK:
            e2 = _e_chunk(x, freq_ptr, 2 * C_CHUNK, F_, C_CHUNK, CDTYPE)
        else:
            e2 = e0
    else:
        e0 = tl.zeros((BLOCK_M, C_CHUNK, BLOCK_N), dtype=CDTYPE)
        e1 = e0
        e2 = e0
    return k0, k1, k2, v0, v1, v2, e0, e1, e2


@triton.jit
def flash_spacetime_bwd_cols_kernel(  # noqa: C901
    # The cyclomatic count is compile-time constexpr specialization
    # (channel-chunk and bias-flag branches Triton resolves before
    # codegen), not runtime control flow.
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    v_ptr: tl.tensor,
    u_ptr: tl.tensor,
    do_ptr: tl.tensor,
    wt_ptr: tl.tensor,
    feats_ptr: tl.tensor,
    seqlen_ptr: tl.tensor,
    lse_ptr: tl.tensor,
    d_ptr: tl.tensor,  # [B, H, L] softmax row sums from the row kernel
    dob_ptr: tl.tensor,
    dk_ptr: tl.tensor,
    dv_ptr: tl.tensor,
    cu_ptr: tl.tensor,  # [B+1] token offsets (PACKED only)
    freq_ptr: tl.tensor,
    scale: float,
    L: tl.constexpr,
    H: tl.constexpr,
    FEAT_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    G_PAD: tl.constexpr,
    C_: tl.constexpr,
    C_CHUNK: tl.constexpr,
    F_: tl.constexpr,
    USE_ATTN_BIAS: tl.constexpr,
    USE_ACT_BIAS: tl.constexpr,
    PACKED: tl.constexpr,
    CDTYPE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    TIME_SCALE_C: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
    CLIP: tl.constexpr,
) -> None:
    """Column-owned backward: one CTA owns BLOCK_N key rows; writes dk, dv."""
    pid_n = tl.program_id(0)
    b = tl.program_id(1)
    n0 = pid_n * BLOCK_N

    seqlen = tl.load(seqlen_ptr + b)
    if n0 >= seqlen:
        return
    if PACKED:
        tok0 = tl.load(cu_ptr + b)
    else:
        tok0 = 0
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_g = tl.arange(0, G_PAD)
    offs_cc = tl.arange(0, C_CHUNK)
    col_valid = offs_n < seqlen
    head_live = offs_g < H

    if PACKED:
        col_off = (
            (tok0 + offs_n[:, None, None]) * H + offs_g[None, :, None]
        ) * C_ + offs_cc[None, None, :]
    else:
        col_off = (
            (b * H + offs_g[None, :, None]) * L + offs_n[:, None, None]
        ) * C_ + offs_cc[None, None, :]
    col_mask = col_valid[:, None, None] & head_live[None, :, None]
    k0 = tl.load(k_ptr + col_off + 0 * C_CHUNK, mask=col_mask, other=0.0).to(
        CDTYPE
    )
    if C_ > C_CHUNK:
        k1 = tl.load(
            k_ptr + col_off + 1 * C_CHUNK, mask=col_mask, other=0.0
        ).to(CDTYPE)
    else:
        k1 = k0
    if C_ > 2 * C_CHUNK:
        k2 = tl.load(
            k_ptr + col_off + 2 * C_CHUNK, mask=col_mask, other=0.0
        ).to(CDTYPE)
    else:
        k2 = k0
    v0 = tl.load(v_ptr + col_off + 0 * C_CHUNK, mask=col_mask, other=0.0).to(
        CDTYPE
    )
    if C_ > C_CHUNK:
        v1 = tl.load(
            v_ptr + col_off + 1 * C_CHUNK, mask=col_mask, other=0.0
        ).to(CDTYPE)
    else:
        v1 = v0
    if C_ > 2 * C_CHUNK:
        v2 = tl.load(
            v_ptr + col_off + 2 * C_CHUNK, mask=col_mask, other=0.0
        ).to(CDTYPE)

    else:
        v2 = v0
    if PACKED:
        feats_j = feats_ptr + (tok0 + offs_n) * FEAT_STRIDE
    else:
        feats_j = feats_ptr + b * L * FEAT_STRIDE + offs_n * FEAT_STRIDE
    pjx = tl.load(feats_j + 0, mask=col_valid, other=0.0)
    pjy = tl.load(feats_j + 1, mask=col_valid, other=0.0)
    pjz = tl.load(feats_j + 2, mask=col_valid, other=0.0)
    pjt = tl.load(feats_j + 3, mask=col_valid, other=0.0)

    dk0 = tl.zeros((BLOCK_N, G_PAD, C_CHUNK), dtype=tl.float32)
    if C_ > C_CHUNK:
        dk1 = tl.zeros((BLOCK_N, G_PAD, C_CHUNK), dtype=tl.float32)
    if C_ > 2 * C_CHUNK:
        dk2 = tl.zeros((BLOCK_N, G_PAD, C_CHUNK), dtype=tl.float32)
    dv0 = tl.zeros((BLOCK_N, G_PAD, C_CHUNK), dtype=tl.float32)
    if C_ > C_CHUNK:
        dv1 = tl.zeros((BLOCK_N, G_PAD, C_CHUNK), dtype=tl.float32)
    if C_ > 2 * C_CHUNK:
        dv2 = tl.zeros((BLOCK_N, G_PAD, C_CHUNK), dtype=tl.float32)

    for m0 in range(0, L, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        row_valid = offs_m < seqlen
        if m0 < seqlen:
            if PACKED:
                row_off = (
                    (tok0 + offs_m[:, None, None]) * H + offs_g[None, :, None]
                ) * C_ + offs_cc[None, None, :]
            else:
                row_off = (
                    (b * H + offs_g[None, :, None]) * L + offs_m[:, None, None]
                ) * C_ + offs_cc[None, None, :]
            row_mask = row_valid[:, None, None] & head_live[None, :, None]
            qt0 = (
                tl.load(
                    q_ptr + row_off + 0 * C_CHUNK, mask=row_mask, other=0.0
                )
                * scale
            ).to(CDTYPE)
            if C_ > C_CHUNK:
                qt1 = (
                    tl.load(
                        q_ptr + row_off + 1 * C_CHUNK, mask=row_mask, other=0.0
                    )
                    * scale
                ).to(CDTYPE)
            else:
                qt1 = qt0
            if C_ > 2 * C_CHUNK:
                qt2 = (
                    tl.load(
                        q_ptr + row_off + 2 * C_CHUNK, mask=row_mask, other=0.0
                    )
                    * scale
                ).to(CDTYPE)
            else:
                qt2 = qt0
            do0 = tl.load(
                do_ptr + row_off + 0 * C_CHUNK, mask=row_mask, other=0.0
            ).to(CDTYPE)
            if C_ > C_CHUNK:
                do1 = tl.load(
                    do_ptr + row_off + 1 * C_CHUNK, mask=row_mask, other=0.0
                ).to(CDTYPE)
            else:
                do1 = do0
            if C_ > 2 * C_CHUNK:
                do2 = tl.load(
                    do_ptr + row_off + 2 * C_CHUNK, mask=row_mask, other=0.0
                ).to(CDTYPE)
            else:
                do2 = do0
            if USE_ATTN_BIAS:
                u0 = tl.load(
                    u_ptr + row_off + 0 * C_CHUNK, mask=row_mask, other=0.0
                ).to(CDTYPE)
                if C_ > C_CHUNK:
                    u1 = tl.load(
                        u_ptr + row_off + 1 * C_CHUNK, mask=row_mask, other=0.0
                    ).to(CDTYPE)
                else:
                    u1 = u0
                if C_ > 2 * C_CHUNK:
                    u2 = tl.load(
                        u_ptr + row_off + 2 * C_CHUNK, mask=row_mask, other=0.0
                    ).to(CDTYPE)
                else:
                    u2 = u0
            else:
                u0 = qt0
                if C_ > C_CHUNK:
                    u1 = qt1
                else:
                    u1 = u0
                if C_ > 2 * C_CHUNK:
                    u2 = qt2
                else:
                    u2 = u0
            if USE_ACT_BIAS:
                wt0 = tl.load(
                    wt_ptr + row_off + 0 * C_CHUNK, mask=row_mask, other=0.0
                ).to(CDTYPE)
                if C_ > C_CHUNK:
                    wt1 = tl.load(
                        wt_ptr + row_off + 1 * C_CHUNK,
                        mask=row_mask,
                        other=0.0,
                    ).to(CDTYPE)
                else:
                    wt1 = wt0
                if C_ > 2 * C_CHUNK:
                    wt2 = tl.load(
                        wt_ptr + row_off + 2 * C_CHUNK,
                        mask=row_mask,
                        other=0.0,
                    ).to(CDTYPE)
                else:
                    wt2 = wt0
            else:
                wt0 = qt0
                if C_ > C_CHUNK:
                    wt1 = qt1
                else:
                    wt1 = wt0
                if C_ > 2 * C_CHUNK:
                    wt2 = qt2

                else:
                    wt2 = wt0
            if PACKED:
                row_vec = (tok0 + offs_m[:, None]) * H + offs_g[None, :]
            else:
                row_vec = (b * H + offs_g[None, :]) * L + offs_m[:, None]
            row_vec_mask = row_valid[:, None] & head_live[None, :]
            lse = tl.load(lse_ptr + row_vec, mask=row_vec_mask, other=0.0)
            d_row = tl.load(d_ptr + row_vec, mask=row_vec_mask, other=0.0)
            if USE_ACT_BIAS:
                dob = tl.load(dob_ptr + row_vec, mask=row_vec_mask, other=0.0)
            else:
                dob = lse * 0.0

            if PACKED:
                feats_i = feats_ptr + (tok0 + offs_m) * FEAT_STRIDE
            else:
                feats_i = (
                    feats_ptr + b * L * FEAT_STRIDE + offs_m * FEAT_STRIDE
                )
            pix = tl.load(feats_i + 0, mask=row_valid, other=0.0)
            piy = tl.load(feats_i + 1, mask=row_valid, other=0.0)
            piz = tl.load(feats_i + 2, mask=row_valid, other=0.0)
            pit = tl.load(feats_i + 3, mask=row_valid, other=0.0)

            if USE_ATTN_BIAS or USE_ACT_BIAS:
                x = _pair_angle(
                    pix,
                    piy,
                    piz,
                    pit,
                    pjx,
                    pjy,
                    pjz,
                    pjt,
                    TIME_SCALE_C,
                    INPUT_SCALE,
                    CLIP,
                )
                e0 = _e_chunk(x, freq_ptr, 0 * C_CHUNK, F_, C_CHUNK, CDTYPE)
                if C_ > C_CHUNK:
                    e1 = _e_chunk(
                        x, freq_ptr, 1 * C_CHUNK, F_, C_CHUNK, CDTYPE
                    )
                else:
                    e1 = e0
                if C_ > 2 * C_CHUNK:
                    e2 = _e_chunk(
                        x, freq_ptr, 2 * C_CHUNK, F_, C_CHUNK, CDTYPE
                    )
                else:
                    e2 = e0
            else:
                e0 = tl.zeros((BLOCK_M, C_CHUNK, BLOCK_N), dtype=CDTYPE)
                e1 = e0
                e2 = e0
            p, dpr = _recompute_p_dpr(
                qt0,
                qt1,
                qt2,
                u0,
                u1,
                u2,
                do0,
                do1,
                do2,
                k0,
                k1,
                k2,
                v0,
                v1,
                v2,
                e0,
                e1,
                e2,
                wt0,
                wt1,
                wt2,
                dob,
                lse,
                col_valid,
                C_,
                C_CHUNK,
                USE_ATTN_BIAS,
                USE_ACT_BIAS,
                INPUT_PRECISION,
            )
            ds = p * (dpr - d_row[:, :, None])
            ds = tl.where(col_valid[None, None, :], ds, 0.0)
            pcd = p.to(CDTYPE)
            dscd = ds.to(CDTYPE)

            # dv_j += P^T dO_i ; dk_j += dS^T qt_i  (per chunk, [N, G, CC]).
            pT = tl.trans(pcd, 1, 2, 0)  # [G, N, M]
            dsT = tl.trans(dscd, 1, 2, 0)
            dv0 += tl.trans(
                tl.dot(
                    pT, tl.trans(do0, 1, 0, 2), input_precision=INPUT_PRECISION
                ),
                1,
                0,
                2,
            )
            if C_ > C_CHUNK:
                dv1 += tl.trans(
                    tl.dot(
                        pT,
                        tl.trans(do1, 1, 0, 2),
                        input_precision=INPUT_PRECISION,
                    ),
                    1,
                    0,
                    2,
                )
            if C_ > 2 * C_CHUNK:
                dv2 += tl.trans(
                    tl.dot(
                        pT,
                        tl.trans(do2, 1, 0, 2),
                        input_precision=INPUT_PRECISION,
                    ),
                    1,
                    0,
                    2,
                )
            dk0 += tl.trans(
                tl.dot(
                    dsT,
                    tl.trans(qt0, 1, 0, 2),
                    input_precision=INPUT_PRECISION,
                ),
                1,
                0,
                2,
            )
            if C_ > C_CHUNK:
                dk1 += tl.trans(
                    tl.dot(
                        dsT,
                        tl.trans(qt1, 1, 0, 2),
                        input_precision=INPUT_PRECISION,
                    ),
                    1,
                    0,
                    2,
                )
            if C_ > 2 * C_CHUNK:
                dk2 += tl.trans(
                    tl.dot(
                        dsT,
                        tl.trans(qt2, 1, 0, 2),
                        input_precision=INPUT_PRECISION,
                    ),
                    1,
                    0,
                    2,
                )

    tl.store(dk_ptr + col_off + 0 * C_CHUNK, dk0, mask=col_mask)
    if C_ > C_CHUNK:
        tl.store(dk_ptr + col_off + 1 * C_CHUNK, dk1, mask=col_mask)
    if C_ > 2 * C_CHUNK:
        tl.store(dk_ptr + col_off + 2 * C_CHUNK, dk2, mask=col_mask)
    tl.store(dv_ptr + col_off + 0 * C_CHUNK, dv0, mask=col_mask)
    if C_ > C_CHUNK:
        tl.store(dv_ptr + col_off + 1 * C_CHUNK, dv1, mask=col_mask)
    if C_ > 2 * C_CHUNK:
        tl.store(dv_ptr + col_off + 2 * C_CHUNK, dv2, mask=col_mask)


@triton.jit
def flash_spacetime_bwd_rows_kernel(  # noqa: C901
    # The cyclomatic count is compile-time constexpr specialization
    # (PACKED / bias-flag branches Triton resolves before codegen), not
    # runtime control flow.
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    v_ptr: tl.tensor,
    u_ptr: tl.tensor,
    do_ptr: tl.tensor,
    wt_ptr: tl.tensor,
    feats_ptr: tl.tensor,
    seqlen_ptr: tl.tensor,
    lse_ptr: tl.tensor,
    d_ptr: tl.tensor,  # [B, H, L] softmax row sums, written here
    dob_ptr: tl.tensor,
    dqp_ptr: tl.tensor,
    hacc_ptr: tl.tensor,
    gacc_ptr: tl.tensor,
    sig_ptr: tl.tensor,
    cu_ptr: tl.tensor,  # [B+1] token offsets (PACKED only)
    freq_ptr: tl.tensor,
    scale: float,
    L: tl.constexpr,
    H: tl.constexpr,
    FEAT_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    G_PAD: tl.constexpr,
    C_: tl.constexpr,
    C_CHUNK: tl.constexpr,
    F_: tl.constexpr,
    USE_ATTN_BIAS: tl.constexpr,
    USE_ACT_BIAS: tl.constexpr,
    PACKED: tl.constexpr,
    CDTYPE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    TIME_SCALE_C: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
    CLIP: tl.constexpr,
) -> None:
    """Row-owned backward: D, then dq's dS@k part plus H, G, sigma."""
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    m0 = pid_m * BLOCK_M

    seqlen = tl.load(seqlen_ptr + b)
    if m0 >= seqlen:
        return
    if PACKED:
        tok0 = tl.load(cu_ptr + b)
    else:
        tok0 = 0
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_g = tl.arange(0, G_PAD)
    offs_cc = tl.arange(0, C_CHUNK)
    row_valid = offs_m < seqlen
    head_live = offs_g < H

    if PACKED:
        row_off = (
            (tok0 + offs_m[:, None, None]) * H + offs_g[None, :, None]
        ) * C_ + offs_cc[None, None, :]
    else:
        row_off = (
            (b * H + offs_g[None, :, None]) * L + offs_m[:, None, None]
        ) * C_ + offs_cc[None, None, :]
    row_mask = row_valid[:, None, None] & head_live[None, :, None]
    qt0 = (
        tl.load(q_ptr + row_off + 0 * C_CHUNK, mask=row_mask, other=0.0)
        * scale
    ).to(CDTYPE)
    if C_ > C_CHUNK:
        qt1 = (
            tl.load(q_ptr + row_off + 1 * C_CHUNK, mask=row_mask, other=0.0)
            * scale
        ).to(CDTYPE)
    else:
        qt1 = qt0
    if C_ > 2 * C_CHUNK:
        qt2 = (
            tl.load(q_ptr + row_off + 2 * C_CHUNK, mask=row_mask, other=0.0)
            * scale
        ).to(CDTYPE)
    else:
        qt2 = qt0
    do0 = tl.load(do_ptr + row_off + 0 * C_CHUNK, mask=row_mask, other=0.0).to(
        CDTYPE
    )
    if C_ > C_CHUNK:
        do1 = tl.load(
            do_ptr + row_off + 1 * C_CHUNK, mask=row_mask, other=0.0
        ).to(CDTYPE)
    else:
        do1 = do0
    if C_ > 2 * C_CHUNK:
        do2 = tl.load(
            do_ptr + row_off + 2 * C_CHUNK, mask=row_mask, other=0.0
        ).to(CDTYPE)
    else:
        do2 = do0
    if USE_ATTN_BIAS:
        u0 = tl.load(
            u_ptr + row_off + 0 * C_CHUNK, mask=row_mask, other=0.0
        ).to(CDTYPE)
        if C_ > C_CHUNK:
            u1 = tl.load(
                u_ptr + row_off + 1 * C_CHUNK, mask=row_mask, other=0.0
            ).to(CDTYPE)
        else:
            u1 = u0
        if C_ > 2 * C_CHUNK:
            u2 = tl.load(
                u_ptr + row_off + 2 * C_CHUNK, mask=row_mask, other=0.0
            ).to(CDTYPE)
        else:
            u2 = u0
    else:
        u0 = qt0
        if C_ > C_CHUNK:
            u1 = qt1
        else:
            u1 = u0
        if C_ > 2 * C_CHUNK:
            u2 = qt2
        else:
            u2 = u0
    if USE_ACT_BIAS:
        wt0 = tl.load(
            wt_ptr + row_off + 0 * C_CHUNK, mask=row_mask, other=0.0
        ).to(CDTYPE)
        if C_ > C_CHUNK:
            wt1 = tl.load(
                wt_ptr + row_off + 1 * C_CHUNK, mask=row_mask, other=0.0
            ).to(CDTYPE)
        else:
            wt1 = wt0
        if C_ > 2 * C_CHUNK:
            wt2 = tl.load(
                wt_ptr + row_off + 2 * C_CHUNK, mask=row_mask, other=0.0
            ).to(CDTYPE)
        else:
            wt2 = wt0
    else:
        wt0 = qt0
        if C_ > C_CHUNK:
            wt1 = qt1
        else:
            wt1 = wt0
        if C_ > 2 * C_CHUNK:
            wt2 = qt2

        else:
            wt2 = wt0
    if PACKED:
        row_vec = (tok0 + offs_m[:, None]) * H + offs_g[None, :]
    else:
        row_vec = (b * H + offs_g[None, :]) * L + offs_m[:, None]
    row_vec_mask = row_valid[:, None] & head_live[None, :]
    lse = tl.load(lse_ptr + row_vec, mask=row_vec_mask, other=0.0)
    if USE_ACT_BIAS:
        dob = tl.load(dob_ptr + row_vec, mask=row_vec_mask, other=0.0)
    else:
        dob = lse * 0.0

    if PACKED:
        feats_i = feats_ptr + (tok0 + offs_m) * FEAT_STRIDE
    else:
        feats_i = feats_ptr + b * L * FEAT_STRIDE + offs_m * FEAT_STRIDE
    pix = tl.load(feats_i + 0, mask=row_valid, other=0.0)
    piy = tl.load(feats_i + 1, mask=row_valid, other=0.0)
    piz = tl.load(feats_i + 2, mask=row_valid, other=0.0)
    pit = tl.load(feats_i + 3, mask=row_valid, other=0.0)

    # First pass: the softmax row sum from the same P and dP that the second
    # pass turns into dS, so that rowsum(dS) is zero identically. It is the
    # P-weighted mean of dP, normalised by the recomputed weights rather
    # than assumed to have unit mass: at large logits the fp32 LSE from the
    # forward and the logits recomputed here differ by order one, which
    # scales every P of a row by the same factor, and the mean is invariant
    # to that where the bare sum is not.
    dsum = tl.zeros((BLOCK_M, G_PAD), dtype=tl.float32)
    psum = tl.zeros((BLOCK_M, G_PAD), dtype=tl.float32)
    for n0 in range(0, L, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        col_valid = offs_n < seqlen
        if n0 < seqlen:
            if PACKED:
                col_off = (
                    (tok0 + offs_n[:, None, None]) * H + offs_g[None, :, None]
                ) * C_ + offs_cc[None, None, :]
                feats_j = feats_ptr + (tok0 + offs_n) * FEAT_STRIDE
            else:
                col_off = (
                    (b * H + offs_g[None, :, None]) * L + offs_n[:, None, None]
                ) * C_ + offs_cc[None, None, :]
                feats_j = (
                    feats_ptr + b * L * FEAT_STRIDE + offs_n * FEAT_STRIDE
                )
            col_mask = col_valid[:, None, None] & head_live[None, :, None]
            k0, k1, k2, v0, v1, v2, e0, e1, e2 = _col_tiles(
                k_ptr,
                v_ptr,
                freq_ptr,
                col_off,
                col_mask,
                feats_j,
                col_valid,
                pix,
                piy,
                piz,
                pit,
                BLOCK_M,
                BLOCK_N,
                C_,
                C_CHUNK,
                F_,
                USE_ATTN_BIAS,
                USE_ACT_BIAS,
                CDTYPE,
                TIME_SCALE_C,
                INPUT_SCALE,
                CLIP,
            )
            p, dpr = _recompute_p_dpr(
                qt0,
                qt1,
                qt2,
                u0,
                u1,
                u2,
                do0,
                do1,
                do2,
                k0,
                k1,
                k2,
                v0,
                v1,
                v2,
                e0,
                e1,
                e2,
                wt0,
                wt1,
                wt2,
                dob,
                lse,
                col_valid,
                C_,
                C_CHUNK,
                USE_ATTN_BIAS,
                USE_ACT_BIAS,
                INPUT_PRECISION,
            )
            dsum += tl.sum(p * dpr, axis=2)
            psum += tl.sum(p, axis=2)
    d_row = tl.where(psum > 0.0, dsum / psum, 0.0)
    tl.store(d_ptr + row_vec, d_row, mask=row_vec_mask)

    dqp0 = tl.zeros((BLOCK_M, G_PAD, C_CHUNK), dtype=tl.float32)
    if C_ > C_CHUNK:
        dqp1 = tl.zeros((BLOCK_M, G_PAD, C_CHUNK), dtype=tl.float32)
    if C_ > 2 * C_CHUNK:
        dqp2 = tl.zeros((BLOCK_M, G_PAD, C_CHUNK), dtype=tl.float32)
    if USE_ATTN_BIAS:
        h0 = tl.zeros((BLOCK_M, C_CHUNK, G_PAD), dtype=tl.float32)
        if C_ > C_CHUNK:
            h1 = tl.zeros((BLOCK_M, C_CHUNK, G_PAD), dtype=tl.float32)
        if C_ > 2 * C_CHUNK:
            h2 = tl.zeros((BLOCK_M, C_CHUNK, G_PAD), dtype=tl.float32)
        sig = tl.zeros((BLOCK_M, G_PAD), dtype=tl.float32)
    if USE_ACT_BIAS:
        g0 = tl.zeros((BLOCK_M, C_CHUNK, G_PAD), dtype=tl.float32)
        if C_ > C_CHUNK:
            g1 = tl.zeros((BLOCK_M, C_CHUNK, G_PAD), dtype=tl.float32)
        if C_ > 2 * C_CHUNK:
            g2 = tl.zeros((BLOCK_M, C_CHUNK, G_PAD), dtype=tl.float32)

    for n0 in range(0, L, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        col_valid = offs_n < seqlen
        if n0 < seqlen:
            if PACKED:
                col_off = (
                    (tok0 + offs_n[:, None, None]) * H + offs_g[None, :, None]
                ) * C_ + offs_cc[None, None, :]
                feats_j = feats_ptr + (tok0 + offs_n) * FEAT_STRIDE
            else:
                col_off = (
                    (b * H + offs_g[None, :, None]) * L + offs_n[:, None, None]
                ) * C_ + offs_cc[None, None, :]
                feats_j = (
                    feats_ptr + b * L * FEAT_STRIDE + offs_n * FEAT_STRIDE
                )
            col_mask = col_valid[:, None, None] & head_live[None, :, None]
            k0, k1, k2, v0, v1, v2, e0, e1, e2 = _col_tiles(
                k_ptr,
                v_ptr,
                freq_ptr,
                col_off,
                col_mask,
                feats_j,
                col_valid,
                pix,
                piy,
                piz,
                pit,
                BLOCK_M,
                BLOCK_N,
                C_,
                C_CHUNK,
                F_,
                USE_ATTN_BIAS,
                USE_ACT_BIAS,
                CDTYPE,
                TIME_SCALE_C,
                INPUT_SCALE,
                CLIP,
            )
            p, dpr = _recompute_p_dpr(
                qt0,
                qt1,
                qt2,
                u0,
                u1,
                u2,
                do0,
                do1,
                do2,
                k0,
                k1,
                k2,
                v0,
                v1,
                v2,
                e0,
                e1,
                e2,
                wt0,
                wt1,
                wt2,
                dob,
                lse,
                col_valid,
                C_,
                C_CHUNK,
                USE_ATTN_BIAS,
                USE_ACT_BIAS,
                INPUT_PRECISION,
            )
            ds = p * (dpr - d_row[:, :, None])
            ds = tl.where(col_valid[None, None, :], ds, 0.0)
            pcd = p.to(CDTYPE)
            dscd = ds.to(CDTYPE)
            dsT = tl.trans(dscd, 1, 0, 2)  # [G, M, N]

            # dqp += dS @ k (per chunk).
            dqp0 += tl.trans(
                tl.dot(
                    dsT, tl.trans(k0, 1, 0, 2), input_precision=INPUT_PRECISION
                ),
                1,
                0,
                2,
            )
            if C_ > C_CHUNK:
                dqp1 += tl.trans(
                    tl.dot(
                        dsT,
                        tl.trans(k1, 1, 0, 2),
                        input_precision=INPUT_PRECISION,
                    ),
                    1,
                    0,
                    2,
                )
            if C_ > 2 * C_CHUNK:
                dqp2 += tl.trans(
                    tl.dot(
                        dsT,
                        tl.trans(k2, 1, 0, 2),
                        input_precision=INPUT_PRECISION,
                    ),
                    1,
                    0,
                    2,
                )
            if USE_ATTN_BIAS:
                # H += dS·E (batched over rows), sigma += rowsum(dS).
                dst_c = tl.trans(dscd, 0, 2, 1)  # [M, N, G]
                h0 += tl.dot(e0, dst_c, input_precision=INPUT_PRECISION)
                if C_ > C_CHUNK:
                    h1 += tl.dot(e1, dst_c, input_precision=INPUT_PRECISION)
                if C_ > 2 * C_CHUNK:
                    h2 += tl.dot(e2, dst_c, input_precision=INPUT_PRECISION)
                sig += tl.sum(ds, axis=2)
            if USE_ACT_BIAS:
                pct_c = tl.trans(pcd, 0, 2, 1)
                g0 += tl.dot(e0, pct_c, input_precision=INPUT_PRECISION)
                if C_ > C_CHUNK:
                    g1 += tl.dot(e1, pct_c, input_precision=INPUT_PRECISION)
                if C_ > 2 * C_CHUNK:
                    g2 += tl.dot(e2, pct_c, input_precision=INPUT_PRECISION)

    tl.store(dqp_ptr + row_off + 0 * C_CHUNK, dqp0, mask=row_mask)
    if C_ > C_CHUNK:
        tl.store(dqp_ptr + row_off + 1 * C_CHUNK, dqp1, mask=row_mask)
    if C_ > 2 * C_CHUNK:
        tl.store(dqp_ptr + row_off + 2 * C_CHUNK, dqp2, mask=row_mask)
    if USE_ATTN_BIAS:
        tl.store(
            hacc_ptr + row_off + 0 * C_CHUNK,
            tl.trans(h0, 0, 2, 1),
            mask=row_mask,
        )
        if C_ > C_CHUNK:
            tl.store(
                hacc_ptr + row_off + 1 * C_CHUNK,
                tl.trans(h1, 0, 2, 1),
                mask=row_mask,
            )
        if C_ > 2 * C_CHUNK:
            tl.store(
                hacc_ptr + row_off + 2 * C_CHUNK,
                tl.trans(h2, 0, 2, 1),
                mask=row_mask,
            )
        tl.store(sig_ptr + row_vec, sig, mask=row_vec_mask)
    if USE_ACT_BIAS:
        tl.store(
            gacc_ptr + row_off + 0 * C_CHUNK,
            tl.trans(g0, 0, 2, 1),
            mask=row_mask,
        )
        if C_ > C_CHUNK:
            tl.store(
                gacc_ptr + row_off + 1 * C_CHUNK,
                tl.trans(g1, 0, 2, 1),
                mask=row_mask,
            )
        if C_ > 2 * C_CHUNK:
            tl.store(
                gacc_ptr + row_off + 2 * C_CHUNK,
                tl.trans(g2, 0, 2, 1),
                mask=row_mask,
            )


def flash_spacetime_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    feats: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    seqlens: Tensor,
    lse: Tensor,
    grad_out: Tensor,
    scale: Optional[float] = None,
    use_attn_bias: bool = True,
    use_activation_bias: bool = True,
    block_m: int = 16,
    block_n: int = 16,
    num_warps: int = 8,
    num_stages: int = 1,
    cu_seqlens: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Deterministic backward; returns (dq, dk, dv, dW, db).

    With `cu_seqlens` the tensors are packed [T, H, D] (see the forward);
    every row is a real token, so no upstream-gradient masking exists.
    """
    packed = cu_seqlens is not None
    if cu_seqlens is not None:
        cu = cu_seqlens.to(torch.int32).contiguous()
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        batch = int(seqlens.numel())
        _, heads, dim = q.shape
        length = _next_pow2(int(seqlens.max()), floor=16)
        do = grad_out.contiguous()
    else:
        batch, heads, length, dim = q.shape
    scale_value = dim**-0.5 if scale is None else scale
    compute_bf16 = q.dtype != torch.float32
    fp = torch.float32

    if not packed:
        valid = torch.arange(length, device=q.device).unsqueeze(
            0
        ) < seqlens.unsqueeze(1)
        # The op's pad-row outputs are the constant zero, so upstream
        # grads there are discardable regardless of caller garbage.
        do = (grad_out * valid[:, None, :, None]).contiguous()
    qc, kc, vc = (t.contiguous() for t in (q, k, v))
    featsc = feats[..., :4].to(torch.float32).contiguous()
    freqs = sinusoidal_frequencies(dim, q.device)
    w32 = weight.to(fp)

    # Softmax row sums, written by the row kernel and read by the column
    # kernel; same layout as the LSE.
    dsum = torch.zeros_like(lse)
    u = (
        ((qc.to(fp) * scale_value) @ w32).to(qc.dtype).contiguous()
        if use_attn_bias
        else qc
    )
    if use_activation_bias:
        wt = (do.to(fp) @ w32).to(do.dtype).contiguous()
        dob = (
            do.to(fp) @ bias.to(fp)
            if bias is not None
            else torch.zeros_like(lse)
        )
    else:
        wt = qc
        dob = lse  # dummy pointer, never read
    dob = dob.contiguous()

    dk = torch.zeros_like(kc, dtype=fp)
    dv = torch.zeros_like(vc, dtype=fp)
    dqp = torch.zeros_like(qc, dtype=fp)
    hacc = torch.zeros_like(qc, dtype=fp) if use_attn_bias else dqp
    gacc = torch.zeros_like(qc, dtype=fp) if use_activation_bias else dqp
    sig = torch.zeros_like(lse) if use_attn_bias else lse

    seq32 = seqlens.to(torch.int32).contiguous()
    cu32 = cu if packed else seq32
    common = dict(
        scale=scale_value,
        L=length,
        H=heads,
        FEAT_STRIDE=4,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        G_PAD=max(16, 1 << (heads - 1).bit_length()),
        C_=dim,
        C_CHUNK=16,
        F_=dim // 2,
        USE_ATTN_BIAS=use_attn_bias,
        USE_ACT_BIAS=use_activation_bias,
        PACKED=packed,
        CDTYPE=tl.bfloat16 if compute_bf16 else tl.float32,
        INPUT_PRECISION="ieee",
        TIME_SCALE_C=TIME_SCALE,
        INPUT_SCALE=SINEMB_INPUT_SCALE,
        CLIP=SINEMB_CLIP,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    flash_spacetime_bwd_rows_kernel[(triton.cdiv(length, block_m), batch)](
        qc,
        kc,
        vc,
        u,
        do,
        wt,
        featsc,
        seq32,
        lse,
        dsum,
        dob,
        dqp,
        hacc,
        gacc,
        sig,
        cu32,
        freqs,
        **common,
    )
    flash_spacetime_bwd_cols_kernel[(triton.cdiv(length, block_n), batch)](
        qc,
        kc,
        vc,
        u,
        do,
        wt,
        featsc,
        seq32,
        lse,
        dsum,
        dob,
        dk,
        dv,
        cu32,
        freqs,
        **common,
    )

    dq = dqp
    if use_attn_bias:
        dq = dq + hacc @ w32.t()
        if bias is not None:
            dq = dq + sig.unsqueeze(-1) * bias.to(fp)
    dq = dq * scale_value

    dw = torch.zeros_like(w32)
    if use_attn_bias:
        qt_flat = (qc.to(fp) * scale_value).reshape(-1, dim)
        dw = dw + qt_flat.t() @ hacc.reshape(-1, dim)
    if use_activation_bias:
        dw = dw + do.to(fp).reshape(-1, dim).t() @ gacc.reshape(-1, dim)
    # dW_ce = sum dR_c E_e: the row side (qt/dO) carries the dR channel
    # index c, the accumulators carry the E index e.
    db: Optional[Tensor] = None
    if bias is not None:
        if use_activation_bias:
            db = do.to(fp).reshape(-1, dim).sum(0)
        else:
            # The logit path's db vanishes analytically (softmax shift
            # invariance).
            db = torch.zeros(dim, dtype=fp, device=q.device)

    return (
        dq.to(q.dtype),
        dk.to(k.dtype),
        dv.to(v.dtype),
        dw.to(weight.dtype),
        db.to(bias.dtype) if (db is not None and bias is not None) else db,
    )
