"""Triton forward kernel for fused spacetime-bias attention.

Implements the op boundary of `flash_spacetime.spacetime_attention_reference`
as a flash-style kernel: the per-pair feature `R_ij` is never materialised.
The `SpacetimeEncoder` projection is folded out of the kernel entirely
(DESIGN.md §6): every use of `R` is a channel contraction, so

- the pre-softmax bias `sum_c qt_c R_ijc` becomes `sum_e u_e E_ije` with the
  host-side GEMM `u = (q * scale) @ W` (the row-constant `qt·b` term shifts
  each softmax row uniformly and is dropped — exact);
- the value-side output `sum_j P_ij R_ij` is accumulated as `A' = sum_j P E_j`
  in-kernel and mapped through `A' @ W^T + b` on the host (`sum_j P = 1` on
  every valid row after flash normalisation).

The kernel therefore only ever builds sin/cos panels of the pair angle
`x_ij = 1024 * clip(sign(I) sqrt(|I|), -4, 4)` in `C_CHUNK`-wide channel
chunks (Triton block shapes must be powers of two; three chunks of 16 cover
C = 48 with no waste).

Padding contract: outputs and LSE are written 0 on padding rows; valid rows
match eager exactly (one-sided pairs masked with -inf; the eager both-padding
quirk affects only padding rows and so is not replicated). Heads are padded
in-register to `G_PAD` (next power of two >= H, min 16 for `tl.dot`).

The backward runs the deterministic two-kernel Triton scheme in
`flash_spacetime_triton_bwd` (set FLASH_ST_REFERENCE_BWD=1 to fall back to
the exact autograd-through-the-reference path, which materialises `R`).
"""

import os
from typing import Any, List, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from graphnet.models.components.flash_spacetime import (
    SINEMB_CLIP,
    SINEMB_INPUT_SCALE,
    TIME_SCALE,
    sinusoidal_frequencies,
    spacetime_attention_reference,
)


@triton.jit
def _pair_angle(
    px: tl.tensor,
    py: tl.tensor,
    pz: tl.tensor,
    pt: tl.tensor,  # [M] query-row coordinates
    qx: tl.tensor,
    qy: tl.tensor,
    qz: tl.tensor,
    qt_: tl.tensor,  # [N] key-column coordinates
    TIME_SCALE_C: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
    CLIP: tl.constexpr,
) -> tl.tensor:
    """`x_ij` of SpacetimeEncoder for one (M, N) tile, fp32."""
    dx = px[:, None] - qx[None, :]
    dy = py[:, None] - qy[None, :]
    dz = pz[:, None] - qz[None, :]
    dt = (pt[:, None] - qt_[None, :]) * TIME_SCALE_C
    interval = dx * dx + dy * dy + dz * dz - dt * dt
    # sign(I) * sqrt(|I|); sign(0) = 0 is preserved by the where.
    dist = tl.where(
        interval > 0.0,
        tl.sqrt(interval),
        -tl.sqrt(-interval),
    )
    dist = tl.where(interval == 0.0, 0.0, dist)
    return INPUT_SCALE * tl.minimum(tl.maximum(dist, -CLIP), CLIP)


@triton.jit
def _e_chunk(
    x: tl.tensor,  # [M, N] pair angle
    freq_ptr: tl.tensor,
    BASE: tl.constexpr,
    F_: tl.constexpr,
    C_CHUNK: tl.constexpr,
    CDTYPE: tl.constexpr,
) -> tl.tensor:
    """One channel chunk of E: sin for gc < F, cos above -> [M, CC, N]."""
    gc = BASE + tl.arange(0, C_CHUNK)
    f = tl.load(freq_ptr + gc % F_)
    # Channel-major layout ([M, CC, N]): every logit-side dot consumes E in
    # this orientation, so no E tensor is ever transposed; the value-side
    # dots transpose the much smaller P tile instead.
    theta = x[:, None, :] * f[None, :, None]
    return tl.where((gc < F_)[None, :, None], tl.sin(theta), tl.cos(theta)).to(
        CDTYPE
    )


@triton.jit
def flash_spacetime_fwd_kernel(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    v_ptr: tl.tensor,
    u_ptr: tl.tensor,  # [B, H, L, C] (u may be dummy when no [A])
    feats_ptr: tl.tensor,  # [B, L, F_STRIDE]
    seqlen_ptr: tl.tensor,  # [B] int32 valid lengths
    o1_ptr: tl.tensor,
    ae_ptr: tl.tensor,
    lse_ptr: tl.tensor,  # [B, H, L, C], [B, H, L, C], [B, H, L]
    freq_ptr: tl.tensor,  # [F]
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
    SKIP_EMPTY_TILES: tl.constexpr,
    CDTYPE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    TIME_SCALE_C: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
    CLIP: tl.constexpr,
) -> None:
    """One CTA: BLOCK_M query rows of one batch element, all heads.

    Every channel-carrying tensor lives in three C_CHUNK-wide chunks
    (3 * C_CHUNK == C): Triton block shapes must be powers of two, the
    16-wide dot slices keep shared-memory staging small enough for the
    fp32 path, and no padded channel is ever loaded or computed.
    """
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    m0 = pid_m * BLOCK_M

    seqlen = tl.load(seqlen_ptr + b)
    # Whole-block early exit: outputs are zero-initialised on the host, so
    # row blocks entirely inside the padding write nothing at all. Without
    # this, long-L padded batches pay O(L * max_len) dead tile loops.
    if m0 >= seqlen:
        return
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_g = tl.arange(0, G_PAD)
    offs_cc = tl.arange(0, C_CHUNK)
    row_valid = offs_m < seqlen
    head_live = offs_g < H

    # Chunked [M, G, CC] loads from the [B, H, L, C] layout; dead heads and
    # padding rows load 0 (their outputs are zero-stored at the end).
    row_off = (
        (b * H + offs_g[None, :, None]) * L + offs_m[:, None, None]
    ) * C_ + offs_cc[None, None, :]
    row_mask = row_valid[:, None, None] & head_live[None, :, None]
    qt0 = (
        tl.load(q_ptr + row_off + 0 * C_CHUNK, mask=row_mask, other=0.0)
        * scale
    ).to(CDTYPE)
    qt1 = (
        tl.load(q_ptr + row_off + 1 * C_CHUNK, mask=row_mask, other=0.0)
        * scale
    ).to(CDTYPE)
    qt2 = (
        tl.load(q_ptr + row_off + 2 * C_CHUNK, mask=row_mask, other=0.0)
        * scale
    ).to(CDTYPE)
    if USE_ATTN_BIAS:
        u0 = tl.load(
            u_ptr + row_off + 0 * C_CHUNK, mask=row_mask, other=0.0
        ).to(CDTYPE)
        u1 = tl.load(
            u_ptr + row_off + 1 * C_CHUNK, mask=row_mask, other=0.0
        ).to(CDTYPE)
        u2 = tl.load(
            u_ptr + row_off + 2 * C_CHUNK, mask=row_mask, other=0.0
        ).to(CDTYPE)

    feats_i = feats_ptr + b * L * FEAT_STRIDE + offs_m * FEAT_STRIDE
    fm = row_valid
    pix = tl.load(feats_i + 0, mask=fm, other=0.0)
    piy = tl.load(feats_i + 1, mask=fm, other=0.0)
    piz = tl.load(feats_i + 2, mask=fm, other=0.0)
    pit = tl.load(feats_i + 3, mask=fm, other=0.0)

    acc0 = tl.zeros((BLOCK_M, G_PAD, C_CHUNK), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, G_PAD, C_CHUNK), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, G_PAD, C_CHUNK), dtype=tl.float32)
    if USE_ACT_BIAS:
        # Held transposed ([M, CC, G]) to match channel-major E; the
        # epilogue store flips them back once.
        ae0 = tl.zeros((BLOCK_M, C_CHUNK, G_PAD), dtype=tl.float32)
        ae1 = tl.zeros((BLOCK_M, C_CHUNK, G_PAD), dtype=tl.float32)
        ae2 = tl.zeros((BLOCK_M, C_CHUNK, G_PAD), dtype=tl.float32)
    m_run = tl.full((BLOCK_M, G_PAD), float("-inf"), dtype=tl.float32)
    l_run = tl.zeros((BLOCK_M, G_PAD), dtype=tl.float32)

    for n0 in range(0, L, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        col_valid = offs_n < seqlen
        # Front-packed padding: a tile has valid keys iff its first column
        # is valid. The skip is perf-only (col_valid masking already zeroes
        # empty tiles' contributions), so it can be compiled out.
        if (not SKIP_EMPTY_TILES) or n0 < seqlen:
            col_off = (
                (b * H + offs_g[None, :, None]) * L + offs_n[:, None, None]
            ) * C_ + offs_cc[None, None, :]
            col_mask = col_valid[:, None, None] & head_live[None, :, None]
            k0 = tl.load(
                k_ptr + col_off + 0 * C_CHUNK, mask=col_mask, other=0.0
            ).to(
                CDTYPE
            )  # [N, G, CC]
            k1 = tl.load(
                k_ptr + col_off + 1 * C_CHUNK, mask=col_mask, other=0.0
            ).to(CDTYPE)
            k2 = tl.load(
                k_ptr + col_off + 2 * C_CHUNK, mask=col_mask, other=0.0
            ).to(CDTYPE)

            # QK per head, chunked over channels:
            # [G, M, CC] @ [G, CC, N] accumulated over the three chunks.
            s = tl.dot(
                tl.trans(qt0, 1, 0, 2),
                tl.trans(k0, 1, 2, 0),
                input_precision=INPUT_PRECISION,
            )
            s += tl.dot(
                tl.trans(qt1, 1, 0, 2),
                tl.trans(k1, 1, 2, 0),
                input_precision=INPUT_PRECISION,
            )
            s += tl.dot(
                tl.trans(qt2, 1, 0, 2),
                tl.trans(k2, 1, 2, 0),
                input_precision=INPUT_PRECISION,
            )  # [G, M, N] fp32 accum

            if USE_ATTN_BIAS or USE_ACT_BIAS:
                feats_j = (
                    feats_ptr + b * L * FEAT_STRIDE + offs_n * FEAT_STRIDE
                )
                cm = col_valid
                pjx = tl.load(feats_j + 0, mask=cm, other=0.0)
                pjy = tl.load(feats_j + 1, mask=cm, other=0.0)
                pjz = tl.load(feats_j + 2, mask=cm, other=0.0)
                pjt = tl.load(feats_j + 3, mask=cm, other=0.0)
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
                )  # [M, N]
                e0 = _e_chunk(x, freq_ptr, 0 * C_CHUNK, F_, C_CHUNK, CDTYPE)
                e1 = _e_chunk(x, freq_ptr, 1 * C_CHUNK, F_, C_CHUNK, CDTYPE)
                e2 = _e_chunk(x, freq_ptr, 2 * C_CHUNK, F_, C_CHUNK, CDTYPE)

            if USE_ATTN_BIAS:
                # sum_e u_e E_e, batched over rows: [M,G,CC] @ [M,CC,N].
                sb = tl.dot(u0, e0, input_precision=INPUT_PRECISION)
                sb += tl.dot(u1, e1, input_precision=INPUT_PRECISION)
                sb += tl.dot(u2, e2, input_precision=INPUT_PRECISION)
                s += tl.trans(sb, 1, 0, 2)  # [M,G,N] -> [G,M,N]

            # One-sided masking only: this CTA's valid rows never see the
            # both-padding case, and padding rows are zero-stored.
            s = tl.where(col_valid[None, None, :], s, float("-inf"))
            s = tl.trans(s, 1, 0, 2)  # [M, G, N]

            m_new = tl.maximum(m_run, tl.max(s, axis=2))
            # Rows with no valid key never occur (diagonal lemma); still,
            # guard exp against -inf - -inf on fully-dead padded rows.
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, :, None])
            p = tl.where(col_valid[None, None, :], p, 0.0)
            alpha = tl.where(
                m_run == float("-inf"), 0.0, tl.exp(m_run - m_safe)
            )
            l_run = l_run * alpha + tl.sum(p, axis=2)
            m_run = m_new
            pb = p.to(CDTYPE)
            pbt = tl.trans(pb, 1, 0, 2)  # [G, M, N]

            v0 = tl.load(
                v_ptr + col_off + 0 * C_CHUNK, mask=col_mask, other=0.0
            ).to(CDTYPE)
            v1 = tl.load(
                v_ptr + col_off + 1 * C_CHUNK, mask=col_mask, other=0.0
            ).to(CDTYPE)
            v2 = tl.load(
                v_ptr + col_off + 2 * C_CHUNK, mask=col_mask, other=0.0
            ).to(CDTYPE)

            # P@V per head, chunked: [G, M, N] @ [G, N, CC] -> back to
            # [M, G, CC].
            acc0 = acc0 * alpha[:, :, None] + tl.trans(
                tl.dot(
                    pbt,
                    tl.trans(v0, 1, 0, 2),
                    input_precision=INPUT_PRECISION,
                ),
                1,
                0,
                2,
            )
            acc1 = acc1 * alpha[:, :, None] + tl.trans(
                tl.dot(
                    pbt,
                    tl.trans(v1, 1, 0, 2),
                    input_precision=INPUT_PRECISION,
                ),
                1,
                0,
                2,
            )
            acc2 = acc2 * alpha[:, :, None] + tl.trans(
                tl.dot(
                    pbt,
                    tl.trans(v2, 1, 0, 2),
                    input_precision=INPUT_PRECISION,
                ),
                1,
                0,
                2,
            )
            if USE_ACT_BIAS:
                pbt_c = tl.trans(pb, 0, 2, 1)  # [M, N, G]
                ae0 = ae0 * alpha[:, None, :] + tl.dot(
                    e0, pbt_c, input_precision=INPUT_PRECISION
                )
                ae1 = ae1 * alpha[:, None, :] + tl.dot(
                    e1, pbt_c, input_precision=INPUT_PRECISION
                )
                ae2 = ae2 * alpha[:, None, :] + tl.dot(
                    e2, pbt_c, input_precision=INPUT_PRECISION
                )

    l_safe = tl.where(l_run == 0.0, 1.0, l_run)
    inv_l = 1.0 / l_safe[:, :, None]
    tl.store(o1_ptr + row_off + 0 * C_CHUNK, acc0 * inv_l, mask=row_mask)
    tl.store(o1_ptr + row_off + 1 * C_CHUNK, acc1 * inv_l, mask=row_mask)
    tl.store(o1_ptr + row_off + 2 * C_CHUNK, acc2 * inv_l, mask=row_mask)
    if USE_ACT_BIAS:
        tl.store(
            ae_ptr + row_off + 0 * C_CHUNK,
            tl.trans(ae0, 0, 2, 1) * inv_l,
            mask=row_mask,
        )
        tl.store(
            ae_ptr + row_off + 1 * C_CHUNK,
            tl.trans(ae1, 0, 2, 1) * inv_l,
            mask=row_mask,
        )
        tl.store(
            ae_ptr + row_off + 2 * C_CHUNK,
            tl.trans(ae2, 0, 2, 1) * inv_l,
            mask=row_mask,
        )
    lse = m_run + tl.log(l_safe)
    lse_off = (b * H + offs_g[None, :]) * L + offs_m[:, None]
    tl.store(
        lse_ptr + lse_off,
        lse,
        mask=row_valid[:, None] & head_live[None, :],
    )


def _next_pow2(n: int, floor: int = 16) -> int:
    p = floor
    while p < n:
        p *= 2
    return p


def flash_spacetime_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    feats: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    seqlens: Tensor,
    scale: Optional[float] = None,
    use_attn_bias: bool = True,
    use_activation_bias: bool = True,
    block_m: int = 16,
    block_n: Optional[int] = None,
    num_warps: int = 8,
    num_stages: int = 1,
    skip_empty_tiles: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Fused forward. Returns (O [B,H,L,D] with pad rows zeroed, LSE).

    Inputs follow `spacetime_attention_reference`; `seqlens` is the [B]
    int tensor of valid lengths (front-packed padding assumed).
    """
    batch, heads, length, dim = q.shape
    c = weight.shape[1]
    if c != dim:
        raise ValueError(f"C (={c}) must equal head_dim (={dim})")
    if c != 48:
        # The kernel unrolls exactly three C_CHUNK=16 channel chunks (the
        # production head size). Other widths need a re-unrolled kernel.
        raise ValueError(f"kernel supports C = 48 only, got {c}")
    scale_value = dim**-0.5 if scale is None else scale

    # fp32 inputs compute in true fp32 (ieee dots) to honour the
    # 2x-eager-error contract; bf16 inputs use bf16 tensor-core math. fp32
    # operands double the tile staging, so the key-block shrinks.
    compute_bf16 = q.dtype != torch.float32
    if block_n is None:
        block_n = 32 if compute_bf16 else 16
    qc, kc, vc = (t.contiguous() for t in (q, k, v))
    featsc = feats[..., :4].to(torch.float32).contiguous()
    freqs = sinusoidal_frequencies(c, q.device)

    if use_attn_bias:
        u = (
            ((qc.to(torch.float32) * scale_value) @ weight.to(torch.float32))
            .to(qc.dtype)
            .contiguous()
        )
    else:
        u = qc  # dummy pointer, never read

    o1 = torch.zeros_like(qc)
    ae = (
        torch.zeros(batch, heads, length, c, device=q.device, dtype=q.dtype)
        if use_activation_bias
        else qc  # dummy pointer, never written
    )
    lse = torch.zeros(
        batch, heads, length, device=q.device, dtype=torch.float32
    )

    grid = (triton.cdiv(length, block_m), batch)
    flash_spacetime_fwd_kernel[grid](
        qc,
        kc,
        vc,
        u,
        featsc,
        seqlens.to(torch.int32).contiguous(),
        o1,
        ae,
        lse,
        freqs,
        scale_value,
        L=length,
        H=heads,
        FEAT_STRIDE=4,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        G_PAD=_next_pow2(heads),
        C_=c,
        C_CHUNK=16,
        F_=c // 2,
        USE_ATTN_BIAS=use_attn_bias,
        USE_ACT_BIAS=use_activation_bias,
        SKIP_EMPTY_TILES=skip_empty_tiles,
        CDTYPE=tl.bfloat16 if compute_bf16 else tl.float32,
        # ieee fp32 dots hit an LLVM assertion in this triton build;
        # tf32x3 (three-pass tf32) reaches fp32-class accuracy (~1e-7
        # relative) through a lowering that works. bf16 operand dots
        # ignore input_precision entirely.
        INPUT_PRECISION="ieee",
        TIME_SCALE_C=TIME_SCALE,
        INPUT_SCALE=SINEMB_INPUT_SCALE,
        CLIP=SINEMB_CLIP,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    out = o1
    if use_activation_bias:
        o2 = ae.to(torch.float32) @ weight.to(torch.float32).t()
        if bias is not None:
            o2 = o2 + bias.to(torch.float32)
        # Pad rows must stay exactly zero after the +bias broadcast.
        idx = torch.arange(length, device=q.device)
        valid = (idx.unsqueeze(0) < seqlens.unsqueeze(1))[:, None, :, None]
        out = out + (o2 * valid).to(out.dtype)
    return out, lse


class _FlashSpacetimeAttention(torch.autograd.Function):
    """Fused forward and deterministic Triton backward.

    FLASH_ST_REFERENCE_BWD=1 selects autograd through the eager
    reference instead (exact but materialises R) — the debugging A/B for
    the Triton backward.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        feats: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        seqlens: Tensor,
        scale: Optional[float],
        use_attn_bias: bool,
        use_activation_bias: bool,
    ) -> Tensor:
        if feats.requires_grad:
            raise ValueError("feats (detector data) must not require grad")
        out, lse = flash_spacetime_forward(
            q,
            k,
            v,
            feats,
            weight,
            bias,
            seqlens,
            scale=scale,
            use_attn_bias=use_attn_bias,
            use_activation_bias=use_activation_bias,
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            feats,
            weight,
            bias if bias is not None else q.new_empty(0),
            seqlens,
            out,
            lse,
        )
        ctx.has_bias = bias is not None
        ctx.scale = scale
        ctx.flags = (use_attn_bias, use_activation_bias)
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_out: Tensor
    ) -> Tuple[Optional[Tensor], ...]:
        q, k, v, feats, weight, bias_t, seqlens, out, lse = ctx.saved_tensors
        bias = bias_t if ctx.has_bias else None
        use_attn_bias, use_activation_bias = ctx.flags
        if os.environ.get("FLASH_ST_REFERENCE_BWD", "0") != "1":
            from graphnet.models.components.flash_spacetime_triton_bwd import (
                flash_spacetime_backward,
            )

            dq, dk, dv, dw, db = flash_spacetime_backward(
                q,
                k,
                v,
                feats,
                weight,
                bias,
                seqlens,
                out,
                lse,
                grad_out,
                scale=ctx.scale,
                use_attn_bias=use_attn_bias,
                use_activation_bias=use_activation_bias,
            )
            # With both biases off the projection never enters the graph;
            # autograd's convention for unused parameters is None, not zeros.
            proj_used = use_attn_bias or use_activation_bias
            return (
                dq if q.requires_grad else None,
                dk if k.requires_grad else None,
                dv if v.requires_grad else None,
                None,
                dw if (weight.requires_grad and proj_used) else None,
                (
                    db
                    if (bias is not None and bias.requires_grad and proj_used)
                    else None
                ),
                None,
                None,
                None,
                None,
            )
        length = q.shape[2]
        idx = torch.arange(length, device=q.device)
        valid = idx.unsqueeze(0) < seqlens.unsqueeze(1)
        mask = torch.zeros(valid.shape, dtype=torch.float32, device=q.device)
        mask[~valid] = float("-inf")

        with torch.enable_grad():
            qd = q.detach().requires_grad_(q.requires_grad)
            kd = k.detach().requires_grad_(k.requires_grad)
            vd = v.detach().requires_grad_(v.requires_grad)
            wd = weight.detach().requires_grad_(weight.requires_grad)
            bd = (
                bias.detach().requires_grad_(bias.requires_grad)
                if bias is not None
                else None
            )
            out = spacetime_attention_reference(
                qd,
                kd,
                vd,
                feats.detach(),
                wd,
                bd,
                key_padding_mask=mask,
                scale=ctx.scale,
                use_attn_bias=use_attn_bias,
                use_activation_bias=use_activation_bias,
            )
            # The kernel zeroes pad rows; the reference leaves eager garbage
            # there. Blank the upstream grad at pad rows so both agree (the
            # contract guarantees it is zero there anyway).
            g = grad_out * valid[:, None, :, None]
            grads = torch.autograd.grad(
                out,
                [
                    t
                    for t, need in (
                        (qd, qd.requires_grad),
                        (kd, kd.requires_grad),
                        (vd, vd.requires_grad),
                        (wd, wd.requires_grad),
                    )
                    if need
                ]
                + ([bd] if bd is not None and bd.requires_grad else []),
                grad_outputs=g,
                allow_unused=True,
            )
        out_grads: List[Optional[Tensor]] = []
        it = iter(grads)
        for t in (q, k, v, weight):
            out_grads.append(next(it) if t.requires_grad else None)
        b_grad = (
            next(it) if (bias is not None and bias.requires_grad) else None
        )
        return (
            *out_grads[:3],
            None,
            out_grads[3],
            b_grad,
            None,
            None,
            None,
            None,
        )


def flash_spacetime_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    feats: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    seqlens: Tensor,
    scale: Optional[float] = None,
    use_attn_bias: bool = True,
    use_activation_bias: bool = True,
) -> Tensor:
    """Fused spacetime-bias attention (public op).

    Semantics of `spacetime_attention_reference` on valid rows; padding
    rows are zeroed. See the module docstring for the padding contract.
    """
    return _FlashSpacetimeAttention.apply(
        q,
        k,
        v,
        feats,
        weight,
        bias,
        seqlens,
        scale,
        use_attn_bias,
        use_activation_bias,
    )
