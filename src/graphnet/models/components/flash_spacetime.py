"""Fused spacetime-bias attention: reference implementation and public op.

The op fuses the `SpacetimeEncoder` relative-position feature `R_ij` with the
`Attention_rel` attention computation into a single kernel boundary, so the
`[B, L, L, C]` per-pair tensor and the `[B, H, L, L]` attention intermediates
never have to be materialised. This module currently provides the exact
pure-PyTorch reference of that boundary, against which the Triton kernel is
verified; the reference is deliberately written to reproduce the composed
eager modules bit-for-bit in a common dtype, including their quirks:

- The interval's "time" coordinate is feature column 3, matching
  `SpacetimeEncoder`. On NuBench-ordered features (x, y, z, charge, t) column
  3 is the charge; the shipped models were trained with exactly that, so the
  op preserves it. Pass a feature tensor with time in column 3 to get a true
  spacetime interval.
- The sinusoidal frequency ladder is computed in float32 regardless of the
  working dtype, as `SinusoidalPosEmb` does.
- Padding uses the `Attention_rel` mask convention: a pair is masked with
  -inf if exactly one of its tokens is padding; pairs where BOTH tokens are
  padding are left unmasked, so fully-padded query rows softmax uniformly
  over padding keys instead of producing NaN. Outputs on padding rows are
  therefore garbage in the eager path; they are never read downstream. The
  fused kernel instead zeroes padding rows — comparisons are defined on
  valid rows only.

The attention logits use the q-scaled convention of `Attention_rel`: q is
multiplied by `scale` before BOTH the content term `q k^T` and the bias term
`sum_c q_c R_c`.
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

# Verbatim expression from SpacetimeEncoder.forward; kept as an expression so
# the float rounding matches the module exactly.
TIME_SCALE = 3e4 / 500 * 3e-1
SINEMB_INPUT_SCALE = 1024.0
SINEMB_CLIP = 4.0
SINEMB_N_FREQ = 10000.0


def sinusoidal_frequencies(dim: int, device: torch.device) -> Tensor:
    """Frequency ladder of `SinusoidalPosEmb`, in float32.

    The module computes `exp(arange(dim/2) * (-log(n_freq) / (dim/2)))`
    in float32 even when the surrounding computation runs in float64, so
    the ladder is fixed to float32 here for exact agreement.
    """
    half = dim / 2
    log_n_freq = torch.log(torch.tensor([SINEMB_N_FREQ], device=device))
    return torch.exp(
        torch.arange(half, device=device) * (-(log_n_freq / half))
    )


def spacetime_pair_features(
    feats: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
) -> Tensor:
    """`R_ij` of `SpacetimeEncoder`: projected sin/cos of the pair interval.

    Args:
        feats: `[B, L, F]` with positions in columns 0-2 and the interval's
            time coordinate in column 3 (`SpacetimeEncoder` convention).
        weight: `[C, C]` projection weight (`nn.Linear` layout).
        bias: `[C]` projection bias, or None.

    Returns:
        `[B, L, L, C]` pair features in the dtype of `feats`.
    """
    pos = feats[:, :, :3]
    time = feats[:, :, 3]
    interval = (pos[:, :, None] - pos[:, None, :]).pow(2).sum(-1) - (
        (time[:, :, None] - time[:, None, :]) * TIME_SCALE
    ).pow(2)
    signed_dist = torch.sign(interval) * torch.sqrt(torch.abs(interval))
    x = SINEMB_INPUT_SCALE * signed_dist.clip(-SINEMB_CLIP, SINEMB_CLIP)
    freqs = sinusoidal_frequencies(weight.shape[1], feats.device)
    ang = x.unsqueeze(-1) * freqs
    emb = torch.cat((torch.sin(ang), torch.cos(ang)), dim=-1)
    return torch.nn.functional.linear(emb, weight, bias)


def pair_mask_bias(key_padding_mask: Tensor) -> Tensor:
    """Additive pair mask of `Attention_rel` from a per-token float mask.

    `key_padding_mask` is `[B, L]` with 0 for valid tokens and -inf for
    padding. A pair is masked iff exactly one of its tokens is padding;
    both-padding pairs stay unmasked so fully-padded rows softmax to a
    uniform distribution instead of NaN.
    """
    m = key_padding_mask
    bias = torch.min(m[:, None, :], m[:, :, None])
    bias = bias.clone()
    bias[torch.max(m[:, None, :], m[:, :, None]) < 0] = 0
    return bias


def spacetime_attention_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    feats: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    key_padding_mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
    use_attn_bias: bool = True,
    use_activation_bias: bool = True,
) -> Tensor:
    """Exact eager reference of the fused op boundary.

    Args:
        q: `[B, H, L, D]` per-head queries, unscaled.
        k: `[B, H, L, D]` per-head keys.
        v: `[B, H, L, D]` per-head values.
        feats: `[B, L, F]` raw features (positions cols 0-2, interval time
            col 3); no gradient flows into `feats`.
        weight: `[C, C]` SpacetimeEncoder projection weight, `C == D`.
        bias: `[C]` SpacetimeEncoder projection bias, or None.
        key_padding_mask: `[B, L]` float mask, 0 valid / -inf padding, or
            None for fully-valid batches.
        scale: Query scale; defaults to `D ** -0.5`.
        use_attn_bias: Add `sum_c q_c R_c` to the pre-softmax logits.
        use_activation_bias: Add `sum_j P_ij R_ij` to the output.

    Returns:
        `[B, H, L, D]` attention output before the output projection.
        Padding rows carry the same garbage the eager modules produce (they
        are compared on valid rows only).
    """
    head_dim = q.shape[-1]
    if weight.shape[1] != head_dim:
        raise ValueError(
            f"C (={weight.shape[1]}) must equal head_dim (={head_dim})"
        )
    scale_value: float = head_dim**-0.5 if scale is None else scale

    qt = q * scale_value
    attn = qt @ k.transpose(-2, -1)

    rel = None
    if use_attn_bias or use_activation_bias:
        rel = spacetime_pair_features(feats, weight, bias)
    if use_attn_bias:
        assert rel is not None
        attn = attn + torch.einsum("bhic,bijc->bhij", qt, rel)
    if key_padding_mask is not None:
        attn = attn + pair_mask_bias(key_padding_mask).unsqueeze(1)

    p = attn.softmax(dim=-1)
    out = p @ v
    if use_activation_bias:
        assert rel is not None
        out = out + torch.einsum("bhij,bijc->bhic", p, rel)
    return out


def valid_row_mask(
    batch_size: int,
    length: int,
    seqlens: Tensor,
) -> Tensor:
    """Boolean `[B, L]` mask of valid (non-padding) rows from lengths."""
    idx = torch.arange(length, device=seqlens.device)
    return idx.unsqueeze(0) < seqlens.unsqueeze(1)


def float_padding_mask(valid: Tensor, dtype: torch.dtype) -> Tensor:
    """`Attention_rel`-style float mask (0 valid / -inf pad) from booleans."""
    mask = torch.zeros(valid.shape, dtype=dtype, device=valid.device)
    mask[~valid] = float("-inf")
    return mask


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    """`[B, L, H*D]` to `[B, H, L, D]`, matching `Attention_rel`'s reshape."""
    b, length, _ = x.shape
    return x.reshape(b, length, num_heads, -1).permute(0, 2, 1, 3)


def merge_heads(x: Tensor) -> Tensor:
    """`[B, H, L, D]` back to `[B, L, H*D]`."""
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)


def attention_rel_oracle_inputs(
    attention: torch.nn.Module,
    x: Tensor,
    kv: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Per-head q, k, v exactly as `Attention_rel.forward` builds them.

    Applies the module's own projection weights (and optional q/v
    biases) followed by the head split, so a reference call on the
    result composes with the module's output projection to reproduce the
    full module.
    """
    kv = x if kv is None else kv
    q = torch.nn.functional.linear(
        x, attention.proj_q.weight, attention.q_bias
    )
    k = torch.nn.functional.linear(kv, attention.proj_k.weight, None)
    v = torch.nn.functional.linear(
        kv, attention.proj_v.weight, attention.v_bias
    )
    num_heads = attention.num_heads
    return (
        split_heads(q, num_heads),
        split_heads(k, num_heads),
        split_heads(v, num_heads),
    )
