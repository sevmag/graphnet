"""Oracle-chain tests for the fused spacetime-bias attention op.

Stage (a) of the verification chain: the extracted pure-PyTorch reference in
`graphnet.models.components.flash_spacetime` must reproduce the composed
eager modules (`SpacetimeEncoder` + `Attention_rel`) to float64 round-off,
forward and backward. The Triton kernel is later verified against this
reference (stage (b), GPU-only tests); together the two stages bound the
kernel against the shipped model bit-for-bit up to dtype round-off.

Comparisons are restricted to valid (non-padding) rows: the eager path
produces garbage on padding rows by construction (both-padding pairs are
left unmasked so fully-padded rows softmax uniformly), and nothing
downstream ever reads them.
"""

from typing import Dict, Optional, Tuple

import pytest
import torch

from graphnet.models.components.embedding import SpacetimeEncoder
from graphnet.models.components.layers import Attention_rel
from graphnet.models.components.flash_spacetime import (
    attention_rel_oracle_inputs,
    float_padding_mask,
    merge_heads,
    spacetime_attention_reference,
    spacetime_pair_features,
    valid_row_mask,
)

# float64 round-off through ~L-long reductions; forward and gradients alike.
ATOL = 1e-11
RTOL = 1e-10


def _make_inputs(
    batch_size: int,
    length: int,
    num_heads: int,
    head_dim: int,
    masked: bool,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    torch.manual_seed(seed)
    hidden = num_heads * head_dim
    x = torch.randn(batch_size, length, hidden, dtype=torch.float64)
    # NuBench-like scales: positions O(1), column 3 (the interval's "time"
    # input, charge on NuBench) O(1), column 4 unused by the op.
    feats = torch.randn(batch_size, length, 5, dtype=torch.float64)
    if masked and length > 1:
        seqlens = torch.randint(1, length + 1, (batch_size,))
        seqlens[0] = length  # keep at least one full-length event
        if batch_size > 1:
            seqlens[-1] = 1  # and one length-1 event
    else:
        seqlens = torch.full((batch_size,), length, dtype=torch.long)
    valid = valid_row_mask(batch_size, length, seqlens)
    # Attention_rel asserts fp32/fp16 masks, so the float mask stays fp32
    # even in the float64 oracle.
    mask = float_padding_mask(valid, torch.float32) if masked else None
    return x, feats, valid, mask


def _make_modules(
    num_heads: int, head_dim: int, flags: Tuple[bool, bool], seed: int
) -> Tuple[SpacetimeEncoder, Attention_rel]:
    torch.manual_seed(seed + 1)
    hidden = num_heads * head_dim
    use_attn_bias, use_activation_bias = flags
    spacetime = SpacetimeEncoder(head_dim).double()
    attention = Attention_rel(
        hidden,
        num_heads,
        use_attn_bias=use_attn_bias,
        use_activation_bias=use_activation_bias,
    ).double()
    return spacetime, attention


def _module_forward(
    spacetime: SpacetimeEncoder,
    attention: Attention_rel,
    x: torch.Tensor,
    feats: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    rel = spacetime(feats)
    return attention(x, x, x, rel_pos_bias=rel, key_padding_mask=mask)


def _reference_forward(
    spacetime: SpacetimeEncoder,
    attention: Attention_rel,
    x: torch.Tensor,
    feats: torch.Tensor,
    mask: Optional[torch.Tensor],
    flags: Tuple[bool, bool],
) -> torch.Tensor:
    q, k, v = attention_rel_oracle_inputs(attention, x)
    out = spacetime_attention_reference(
        q,
        k,
        v,
        feats,
        spacetime.projection.weight,
        spacetime.projection.bias,
        key_padding_mask=mask,
        use_attn_bias=flags[0],
        use_activation_bias=flags[1],
    )
    return attention.proj(merge_heads(out))


SHAPES = [
    # (batch_size, length, num_heads, head_dim)
    (1, 1, 1, 16),
    (2, 2, 4, 16),
    (3, 17, 4, 32),
    (2, 64, 4, 48),
    (2, 65, 2, 48),
    (1, 128, 16, 48),
]
FLAG_COMBINATIONS = [
    (True, True),
    (True, False),
    (False, True),
    (False, False),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("flags", FLAG_COMBINATIONS)
@pytest.mark.parametrize("masked", [False, True])
def test_reference_matches_modules_forward(
    shape: Tuple[int, int, int, int],
    flags: Tuple[bool, bool],
    masked: bool,
) -> None:
    """Reference == SpacetimeEncoder∘Attention_rel on valid rows (fp64)."""
    _, _, num_heads, head_dim = shape
    x, feats, valid, mask = _make_inputs(
        *shape, masked, seed=hash(shape) % 2**31
    )
    spacetime, attention = _make_modules(num_heads, head_dim, flags, seed=1)

    out_module = _module_forward(spacetime, attention, x, feats, mask)
    out_reference = _reference_forward(
        spacetime, attention, x, feats, mask, flags
    )

    torch.testing.assert_close(
        out_reference[valid], out_module[valid], atol=ATOL, rtol=RTOL
    )


@pytest.mark.parametrize("shape", SHAPES[1:4])
@pytest.mark.parametrize("flags", FLAG_COMBINATIONS[:2])
@pytest.mark.parametrize("masked", [False, True])
def test_reference_matches_modules_backward(
    shape: Tuple[int, int, int, int],
    flags: Tuple[bool, bool],
    masked: bool,
) -> None:
    """Gradients through the reference match the module path (fp64).

    Compares d(x), d(projection W), d(projection b) under a loss that
    reads only valid rows, making the comparison independent of padding-
    row garbage.
    """
    _, _, num_heads, head_dim = shape
    x, feats, valid, mask = _make_inputs(
        *shape, masked, seed=hash(shape) % 2**31
    )
    spacetime, attention = _make_modules(num_heads, head_dim, flags, seed=2)

    grads: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in ("module", "reference"):
        xg = x.clone().requires_grad_(True)
        spacetime.zero_grad(set_to_none=True)
        attention.zero_grad(set_to_none=True)
        if name == "module":
            out = _module_forward(spacetime, attention, xg, feats, mask)
        else:
            out = _reference_forward(
                spacetime, attention, xg, feats, mask, flags
            )
        out[valid].pow(2).sum().backward()
        collected: Dict[str, torch.Tensor] = {}
        for key, tensor in (
            ("x", xg.grad),
            ("W", spacetime.projection.weight.grad),
            ("b", spacetime.projection.bias.grad),
            ("proj_q", attention.proj_q.weight.grad),
            ("proj_v", attention.proj_v.weight.grad),
        ):
            assert tensor is not None, f"missing gradient for {key}"
            collected[key] = tensor.clone()
        grads[name] = collected

    for key in grads["module"]:
        torch.testing.assert_close(
            grads["reference"][key],
            grads["module"][key],
            atol=ATOL,
            rtol=RTOL,
            msg=lambda m, key=key: f"gradient mismatch for {key}: {m}",
        )


def test_pair_features_match_spacetime_encoder() -> None:
    """`spacetime_pair_features` == `SpacetimeEncoder.forward` (fp64)."""
    torch.manual_seed(7)
    spacetime = SpacetimeEncoder(48).double()
    feats = torch.randn(3, 33, 5, dtype=torch.float64)
    torch.testing.assert_close(
        spacetime_pair_features(
            feats,
            spacetime.projection.weight,
            spacetime.projection.bias,
        ),
        spacetime(feats),
        atol=ATOL,
        rtol=RTOL,
    )


def test_fully_padded_row_is_finite_in_modules() -> None:
    """Documents the both-padding-pairs-unmasked quirk the op relies on.

    A fully-padded query row softmaxes uniformly over padding keys in
    the eager path (no NaN); the fused kernel instead zeroes such rows.
    Valid rows must be unaffected by how padding rows are handled.
    """
    batch_size, length, num_heads, head_dim = 2, 8, 2, 16
    x, feats, valid, _ = _make_inputs(
        batch_size, length, num_heads, head_dim, masked=False, seed=11
    )
    seqlens = torch.tensor([8, 3])
    valid = valid_row_mask(batch_size, length, seqlens)
    mask = float_padding_mask(valid, torch.float32)
    spacetime, attention = _make_modules(
        num_heads, head_dim, (True, True), seed=3
    )
    out = _module_forward(spacetime, attention, x, feats, mask)
    assert torch.isfinite(out).all()
