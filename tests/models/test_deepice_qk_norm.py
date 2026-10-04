"""Tests for QK normalisation in DeepIce's relative-attention blocks."""

import pytest
import torch

from graphnet.models.components.attention_blocks import Attention_rel
from graphnet.models.gnn import DeepIce

HEADS, HEAD_DIM = 2, 16


def _attention_output(qk_norm: bool, projection_gain: float) -> torch.Tensor:
    """Attention over one fixed input, with q and k projections rescaled."""
    torch.manual_seed(0)
    attention = Attention_rel(
        HEADS * HEAD_DIM, num_heads=HEADS, qk_norm=qk_norm
    ).eval()
    x = torch.randn(3, 11, HEADS * HEAD_DIM)
    bias = torch.randn(3, 11, 11, HEAD_DIM)
    with torch.no_grad():
        attention.proj_q.weight *= projection_gain
        attention.proj_k.weight *= projection_gain
        return attention(x, x, x, rel_pos_bias=bias)


def test_normalised_attention_ignores_the_projection_scale() -> None:
    """The logits, relative bias included, no longer grow with the weights."""
    torch.testing.assert_close(
        _attention_output(True, 1.0),
        _attention_output(True, 7.0),
        atol=1e-5,
        rtol=1e-4,
    )
    assert not torch.allclose(
        _attention_output(False, 1.0), _attention_output(False, 7.0), atol=1e-3
    )


def test_rel_qk_norm_adds_parameters_to_the_relative_blocks_only() -> None:
    """Unset, the model keeps its parameters, so its checkpoints load."""
    kwargs = dict(hidden_dim=64, seq_length=32, depth=1, head_size=16)
    plain = set(DeepIce(depth_rel=2, **kwargs).state_dict())
    normed = set(DeepIce(depth_rel=2, rel_qk_norm=True, **kwargs).state_dict())
    assert plain < normed
    added = normed - plain
    assert added == {
        f"sandwich.{i}.attn.{name}_norm.weight"
        for i in range(2)
        for name in ("q", "k")
    }


def test_flash_attention_rejects_rel_qk_norm() -> None:
    """The fused kernel has no place to normalise."""
    with pytest.raises(ValueError, match="rel_qk_norm"):
        DeepIce(rel_attention="flash", rel_qk_norm=True)
