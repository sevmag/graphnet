"""Tests for `DeepIceRotaryBlock`."""

from typing import Any, Dict, List

import pytest
import torch
from torch_geometric.data import Batch, Data

from graphnet.models.components.attention_blocks import RotaryTransformerBlock
from graphnet.models.transformer import DeepIceRope, DeepIceRotaryBlock

SIZES: Dict[str, Any] = dict(
    hidden_dim=64, seq_length=32, depth=3, head_size=16, n_features=5
)


def _batch(lengths: List[int]) -> Batch:
    """Build a batch of synthetic events, fixed by their lengths."""
    events = []
    for i, n in enumerate(lengths):
        g = torch.Generator().manual_seed(n + i)
        events.append(Data(x=torch.randn(n, 5, generator=g) * 0.3))
    return Batch.from_data_list(events)


def _model(**overrides: Any) -> DeepIceRotaryBlock:
    torch.manual_seed(0)
    return DeepIceRotaryBlock(**{**SIZES, **overrides}).eval()


def test_matches_the_dimensions_of_deepice_rope() -> None:
    """Width, depth, heads and feed-forward width follow `DeepIceRope`'s."""
    model = _model()
    plain = DeepIceRope(**SIZES)
    assert model.nb_outputs == plain.nb_outputs
    assert len(model.blocks) == len(plain.blocks) == 3
    assert not len(model.sandwich)
    for blk in model.blocks:
        assert isinstance(blk, RotaryTransformerBlock)
        assert (blk.nhead, blk.head_dim) == (4, 16)
        assert blk.ffn.w2.in_features == 64 * 4
    assert all(
        (blk.gamma_1 == 1e-5).all() and (blk.gamma_2 == 1e-5).all()
        for blk in model.blocks
    )


def test_the_feedforward_width_and_layerscale_can_be_set() -> None:
    """Each difference from `DeepIceRope`'s block can be turned off."""
    blk = _model(feedforward_dim=100, layerscale_init=1.0).blocks[0]
    assert blk.ffn.w2.in_features == 100
    assert (blk.gamma_1 == 1.0).all()


@pytest.mark.parametrize("rope_per_head", [False, True])
def test_padding_does_not_change_an_event(rope_per_head: bool) -> None:
    """An event scores the same alone as next to a longer one."""
    model = _model(rope_per_head=rope_per_head)
    with torch.no_grad():
        together = model(_batch([9, 30]))
        alone = model(_batch([9]))
    torch.testing.assert_close(together[0], alone[0], atol=1e-5, rtol=1e-4)
    assert together.shape == (2, 64)


def test_rotation_reaches_the_attention() -> None:
    """Different coordinates give different outputs through the rotation."""
    model = _model()
    data = _batch([12])
    shifted = data.clone()
    shifted.x[:, 4] += 0.2 * torch.arange(12.0)
    tables = model._rotation_tables(data, torch.tensor([12]))
    moved = model._rotation_tables(shifted, torch.tensor([12]))
    assert not torch.allclose(tables[0], moved[0])


def test_padding_slots_hold_the_identity_rotation() -> None:
    """A padded key is masked, so its rotation must not produce NaNs."""
    cos, sin = _model()._rotation_tables(_batch([3, 7]), torch.tensor([3, 7]))
    assert cos.shape[0:3] == (2, 1, 7)
    assert (cos[0, :, 3:] == 1).all() and (sin[0, :, 3:] == 0).all()


def test_only_mean_pooling_is_available() -> None:
    """A cls token has no coordinates to rotate by."""
    with pytest.raises(ValueError, match="mean pooling"):
        _model(pooling="cls")
