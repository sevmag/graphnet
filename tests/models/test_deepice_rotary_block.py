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


def _block_output(
    shift: float, rotate_queries_keys: bool, rotate_values: bool
) -> torch.Tensor:
    """One block over fixed tokens, with every rotation angle shifted."""
    torch.manual_seed(0)
    block = RotaryTransformerBlock(
        d_model=32,
        nhead=2,
        dim_feedforward=64,
        dropout=0.0,
        layerscale_init=1.0,
        rotate_queries_keys=rotate_queries_keys,
        rotate_values=rotate_values,
    ).double()
    block.eval()
    g = torch.Generator().manual_seed(2)
    x = torch.randn(2, 7, 32, generator=g, dtype=torch.double)
    angles = torch.randn(2, 1, 7, 8, generator=g, dtype=torch.double) * 3
    plane_shift = shift * torch.arange(1.0, 9.0, dtype=torch.double)
    angles = angles + plane_shift
    with torch.no_grad():
        return block(x, (angles.cos(), angles.sin()))


@pytest.mark.parametrize(
    "rotate_queries_keys, rotate_values",
    [(True, False), (True, True), (False, True)],
)
def test_block_depends_only_on_relative_angles(
    rotate_queries_keys: bool, rotate_values: bool
) -> None:
    """A common shift of every angle leaves the output unchanged."""
    torch.testing.assert_close(
        _block_output(0.0, rotate_queries_keys, rotate_values),
        _block_output(1.7, rotate_queries_keys, rotate_values),
    )


def test_rotated_values_carry_position_into_the_output() -> None:
    """With value rotation the output depends on the rotation angles."""
    plain = _block_output(0.0, False, True)
    torch.manual_seed(0)
    block = (
        RotaryTransformerBlock(
            d_model=32,
            nhead=2,
            dim_feedforward=64,
            dropout=0.0,
            layerscale_init=1.0,
            rotate_queries_keys=False,
            rotate_values=True,
        )
        .double()
        .eval()
    )
    g = torch.Generator().manual_seed(2)
    x = torch.randn(2, 7, 32, generator=g, dtype=torch.double)
    flat = torch.zeros(2, 1, 7, 8, dtype=torch.double)
    with torch.no_grad():
        unrotated = block(x, (flat.cos(), flat.sin()))
    assert not torch.allclose(plain, unrotated)


def test_a_block_without_any_rotation_is_rejected() -> None:
    """Attention with no rotation carries no position."""
    with pytest.raises(ValueError, match="carries no position"):
        RotaryTransformerBlock(
            d_model=32,
            nhead=2,
            dim_feedforward=64,
            rotate_queries_keys=False,
            rotate_values=False,
        )
