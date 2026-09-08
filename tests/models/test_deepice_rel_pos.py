"""Tests for DeepIce's choice of pairwise spacetime encoder."""

from typing import Any, Dict

import pytest
import torch
from torch_geometric.data import Batch, Data

from graphnet.models.components.embedding import (
    DirectionalSpacetimeEncoder,
    SpacetimeEncoder,
    SpacetimeEncoderEPJC,
)
from graphnet.models.gnn import DeepIce

SMALL: Dict[str, Any] = dict(
    hidden_dim=32, seq_length=16, depth=2, head_size=8, depth_rel=1
)
DIRECTIONAL: Dict[str, Any] = dict(
    rel_pos_encoder="directional",
    rel_pos_kwargs={"n_media": 2},
    medium_key="medium",
)


def _batch() -> Batch:
    """Two events of different length, one in each medium."""
    torch.manual_seed(0)
    graphs = []
    for n_pulses, medium in ((5, 0), (3, 1)):
        x = torch.rand(n_pulses, 6)
        x[:, 5] = 0
        graphs.append(Data(x=x, medium=torch.tensor([medium])))
    return Batch.from_data_list(graphs)


def test_default_matches_epjc() -> None:
    """Existing configurations keep the published encoder's bias."""
    model = DeepIce(**SMALL)
    assert isinstance(model.rel_pos, SpacetimeEncoder)
    published = SpacetimeEncoderEPJC(SMALL["head_size"])
    published.load_state_dict(model.rel_pos.state_dict())
    x = torch.rand(2, 5, 6)
    assert torch.allclose(model.rel_pos(x), published(x))


def test_directional_forward() -> None:
    """The directional encoder slots into the relative blocks."""
    model = DeepIce(**SMALL, **DIRECTIONAL)
    assert isinstance(model.rel_pos, DirectionalSpacetimeEncoder)
    assert model(_batch()).shape == (2, 32)


def test_directional_reads_the_medium() -> None:
    """An event's output follows its own medium, not its neighbour's."""
    model = DeepIce(**SMALL, **DIRECTIONAL).eval()
    assert isinstance(model.rel_pos, DirectionalSpacetimeEncoder)
    with torch.no_grad():
        model.rel_pos.log_speed.weight[1] = 0.5
        batch = _batch()
        before = model(batch)
        batch.medium = torch.tensor([0, 0])
        after = model(batch)
    assert torch.allclose(before[0], after[0])
    assert not torch.allclose(before[1], after[1])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(rel_pos_encoder="unknown"),
        dict(medium_key="medium"),
        dict(rel_pos_kwargs={"n_media": 2}),
    ],
)
def test_invalid_combinations(kwargs: Dict[str, Any]) -> None:
    """Options that would silently do nothing are rejected."""
    with pytest.raises(ValueError):
        DeepIce(**SMALL, **kwargs)
