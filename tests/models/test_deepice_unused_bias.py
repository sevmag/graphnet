"""Tests that DeepIce skips the spacetime bias when no block consumes it."""

from typing import Any

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from graphnet.models.gnn import DeepIce

N_FEATURES = 6


def _batch() -> Batch:
    """Build a small batch of variable-length synthetic events."""
    g = torch.Generator().manual_seed(1)
    events = []
    for n in (7, 19, 12):
        x = torch.randn(n, N_FEATURES, generator=g, dtype=torch.float64)
        x[:, 5] = torch.randint(0, 2, (n,), generator=g).to(torch.float64)
        events.append(Data(x=x))
    return Batch.from_data_list(events)


def _model(**overrides: Any) -> DeepIce:
    """Build a small DeepIce with only relative blocks."""
    torch.manual_seed(0)
    return (
        DeepIce(
            hidden_dim=64,
            seq_length=32,
            depth=0,
            head_size=32,
            depth_rel=2,
            n_rel=2,
            include_dynedge=False,
            n_features=N_FEATURES,
            pooling="mean",
            **overrides,
        )
        .double()
        .eval()
    )


def _refuse(*args: Any, **kwargs: Any) -> Tensor:
    raise AssertionError("the spacetime bias was built")


@pytest.mark.parametrize("rel_attention", ["dense", "tiled"])
def test_bias_free_blocks_never_build_the_bias(rel_attention: str) -> None:
    """The O(len^2) bias costs memory and time that nothing would use."""
    model = _model(
        use_attn_bias=False,
        use_activation_bias=False,
        rel_attention=rel_attention,
    )
    expected = model(_batch())
    model.rel_pos.forward = _refuse  # type: ignore[method-assign]
    model.rel_pos.forward_tiled = _refuse  # type: ignore[method-assign]
    torch.testing.assert_close(model(_batch()), expected)


def test_blocks_that_use_the_bias_still_build_it() -> None:
    """Either use of the bias keeps it, so biased models are unchanged."""
    for attn, activation in ((True, False), (False, True)):
        model = _model(use_attn_bias=attn, use_activation_bias=activation)
        model.rel_pos.forward = _refuse  # type: ignore[method-assign]
        with pytest.raises(AssertionError, match="was built"):
            model(_batch())
