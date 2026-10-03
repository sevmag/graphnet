"""Tests for DeepIce's choice of event-level pooling."""

from typing import Any, Dict

import pytest
import torch
from torch_geometric.data import Data, Batch

from graphnet.models.gnn import DeepIce

N_FEATURES = 6


def _synth_batch(nev: int = 6) -> Batch:
    """Build a small batch of variable-length synthetic events."""
    g = torch.Generator().manual_seed(1)
    events = []
    # A single-pulse event is the edge case a mean pool has to survive.
    lengths = [1] + [
        int(torch.randint(15, 50, (1,), generator=g)) for _ in range(nev - 1)
    ]
    for n in lengths:
        x = torch.randn(n, N_FEATURES, generator=g, dtype=torch.float64)
        x[:, 5] = torch.randint(0, 2, (n,), generator=g).to(torch.float64)
        data = Data(x=x)
        data.n_pulses = torch.tensor([n])
        events.append(data)
    return Batch.from_data_list(events)


def _model(**overrides: Any) -> DeepIce:
    """Build a small DeepIce with a fixed init."""
    kwargs: Dict[str, Any] = dict(
        hidden_dim=128,
        seq_length=64,
        depth=2,
        head_size=32,
        depth_rel=3,
        n_rel=2,
        include_dynedge=False,
        n_features=N_FEATURES,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return DeepIce(**kwargs).double().eval()


def test_rejects_unknown_pooling() -> None:
    """An unsupported pooling name fails at construction, not at forward."""
    with pytest.raises(ValueError, match="pooling"):
        _model(pooling="attention")


def test_state_dict_is_interchangeable() -> None:
    """Either pooling loads the other's weights, so a checkpoint transfers."""
    cls_model, mean_model = _model(pooling="cls"), _model(pooling="mean")
    assert set(cls_model.state_dict()) == set(mean_model.state_dict())
    assert "cls_token.weight" in mean_model.state_dict()
    mean_model.load_state_dict(cls_model.state_dict())
    cls_model.load_state_dict(mean_model.state_dict())


def test_mean_pool_matches_per_event_mean() -> None:
    """The pooled output is the mean of that event's real token rows.

    The token stack is taken from the last block's own output rather
    than by re-driving the model, so the test states the pooling
    contract without depending on how the stages are wired together.
    """
    batch = _synth_batch()
    lengths = [int(d.x.shape[0]) for d in batch.to_data_list()]
    model = _model(pooling="mean")
    captured = []
    handle = model.blocks[-1].register_forward_hook(
        lambda _module, _args, output: captured.append(output)
    )
    try:
        with torch.no_grad():
            pooled = model(batch)
    finally:
        handle.remove()
    tokens = captured[-1]
    expected = torch.stack(
        [tokens[i, :n].mean(0) for i, n in enumerate(lengths)]
    )
    assert torch.allclose(pooled, expected, atol=1e-10)


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_nested_matches_padded(pooling: str) -> None:
    """Dropping the padding does not change the event embedding."""
    batch = _synth_batch()
    padded = _model(pooling=pooling)
    nested = _model(pooling=pooling, use_nested_attention=True)
    nested.load_state_dict(padded.state_dict())
    with torch.no_grad():
        assert torch.allclose(padded(batch), nested(batch), atol=1e-10)


def test_mean_pool_is_permutation_invariant() -> None:
    """Reordering an event's pulses leaves its embedding untouched."""
    batch = _synth_batch()
    model = _model(pooling="mean")
    shuffled = Batch.from_data_list(
        [
            Data(
                x=d.x[torch.randperm(d.x.shape[0], generator=g)],
                n_pulses=d.n_pulses,
            )
            for d, g in zip(
                batch.to_data_list(),
                (torch.Generator().manual_seed(i) for i in range(6)),
            )
        ]
    )
    with torch.no_grad():
        assert torch.allclose(model(batch), model(shuffled), atol=1e-10)


def test_cls_token_is_unused_under_mean_pooling() -> None:
    """Mean pooling leaves `cls_token` gradient-less.

    Distributed training rejects a parameter that never receives a
    gradient unless it is told to expect one, so this is a constraint on
    the run rather than an implementation detail.
    """
    model = _model(pooling="mean").train()
    model(_synth_batch()).sum().backward()
    ungradded = [n for n, p in model.named_parameters() if p.grad is None]
    assert ungradded == ["cls_token.weight"]
