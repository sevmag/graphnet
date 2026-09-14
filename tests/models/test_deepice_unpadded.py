"""Tests for DeepIce's packed, unpadded forward.

The padded routes size every tensor by the batch's longest event, so one long
event inflates the whole batch and `max_pulses` exists to bound that. The
unpadded route carries packed sequences end to end, which removes the bound
entirely -- so the tests are that it agrees with the padded route where both
are valid, and that it survives an event far past any cap.
"""

from typing import Any, Dict, Sequence

import pytest
import torch
from torch_geometric.data import Data, Batch

from graphnet.models.gnn import DeepIce

pytest.importorskip("flash_spacetime")
CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused kernel is GPU-only"
)
BASE: Dict[str, Any] = dict(
    hidden_dim=128,
    depth=3,
    head_size=16,
    depth_rel=2,
    n_rel=2,
    seq_length=192,
    include_dynedge=False,
    n_features=5,
    pooling="mean",
    rel_attention="flash",
    use_nested_attention=True,
)


def _batch(lengths: Sequence[int]) -> Batch:
    """Variable-length events, including a single-pulse one."""
    g = torch.Generator().manual_seed(5)
    events = []
    for n in lengths:
        d = Data(x=torch.randn(n, 5, generator=g))
        d.n_pulses = torch.tensor([n])
        events.append(d)
    return Batch.from_data_list(events)


def _pair() -> tuple:
    """Padded and unpadded models sharing one set of weights."""
    torch.manual_seed(0)
    padded = DeepIce(**BASE)
    torch.manual_seed(0)
    unpadded = DeepIce(**BASE, unpadded=True)
    unpadded.load_state_dict(padded.state_dict())
    return padded, unpadded


def test_unpadded_needs_flash_and_mean_pooling() -> None:
    """The preconditions fail at construction, not at the first forward."""
    with pytest.raises(ValueError, match="unpadded"):
        DeepIce(**{**BASE, "rel_attention": "dense"}, unpadded=True)
    with pytest.raises(ValueError, match="unpadded"):
        DeepIce(**{**BASE, "pooling": "cls"}, unpadded=True)


def test_unpadded_shares_the_parameters() -> None:
    """It is a different route through the same weights, not a new model."""
    padded, unpadded = _pair()
    assert set(padded.state_dict()) == set(unpadded.state_dict())


@CUDA
def test_unpadded_matches_padded() -> None:
    """Both routes compute the same function."""
    batch = _batch([1, 15, 97, 400]).to("cuda")
    padded, unpadded = _pair()
    padded, unpadded = padded.cuda().eval(), unpadded.cuda().eval()
    with torch.no_grad():
        assert torch.allclose(padded(batch), unpadded(batch), atol=1e-4)


@CUDA
def test_unpadded_gradients_match_padded() -> None:
    """And the same gradients: a forward-only check would miss a backward bug."""
    batch = _batch([1, 15, 97, 400]).to("cuda")
    grads = []
    for model in _pair():
        model = model.cuda().train()
        model(batch).square().mean().backward()
        grads.append(
            {
                k: p.grad.detach().clone()
                for k, p in model.named_parameters()
                if p.grad is not None
            }
        )
    shared = set(grads[0]) & set(grads[1])
    assert shared
    worst = max((grads[0][k] - grads[1][k]).abs().max().item() for k in shared)
    assert worst < 1e-5, worst


@CUDA
def test_event_length_is_unbounded() -> None:
    """An event far past `max_pulses` runs, which is the point of the route."""
    batch = _batch([1, 50, 12_000]).to("cuda")
    _, unpadded = _pair()
    unpadded = unpadded.cuda().eval()
    with torch.no_grad():
        out = unpadded(batch)
    assert out.shape == (3, 128)
    assert torch.isfinite(out).all()
