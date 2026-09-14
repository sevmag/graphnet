"""Tests for DeepIce's fused relative-attention path.

`rel_attention="flash"` must compute the same function as the dense
path: the kernel rebuilds the spacetime pair features inside each tile
instead of materialising the `[B, L, L, C]` tensor, which changes where
the arithmetic happens but not what it is.
"""

import importlib.util
import sys
from typing import Any, Dict, Tuple

import pytest
import torch
from torch_geometric.data import Data, Batch

from graphnet.models.gnn import DeepIce

N_FEATURES = 6
HAS_PACKAGE = importlib.util.find_spec("flash_spacetime") is not None
# Building a flash model reads the band constants from the package, so even
# the construction-time checks need it; the kernel itself is GPU-only.
PACKAGE = pytest.mark.skipif(
    not HAS_PACKAGE, reason="needs the flash_spacetime package"
)
CUDA = pytest.mark.skipif(
    not (HAS_PACKAGE and torch.cuda.is_available()),
    reason="the fused kernel needs flash_spacetime and a GPU",
)


def _batch(lengths: Tuple[int, ...] = (1, 15, 39, 97)) -> Batch:
    """Variable-length synthetic events, including a single-pulse one."""
    g = torch.Generator().manual_seed(3)
    events = []
    for n in lengths:
        x = torch.randn(n, N_FEATURES, generator=g)
        x[:, 5] = torch.randint(0, 2, (n,), generator=g).to(x.dtype)
        d = Data(x=x)
        d.n_pulses = torch.tensor([n])
        events.append(d)
    return Batch.from_data_list(events)


def _model(**overrides: Any) -> DeepIce:
    """A small DeepIce with a fixed init."""
    kwargs: Dict[str, Any] = dict(
        hidden_dim=128,
        seq_length=192,
        depth=2,
        head_size=16,
        depth_rel=2,
        n_rel=2,
        include_dynedge=False,
        n_features=N_FEATURES,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return DeepIce(**kwargs).eval()


def test_rejects_unknown_rel_attention() -> None:
    """An unsupported name fails at construction, not at the first forward."""
    with pytest.raises(ValueError, match="rel_attention"):
        _model(rel_attention="sparse")


def test_flash_needs_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the kernel package the flash model fails to build, by name."""
    monkeypatch.setitem(sys.modules, "flash_spacetime", None)
    with pytest.raises(ImportError, match="flash_spacetime"):
        _model(rel_attention="flash")


@pytest.mark.parametrize(
    "overrides",
    [dict(rel_pos_encoder="directional"), dict(alibi_bias=True)],
    ids=["directional", "alibi"],
)
def test_flash_rejects_other_pair_encodings(overrides: Dict[str, Any]) -> None:
    """The kernel rebuilds the EPJC embedding; nothing else can drive it."""
    with pytest.raises(ValueError, match="rel_attention='flash'"):
        _model(rel_attention="flash", **overrides)


@PACKAGE
def test_flash_pins_the_spacetime_band() -> None:
    """A band other than the one compiled into the kernel is refused."""
    with pytest.raises(ValueError, match="spacetime_scale"):
        _model(rel_attention="flash", spacetime_scale=512.0)


@PACKAGE
def test_flash_leaves_the_parameters_alone() -> None:
    """The fused path is a different route through the same weights."""
    dense, flash = _model(), _model(rel_attention="flash")
    assert set(dense.state_dict()) == set(flash.state_dict())
    flash.load_state_dict(dense.state_dict())


@CUDA
def test_flash_matches_dense() -> None:
    """Both paths compute the same function on the same weights."""
    batch = _batch().to("cuda")
    dense = _model().cuda()
    flash = _model(rel_attention="flash").cuda()
    flash.load_state_dict(dense.state_dict())
    with torch.no_grad():
        a, b = dense(batch), flash(batch)
    assert torch.allclose(a, b, atol=2e-4, rtol=2e-4)


@CUDA
def test_flash_gradients_match_dense() -> None:
    """And the same gradients, which a forward-only check would not catch."""
    batch = _batch().to("cuda")
    grads = {}
    for name, kw in (("dense", {}), ("flash", dict(rel_attention="flash"))):
        m = _model(**kw).cuda().train()
        if name == "flash":
            m.load_state_dict(_model().cuda().state_dict())
        m(batch).square().mean().backward()
        grads[name] = {
            k: p.grad.detach().clone()
            for k, p in m.named_parameters()
            if p.grad is not None
        }
    shared = set(grads["dense"]) & set(grads["flash"])
    assert shared
    worst = max(
        (grads["dense"][k] - grads["flash"][k]).abs().max().item()
        for k in shared
    )
    assert worst < 5e-3, worst
