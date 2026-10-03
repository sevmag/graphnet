"""Tests for the width of DeepIce's Fourier projection."""

from typing import Any, Dict

import torch
from torch_geometric.data import Data, Batch

from graphnet.models.gnn import DeepIce

N_FEATURES = 6
SEQ_LENGTH = 64
# `FourierEncoderEPJC` concatenates 6 * seq_length features at n_features >= 6.
CONCAT_DIM = 6 * SEQ_LENGTH
SCHEMA = {"x": 4096.0, "y": 4096.0, "z": 4096.0, "t": 4096.0, "q": 1024.0}
NAMES = ["x", "y", "z", "t", "q", "aux"]


def _synth_batch(nev: int = 4) -> Batch:
    """Build a small batch of variable-length synthetic events."""
    g = torch.Generator().manual_seed(1)
    events = []
    for _ in range(nev):
        n = int(torch.randint(15, 50, (1,), generator=g))
        x = torch.randn(n, N_FEATURES, generator=g)
        x[:, 5] = torch.randint(0, 2, (n,), generator=g).float()
        data = Data(x=x)
        data.n_pulses = torch.tensor([n])
        events.append(data)
    return Batch.from_data_list(events)


def _model(**overrides: Any) -> DeepIce:
    """Build a small DeepIce with a fixed init."""
    kwargs: Dict[str, Any] = dict(
        hidden_dim=128,
        seq_length=SEQ_LENGTH,
        depth=2,
        head_size=32,
        depth_rel=2,
        n_rel=1,
        include_dynedge=False,
        n_features=N_FEATURES,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return DeepIce(**kwargs).eval()


def _fourier_params(model: DeepIce) -> int:
    """Count the parameters of whichever projection the model built."""
    n = sum(p.numel() for p in model.fourier_ext.parameters())
    if model.fourier_mlp is not None:
        n += sum(p.numel() for p in model.fourier_mlp.parameters())
    return n


def test_default_is_the_concatenation_width() -> None:
    """Unset, the hidden width is the concatenation's own, on both paths."""
    epjc = _model()
    assert epjc.fourier_ext.mlp[0].out_features == CONCAT_DIM
    schema = _model(fourier_schema=SCHEMA, input_feature_names=NAMES)
    assert schema.fourier_mlp[0].out_features == schema.fourier_ext.output_dim


def test_narrowing_shrinks_both_input_paths() -> None:
    """Both the EPJC and the schema-driven projection honour the width."""
    for extra in ({}, dict(fourier_schema=SCHEMA, input_feature_names=NAMES)):
        wide, narrow = _model(**extra), _model(fourier_mlp_dim=64, **extra)
        assert _fourier_params(narrow) < _fourier_params(wide)
        layer = (
            narrow.fourier_mlp[0]
            if narrow.fourier_mlp is not None
            else narrow.fourier_ext.mlp[0]
        )
        assert layer.out_features == 64


def test_narrowing_leaves_the_transformer_alone() -> None:
    """Only the projection changes size; the blocks keep their parameters."""
    wide, narrow = _model(), _model(fourier_mlp_dim=64)
    for part in ("sandwich", "blocks"):
        assert sum(p.numel() for p in getattr(wide, part).parameters()) == sum(
            p.numel() for p in getattr(narrow, part).parameters()
        )


def test_forward_shape_is_unchanged() -> None:
    """A narrower projection still emits one `hidden_dim` vector per event."""
    batch = _synth_batch()
    with torch.no_grad():
        out = _model(fourier_mlp_dim=64)(batch)
    assert out.shape == (4, 128)
    assert torch.isfinite(out).all()
