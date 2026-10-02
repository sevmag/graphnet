"""Tests for DeepIce's query-tiled relative-attention path."""

from typing import Any, Dict

import pytest
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph

from graphnet.models.components.embedding import (
    SpacetimeEncoder,
    SpacetimeEncoderEPJC,
)
from graphnet.models.gnn import DeepIce

N_FEATURES = 6


def _synth_batch(
    nev: int = 4,
    with_edges: bool = False,
    dtype: torch.dtype = torch.float64,
) -> Batch:
    """Build a small batch of variable-length synthetic events."""
    g = torch.Generator().manual_seed(1)
    events = []
    for _ in range(nev):
        n = int(torch.randint(15, 50, (1,), generator=g))
        x = torch.randn(n, N_FEATURES, generator=g, dtype=dtype)
        x[:, 5] = torch.randint(0, 2, (n,), generator=g).to(dtype)
        data = Data(x=x)
        if with_edges:
            data.edge_index = knn_graph(x[:, :3].float(), k=8)
        data.n_pulses = torch.tensor([n])
        events.append(data)
    return Batch.from_data_list(events)


def _kwargs(include_dynedge: bool = False) -> Dict[str, Any]:
    kw: Dict[str, Any] = dict(
        hidden_dim=128,
        seq_length=64,
        depth=2,
        head_size=32,
        depth_rel=3,
        n_rel=2,
        scaled_emb=True,
        include_dynedge=include_dynedge,
        n_features=N_FEATURES,
    )
    if include_dynedge:
        kw["dynedge_args"] = {
            "nb_inputs": N_FEATURES,
            "nb_neighbours": 8,
            "post_processing_layer_sizes": [64, 64],
            "activation_layer": "gelu",
            "add_norm_layer": True,
            "skip_readout": True,
        }
    return kw


@pytest.mark.parametrize("q_tile", [8, 64, 1000])
@pytest.mark.parametrize("include_dynedge", [False, True])
def test_tiled_bit_identical_to_dense(
    q_tile: int, include_dynedge: bool
) -> None:
    """Same weights -> identical output, at any tile size."""
    torch.manual_seed(0)
    dense = DeepIce(**_kwargs(include_dynedge)).double().eval()
    tiled = (
        DeepIce(
            **_kwargs(include_dynedge),
            rel_attention="tiled",
            q_tile=q_tile,
        )
        .double()
        .eval()
    )
    # A stock (dense) checkpoint loads into a tiled model unchanged.
    tiled.load_state_dict(dense.state_dict())

    batch = _synth_batch(with_edges=include_dynedge)
    with torch.no_grad():
        out_dense = dense(batch)
        out_tiled = tiled(batch)
    assert (out_dense - out_tiled).abs().max().item() == 0.0


def test_tiled_gradients_match_dense() -> None:
    """Gradients agree, including the checkpointed per-tile recompute.

    Run in train mode (all dropouts / drop-paths are 0, so it is
    deterministic).
    """
    torch.manual_seed(0)
    dense = DeepIce(**_kwargs()).double().train()
    tiled = (
        DeepIce(**_kwargs(), rel_attention="tiled", q_tile=16).double().train()
    )
    tiled.load_state_dict(dense.state_dict())

    batch = _synth_batch()
    dense(batch).pow(2).sum().backward()
    tiled(batch).pow(2).sum().backward()

    dense_grads = dict(dense.named_parameters())
    max_err = 0.0
    for name, p in tiled.named_parameters():
        if p.grad is None:
            continue
        gd = dense_grads[name].grad
        assert gd is not None
        max_err = max(max_err, (p.grad - gd).abs().max().item())
    assert max_err < 1e-8, max_err


def test_tiled_trains_with_checkpoint() -> None:
    """Train mode: finite loss, gradients flow through checkpointed tiles."""
    torch.manual_seed(0)
    tiled = (
        DeepIce(**_kwargs(), rel_attention="tiled", q_tile=8).double().train()
    )
    out = tiled(_synth_batch())
    out.pow(2).sum().backward()
    gsum = sum(
        p.grad.abs().sum().item()
        for p in tiled.parameters()
        if p.grad is not None
    )
    assert torch.isfinite(out).all() and gsum > 0


@pytest.mark.parametrize("q_tile", [1, 7, 16, 1000])
def test_encoder_tiles_match_untiled(q_tile: int) -> None:
    """Both encoders reproduce `forward` from a sequence of bands.

    `forward` is defined as the whole-sequence case of `forward_tiled`, so
    this pins the property that makes that definition safe: the bands, when
    stitched back together along the query axis, are the untiled result --
    including for a band size that does not divide the sequence length.
    """
    torch.manual_seed(0)
    length = 40
    x = torch.randn(3, length, 5, dtype=torch.float64)

    encoders = {
        "EPJC": SpacetimeEncoderEPJC(seq_length=32).double().eval(),
        # Non-default on every axis the generalisation exposes, so a band
        # that silently used the EPJC constants would not match.
        "configurable": SpacetimeEncoder(
            seq_length=32,
            output_dim=8,
            columns=(2, 3, 4, 0),
            time_scale=1.0,
            scale=97.0,
            clip=2.5,
            n_freq=188.0,
        )
        .double()
        .eval(),
    }

    for name, encoder in encoders.items():
        with torch.no_grad():
            full = encoder(x)
            stitched = torch.cat(
                [
                    encoder.forward_tiled(x, s, min(s + q_tile, length))
                    for s in range(0, length, q_tile)
                ],
                dim=1,
            )
        assert stitched.shape == full.shape, name
        assert torch.equal(stitched, full), name
