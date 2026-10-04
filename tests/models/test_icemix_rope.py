"""Tests for the options of the rotary DeepIce models."""

import math
from typing import Any, Dict, List

import pytest
import torch
from torch_geometric.data import Batch, Data

from graphnet.models.components.attention_blocks import apply_spacetime_rope
from graphnet.models.transformer import DeepIceRope, DeepIceRopeND

# A head size both constructions accept: divisible by 8 and by 10.
BOTH = pytest.mark.parametrize(
    "model_class, sizes",
    [
        (DeepIceRope, dict(hidden_dim=64, head_size=16)),
        (DeepIceRopeND, dict(hidden_dim=80, head_size=40)),
    ],
)

HIDDEN_DIM = 64


def _batch(lengths: List[int] = [17, 30, 9]) -> Batch:
    """Build a batch of variable-length synthetic events."""
    generator = torch.Generator().manual_seed(1)
    return Batch.from_data_list(
        [Data(x=torch.randn(n, 5, generator=generator) * 0.3) for n in lengths]
    )


def _rope(model_class: type = DeepIceRope, **overrides: Any) -> DeepIceRope:
    kwargs: Dict[str, Any] = dict(
        hidden_dim=HIDDEN_DIM, seq_length=32, depth=3, head_size=16
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return model_class(**kwargs).eval()


def test_depth_counts_every_block() -> None:
    """`depth` alone sets the size of the model."""
    model = _rope()
    assert len(model.blocks) == 3
    assert len(model.sandwich) == 0


def test_depth_rel_only_changes_the_state_dict_layout() -> None:
    """Blocks named by `depth_rel` are ordinary leading blocks."""
    split = _rope(depth=2, depth_rel=1)
    assert [len(split.sandwich), len(split.blocks)] == [1, 2]

    renamed = {}
    for key, value in split.state_dict().items():
        part, _, rest = key.partition(".")
        if part == "sandwich":
            key = f"blocks.{rest}"
        elif part == "blocks":
            index, _, tail = rest.partition(".")
            key = f"blocks.{int(index) + 1}.{tail}"
        renamed[key] = value
    joined = _rope(depth=3)
    joined.load_state_dict(renamed)

    batch = _batch()
    with torch.no_grad():
        torch.testing.assert_close(joined(batch), split(batch))


@BOTH
def test_qk_norm_reaches_every_block(
    model_class: type, sizes: Dict[str, int]
) -> None:
    """Unset, no block normalises; set, each one does."""
    plain = _rope(model_class, **sizes)
    assert all(block.q_norm is None for block in plain.blocks)

    normed = _rope(model_class, qk_norm=True, **sizes)
    assert all(
        block.q_norm is not None and block.k_norm is not None
        for block in normed.blocks
    )
    with torch.no_grad():
        out = normed(_batch())
    assert out.shape == (3, sizes["hidden_dim"])
    assert torch.isfinite(out).all()


@BOTH
def test_mean_pooling_reads_each_event_alone(
    model_class: type, sizes: Dict[str, int]
) -> None:
    """An event's vector is the same alone as inside a batch."""
    model = _rope(model_class, pooling="mean", **sizes)
    batch = _batch()
    with torch.no_grad():
        together = model(batch)
        for i, event in enumerate(batch.to_data_list()):
            alone = model(Batch.from_data_list([event]))
            torch.testing.assert_close(
                alone[0], together[i], atol=1e-5, rtol=1e-4
            )


@BOTH
def test_mean_pooling_ignores_the_pulse_order(
    model_class: type, sizes: Dict[str, int]
) -> None:
    """Pulses are a set: only their coordinates place them."""
    model = _rope(model_class, pooling="mean", **sizes)
    event = _batch([25])
    shuffled = event.clone()
    shuffled.x = event.x[torch.randperm(25, generator=torch.manual_seed(3))]
    with torch.no_grad():
        torch.testing.assert_close(
            model(shuffled), model(event), atol=1e-5, rtol=1e-4
        )


@BOTH
def test_mean_pooling_runs_no_class_token(
    model_class: type, sizes: Dict[str, int]
) -> None:
    """The rotation table then has one row per pulse and no more."""
    batch = _batch()
    n_pulses, n_events = batch.x.shape[0], 3
    mean = _rope(model_class, pooling="mean", **sizes)
    cos, _ = mean._rope_angles(batch.x, batch.batch, n_events)
    assert cos.shape[0] == n_pulses

    cls = _rope(model_class, **sizes)
    cos, _ = cls._rope_angles(batch.x, batch.batch, n_events)
    assert cos.shape[0] == n_pulses + n_events
    cls.load_state_dict(mean.state_dict())
    with torch.no_grad():
        assert not torch.allclose(mean(batch), cls(batch), atol=1e-3)


def test_unknown_pooling_raises() -> None:
    """Only the two readouts exist."""
    with pytest.raises(ValueError, match="pooling"):
        _rope(pooling="max")


def test_rope_per_head_spreads_each_band_over_the_heads() -> None:
    """Together the heads hold one geometric ladder per axis."""
    bands = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (0.4, 4000.0)]
    model = _rope(rope_axis_bands=bands, rope_per_head=True)
    n_heads, pairs = 4, 2
    omega = model.rope_omega.view(n_heads, 4, pairs)
    for axis, (lo, hi) in enumerate(bands):
        ladder = torch.logspace(
            math.log10(lo), math.log10(hi), n_heads * pairs, dtype=omega.dtype
        )
        # Head h holds the h-th frequency and every n_heads-th after it.
        torch.testing.assert_close(omega[:, axis].T.flatten(), ladder)


def test_rope_per_head_logit_depends_only_on_displacement() -> None:
    """Each head's own table still makes the encoding relative."""
    torch.manual_seed(0)
    model = _rope(rope_axis_bands=[(0.5, 50.0)] * 4, rope_per_head=True)
    heads, head_dim = 4, 16
    q = torch.randn(2, heads * head_dim)
    k = torch.randn(2, heads * head_dim)

    def logits(feats: torch.Tensor) -> torch.Tensor:
        # One event, so one class-token slot precedes the two pulse rows.
        cos, sin = model._rope_angles(feats, torch.zeros(2).long(), 1)
        cos, sin = cos[1:3], sin[1:3]
        assert cos.shape == (2, heads, head_dim // 2)
        qh = apply_spacetime_rope(q, cos, sin, heads, head_dim)
        kh = apply_spacetime_rope(k, cos, sin, heads, head_dim)
        return (qh[0] * kh[1]).unflatten(-1, [heads, head_dim]).sum(-1)

    feats = torch.randn(2, 5)
    # Column 3 is charge in the default NuBench order; time is column 4.
    shift = torch.tensor([0.7, -1.3, 0.2, 0.0, 0.4])
    torch.testing.assert_close(
        logits(feats), logits(feats + shift), atol=1e-3, rtol=1e-3
    )
    moved = feats.clone()
    moved[1, 0] += 0.5
    assert not torch.allclose(logits(feats), logits(moved), atol=1e-2)


def test_rope_per_head_needs_per_axis_bands() -> None:
    """There is no per-axis band to spread under the single shared ladder."""
    with pytest.raises(ValueError, match="rope_per_head"):
        _rope(rope_per_axis=False, rope_per_head=True)


def test_rope_per_head_forward_is_finite() -> None:
    """End to end, with the normalisation and the mean readout."""
    model = _rope(rope_per_head=True, qk_norm=True, pooling="mean")
    with torch.no_grad():
        out = model(_batch())
    assert out.shape == (3, HIDDEN_DIM)
    assert torch.isfinite(out).all()
