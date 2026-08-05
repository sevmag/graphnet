"""Tests for the nD-RoPE spacetime variant (arXiv:2606.12146).

Three things are worth pinning: the wave-vector geometry the paper's
isotropy claim rests on, the relative-position property that makes any
RoPE a RoPE (attention depends on displacement, not absolute position),
and the fact that the per-head rotations are actually distinct and
reproducible.
"""

import math
from typing import List

import pytest
import torch
from torch_geometric.data import Data

from graphnet.models.components.layers import apply_spacetime_rope
from graphnet.models.transformer.icemix_rope_nd import (
    DeepIceRopeND,
    regular_simplex_directions,
    random_rotations,
)

N_DIMS = 4


def _make_batch(lengths: List[int], seed: int) -> Data:
    gen = torch.Generator().manual_seed(seed)
    total = sum(lengths)
    x = torch.randn(total, 5, generator=gen, dtype=torch.float32)
    batch_idx = torch.repeat_interleave(
        torch.arange(len(lengths)), torch.tensor(lengths)
    )
    data = Data(x=x, batch=batch_idx)
    data.num_graphs = len(lengths)
    return data


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_simplex_geometry(n: int) -> None:
    """The paper's defining simplex conditions (Eq.

    18).
    """
    w = regular_simplex_directions(n)
    assert w.shape == (n + 1, n)
    torch.testing.assert_close(
        w.sum(0), torch.zeros(n, dtype=w.dtype), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        w.norm(dim=1), torch.ones(n + 1, dtype=w.dtype), atol=1e-12, rtol=0
    )
    gram = w @ w.T
    off_diagonal = gram[~torch.eye(n + 1, dtype=torch.bool)]
    torch.testing.assert_close(
        off_diagonal,
        torch.full_like(off_diagonal, -1.0 / n),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("n", [2, 3, 4])
def test_second_order_isotropy(n: int) -> None:
    """`sum_i w_i w_i^T = ((n+1)/n) I` — equal response in every direction."""
    w = regular_simplex_directions(n)
    torch.testing.assert_close(
        w.T @ w,
        (n + 1) / n * torch.eye(n, dtype=w.dtype),
        atol=1e-12,
        rtol=0,
    )


def test_rotations_are_proper_and_preserve_geometry() -> None:
    """Per-head rotations are in SO(n) and keep the simplex a simplex."""
    gen = torch.Generator().manual_seed(0)
    rotations = random_rotations(N_DIMS, 8, gen)
    eye = torch.eye(N_DIMS, dtype=torch.float64)
    for q in rotations:
        torch.testing.assert_close(q @ q.T, eye, atol=1e-12, rtol=0)
        assert torch.det(q) > 0
    w = regular_simplex_directions(N_DIMS)
    for q in rotations:
        rotated = w @ q.T
        torch.testing.assert_close(
            rotated.T @ rotated,
            (N_DIMS + 1) / N_DIMS * eye,
            atol=1e-12,
            rtol=0,
        )


def test_per_head_wave_vectors_differ() -> None:
    """Heads must not share a preferred direction (the point of step 2)."""
    model = DeepIceRopeND(hidden_dim=96, head_size=48, depth=1, depth_rel=1)
    wave = model.rope_wave
    assert wave.shape == (2, N_DIMS + 1, N_DIMS)
    assert not torch.allclose(wave[0], wave[1], atol=1e-4)


def test_scale_ladder_and_plane_budget() -> None:
    """S = floor(half / (n+1)) scales; leftover planes stay unrotated."""
    model = DeepIceRopeND(hidden_dim=96, head_size=48, depth=1, depth_rel=1)
    # head_size 48 -> 24 planes -> 4 scales x 5 wave vectors = 20 used.
    assert model.rope_scales.numel() == 4
    assert model.rope_planes == 20
    expected_base = math.exp(48 / (2 * (N_DIMS + 1) * N_DIMS))
    assert model.rope_base == pytest.approx(expected_base)
    # The ladder starts at alpha_0 = base^0 = 1.
    assert model.rope_scales[0].item() == pytest.approx(1.0)


def test_angles_identity_on_cls_and_unused_planes() -> None:
    """Cls slots and planes beyond the budget carry no rotation."""
    model = DeepIceRopeND(hidden_dim=96, head_size=48, depth=1, depth_rel=1)
    data = _make_batch([3, 5], seed=4)
    cos, sin = model._rope_angles(data.x, data.batch, 2)
    assert cos.shape == (10, 2, 24)
    # cls slots sit at the start of each event.
    for slot in (0, 4):
        torch.testing.assert_close(cos[slot], torch.ones_like(cos[slot]))
        torch.testing.assert_close(sin[slot], torch.zeros_like(sin[slot]))
    torch.testing.assert_close(
        cos[:, :, model.rope_planes :],
        torch.ones_like(cos[:, :, model.rope_planes :]),
    )
    torch.testing.assert_close(
        sin[:, :, model.rope_planes :],
        torch.zeros_like(sin[:, :, model.rope_planes :]),
    )


def test_attention_logit_depends_only_on_displacement() -> None:
    """The RoPE property: translating an event leaves q.k unchanged.

    Rotating q at position x_i and k at x_j by the nD-RoPE phases makes
    their inner product a function of x_i - x_j alone, which is what makes
    the encoding relative.
    """
    torch.manual_seed(0)
    model = DeepIceRopeND(hidden_dim=96, head_size=48, depth=1, depth_rel=1)
    heads, head_dim = 2, 48
    q = torch.randn(2, heads * head_dim, dtype=torch.float32)
    k = torch.randn(2, heads * head_dim, dtype=torch.float32)

    def logits(feats: torch.Tensor) -> torch.Tensor:
        batch = torch.zeros(2, dtype=torch.long)
        # One event, so one cls slot precedes the two pulse rows.
        cos, sin = model._rope_angles(feats, batch, 1)
        # _rope_angles reserves a cls slot per event; take the pulse rows.
        cos, sin = cos[1:3], sin[1:3]
        qr = apply_spacetime_rope(q, cos, sin, heads, head_dim)
        kr = apply_spacetime_rope(k, cos, sin, heads, head_dim)
        qh = qr.unflatten(-1, [heads, head_dim])
        kh = kr.unflatten(-1, [heads, head_dim])
        # q of token 0 against k of token 1, per head.
        return (qh[0] * kh[1]).sum(-1)

    feats = torch.randn(2, 5)
    # A translation in space and in time; the time column carries the
    # largest axis scale, so it is the strongest test of phase accuracy.
    shift = torch.tensor([0.7, -1.3, 0.2, 0.0, 5.0])
    torch.testing.assert_close(
        logits(feats), logits(feats + shift), atol=1e-4, rtol=1e-4
    )


def test_absolute_position_still_matters_within_a_pair() -> None:
    """A guard against the trivial way to pass the previous test."""
    torch.manual_seed(1)
    model = DeepIceRopeND(hidden_dim=96, head_size=48, depth=1, depth_rel=1)
    heads, head_dim = 2, 48
    q = torch.randn(2, heads * head_dim)
    k = torch.randn(2, heads * head_dim)
    feats = torch.randn(2, 5)

    def logits(f: torch.Tensor) -> torch.Tensor:
        cos, sin = model._rope_angles(f, torch.zeros(2, dtype=torch.long), 1)
        qr = apply_spacetime_rope(q, cos[1:3], sin[1:3], heads, head_dim)
        kr = apply_spacetime_rope(k, cos[1:3], sin[1:3], heads, head_dim)
        qh = qr.unflatten(-1, [heads, head_dim])
        kh = kr.unflatten(-1, [heads, head_dim])
        return (qh[0] * kh[1]).sum(-1)

    moved = feats.clone()
    moved[1, 0] += 0.5  # change the displacement, not a global shift
    assert not torch.allclose(logits(feats), logits(moved), atol=1e-3)


def test_construction_is_reproducible_and_checkpointable() -> None:
    """Same seed -> same wave vectors; state dict carries them."""
    a = DeepIceRopeND(hidden_dim=96, head_size=48, depth=1, depth_rel=1)
    b = DeepIceRopeND(hidden_dim=96, head_size=48, depth=1, depth_rel=1)
    torch.testing.assert_close(a.rope_wave, b.rope_wave, atol=0, rtol=0)
    c = DeepIceRopeND(
        hidden_dim=96, head_size=48, depth=1, depth_rel=1, rope_seed=7
    )
    assert not torch.allclose(a.rope_wave, c.rope_wave, atol=1e-4)
    assert "rope_wave" in a.state_dict()
    c.load_state_dict(a.state_dict())
    torch.testing.assert_close(a.rope_wave, c.rope_wave, atol=0, rtol=0)


def test_forward_runs_and_is_finite() -> None:
    """End-to-end forward on a mixed-length batch."""
    model = DeepIceRopeND(
        hidden_dim=96, seq_length=64, depth=2, head_size=48, depth_rel=2
    )
    data = _make_batch([9, 1, 17], seed=8)
    out = model(data)
    assert out.shape == (3, 96)
    assert torch.isfinite(out).all()


def test_rejects_too_small_head_size() -> None:
    """A head with fewer planes than wave vectors cannot host the simplex."""
    with pytest.raises(ValueError, match="wave vectors"):
        DeepIceRopeND(hidden_dim=16, head_size=8, depth=1, depth_rel=1)
