"""Unit tests for `DirectionalSpacetimeEncoder`."""

from typing import Any, Dict

import pytest
import torch

from graphnet.models.components.embedding import DirectionalSpacetimeEncoder


def _steps() -> torch.Tensor:
    """Two events of four steps; the last two share a position and a time."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 4) * 0.3
    x[:, 3] = x[:, 2]
    return x


PAIR_FEATURES = pytest.mark.parametrize(
    "pair_features", ["polar", "cartesian", "differences"]
)


@PAIR_FEATURES
def test_output_shape(pair_features: str) -> None:
    """One feature vector per ordered pair of steps."""
    encoder = DirectionalSpacetimeEncoder(
        seq_length=8, output_dim=6, pair_features=pair_features
    )
    assert encoder(_steps()).shape == (2, 4, 4, 6)


def test_cartesian_embeds_six_scalars() -> None:
    """Three components, range, time difference and interval."""
    encoder = DirectionalSpacetimeEncoder(
        seq_length=8, pair_features="cartesian"
    )
    assert encoder.mlp[0].in_features == 6 * 8


def test_differences_embed_four_scalars() -> None:
    """Three components and the time difference, nothing derived."""
    encoder = DirectionalSpacetimeEncoder(
        seq_length=8, pair_features="differences"
    )
    assert encoder.mlp[0].in_features == 4 * 8


def test_unknown_pair_features() -> None:
    """A misspelt choice is an error, not a silent fallback."""
    with pytest.raises(ValueError):
        DirectionalSpacetimeEncoder(seq_length=8, pair_features="spherical")


def test_single_medium_has_no_film() -> None:
    """A constant modulation is redundant, so none is built."""
    encoder = DirectionalSpacetimeEncoder(seq_length=8)
    assert encoder.film is None and encoder.medium_code is None


def test_media_start_identical() -> None:
    """The per-medium speed and FiLM both start neutral."""
    encoder = DirectionalSpacetimeEncoder(seq_length=8, n_media=3)
    x = _steps()
    same = encoder(x, torch.tensor([0, 0]))
    other = encoder(x, torch.tensor([2, 1]))
    assert torch.allclose(same, other)


def test_media_diverge_once_trained() -> None:
    """Either conditioning path, once moved, separates the media."""
    x = _steps()
    for path in ("speed", "film"):
        encoder = DirectionalSpacetimeEncoder(seq_length=8, n_media=2)
        with torch.no_grad():
            if path == "speed":
                encoder.log_speed.weight[1] = 0.1
            else:
                assert encoder.film is not None
                # The bias is shared by every medium; the weights read
                # the medium's code.
                encoder.film.weight.fill_(0.1)
        first = encoder(x, torch.tensor([0, 0]))
        second = encoder(x, torch.tensor([1, 1]))
        assert not torch.allclose(first, second), path


@PAIR_FEATURES
def test_gradients_finite_for_coincident_steps(pair_features: str) -> None:
    """Pairs with no separation must not produce 0/0 gradients."""
    encoder = DirectionalSpacetimeEncoder(
        seq_length=8, n_media=2, pair_features=pair_features
    )
    encoder(_steps(), torch.tensor([0, 1])).sum().backward()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@PAIR_FEATURES
def test_distinguishes_direction(pair_features: str) -> None:
    """Mirrored pairs share range and interval but not direction."""
    encoder = DirectionalSpacetimeEncoder(
        seq_length=8, pair_features=pair_features
    )
    x = torch.zeros(1, 2, 4)
    x[0, 1, 0] = 0.1
    mirrored = x.clone()
    mirrored[0, 1, 0] = -0.1
    assert not torch.allclose(encoder(x)[0, 0, 1], encoder(mirrored)[0, 0, 1])


@PAIR_FEATURES
def test_distinguishes_time_order(pair_features: str) -> None:
    """The interval is symmetric in time; this encoder is not."""
    encoder = DirectionalSpacetimeEncoder(
        seq_length=8, pair_features=pair_features
    )
    x = torch.zeros(1, 2, 4)
    x[0, 1, 3] = 0.01
    reversed_order = x.clone()
    reversed_order[0, 1, 3] = -0.01
    assert not torch.allclose(
        encoder(x)[0, 0, 1], encoder(reversed_order)[0, 0, 1]
    )


@PAIR_FEATURES
def test_columns(pair_features: str) -> None:
    """Reordered input columns give the same output when `columns` says so."""
    encoder = DirectionalSpacetimeEncoder(
        seq_length=8, pair_features=pair_features
    )
    reordered = DirectionalSpacetimeEncoder(
        seq_length=8, columns=(1, 2, 3, 0), pair_features=pair_features
    )
    reordered.load_state_dict(encoder.state_dict())
    x = _steps()
    assert torch.allclose(encoder(x), reordered(x[:, :, [3, 0, 1, 2]]))


def _pointing_steps() -> torch.Tensor:
    """Two sensors at one position and time, facing up and sideways."""
    x = torch.zeros(1, 2, 7)
    x[0, 0, 6] = 1.0
    x[0, 1, 4] = 1.0
    return x


@PAIR_FEATURES
def test_sensor_directions_widen_the_input(pair_features: str) -> None:
    """Three difference components, each on the ladder."""
    plain = DirectionalSpacetimeEncoder(
        seq_length=8, pair_features=pair_features
    )
    pointing = DirectionalSpacetimeEncoder(
        seq_length=8,
        pair_features=pair_features,
        direction_columns=(4, 5, 6),
    )
    widened = pointing.mlp[0].in_features - plain.mlp[0].in_features
    assert widened == 3 * 8


@PAIR_FEATURES
def test_distinguishes_sensor_orientation(pair_features: str) -> None:
    """A coincident pair differs by how its two sensors face each other."""
    encoder = DirectionalSpacetimeEncoder(
        seq_length=8,
        pair_features=pair_features,
        direction_columns=(4, 5, 6),
    )
    crossed = _pointing_steps()
    aligned = crossed.clone()
    aligned[0, 1, 4:7] = aligned[0, 0, 4:7]
    assert not torch.allclose(
        encoder(crossed)[0, 0, 1], encoder(aligned)[0, 0, 1]
    )


def test_sensor_orientation_ignored_by_default() -> None:
    """Without direction columns the extra columns do not reach the output."""
    encoder = DirectionalSpacetimeEncoder(seq_length=8)
    crossed = _pointing_steps()
    aligned = crossed.clone()
    aligned[0, 1, 4:7] = aligned[0, 0, 4:7]
    assert torch.allclose(encoder(crossed), encoder(aligned))


def test_distinguishes_orientation_order() -> None:
    """Swapping the two sensors flips the sign of their difference."""
    encoder = DirectionalSpacetimeEncoder(
        seq_length=8, direction_columns=(4, 5, 6)
    )
    x = _pointing_steps()
    swapped = x[:, [1, 0]]
    assert not torch.allclose(encoder(x)[0, 0, 1], encoder(swapped)[0, 0, 1])


@pytest.mark.parametrize("n_media", [1, 2])
@pytest.mark.parametrize("pointing", [False, True, "own ladder"])
def test_separable_matches_the_pair_features(
    n_media: int, pointing: object
) -> None:
    """The per-step factorisation is the same function, gradients included."""
    torch.manual_seed(3)
    kwargs: Dict[str, Any] = dict(
        seq_length=8,
        pair_features="differences",
        clip=None,
        time_scale=3.0,
        n_media=n_media,
        direction_columns=(4, 5, 6) if pointing else None,
    )
    if pointing == "own ladder":
        kwargs.update(direction_scale=2.5, direction_n_freq=40.0)
    dense = DirectionalSpacetimeEncoder(**kwargs).double()
    separable = DirectionalSpacetimeEncoder(**kwargs, separable=True).double()
    # The comparison is about the algebra, so both evaluate their phases in
    # the input's precision.
    dense.sin_emb.phase_dtype = None
    dense.direction_emb.phase_dtype = None
    with torch.no_grad():
        for parameter in dense.parameters():
            parameter.normal_(std=0.3)
    separable.load_state_dict(dense.state_dict())

    x = torch.randn(2, 6, 7, dtype=torch.float64) * 0.3
    x[:, :, 4:] = x[:, :, 4:] / x[:, :, 4:].norm(dim=-1, keepdim=True)
    x[:, 5] = x[:, 4]
    medium = torch.tensor([0, n_media - 1])
    out_dense = dense(x, medium)
    out_separable = separable(x, medium)
    torch.testing.assert_close(out_separable, out_dense, atol=1e-9, rtol=1e-9)

    target = torch.randn_like(out_dense)
    (out_dense * target).sum().backward()
    (out_separable * target).sum().backward()
    for (name, a), b in zip(dense.named_parameters(), separable.parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(b.grad, a.grad, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize(
    "kwargs", [dict(pair_features="cartesian", clip=None), dict(clip=4.0)]
)
def test_separable_needs_unclipped_differences(kwargs: Dict[str, Any]) -> None:
    """Range, interval and a clip do not split into per-step factors."""
    arguments: Dict[str, Any] = {"pair_features": "differences", **kwargs}
    with pytest.raises(ValueError, match="separable"):
        DirectionalSpacetimeEncoder(seq_length=8, separable=True, **arguments)


def test_direction_ladder_is_its_own() -> None:
    """The direction differences get their band; the separation keeps its."""
    shared = DirectionalSpacetimeEncoder(
        seq_length=8, pair_features="differences", direction_columns=(4, 5, 6)
    )
    own = DirectionalSpacetimeEncoder(
        seq_length=8,
        pair_features="differences",
        direction_columns=(4, 5, 6),
        direction_scale=2.0,
        direction_n_freq=40.0,
    )
    assert shared.direction_emb is shared.sin_emb
    assert own.direction_emb is not own.sin_emb
    torch.testing.assert_close(own.sin_emb.freqs, shared.sin_emb.freqs)
    top = own.direction_scale * own.direction_emb.freqs
    assert top.max().item() == pytest.approx(2.0)
    assert top.max().item() / top.min().item() == pytest.approx(
        40.0 ** (3 / 4)
    )
    # The ladder has no parameters, so the state dict does not change.
    assert set(own.state_dict()) == set(shared.state_dict())


def test_direction_ladder_needs_direction_columns() -> None:
    """Without directions there is nothing for the ladder to embed."""
    with pytest.raises(ValueError, match="direction_columns"):
        DirectionalSpacetimeEncoder(seq_length=8, direction_scale=2.0)
