"""Unit tests for `DirectionalSpacetimeEncoder`."""

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
