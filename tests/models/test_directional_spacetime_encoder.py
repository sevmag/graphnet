"""Unit tests for `DirectionalSpacetimeEncoder`."""

import torch

from graphnet.models.components.embedding import DirectionalSpacetimeEncoder


def _steps() -> torch.Tensor:
    """Two events of four steps; the last two share a position and a time."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 4) * 0.3
    x[:, 3] = x[:, 2]
    return x


def test_output_shape() -> None:
    """One feature vector per ordered pair of steps."""
    encoder = DirectionalSpacetimeEncoder(seq_length=8, output_dim=6)
    assert encoder(_steps()).shape == (2, 4, 4, 6)


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


def test_gradients_finite_for_coincident_steps() -> None:
    """Pairs with no separation must not produce 0/0 gradients."""
    encoder = DirectionalSpacetimeEncoder(seq_length=8, n_media=2)
    encoder(_steps(), torch.tensor([0, 1])).sum().backward()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_distinguishes_direction() -> None:
    """Mirrored pairs share range and interval but not direction."""
    encoder = DirectionalSpacetimeEncoder(seq_length=8)
    x = torch.zeros(1, 2, 4)
    x[0, 1, 0] = 0.1
    mirrored = x.clone()
    mirrored[0, 1, 0] = -0.1
    assert not torch.allclose(encoder(x)[0, 0, 1], encoder(mirrored)[0, 0, 1])


def test_distinguishes_time_order() -> None:
    """The interval is symmetric in time; this encoder is not."""
    encoder = DirectionalSpacetimeEncoder(seq_length=8)
    x = torch.zeros(1, 2, 4)
    x[0, 1, 3] = 0.01
    reversed_order = x.clone()
    reversed_order[0, 1, 3] = -0.01
    assert not torch.allclose(
        encoder(x)[0, 0, 1], encoder(reversed_order)[0, 0, 1]
    )


def test_columns() -> None:
    """Reordered input columns give the same output when `columns` says so."""
    encoder = DirectionalSpacetimeEncoder(seq_length=8)
    reordered = DirectionalSpacetimeEncoder(seq_length=8, columns=(1, 2, 3, 0))
    reordered.load_state_dict(encoder.state_dict())
    x = _steps()
    assert torch.allclose(encoder(x), reordered(x[:, :, [3, 0, 1, 2]]))
