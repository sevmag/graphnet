"""Tests for how the transformer models read their input columns.

Three properties carry the feature. A schema that spells out the layout of
`FourierEncoderEPJC` reproduces it, so the schema-driven path is a
generalisation rather than a second model. Naming the columns makes a model
indifferent to the order they arrive in. And left unset, every model keeps
the parameters it had, so its checkpoints still load.
"""

from typing import Any, Dict, List

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from graphnet.models.components.embedding import FourierEncoder
from graphnet.models.transformer import DeepIce, DeepIceRope, DeepIceRopeND
from graphnet.models.transformer.inputs import (
    build_fourier_tokenizer,
    embed_pulses,
    resolve_coordinates,
)

SEQ_LENGTH = 32
HIDDEN_DIM = 64
# `FourierEncoderEPJC` embeds columns 0-2 and 3 with 4096 and column 4 with
# 1024, concatenated as position, column 4, column 3.
KAGGLE_NAMES = ["x", "y", "z", "t", "q"]
KAGGLE_SCHEMA = {
    "x": 4096.0,
    "y": 4096.0,
    "z": 4096.0,
    "q": 1024.0,
    "t": 4096.0,
}
# The same layout read off the NuBench order, where column 3 is charge.
NUBENCH_NAMES = ["x", "y", "z", "q", "t"]
NUBENCH_SCHEMA = {
    "x": 4096.0,
    "y": 4096.0,
    "z": 4096.0,
    "t": 1024.0,
    "q": 4096.0,
}
COORDINATES = ["x", "y", "z", "t"]


def _batch(n_features: int = 5, lengths: List[int] = [17, 30, 9]) -> Batch:
    """Build a batch of variable-length synthetic events."""
    generator = torch.Generator().manual_seed(1)
    return Batch.from_data_list(
        [
            Data(x=torch.randn(n, n_features, generator=generator) * 0.3)
            for n in lengths
        ]
    )


def _reordered(batch: Batch, names: List[str], order: List[str]) -> Batch:
    """Return `batch` with its columns arranged as `order`."""
    out = batch.clone()
    out.x = batch.x[:, [names.index(name) for name in order]]
    return out


def _schema_weights(epjc: torch.nn.Module) -> Dict[str, Tensor]:
    """State dict of an EPJC-tokenized model, renamed for the schema path."""
    return {
        key.replace("fourier_ext.mlp.", "fourier_mlp."): value
        for key, value in epjc.state_dict().items()
    }


def _deepice(**overrides: Any) -> DeepIce:
    kwargs: Dict[str, Any] = dict(
        hidden_dim=HIDDEN_DIM,
        seq_length=SEQ_LENGTH,
        depth=2,
        head_size=16,
        depth_rel=2,
        n_rel=1,
        n_features=5,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return DeepIce(**kwargs).eval()


def _rope(model_class: type = DeepIceRope, **overrides: Any) -> DeepIceRope:
    kwargs: Dict[str, Any] = dict(
        hidden_dim=HIDDEN_DIM,
        seq_length=SEQ_LENGTH,
        depth=3,
        head_size=16,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return model_class(**kwargs).eval()


def test_fourier_encoder_jagged_matches_padded() -> None:
    """The jagged path embeds each real pulse as the padded one does."""
    batch = _batch()
    lengths = torch.bincount(batch.batch)
    padded = torch.zeros(len(lengths), int(lengths.max()), batch.x.shape[1])
    mask = torch.arange(padded.shape[1])[None] < lengths[:, None]
    padded[mask] = batch.x
    offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    jagged = torch.nested.nested_tensor_from_jagged(batch.x, offsets)

    encoder = FourierEncoder({0: 4096.0, 3: (1024.0, 500.0)}, SEQ_LENGTH)
    out = encoder(jagged, lengths)
    assert out.is_nested
    torch.testing.assert_close(out.values(), encoder(padded, lengths)[mask])


def test_embed_pulses_projects_jagged_tokens() -> None:
    """Tokens come back jagged, one of the model width per pulse."""
    batch = _batch()
    lengths = torch.bincount(batch.batch)
    offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    jagged = torch.nested.nested_tensor_from_jagged(batch.x, offsets)
    encoder, projection = build_fourier_tokenizer(
        SEQ_LENGTH,
        HIDDEN_DIM,
        fourier_schema=KAGGLE_SCHEMA,
        input_feature_names=KAGGLE_NAMES,
    )
    tokens = embed_pulses(encoder, projection, jagged, lengths)
    assert tokens.is_nested
    assert tokens.values().shape == (batch.x.shape[0], HIDDEN_DIM)


def test_unset_schema_keeps_the_epjc_parameters() -> None:
    """Without a schema no model gains or renames a parameter."""
    rope_nd = _rope(DeepIceRopeND, head_size=40, hidden_dim=80)
    for model in (_deepice(), _rope(), rope_nd):
        assert model.fourier_mlp is None
        keys = [k for k in model.state_dict() if k.startswith("fourier_")]
        assert keys and all(k.startswith("fourier_ext.mlp.") for k in keys)


def test_deepice_schema_reproduces_epjc() -> None:
    """The EPJC layout written as a schema gives the EPJC output."""
    batch = _batch()
    epjc = _deepice()
    schema = _deepice(
        fourier_schema=KAGGLE_SCHEMA, input_feature_names=KAGGLE_NAMES
    )
    schema.load_state_dict(_schema_weights(epjc))
    with torch.no_grad():
        torch.testing.assert_close(schema(batch), epjc(batch))


def test_rope_schema_reproduces_epjc() -> None:
    """The same holds on the jagged path of the rotary models."""
    batch = _batch()
    epjc = _rope()
    schema = _rope(
        fourier_schema=NUBENCH_SCHEMA, input_feature_names=NUBENCH_NAMES
    )
    schema.load_state_dict(_schema_weights(epjc))
    with torch.no_grad():
        torch.testing.assert_close(schema(batch), epjc(batch))


def test_deepice_follows_named_columns() -> None:
    """Reordered input, named accordingly, gives the same event vector."""
    order = ["q", "t", "z", "y", "x"]
    batch = _batch()
    reference = _deepice(
        fourier_schema=KAGGLE_SCHEMA, input_feature_names=KAGGLE_NAMES
    )
    reordered = _deepice(
        fourier_schema=KAGGLE_SCHEMA,
        input_feature_names=order,
        spacetime_features=COORDINATES,
    )
    reordered.load_state_dict(reference.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(
            reordered(_reordered(batch, KAGGLE_NAMES, order)),
            reference(batch),
        )


@pytest.mark.parametrize(
    "model_class, head_size", [(DeepIceRope, 16), (DeepIceRopeND, 40)]
)
def test_rope_follows_named_columns(model_class: type, head_size: int) -> None:
    """The rotation reads the coordinates wherever they are named to be."""
    order = ["t", "q", "z", "y", "x"]
    batch = _batch()
    reference = _rope(
        model_class,
        head_size=head_size,
        hidden_dim=2 * head_size,
        fourier_schema=NUBENCH_SCHEMA,
        input_feature_names=NUBENCH_NAMES,
    )
    reordered = _rope(
        model_class,
        head_size=head_size,
        hidden_dim=2 * head_size,
        fourier_schema=NUBENCH_SCHEMA,
        input_feature_names=order,
        coordinate_features=COORDINATES,
    )
    reordered.load_state_dict(reference.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(
            reordered(_reordered(batch, NUBENCH_NAMES, order)),
            reference(batch),
        )


def test_rope_needs_no_charge_column_when_coordinates_are_named() -> None:
    """Four named columns suffice; the NuBench order is only the default."""
    with pytest.raises(ValueError, match="NuBench feature order"):
        _rope(n_features=4)
    model = _rope(
        fourier_schema={name: 4096.0 for name in COORDINATES},
        input_feature_names=COORDINATES,
        coordinate_features=COORDINATES,
    )
    with torch.no_grad():
        assert model(_batch(n_features=4)).shape == (3, HIDDEN_DIM)


def test_rope_axis_bands_set_the_rotation_frequencies() -> None:
    """Each axis's ladder runs between the two frequencies given for it."""
    bands = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0)]
    model = _rope(rope_axis_bands=bands)
    per_axis = model.rope_omega.view(4, -1)
    torch.testing.assert_close(
        per_axis[:, [0, -1]], torch.tensor(bands, dtype=per_axis.dtype)
    )
    assert not torch.equal(model.rope_omega, _rope().rope_omega)


@pytest.mark.parametrize(
    "bands",
    [
        [(1.0, 10.0)] * 3,
        [(0.0, 10.0)] * 4,
        [(10.0, 1.0)] * 4,
    ],
)
def test_rope_axis_bands_are_validated(bands: List[Any]) -> None:
    """A band is one positive, ordered pair per coordinate."""
    with pytest.raises(ValueError, match="rope_axis_bands"):
        _rope(rope_axis_bands=bands)


def test_rope_axis_bands_need_per_axis_ladders() -> None:
    """Bands cannot shape the single shared ladder."""
    with pytest.raises(ValueError, match="rope_per_axis"):
        _rope(rope_per_axis=False, rope_axis_bands=[(1.0, 10.0)] * 4)


def test_fourier_kwargs_reach_the_encoder() -> None:
    """Encoder settings beyond the schema pass through, and only with one."""
    model = _deepice(
        fourier_schema=KAGGLE_SCHEMA,
        input_feature_names=KAGGLE_NAMES,
        fourier_kwargs={"add_sequence_length": False},
    )
    assert model.fourier_ext.output_dim == len(KAGGLE_SCHEMA) * SEQ_LENGTH
    with pytest.raises(ValueError, match="fourier_kwargs"):
        _deepice(fourier_kwargs={"add_sequence_length": False})


def test_unknown_names_raise() -> None:
    """A name that is not an input column is an error, not a guess."""
    with pytest.raises(ValueError, match="fourier_schema names"):
        _rope(fourier_schema=NUBENCH_SCHEMA, input_feature_names=["x", "y"])
    with pytest.raises(ValueError, match="spacetime_features names"):
        _deepice(
            input_feature_names=KAGGLE_NAMES,
            spacetime_features=["x", "y", "z", "time"],
        )
    with pytest.raises(ValueError, match="x, y, z and time"):
        resolve_coordinates(["x", "y", "z"], KAGGLE_NAMES, "arg", (0, 1, 2, 3))


def test_iseecube_runs_on_both_tokenizers() -> None:
    """ISeeCube embeds its pulses with either encoder."""
    pytest.importorskip("torchscale")
    from graphnet.models.transformer import ISeeCube

    names = KAGGLE_NAMES + ["aux"]
    generator = torch.Generator().manual_seed(1)
    events = []
    for _ in range(3):
        x = torch.randn(16, 6, generator=generator) * 0.3
        x[:, 5] = torch.randint(0, 2, (16,), generator=generator).float()
        events.append(Data(x=x))
    batch = Batch.from_data_list(events)
    extras: List[Dict[str, Any]] = [
        {},
        dict(fourier_schema=KAGGLE_SCHEMA, input_feature_names=names),
    ]
    for extra in extras:
        torch.manual_seed(0)
        model = ISeeCube(
            hidden_dim=48,
            seq_length=16,
            num_layers=2,
            num_heads=4,
            mlp_dim=64,
            **extra,
        ).eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (3, 48)
        assert torch.isfinite(out).all()
