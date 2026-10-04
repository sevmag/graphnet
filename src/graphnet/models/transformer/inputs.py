"""How the transformer models read their input columns.

Every model here turns a pulse into a token with a Fourier encoder followed
by a projection to the model width, and reads the pulse's coordinates again
for its relative spacetime encoding. `FourierEncoderEPJC` fixes the column
layout and the multipliers to those of the Kaggle dataset and holds its own
projection; `FourierEncoder` embeds the columns a schema names, over the bands
it states, and leaves the projection to the model. The functions here build
and apply either from one set of arguments, and resolve feature names to
columns, so that each model offers the same choices without restating them.
"""

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from torch import Tensor

from graphnet.models.components.embedding import (
    FourierEncoder,
    FourierEncoderEPJC,
)

# A mapping rather than a dict, so that a plain `{name: multiplier}` is one.
FourierSchema = Mapping[str, Union[float, Tuple[float, float]]]


def resolve_columns(
    features: Sequence[str],
    input_feature_names: Optional[Sequence[str]],
    argument: str,
) -> List[int]:
    """Return the input column of each named feature.

    Naming the features is what makes a change of input order raise rather
    than shift every setting onto a neighbouring column.

    Args:
        features: Feature names to look up.
        input_feature_names: Input column names, in order.
        argument: Name of the argument `features` came from, for the error.

    Returns:
        The column index of each feature, in the order given.

    Raises:
        ValueError: If a feature is not among the input columns.
    """
    names = list(input_feature_names or [])
    unknown = sorted(set(features) - set(names))
    if unknown:
        raise ValueError(
            f"{argument} names {unknown}, not among the input features "
            f"{names}."
        )
    return [names.index(feature) for feature in features]


def resolve_coordinates(
    features: Optional[Sequence[str]],
    input_feature_names: Optional[Sequence[str]],
    argument: str,
    default: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """Return the input columns holding `(x, y, z, t)`.

    Args:
        features: Names of the x, y, z and time features, in that order, or
            None to keep `default`.
        input_feature_names: Input column names, in order.
        argument: Name of the argument `features` came from, for the error.
        default: Columns assumed when no names are given.

    Returns:
        The four column indices.

    Raises:
        ValueError: If `features` does not name exactly four input columns.
    """
    if features is None:
        return default
    if len(features) != 4:
        raise ValueError(
            f"{argument} must name the x, y, z and time features, got "
            f"{list(features)}."
        )
    x, y, z, t = resolve_columns(features, input_feature_names, argument)
    return x, y, z, t


def build_fourier_tokenizer(
    seq_length: int,
    output_dim: int,
    scaled: bool = False,
    n_features: int = 6,
    fourier_schema: Optional[FourierSchema] = None,
    input_feature_names: Optional[Sequence[str]] = None,
    mlp_dim: Optional[int] = None,
    fourier_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Optional[nn.Module]]:
    """Build a model's Fourier encoder and, if it needs one, its projection.

    Args:
        seq_length: Width of the sinusoidal embedding of one column.
        output_dim: Width of a token.
        scaled: Whether to scale the sinusoidal embeddings.
        n_features: Number of input columns. Read by `FourierEncoderEPJC`
            only; a schema states its own columns.
        fourier_schema: `{feature name: multiplier}` or
            `{feature name: (multiplier, n_freq)}` for the columns to embed,
            concatenated in the order given and resolved against
            `input_feature_names`. Unset, the encoder is
            `FourierEncoderEPJC` with its fixed layout.
        input_feature_names: Input column names, in order. Required with
            `fourier_schema`.
        mlp_dim: Hidden width of the projection. Unset, it is the width of
            the concatenated embeddings.
        fourier_kwargs: Further arguments of `FourierEncoder`: `n_freq`,
            `add_sequence_length` and `phase_dtype`. Only with
            `fourier_schema`, since `FourierEncoderEPJC` fixes them.

    Returns:
        The encoder and the projection from its output to `output_dim`. The
        projection is None for `FourierEncoderEPJC`, which holds its own.

    Raises:
        ValueError: If `fourier_kwargs` is given without `fourier_schema`,
            or the schema names a feature that is not an input column.
    """
    if fourier_schema is None:
        if fourier_kwargs:
            raise ValueError(
                "fourier_kwargs configure `FourierEncoder`, which is built "
                "only when a fourier_schema is given."
            )
        return (
            FourierEncoderEPJC(
                seq_length=seq_length,
                mlp_dim=mlp_dim,
                output_dim=output_dim,
                scaled=scaled,
                n_features=n_features,
            ),
            None,
        )
    columns = resolve_columns(
        list(fourier_schema), input_feature_names, "fourier_schema"
    )
    encoder = FourierEncoder(
        schema={
            column: (
                (float(v[0]), float(v[1]))
                if isinstance(v, (tuple, list))
                else float(v)
            )
            for column, v in zip(columns, fourier_schema.values())
        },
        seq_length=seq_length,
        scaled=scaled,
        **(fourier_kwargs or {}),
    )
    concat_dim = encoder.output_dim
    hidden_dim = concat_dim if mlp_dim is None else mlp_dim
    projection = nn.Sequential(
        nn.Linear(concat_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )
    return encoder, projection


def embed_pulses(
    encoder: nn.Module,
    projection: Optional[nn.Module],
    x: Tensor,
    seq_length: Tensor,
) -> Tensor:
    """Turn pulses into tokens of the model width.

    Args:
        encoder: The encoder returned by `build_fourier_tokenizer`.
        projection: The projection it returned, or None.
        x: Pulses, padded `[B, L, D]` or a jagged `NestedTensor`.
        seq_length: `[B]` unpadded length of each event.

    Returns:
        The tokens, padded or jagged like `x`.
    """
    x = encoder(x, seq_length)
    if projection is None:
        return x
    if x.is_nested:
        # The projection is per pulse, so on jagged input it runs on the
        # dense values buffer: no compute on padding, and dense ops cover
        # what jagged eager kernels do not.
        return torch.nested.nested_tensor_from_jagged(
            projection(x.values()),
            x.offsets(),
            min_seqlen=x._get_min_seqlen(),
            max_seqlen=x._get_max_seqlen(),
        )
    return projection(x)
