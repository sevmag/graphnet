"""DeepIce with a pair bias that runs inside fused attention.

`DeepIce` adds a learned function of each pair's separation to the attention
logits and output, which needs a `[batch, length, length, channels]` tensor
and padded attention. Here the bias is linear in the sinusoidally embedded
coordinate differences (`LinearDifferencesBias`), which makes it expressible
as ordinary attention over queries, keys and values widened by per-step
sines and cosines. The whole stack therefore runs on jagged `NestedTensor`s
through the fused variable-length kernels, with memory linear in the number
of steps, while every block still sees the separation of every pair.
"""

from typing import Any, Dict, List, Optional, Sequence

import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from graphnet.models.components.attention_blocks import Block
from graphnet.models.components.embedding import LinearDifferencesBias
from graphnet.models.gnn.gnn import GNN
from graphnet.models.transformer.icemix import DeepIce
from graphnet.models.transformer.inputs import (
    FourierSchema,
    build_fourier_tokenizer,
    embed_pulses,
    resolve_columns,
    resolve_coordinates,
)
from graphnet.models.utils import array_to_sequence


class DeepIceLinearBias(GNN):
    """DeepIce whose every block attends with a linear differences bias."""

    def __init__(
        self,
        hidden_dim: int = 384,
        mlp_ratio: int = 4,
        seq_length: int = 192,
        depth: int = 16,
        head_size: int = 32,
        scaled_emb: bool = False,
        n_features: int = 6,
        fourier_schema: Optional[FourierSchema] = None,
        input_feature_names: Optional[List[str]] = None,
        fourier_mlp_dim: Optional[int] = None,
        fourier_kwargs: Optional[Dict[str, Any]] = None,
        coordinate_features: Optional[Sequence[str]] = None,
        direction_features: Optional[Sequence[str]] = None,
        direction_scale: Optional[float] = None,
        direction_n_freq: Optional[float] = None,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        bias_seq_length: Optional[int] = None,
        bias_scale: float = 1024.0,
        bias_n_freq: float = 10000.0,
        bias_time_scale: float = 3e4 / 500 * 3e-1,
    ):
        """Construct `DeepIceLinearBias`.

        Args:
            hidden_dim: The latent feature dimension.
            mlp_ratio: Mlp expansion ratio of the transformer blocks.
            seq_length: The base feature dimension of the tokenizer.
            depth: The number of transformer blocks; every one carries the
                pair bias.
            head_size: The size of the attention heads.
            scaled_emb: Whether to scale the sinusoidal positional embeddings.
            n_features: The number of features in the input data, read by
                `FourierEncoderEPJC` when no `fourier_schema` is given.
            fourier_schema: See `DeepIce`.
            input_feature_names: Input column names, in order. Required with
                `fourier_schema` and `coordinate_features`.
            fourier_mlp_dim: See `DeepIce`.
            fourier_kwargs: See `DeepIce`.
            coordinate_features: Names of the x, y, z and time features, in
                that order, resolved against `input_feature_names`: the
                columns the pair bias takes its differences from. Unset,
                they are columns 0-3.
            direction_features: Names of the x, y and z components of each
                pulse's sensor pointing direction, resolved against
                `input_feature_names`. Given, the pair bias also embeds the
                difference of the two pulses' directions; see
                `LinearDifferencesBias`.
            direction_scale: Multiplier before the direction differences'
                own ladder; see `LinearDifferencesBias`.
            direction_n_freq: Span of that ladder.
            qk_norm: Per-head RMSNorm on queries and keys, applied before the
                bias is formed from the query.
            init_values: Initial LayerScale of the blocks, or None for none.
            bias_seq_length: Width of the sinusoidal embedding of each
                difference. Defaults to `head_size`, as the spacetime bias of
                `DeepIce`.
            bias_scale: Multiplier applied before that embedding's ladder.
            bias_n_freq: Span of that ladder.
            bias_time_scale: `t_scale * c / pos_scale` for the normalisation
                in use; a learnable multiplier on it starts at one. The
                default is the IceCube value, as in `DeepIce`.
        """
        super().__init__(seq_length, hidden_dim)
        self.fourier_ext, self.fourier_mlp = build_fourier_tokenizer(
            seq_length=seq_length,
            output_dim=hidden_dim,
            scaled=scaled_emb,
            n_features=n_features,
            fourier_schema=fourier_schema,
            input_feature_names=input_feature_names,
            mlp_dim=fourier_mlp_dim,
            fourier_kwargs=fourier_kwargs,
        )
        self._coordinate_columns = list(
            resolve_coordinates(
                coordinate_features,
                input_feature_names,
                "coordinate_features",
                default=(0, 1, 2, 3),
            )
        )
        self._direction_columns: Optional[List[int]] = None
        if direction_features is not None:
            if len(direction_features) != 3:
                raise ValueError(
                    "direction_features names the x, y and z components, got "
                    f"{list(direction_features)}"
                )
            self._direction_columns = resolve_columns(
                direction_features, input_feature_names, "direction_features"
            )
        self.pair_bias = LinearDifferencesBias(
            directions=direction_features is not None,
            direction_scale=direction_scale,
            direction_n_freq=direction_n_freq,
            head_dim=head_size,
            seq_length=bias_seq_length or head_size,
            time_scale=bias_time_scale,
            scale=bias_scale,
            n_freq=bias_n_freq,
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    input_dim=hidden_dim,
                    num_heads=hidden_dim // head_size,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    qk_norm=qk_norm,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass.

        The event vector is the mean of the pulses' outputs.
        """
        x, _, seq_length = array_to_sequence(data.x, data.batch, nested=True)
        x = embed_pulses(self.fourier_ext, self.fourier_mlp, x, seq_length)
        directions = (
            data.x[:, self._direction_columns]
            if self._direction_columns is not None
            else None
        )
        sin, cos = self.pair_bias.phases(
            data.x[:, self._coordinate_columns], directions
        )
        offsets = x.offsets()
        min_seqlen = x._get_min_seqlen()
        max_seqlen = x._get_max_seqlen()
        values = x.values()
        for blk in self.blocks:
            values = blk.forward_jagged(
                values,
                offsets,
                min_seqlen,
                max_seqlen,
                pair_bias=(self.pair_bias, sin, cos),
            )
        return DeepIce._mean_pool(values, data.batch, seq_length)
