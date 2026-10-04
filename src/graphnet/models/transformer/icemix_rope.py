"""DeepIce variant with rotary spacetime attention instead of bias terms.

The relative-attention sandwich and its `SpacetimeEncoder` bias are
replaced by plain bidirectional blocks whose queries and keys are rotated
by fixed-frequency multiples of each pulse's (x, y, z, t) coordinates, so
the QK product depends on the coordinates only through their differences
-- relative spacetime geometry at zero extra attention cost. The whole
stack runs on jagged `NestedTensor`s and dispatches to fused
variable-length (flash) kernels; the padded [B, L, D] tensor is never
materialised.
"""

import math

import torch
import torch._dynamo
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from graphnet.models.components.attention_blocks import Block
from graphnet.models.gnn.gnn import GNN
from graphnet.models.transformer.icemix import DeepIce
from graphnet.models.transformer.inputs import (
    FourierSchema,
    build_fourier_tokenizer,
    embed_pulses,
    resolve_coordinates,
)
from graphnet.models.utils import array_to_sequence

from torch_geometric.data import Data
from torch import Tensor

# One geometric frequency band per coordinate (x, y, z, t), spanning the range
# where that axis's measured pulse-pair coordinate differences actually vary
# on the hexagon detector. A single shared ladder wastes most frequencies
# out-of-band per axis; matched bands place every frequency where it resolves.
HEXAGON_AXIS_BANDS = (
    (0.50, 8.23),
    (0.47, 16.12),
    (5.84, 193.57),
    (1237.0, 68921.0),
)


class DeepIceRope(GNN):
    """DeepIce with per-coordinate rotary attention, jagged end-to-end."""

    def __init__(
        self,
        hidden_dim: int = 384,
        mlp_ratio: int = 4,
        seq_length: int = 192,
        depth: int = 16,
        head_size: int = 32,
        depth_rel: int = 0,
        scaled_emb: bool = False,
        n_features: int = 5,
        rope_per_axis: bool = True,
        compile_blocks: bool = False,
        fourier_schema: Optional[FourierSchema] = None,
        input_feature_names: Optional[List[str]] = None,
        fourier_mlp_dim: Optional[int] = None,
        fourier_kwargs: Optional[Dict[str, Any]] = None,
        coordinate_features: Optional[Sequence[str]] = None,
        rope_axis_bands: Optional[Sequence[Tuple[float, float]]] = None,
        qk_norm: bool = False,
        pooling: str = "cls",
    ):
        """Construct `DeepIceRope`.

        Args:
            hidden_dim: The latent feature dimension.
            mlp_ratio: Mlp expansion ratio of FourierEncoderEPJC and
                Transformer.
            seq_length: The base feature dimension.
            depth: The number of transformer blocks. Every block rotates.
            head_size: The size of the attention heads. Must be divisible
                by 8 (2D rotation pairs split over 4 coordinates).
            depth_rel: Further blocks, run ahead of the `depth` ones and
                held in a separate module list. They are the same blocks;
                the split only reproduces the state-dict layout of
                `DeepIce`, whose leading blocks are of another kind, for
                checkpoints and callers built on that layout.
            scaled_emb: Whether to scale the sinusoidal positional embeddings.
            n_features: The number of features in the input data, read by
                `FourierEncoderEPJC`. Without `coordinate_features` at least
                5, in the NuBench order (x, y, z, charge, t): the rotation
                then reads the coordinates from columns 0-2 and time from
                column 4.
            rope_per_axis: Use a separate geometric RoPE frequency band per
                coordinate (x, y, z, t) instead of one shared ladder repeated
                across axes. The bands are `rope_axis_bands`.
            compile_blocks: Wrap the transformer block stack in
                `torch.compile`. The jagged path issues many small ops per
                step; compiling the whole stack as one graph is what turns
                it from slower-than-padded (eager) into faster. No effect
                on numerics.
            fourier_schema: `{feature name: multiplier}` or
                `{feature name: (multiplier, n_freq)}` for the columns to
                embed, resolved against `input_feature_names`. Unset, the
                encoder is `FourierEncoderEPJC` with its fixed layout.
            input_feature_names: Input column names, in order. Required with
                `fourier_schema` and `coordinate_features`.
            fourier_mlp_dim: Hidden width of the projection that turns the
                concatenated sinusoidal features into `hidden_dim`. Unset, it
                is the concatenation's own width.
            fourier_kwargs: Further arguments of `FourierEncoder`: `n_freq`,
                `add_sequence_length` and `phase_dtype`. Only with
                `fourier_schema`.
            coordinate_features: Names of the x, y, z and time features, in
                that order, resolved against `input_feature_names`: the
                columns the rotation reads. Unset, they are columns 0-2 and
                4.
            rope_axis_bands: Lowest and highest rotation frequency for each
                of x, y, z and t, in radians per unit of the normalised
                coordinate. A band should span the range over which that
                coordinate's pulse-pair differences vary, so it depends on
                the detector and its normalisation. Defaults to the bands
                measured on the hexagon detector. Only with `rope_per_axis`.
            qk_norm: Per-head RMSNorm on queries and keys before they are
                rotated, bounding the growth of the attention logits. The
                rotation leaves the normalised length unchanged.
            pooling: How the per-pulse embeddings become the one event
                vector the task head reads. `"cls"` prepends a learned
                token, which carries no coordinates and is not rotated, and
                returns its output; `"mean"` averages the pulses and runs no
                extra token. As on `DeepIce`, the `cls_token` parameter
                exists either way, so a checkpoint loads under both.
        """
        super().__init__(seq_length, hidden_dim)
        if head_size % 8 != 0:
            raise ValueError(
                "DeepIceRope needs head_size divisible by 8 "
                "(2D rotation pairs split over 4 coordinates), got "
                f"{head_size}."
            )
        if pooling not in ("cls", "mean"):
            raise ValueError(
                f"pooling must be 'cls' or 'mean', got {pooling!r}"
            )
        self.pooling = pooling
        if coordinate_features is None and n_features < 5:
            raise ValueError(
                "DeepIceRope assumes the NuBench feature order "
                "(x, y, z, charge, t) and reads the time coordinate "
                f"from column 4; got n_features={n_features}."
            )
        if rope_axis_bands is not None:
            if not rope_per_axis:
                raise ValueError(
                    "rope_axis_bands set the per-axis bands, which "
                    "rope_per_axis=False replaces with one shared ladder"
                )
            if len(rope_axis_bands) != 4 or any(
                not 0 < lo <= hi for lo, hi in rope_axis_bands
            ):
                raise ValueError(
                    "rope_axis_bands needs a (lowest, highest) pair of "
                    "positive frequencies for each of x, y, z and t, got "
                    f"{list(rope_axis_bands)}"
                )
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
                default=(0, 1, 2, 4),
            )
        )
        # The `sandwich`/`blocks` split mirrors `DeepIce`'s module layout,
        # so weights transfer between the two classes via plain state_dicts.
        self.sandwich = nn.ModuleList(
            [
                Block(
                    input_dim=hidden_dim,
                    num_heads=hidden_dim // head_size,
                    mlp_ratio=mlp_ratio,
                    init_values=1,
                    qk_norm=qk_norm,
                )
                for _ in range(depth_rel)
            ]
        )
        self.cls_token = nn.Linear(hidden_dim, 1, bias=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    input_dim=hidden_dim,
                    num_heads=hidden_dim // head_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=0.0 * (i / max(depth - 1, 1)),
                    init_values=1,
                    qk_norm=qk_norm,
                )
                for i in range(depth)
            ]
        )

        pairs_per_axis = head_size // 2 // 4
        if rope_per_axis:
            axis_bands = rope_axis_bands or HEXAGON_AXIS_BANDS
            omega = torch.cat(
                [
                    torch.exp(
                        torch.linspace(
                            math.log(lo), math.log(hi), pairs_per_axis
                        )
                    )
                    for lo, hi in axis_bands
                ]
            )
        else:
            decay = torch.arange(pairs_per_axis, dtype=torch.float32) / max(
                pairs_per_axis - 1, 1
            )
            # Frequency ladder per coordinate, matching the FourierEncoder
            # input scales (4096 down to ~0.4).
            ladder = 4096.0 * torch.pow(torch.tensor(10000.0), -decay)
            omega = ladder.repeat(4)
        # Buffers are non-persistent: fixed values, and their absence
        # from the state_dict keeps old checkpoints loadable.
        self.register_buffer("rope_omega", omega, persistent=False)
        self.register_buffer(
            "rope_axis",
            torch.arange(4).repeat_interleave(pairs_per_axis),
            persistent=False,
        )

        self._blocks_fn: Callable[..., Tensor] = self._run_blocks
        if compile_blocks:
            # DDP's graph-splitting optimizer overlaps gradient all-reduce by
            # cutting the dynamo graph at bucket boundaries, but the split
            # subgraphs lose the jagged NestedTensor's dynamic-shape symbol
            # and fail to compile (KeyError: s0). Compile the block stack as
            # one graph instead; DDP still hooks gradients as usual.
            torch._dynamo.config.optimize_ddp = False
            # dynamic=True forces a single shape-polymorphic graph. Otherwise a
            # batch whose max sequence length exceeds every prior one triggers a
            # recompile, and under DDP that recompile fires on only the ranks
            # that saw the longer batch -- the others race ahead to the next
            # all-reduce and the collective deadlocks until the NCCL watchdog
            # aborts the job.
            self._blocks_fn = torch.compile(self._run_blocks, dynamic=True)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_tocken should not be subject to weight decay during training."""
        return {"cls_token"}

    def _prepend_cls_nested(self, x: Tensor, batch_idx: Tensor) -> Tensor:
        """Prepend the cls token to each event of a jagged `NestedTensor`.

        Every event grows by one leading slot holding the cls token, so a
        pulse at flat position ``i`` in event ``b`` moves to ``i + b + 1``.
        """
        values = x.values()
        offsets = x.offsets()
        n_pulses, dim = values.shape
        batch_size = offsets.numel() - 1
        new_offsets = offsets + torch.arange(
            batch_size + 1, device=offsets.device
        )
        out = values.new_empty((n_pulses + batch_size, dim))
        # Under autocast `values` may be half precision while the parameter
        # is fp32; CUDA index_put requires matching dtypes.
        out[new_offsets[:-1]] = self.cls_token.weight.to(out.dtype)
        out[torch.arange(n_pulses, device=values.device) + batch_idx + 1] = (
            values
        )
        return torch.nested.nested_tensor_from_jagged(
            out,
            new_offsets,
            min_seqlen=x._get_min_seqlen() + 1,
            max_seqlen=x._get_max_seqlen() + 1,
        )

    def _rope_table(
        self, cos: Tensor, sin: Tensor, batch_idx: Tensor, batch_size: int
    ) -> Tuple[Tensor, Tensor]:
        """Lay the pulses' rotations out over the token stream.

        With mean pooling the tokens are the pulses. With a class token
        every event has one more leading slot, which keeps cos=1 / sin=0,
        the identity rotation: the token has no coordinates.
        """
        if self.pooling == "mean":
            return cos, sin
        n_pulses = cos.shape[0]
        shape = (n_pulses + batch_size, *cos.shape[1:])
        rope_cos = cos.new_ones(shape)
        rope_sin = sin.new_zeros(shape)
        pos = torch.arange(n_pulses, device=cos.device) + batch_idx + 1
        rope_cos[pos] = cos
        rope_sin[pos] = sin
        return rope_cos, rope_sin

    def _rope_angles(
        self, features: Tensor, batch_idx: Tensor, batch_size: int
    ) -> Tuple[Tensor, Tensor]:
        """Per-token cosine and sine of the rotation angles.

        Coordinates are (x, y, z, t), read from the columns
        `coordinate_features` names; by default 0-2 and 4, the NuBench
        order, whose column 3 is charge.
        """
        coords = features[:, self._coordinate_columns].float()
        angles = coords[:, self.rope_axis] * self.rope_omega
        return self._rope_table(
            torch.cos(angles), torch.sin(angles), batch_idx, batch_size
        )

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        x, _, seq_length = array_to_sequence(data.x, data.batch, nested=True)
        x = embed_pulses(self.fourier_ext, self.fourier_mlp, x, seq_length)
        if self.pooling == "cls":
            x = self._prepend_cls_nested(x, data.batch)
        rope_cos, rope_sin = self._rope_angles(
            data.x, data.batch, seq_length.numel()
        )
        x = self._blocks_fn(x, rope_cos=rope_cos, rope_sin=rope_sin)
        if self.pooling == "mean":
            return DeepIce._mean_pool(x.values(), data.batch, seq_length)
        # The cls token output of each event sits at its sequence start.
        return x.values()[x.offsets()[:-1]]

    def _run_blocks(
        self,
        x: Tensor,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the full transformer block stack on the jagged input.

        Kept as a single method so the whole stack can be wrapped in one
        `torch.compile` region instead of one graph per block. The blocks
        run on the dense value buffer (see `Block.forward_jagged`); the
        sequence-length metadata is read once here and threaded through,
        rather than re-derived per block.
        """
        offsets = x.offsets()
        min_seqlen = x._get_min_seqlen()
        max_seqlen = x._get_max_seqlen()
        values = x.values()
        for blk in [*self.sandwich, *self.blocks]:
            values = blk.forward_jagged(
                values,
                offsets,
                min_seqlen,
                max_seqlen,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
        return torch.nested.nested_tensor_from_jagged(
            values, offsets, min_seqlen=min_seqlen, max_seqlen=max_seqlen
        )
