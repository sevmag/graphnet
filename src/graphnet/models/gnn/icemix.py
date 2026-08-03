"""Implementation of IceMix.

This method was a solution submitted to the IceCube - Neutrinos in Deep Ice
Kaggle competition.

Solution by DrHB: https://github.com/DrHB/icecube-2nd-place
"""

import math

import torch
import torch._dynamo
import torch.nn as nn
from typing import Set, Dict, Any, Optional, Callable

from graphnet.models.components.layers import (
    Block_rel,
    Block,
)
from graphnet.models.components.embedding import (
    FourierEncoder,
    SpacetimeEncoder,
)
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import array_to_sequence

from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from torch import Tensor


class DeepIce(GNN):
    """DeepIce model."""

    def __init__(
        self,
        hidden_dim: int = 384,
        mlp_ratio: int = 4,
        seq_length: int = 192,
        depth: int = 12,
        head_size: int = 32,
        depth_rel: int = 4,
        n_rel: int = 1,
        scaled_emb: bool = False,
        include_dynedge: bool = False,
        dynedge_args: Optional[Dict[str, Any]] = None,
        n_features: int = 6,
        use_nested_attention: bool = False,
        vanilla_only: bool = False,
        compile_blocks: bool = False,
        qk_norm: bool = False,
        spacetime_rope: bool = False,
        rope_per_axis: bool = False,
        rel_attn_bias: bool = True,
        rel_activation_bias: bool = True,
    ):
        """Construct `DeepIce`.

        Args:
            hidden_dim: The latent feature dimension.
            mlp_ratio: Mlp expansion ratio of FourierEncoder and Transformer.
            seq_length: The base feature dimension.
            depth: The depth of the transformer.
            head_size: The size of the attention heads.
            depth_rel: The depth of the relative transformer.
            n_rel: The number of relative transformer layers to use.
            scaled_emb: Whether to scale the sinusoidal positional embeddings.
            include_dynedge: If True, pulse-level predictions from `DynEdge`
                will be added as features to the model.
            dynedge_args: Initialization arguments for DynEdge. If not
                provided, DynEdge will be initialized with the original Kaggle
                Competition settings. If `include_dynedge` is False, this
                argument have no impact.
            n_features: The number of features in the input data.
            use_nested_attention: Run the transformer blocks on jagged
                `NestedTensor`s instead of padded sequences with an
                attention mask. Removes all compute on padding and lets
                attention dispatch to fused variable-length (flash)
                kernels on CUDA with fp16/bf16. The relative-attention
                blocks are unaffected, as their attention bias requires
                padded sequences.
            vanilla_only: Replace the relative-attention sandwich with
                plain bidirectional blocks, so the whole transformer runs
                on jagged `NestedTensor`s and dispatches to fused (flash)
                kernels. The `SpacetimeEncoder` relative-position bias is
                dropped entirely; the model keeps its total depth
                (`depth_rel` + `depth` blocks) but loses pairwise
                space-time geometry in attention. Implies the nested path.
            compile_blocks: Wrap the transformer block stack in
                `torch.compile`. The jagged path issues many small ops per
                step; compiling the whole stack as one graph is what turns
                the nested path from slower-than-padded (eager) into
                faster. No effect on numerics.
            qk_norm: Per-head RMSNorm on queries and keys in every plain
                `Block`. Bounds the attention-logit scale; without the
                relative attention bias the plain blocks otherwise grow
                unbounded QK products to synthesize a locality kernel,
                collapsing attention entropy and destabilizing training.
                Requires the jagged path (`vanilla_only` or
                `use_nested_attention`); the padded block forward raises.
            spacetime_rope: Rotary embedding over the pulse coordinates
                (x, y, z, t): per-head 2D subspaces of q and k are rotated
                by fixed-frequency multiples of each coordinate, so the
                QK product depends on the coordinates only through their
                *differences* -- relative spacetime geometry at zero extra
                attention cost, replacing what the dropped SpacetimeEncoder
                bias provided. The frequency ladder matches the
                FourierEncoder input scales. The cls token gets the
                identity rotation. Requires `vanilla_only`.
            rope_per_axis: Use a separate geometric RoPE frequency band per
                coordinate (x, y, z, t), matched to the measured range of
                pulse-pair coordinate differences on the hexagon detector,
                instead of one shared ladder repeated across axes. Only has an
                effect when `spacetime_rope` is set.
            rel_attn_bias: In the relative-attention sandwich, inject the
                `SpacetimeEncoder` bias into the pre-softmax attention logits.
                Setting False ablates the attention-weight geometry term while
                keeping the value-side one. Defaults to True. No effect when
                `vanilla_only` (the sandwich holds plain blocks).
            rel_activation_bias: In the relative-attention sandwich, inject the
                `SpacetimeEncoder` bias into the post-softmax output
                activations. Setting False ablates the value-side geometry term
                while keeping the attention-weight one. Defaults to True. No
                effect when `vanilla_only`.
        """
        super().__init__(seq_length, hidden_dim)
        fourier_out_dim = hidden_dim // 2 if include_dynedge else hidden_dim
        self.fourier_ext = FourierEncoder(
            seq_length=seq_length,
            mlp_dim=None,
            output_dim=fourier_out_dim,
            scaled=scaled_emb,
            n_features=n_features,
        )
        if vanilla_only:
            # The relative sandwich is replaced by plain bidirectional
            # blocks; the space-time encoder is dropped entirely. Leaving it
            # constructed would register parameters that never receive a
            # gradient, which DDP rejects.
            self.rel_pos = None
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
        else:
            self.rel_pos = SpacetimeEncoder(head_size)
            self.sandwich = nn.ModuleList(
                [
                    Block_rel(
                        input_dim=hidden_dim,
                        num_heads=hidden_dim // head_size,
                        use_attn_bias=rel_attn_bias,
                        use_activation_bias=rel_activation_bias,
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
                    drop_path=0.0 * (i / (depth - 1)),
                    init_values=1,
                    qk_norm=qk_norm,
                )
                for i in range(depth)
            ]
        )
        self.n_rel = n_rel

        if include_dynedge and dynedge_args is None:
            self.warning_once("Running with default DynEdge settings")
            self.dyn_edge = DynEdge(
                nb_inputs=9,
                nb_neighbours=9,
                post_processing_layer_sizes=[336, hidden_dim // 2],
                dynedge_layer_sizes=[
                    (128, 256),
                    (336, 256),
                    (336, 256),
                    (336, 256),
                ],
                global_pooling_schemes=None,
                activation_layer="gelu",
                add_norm_layer=True,
                skip_readout=True,
            )
        elif include_dynedge and not (dynedge_args is None):
            self.dyn_edge = DynEdge(**dynedge_args)

        self.spacetime_rope = spacetime_rope
        if spacetime_rope:
            if not vanilla_only:
                raise ValueError(
                    "spacetime_rope requires vanilla_only: the rotation is "
                    "only implemented for the jagged block path."
                )
            if head_size % 8 != 0:
                raise ValueError(
                    "spacetime_rope needs head_size divisible by 8 "
                    "(2D rotation pairs split over 4 coordinates), got "
                    f"{head_size}."
                )
            if n_features < 5:
                raise ValueError(
                    "spacetime_rope assumes the NuBench feature order "
                    "(x, y, z, charge, t) and reads the time coordinate "
                    f"from column 4; got n_features={n_features}."
                )
            pairs_per_axis = head_size // 2 // 4
            if rope_per_axis:
                # One geometric frequency band per coordinate (x, y, z, t),
                # spanning the range where that axis's measured pulse-pair
                # coordinate differences actually vary on the hexagon detector.
                # A single shared ladder wastes most frequencies out-of-band per
                # axis; matched bands place every frequency where it resolves.
                axis_bands = [
                    (0.50, 8.23),
                    (0.47, 16.12),
                    (5.84, 193.57),
                    (1237.0, 68921.0),
                ]
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
                decay = torch.arange(
                    pairs_per_axis, dtype=torch.float32
                ) / max(pairs_per_axis - 1, 1)
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

        self.include_dynedge = include_dynedge
        self.vanilla_only = vanilla_only
        # vanilla_only runs the whole stack jagged, so it is a nested path.
        self.use_nested_attention = use_nested_attention or vanilla_only
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

    def _to_nested_with_cls(
        self, x: Tensor, batch_idx: Tensor, seq_length: Tensor
    ) -> Tensor:
        """Repack padded `x` as a jagged `NestedTensor`, prepending `cls`.

        Only the real (non-padding) positions of `x` are kept, so the
        transformer blocks never compute on padding. Each event's sequence
        starts with the cls token, mirroring the `torch.cat` in the padded
        path. All indexing is arithmetic on `batch_idx` rather than boolean
        masks, which would force host/device synchronisations every step.
        """
        batch_size, max_len, num_features = x.shape
        n_pulses = batch_idx.shape[0]
        offsets = torch.zeros(
            batch_size + 1, dtype=torch.long, device=x.device
        )
        offsets[1:] = torch.cumsum(seq_length + 1, dim=0)
        # Position of each pulse within its event.
        pos = torch.arange(n_pulses, device=x.device) - (
            offsets[batch_idx] - batch_idx
        )
        values = x.new_empty((n_pulses + batch_size, num_features))
        # Under autocast `x` may be half precision while the parameter is
        # fp32; CUDA index_put requires matching dtypes.
        values[offsets[:-1]] = self.cls_token.weight.to(values.dtype)
        values[offsets[batch_idx] + 1 + pos] = x.reshape(-1, num_features)[
            batch_idx * max_len + pos
        ]
        # Cache the seqlen extremes on the NestedTensor: the fused varlen
        # attention kernels need max_seqlen, and torch.compile requires
        # this metadata to be present consistently when tracing. `x` is
        # padded to the longest event, so `max_len` is exact.
        return torch.nested.nested_tensor_from_jagged(
            values,
            offsets,
            min_seqlen=1,
            max_seqlen=max_len + 1,
        )

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

    def _rope_angles(
        self, features: Tensor, batch_idx: Tensor, batch_size: int
    ) -> tuple:
        """Per-token rotation angles for the cls-prepended jagged stream.

        Coordinates are (x, y, z, t); with the NuBench feature order the
        time column is index 4, not 3 (index 3 is charge). The cls slot
        of every event keeps cos=1 / sin=0, the identity rotation.
        """
        coords = features[:, [0, 1, 2, 4]].float()
        angles = coords[:, self.rope_axis] * self.rope_omega
        n_pulses = coords.shape[0]
        rope_cos = angles.new_ones((n_pulses + batch_size, angles.shape[1]))
        rope_sin = angles.new_zeros((n_pulses + batch_size, angles.shape[1]))
        pos = torch.arange(n_pulses, device=angles.device) + batch_idx + 1
        rope_cos[pos] = torch.cos(angles)
        rope_sin[pos] = torch.sin(angles)
        return rope_cos, rope_sin

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        if self.vanilla_only:
            # No relative sandwich and no space-time bias: the pipeline is
            # jagged end-to-end (encoder included), so the padded [B, L, D]
            # tensor -- sized by the single longest event in the batch --
            # is never materialised.
            x, _, seq_length = array_to_sequence(
                data.x, data.batch, nested=True
            )
            x = self.fourier_ext(x, seq_length)
            if self.include_dynedge:
                # `dyn_edge` returns per-pulse rows in the same order as
                # `data.x`, so the feature concat happens directly on the
                # jagged values buffer.
                graph = self.dyn_edge(data)
                x = torch.nested.nested_tensor_from_jagged(
                    torch.cat([x.values(), graph], dim=1),
                    x.offsets(),
                    min_seqlen=x._get_min_seqlen(),
                    max_seqlen=x._get_max_seqlen(),
                )
            x = self._prepend_cls_nested(x, data.batch)
            rope_cos = rope_sin = None
            if self.spacetime_rope:
                rope_cos, rope_sin = self._rope_angles(
                    data.x, data.batch, seq_length.numel()
                )
            x = self._blocks_fn(x, rope_cos=rope_cos, rope_sin=rope_sin)
            return x.values()[x.offsets()[:-1]]

        x0, mask, seq_length = array_to_sequence(
            data.x, data.batch, padding_value=0
        )
        assert mask is not None
        x = self.fourier_ext(x0, seq_length)
        batch_size = mask.shape[0]
        if self.include_dynedge:
            graph = self.dyn_edge(data)
            graph, _ = to_dense_batch(graph, data.batch)
            x = torch.cat([x, graph], 2)

        # `rel_pos` is only None in the vanilla_only path, which returns above.
        assert self.rel_pos is not None
        rel_pos_bias = self.rel_pos(x0)
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf

        for i, blk in enumerate(self.sandwich):
            x = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None

        if self.use_nested_attention:
            x = self._to_nested_with_cls(x, data.batch, seq_length)
            x = self._blocks_fn(x)
            # The cls token output of each event sits at its sequence start.
            return x.values()[x.offsets()[:-1]]

        mask = torch.cat(
            [
                torch.ones(
                    batch_size, 1, dtype=mask.dtype, device=mask.device
                ),
                mask,
            ],
            1,
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        x = torch.cat([cls_token, x], 1)
        x = self._blocks_fn(x, attn_mask)

        return x[:, 0]

    def _run_blocks(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the transformer block stack.

        Kept as a single method so the whole stack can be wrapped in one
        `torch.compile` region instead of one graph per block. On a jagged
        input the blocks run on the dense value buffer (see
        `Block.forward_jagged`); the sequence-length metadata is read once
        here and threaded through, rather than re-derived per block.
        """
        if x.is_nested:
            offsets = x.offsets()
            min_seqlen = x._get_min_seqlen()
            max_seqlen = x._get_max_seqlen()
            values = x.values()
            # In vanilla_only the sandwich holds plain blocks standing in
            # for the relative ones, run jagged alongside the main stack.
            blocks = (
                [*self.sandwich, *self.blocks]
                if self.vanilla_only
                else list(self.blocks)
            )
            for blk in blocks:
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
        for blk in self.blocks:
            x = blk(x, None, key_padding_mask)
        return x
