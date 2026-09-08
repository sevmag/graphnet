"""Implementation of IceMix.

This method was a solution submitted to the IceCube - Neutrinos in Deep Ice
Kaggle competition.

Solution by DrHB: https://github.com/DrHB/icecube-2nd-place
"""

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
    SpacetimeDistance,
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
        qk_norm: bool = False,
        use_attn_bias: bool = True,
        use_activation_bias: bool = True,
        alibi_bias: bool = False,
        spacetime_time_scale: float = 3e4 / 500 * 3e-1,
        compile_blocks: bool = False,
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
            use_nested_attention: Run the plain transformer blocks on jagged
                `NestedTensor`s instead of padded sequences with an
                attention mask. Removes all compute on padding and lets
                attention dispatch to fused variable-length (flash)
                kernels on CUDA with fp16/bf16. The relative-attention
                blocks are unaffected, as their attention bias requires
                padded sequences.
            qk_norm: Apply a per-head RMSNorm to queries and keys before the
                dot product in the plain blocks, bounding the growth of the
                attention logits. The relative blocks are unaffected: their
                bias enters the logits through a separate term.
            use_attn_bias: Add the spacetime bias to the pre-softmax logits
                of the relative blocks, as `<q_i, R_ij>`.
            use_activation_bias: Add the spacetime bias to the post-softmax
                output of the relative blocks, as `sum_j P_ij R_ij`. The two
                channels are independent, so either may be disabled alone.
            alibi_bias: Replace the embedded per-pair spacetime feature with
                the raw signed four-distance scaled by a learned per-head
                slope, ALiBi-style. The bias no longer depends on the query,
                which costs expressivity but makes the `[L, L]` term a closed
                form of the coordinates rather than an `[L, L, C]` tensor.
                Disables the activation bias, which has no scalar analogue.
            spacetime_time_scale: Factor converting the normalised time
                coordinate into the normalised length unit, `t_scale * c /
                pos_scale` for the `Detector` in use. The default is the
                IceCube value; pass 1.0 with a `Detector` that already emits
                time in length units, such as
                `NuBenchSpacetimeDetector`.
            compile_blocks: Wrap the transformer block stack in
                `torch.compile`. The jagged path issues many small ops per
                step; compiling the whole stack as one graph is what turns
                the nested path from slower-than-padded (eager) into
                faster. No effect on numerics.
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
        self.rel_pos: nn.Module = (
            SpacetimeDistance(time_scale=spacetime_time_scale)
            if alibi_bias
            else SpacetimeEncoder(head_size, time_scale=spacetime_time_scale)
        )
        self.sandwich = nn.ModuleList(
            [
                Block_rel(
                    input_dim=hidden_dim,
                    num_heads=hidden_dim // head_size,
                    use_attn_bias=use_attn_bias,
                    use_activation_bias=use_activation_bias,
                    alibi=alibi_bias,
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

        self.include_dynedge = include_dynedge
        self.use_nested_attention = use_nested_attention
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

    @staticmethod
    def _additive_mask(keep: Tensor) -> Tensor:
        """Turn a boolean keep-mask into an additive mask.

        Attention takes 0 where a key is kept and -inf where it is
        masked.
        """
        attn_mask = torch.zeros(keep.shape, device=keep.device)
        attn_mask[~keep] = -torch.inf
        return attn_mask

    def _run_rel_blocks(
        self, x: Tensor, x0: Tensor, attn_mask: Tensor
    ) -> Tensor:
        """Apply the relative-attention stack over padded sequences.

        Only the leading `n_rel` blocks receive the spacetime bias; the rest
        run as plain attention. With no relative blocks at all the bias is
        never built -- it is an O(len^2) tensor and would otherwise be the
        model's dominant cost with nothing consuming it.
        """
        if not self.sandwich:
            return x
        rel_pos_bias = self.rel_pos(x0)
        for i, blk in enumerate(self.sandwich):
            x = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None
        return x

    def _to_nested_with_cls(
        self, x: Tensor, batch_idx: Tensor, seq_length: Tensor
    ) -> Tensor:
        """Repack padded `x` as a jagged `NestedTensor`, prepending `cls`.

        Padding positions are dropped, so the blocks never compute on them.
        Indexing is arithmetic on `batch_idx` rather than boolean masks, which
        would force a host sync every step.
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
        # The fused varlen kernels need max_seqlen, and torch.compile requires
        # it to be present consistently when tracing; `x` is padded to the
        # longest event, so `max_len` is exact.
        return torch.nested.nested_tensor_from_jagged(
            values,
            offsets,
            min_seqlen=1,
            max_seqlen=max_len + 1,
        )

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        x0, mask, seq_length = array_to_sequence(
            data.x, data.batch, padding_value=0
        )
        assert mask is not None
        x = self.fourier_ext(x0, seq_length)
        if self.include_dynedge:
            graph, _ = to_dense_batch(self.dyn_edge(data), data.batch)
            x = torch.cat([x, graph], 2)

        x = self._run_rel_blocks(x, x0, self._additive_mask(mask))

        if self.use_nested_attention:
            x = self._blocks_fn(
                self._to_nested_with_cls(x, data.batch, seq_length)
            )
            # each event's cls output sits at its sequence start
            return x.values()[x.offsets()[:-1]]

        cls_token = self.cls_token.weight.unsqueeze(0).expand(
            mask.shape[0], -1, -1
        )
        keep = torch.cat([mask.new_ones((mask.shape[0], 1)), mask], 1)
        x = self._blocks_fn(
            torch.cat([cls_token, x], 1), self._additive_mask(keep)
        )
        return x[:, 0]

    def _run_blocks(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the plain transformer block stack.

        One method so the whole stack compiles as a single graph rather than
        one per block. On jagged input the blocks run on the dense value
        buffer (`Block.forward_jagged`), with the seqlen metadata read once
        here instead of re-derived per block.
        """
        if x.is_nested:
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
                )
            return torch.nested.nested_tensor_from_jagged(
                values, offsets, min_seqlen=min_seqlen, max_seqlen=max_seqlen
            )
        for blk in self.blocks:
            x = blk(x, None, key_padding_mask)
        return x
