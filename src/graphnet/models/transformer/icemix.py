"""Implementation of IceMix.

This method was a solution submitted to the IceCube - Neutrinos in Deep Ice
Kaggle competition.

Solution by DrHB: https://github.com/DrHB/icecube-2nd-place
"""

import torch
import torch._dynamo
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Set, Dict, Any, Optional, Callable, List, Tuple

from graphnet.models.components.layers import (
    Block_rel,
    Block,
)
from graphnet.models.components.embedding import (
    FourierEncoder,
    SpacetimeEncoderEPJC,
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
        bucketed_rel: bool = False,
        rel_pair_budget: int = 1 << 26,
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
            bucketed_rel: Run the relative-attention blocks per
                length-sorted event bucket instead of over the full padded
                batch. Every per-pair tensor (the spacetime bias and the
                attention matrix) then costs each event roughly its own
                length squared rather than the batch maximum squared, which
                is what makes heavy-tailed, uncapped event lengths
                tractable. Same modules, same math per event; only the
                padding an event is grouped with changes.
            rel_pair_budget: Upper bound on ``bucket_size * max_len**2``
                when forming buckets, i.e. on the element count of any
                bucket's pairwise tensors. Bounds peak memory of the
                checkpointed recompute; a single event longer than the
                budget still forms its own bucket.
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
        self.rel_pos = SpacetimeEncoderEPJC(head_size)
        self.sandwich = nn.ModuleList(
            [
                Block_rel(
                    input_dim=hidden_dim,
                    num_heads=hidden_dim // head_size,
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
        self.bucketed_rel = bucketed_rel
        self.rel_pair_budget = rel_pair_budget
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

    def _rel_stack(self, x: Tensor, x0: Tensor, attn_mask: Tensor) -> Tensor:
        """Run the relative blocks over one already-padded slice.

        The spacetime bias is computed inside so that, under activation
        checkpointing, its O(len^2) intermediates are recomputed in the
        backward pass instead of stored.
        """
        rel_pos_bias = self.rel_pos(x0)
        for i, blk in enumerate(self.sandwich):
            x = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None
        return x

    def run_rel_blocks_bucketed(
        self, x: Tensor, x0: Tensor, seq_length: Tensor
    ) -> Tensor:
        """Apply the relative blocks bucketed by event length.

        Events are sorted by length and grouped greedily so that no
        bucket's pairwise tensors exceed ``rel_pair_budget`` elements; each
        bucket is padded only to its own longest event. With heavy-tailed
        lengths this brings the total pairwise cost down from
        ``batch * max_len**2`` to roughly ``sum(len_i**2)``. Each bucket
        runs under activation checkpointing, so peak memory is one bucket's
        forward rather than the sum over buckets.

        Args:
            x: Padded embeddings [batch, max_len, dim] (post Fourier).
            x0: Padded raw features [batch, max_len, n_features], consumed
                by the spacetime encoder.
            seq_length: Real event lengths [batch].

        Returns:
            Padded embeddings [batch, max_len, dim]; padding positions are
            zero (they carry no information downstream either way).
        """
        lengths = seq_length.to(device=x.device)
        ll: List[int] = lengths.tolist()
        order = sorted(range(len(ll)), key=ll.__getitem__)
        buckets: List[Tuple[List[int], int]] = []
        cur: List[int] = []
        cur_max = 0
        for idx in order:
            cap = max(cur_max, ll[idx])
            if cur and (len(cur) + 1) * cap * cap > self.rel_pair_budget:
                buckets.append((cur, cur_max))
                cur, cur_max = [idx], ll[idx]
            else:
                cur.append(idx)
                cur_max = cap
        if cur:
            buckets.append((cur, cur_max))
        out = torch.zeros_like(x)
        for idxs, max_len in buckets:
            bucket_len = max(int(max_len), 1)
            bi = torch.as_tensor(idxs, device=x.device)
            xb = x.index_select(0, bi)[:, :bucket_len]
            x0b = x0.index_select(0, bi)[:, :bucket_len]
            mb = (
                torch.arange(bucket_len, device=x.device)[None]
                < lengths[bi][:, None]
            )
            attn_mask = torch.zeros(mb.shape, device=x.device)
            attn_mask[~mb] = -torch.inf
            if torch.is_grad_enabled() and x.requires_grad:
                yb = checkpoint(
                    self._rel_stack, xb, x0b, attn_mask, use_reentrant=False
                )
            else:
                yb = self._rel_stack(xb, x0b, attn_mask)
            out[bi, :bucket_len] = yb
        return out

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
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

        if self.bucketed_rel:
            x = self.run_rel_blocks_bucketed(x, x0, seq_length)
        else:
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
    ) -> Tensor:
        """Apply the plain transformer block stack.

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
