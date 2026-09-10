"""Implementation of IceMix.

This method was a solution submitted to the IceCube - Neutrinos in Deep Ice
Kaggle competition.

Solution by DrHB: https://github.com/DrHB/icecube-2nd-place
"""

import torch
import torch._dynamo
import torch.nn as nn
from typing import Set, Dict, Any, List, Optional, Tuple, Union, Callable

from graphnet.models.components.layers import (
    Block_rel,
    Block,
)
from graphnet.models.components.embedding import (
    FourierEncoder,
    FourierEncoderEPJC,
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
        spacetime_scale: float = 1024.0,
        spacetime_n_freq: float = 10000.0,
        spacetime_clip: float = 4.0,
        fourier_schema: Optional[
            Dict[str, Union[float, Tuple[float, float]]]
        ] = None,
        input_feature_names: Optional[List[str]] = None,
        compile_blocks: bool = False,
        rel_attention: str = "dense",
        q_tile: int = 64,
        tiled_checkpoint: bool = True,
    ):
        """Construct `DeepIce`.

        Args:
            hidden_dim: The latent feature dimension.
            mlp_ratio: Mlp expansion ratio of FourierEncoderEPJC and
                Transformer.
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
            use_nested_attention: Run the plain blocks on jagged
                `NestedTensor`s, dropping all compute on padding. The
                relative blocks are unaffected: their bias needs padding.
            qk_norm: Per-head RMSNorm on queries and keys in the plain
                blocks, bounding the growth of the attention logits.
            use_attn_bias: Add the spacetime bias to the relative blocks'
                pre-softmax logits, as `<q_i, R_ij>`.
            use_activation_bias: Add it to their post-softmax output, as
                `sum_j P_ij R_ij`. Independent of `use_attn_bias`.
            alibi_bias: Scale the raw signed four-distance by a learned
                per-head slope instead of embedding it per pair. Being
                query-independent, the `[L, L]` term is then a closed form of
                the coordinates rather than an `[L, L, C]` tensor. Disables
                the activation bias, which has no scalar analogue.
            spacetime_time_scale: `t_scale * c / pos_scale` for the
                `Detector` in use. The default is the IceCube value; pass 1.0
                with `NuBenchSpacetimeDetector`.
            spacetime_scale: Multiplier on the four-distance before its
                sinusoidal ladder. With `spacetime_n_freq` it sets the band
                of separations the bias resolves: wavelengths from
                `2 * pi / spacetime_scale` up, in the normalised length unit.
            spacetime_n_freq: Span of that ladder.
            spacetime_clip: Bound on the four-distance before embedding.
            fourier_schema: `{feature name: multiplier}` or
                `{feature name: (multiplier, n_freq)}` for the columns to
                embed, resolved against `input_feature_names`. Unset, the
                encoder is `FourierEncoderEPJC` with its fixed layout.
            input_feature_names: Input column names, in order. Required with
                `fourier_schema`.
            compile_blocks: Compile the block stack as one graph. No effect
                on numerics.
            rel_attention: How the relative blocks build their spacetime
                bias. `"dense"` materialises the whole `[B, L, L, C]` tensor;
                `"tiled"` builds it one band of query rows at a time, which
                bounds peak memory at `q_tile * L` without changing the
                result. Note that tiling bounds memory, not compute: the
                padded pairs are still formed and then masked.
            q_tile: Query rows per band when `rel_attention="tiled"`.
            tiled_checkpoint: Recompute each band in the backward pass. Without
                it autograd retains every band, which sums to the dense tensor
                the tiling exists to avoid, so it only saves memory in training
                when this is set.
        """
        super().__init__(seq_length, hidden_dim)
        fourier_out_dim = hidden_dim // 2 if include_dynedge else hidden_dim
        self.fourier_mlp: Optional[nn.Module] = None
        if fourier_schema is None:
            self.fourier_ext: nn.Module = FourierEncoderEPJC(
                seq_length=seq_length,
                mlp_dim=None,
                output_dim=fourier_out_dim,
                scaled=scaled_emb,
                n_features=n_features,
            )
        else:
            # Naming the features is what makes a change of input order raise
            # rather than shift every multiplier onto a neighbouring column.
            names = input_feature_names or []
            unknown = set(fourier_schema) - set(names)
            if unknown:
                raise ValueError(
                    f"fourier_schema names {sorted(unknown)}, not among the "
                    f"input features {names}."
                )
            self.fourier_ext = FourierEncoder(
                schema={
                    names.index(n): (
                        (float(v[0]), float(v[1]))
                        if isinstance(v, (tuple, list))
                        else float(v)
                    )
                    for n, v in fourier_schema.items()
                },
                seq_length=seq_length,
                scaled=scaled_emb,
            )
            # The general encoder leaves this projection to the model.
            concat_dim = self.fourier_ext.output_dim
            self.fourier_mlp = nn.Sequential(
                nn.Linear(concat_dim, concat_dim),
                nn.LayerNorm(concat_dim),
                nn.GELU(),
                nn.Linear(concat_dim, fourier_out_dim),
            )
        if rel_attention not in ("dense", "tiled"):
            raise ValueError(
                f"rel_attention must be 'dense' or 'tiled', "
                f"got {rel_attention!r}"
            )
        if rel_attention == "tiled" and alibi_bias:
            # The tiled path contracts a per-pair feature vector with the
            # query; ALiBi's bias is a scalar per pair and is consumed by
            # a different code path, so the two cannot be combined.
            raise ValueError("rel_attention='tiled' cannot use alibi_bias")
        self.rel_pos: nn.Module = (
            SpacetimeDistance(
                clip=spacetime_clip, time_scale=spacetime_time_scale
            )
            if alibi_bias
            else SpacetimeEncoder(
                head_size,
                time_scale=spacetime_time_scale,
                scale=spacetime_scale,
                clip=spacetime_clip,
                n_freq=spacetime_n_freq,
            )
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
        self.rel_attention = rel_attention
        self.q_tile = q_tile
        self.tiled_checkpoint = tiled_checkpoint

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
            # DDP's graph splitting loses the jagged tensor's dynamic-shape
            # symbol and fails to compile (KeyError: s0).
            torch._dynamo.config.optimize_ddp = False
            # A single shape-polymorphic graph. Otherwise a longer-than-ever
            # batch recompiles on only the ranks that saw it, and the next
            # all-reduce deadlocks.
            self._blocks_fn = torch.compile(self._run_blocks, dynamic=True)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_tocken should not be subject to weight decay during training."""
        return {"cls_token"}

    @staticmethod
    def _additive_mask(keep: Tensor, dtype: torch.dtype) -> Tensor:
        """Turn a boolean keep-mask into the additive mask attention takes.

        The mask must carry the dtype of the activations it will be added
        to: attention reads a float mask as data, and a mismatched one is
        silently misread rather than promoted.
        """
        attn_mask = torch.zeros(keep.shape, device=keep.device, dtype=dtype)
        attn_mask[~keep] = -torch.inf
        return attn_mask

    def _run_rel_blocks(
        self, x: Tensor, x0: Tensor, attn_mask: Tensor
    ) -> Tensor:
        """Apply the relative-attention stack over padded sequences.

        Only the leading `n_rel` blocks receive the spacetime bias. With no
        relative blocks the O(len^2) bias is never built, as nothing would
        consume it.
        """
        if not self.sandwich:
            return x
        tiled = self.rel_attention == "tiled"
        rel_pos_bias = None if tiled else self.rel_pos(x0)
        for i, blk in enumerate(self.sandwich):
            if tiled and i < self.n_rel:
                x = blk.forward_tiled(
                    x,
                    self.rel_pos,
                    x0,
                    key_padding_mask=attn_mask,
                    q_tile=self.q_tile,
                    use_checkpoint=self.tiled_checkpoint and self.training,
                )
            else:
                x = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None
        return x

    def _to_nested_with_cls(
        self, x: Tensor, batch_idx: Tensor, seq_length: Tensor
    ) -> Tensor:
        """Repack padded `x` as a jagged `NestedTensor`, prepending `cls`.

        Indexing is arithmetic on `batch_idx` rather than boolean masks,
        which would force a host sync every step.
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
        # The fused varlen kernels need max_seqlen; `x` is padded to the
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
        if self.fourier_mlp is not None:
            x = self.fourier_mlp(x)
        if self.include_dynedge:
            graph, _ = to_dense_batch(self.dyn_edge(data), data.batch)
            x = torch.cat([x, graph], 2)

        x = self._run_rel_blocks(x, x0, self._additive_mask(mask, x.dtype))

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
            torch.cat([cls_token, x], 1), self._additive_mask(keep, x.dtype)
        )
        return x[:, 0]

    def _run_blocks(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the plain transformer block stack.

        One method so the whole stack compiles as a single graph. Jagged
        input runs on the dense value buffer, with the seqlen metadata
        read once here rather than per block.
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
