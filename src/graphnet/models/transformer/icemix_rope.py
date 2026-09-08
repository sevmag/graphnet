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
from typing import Callable, Optional, Set

from graphnet.models.components.layers import Block
from graphnet.models.components.embedding import FourierEncoderEPJC
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import array_to_sequence

from torch_geometric.data import Data
from torch import Tensor


class DeepIceRope(GNN):
    """DeepIce with per-coordinate rotary attention, jagged end-to-end."""

    def __init__(
        self,
        hidden_dim: int = 384,
        mlp_ratio: int = 4,
        seq_length: int = 192,
        depth: int = 12,
        head_size: int = 32,
        depth_rel: int = 4,
        scaled_emb: bool = False,
        n_features: int = 5,
        rope_per_axis: bool = True,
        compile_blocks: bool = False,
    ):
        """Construct `DeepIceRope`.

        Args:
            hidden_dim: The latent feature dimension.
            mlp_ratio: Mlp expansion ratio of FourierEncoderEPJC and
                Transformer.
            seq_length: The base feature dimension.
            depth: The depth of the transformer.
            head_size: The size of the attention heads. Must be divisible
                by 8 (2D rotation pairs split over 4 coordinates).
            depth_rel: The number of blocks standing in for `DeepIce`'s
                relative-attention sandwich, keeping the total depth at
                `depth_rel + depth`.
            scaled_emb: Whether to scale the sinusoidal positional embeddings.
            n_features: The number of features in the input data. At least
                5, in the NuBench order (x, y, z, charge, t): the rotation
                reads the coordinates from columns 0-2 and time from
                column 4.
            rope_per_axis: Use a separate geometric RoPE frequency band per
                coordinate (x, y, z, t), matched to the measured range of
                pulse-pair coordinate differences on the hexagon detector,
                instead of one shared ladder repeated across axes.
            compile_blocks: Wrap the transformer block stack in
                `torch.compile`. The jagged path issues many small ops per
                step; compiling the whole stack as one graph is what turns
                it from slower-than-padded (eager) into faster. No effect
                on numerics.
        """
        super().__init__(seq_length, hidden_dim)
        if head_size % 8 != 0:
            raise ValueError(
                "DeepIceRope needs head_size divisible by 8 "
                "(2D rotation pairs split over 4 coordinates), got "
                f"{head_size}."
            )
        if n_features < 5:
            raise ValueError(
                "DeepIceRope assumes the NuBench feature order "
                "(x, y, z, charge, t) and reads the time coordinate "
                f"from column 4; got n_features={n_features}."
            )
        self.fourier_ext = FourierEncoderEPJC(
            seq_length=seq_length,
            mlp_dim=None,
            output_dim=hidden_dim,
            scaled=scaled_emb,
            n_features=n_features,
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
        x, _, seq_length = array_to_sequence(data.x, data.batch, nested=True)
        x = self.fourier_ext(x, seq_length)
        x = self._prepend_cls_nested(x, data.batch)
        rope_cos, rope_sin = self._rope_angles(
            data.x, data.batch, seq_length.numel()
        )
        x = self._blocks_fn(x, rope_cos=rope_cos, rope_sin=rope_sin)
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
