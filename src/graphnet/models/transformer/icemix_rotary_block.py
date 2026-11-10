"""DeepIce with rotary attention in Neptune's transformer block.

The tokenizer and rotation angles are those of `DeepIceRope`; only the block
around the rotated attention differs (see `RotaryTransformerBlock`), so the two
models can be compared at equal width, depth and frequencies.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from graphnet.models.components.attention_blocks import RotaryTransformerBlock
from graphnet.models.transformer.icemix_rope import DeepIceRope
from graphnet.models.transformer.inputs import embed_pulses
from graphnet.models.utils import array_to_sequence


class DeepIceRotaryBlock(DeepIceRope):
    """`DeepIceRope` built from `RotaryTransformerBlock`s."""

    def __init__(
        self,
        hidden_dim: int = 384,
        mlp_ratio: int = 4,
        depth: int = 16,
        head_size: int = 32,
        qk_norm: bool = True,
        pooling: str = "mean",
        feedforward_dim: Optional[int] = None,
        layerscale_init: float = 1e-5,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        **kwargs: Any,
    ):
        """Construct `DeepIceRotaryBlock`.

        Args:
            hidden_dim: Token width.
            mlp_ratio: Sets the feed-forward width when `feedforward_dim` is
                unset, and the tokenizer as in `DeepIceRope`.
            depth: Number of transformer blocks.
            head_size: Width of each attention head.
            qk_norm: Per-head RMSNorm on queries and keys before rotation.
            pooling: Only `"mean"`: the block rotates every token, and a cls
                token has no coordinates to rotate by.
            feedforward_dim: Inner width of the SwiGLU network. Unset, it is
                `hidden_dim * mlp_ratio`, the width of `DeepIceRope`'s MLP.
            layerscale_init: Initial value of the LayerScale parameters.
            dropout: Dropout on the attention and feed-forward branches.
            drop_path_rate: Stochastic-depth rate of the last block, ramped
                linearly from zero.
            **kwargs: Further arguments of `DeepIceRope`.
        """
        if pooling != "mean":
            raise ValueError(
                f"DeepIceRotaryBlock only supports mean pooling, got "
                f"{pooling!r}"
            )
        super().__init__(
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            depth=depth,
            head_size=head_size,
            qk_norm=qk_norm,
            pooling=pooling,
            **kwargs,
        )
        self.sandwich = nn.ModuleList()
        self.blocks = nn.ModuleList(
            RotaryTransformerBlock(
                d_model=hidden_dim,
                nhead=hidden_dim // head_size,
                dim_feedforward=feedforward_dim or hidden_dim * mlp_ratio,
                dropout=dropout,
                qk_norm=qk_norm,
                drop_path_rate=rate,
                layerscale_init=layerscale_init,
            )
            for rate in torch.linspace(0, drop_path_rate, depth).tolist()
        )

    @staticmethod
    def _pad_rotation(
        table: Tensor, batch: Tensor, seq_length: Tensor, fill: float
    ) -> Tensor:
        """Lay a per-pulse rotation table out as `[B, H, L, planes]`.

        Padding slots hold the identity rotation. Positions come from
        arithmetic on `batch` because a boolean gather would force a host
        sync every step.
        """
        starts = torch.zeros_like(seq_length)
        starts[1:] = seq_length[:-1].cumsum(0)
        position = torch.arange(batch.shape[0], device=batch.device)
        position = position - starts[batch]
        shape = (seq_length.shape[0], int(seq_length.max()), *table.shape[1:])
        out = table.new_full(shape, fill)
        out[batch, position] = table
        return out.movedim(2, 1) if out.ndim == 4 else out.unsqueeze(1)

    def _rotation_tables(
        self, data: Data, seq_length: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """`(cos, sin)` tables of the rotation for the padded tokens."""
        cos, sin = self._rope_angles(data.x, data.batch, seq_length.numel())
        return (
            self._pad_rotation(cos, data.batch, seq_length, 1.0),
            self._pad_rotation(sin, data.batch, seq_length, 0.0),
        )

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        x0, mask, seq_length = array_to_sequence(
            data.x, data.batch, padding_value=0
        )
        assert mask is not None
        x = embed_pulses(self.fourier_ext, self.fourier_mlp, x0, seq_length)
        tables = self._rotation_tables(data, seq_length)
        attn_mask = mask[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, tables, attn_mask=attn_mask)
        # float32 accumulation: a bfloat16 sum over hundreds of pulses loses
        # several bits to rounding.
        total = (x * mask.unsqueeze(-1)).sum(1, dtype=torch.float32)
        return (total / seq_length.unsqueeze(-1)).to(x.dtype)
