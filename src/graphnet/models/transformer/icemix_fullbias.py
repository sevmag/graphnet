"""DeepIce variant with the spacetime interval bias in every block.

Standard DeepIce computes the pairwise spacetime-interval bias once and
injects it into the first ``n_rel`` relative blocks only; every remaining
block runs plain content attention. Here both stacks consist of
bias-receiving blocks and all of them get the same interval bias, so
pairwise geometry is available at full depth — the additive counterpart of
applying spacetime RoPE in every block.
"""

import torch
import torch.nn as nn
from torch.functional import Tensor

from torch_geometric.data import Data

from graphnet.models.components.layers import Block_rel
from graphnet.models.gnn.icemix import DeepIce
from graphnet.models.utils import array_to_sequence


class DeepIceFullBias(DeepIce):
    """`DeepIce` with the `SpacetimeEncoder` bias injected in all blocks.

    Identical tokenizer and block dimensions to `DeepIce`; the ``depth``
    plain blocks are replaced by `Block_rel` so every layer of both stacks
    receives the (shared, computed-once) interval bias. Padded execution
    only: the dense [B, L, L, C] bias requires padded sequences, which is
    why the padded pulse cap applies. The cls token, prepended after the
    first stack, gets a zero bias row and column — no geometric prior
    toward or from any pulse, the additive analogue of the identity
    rotation it receives under spacetime RoPE.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        mlp_ratio: int = 4,
        seq_length: int = 192,
        depth: int = 12,
        head_size: int = 32,
        depth_rel: int = 4,
        scaled_emb: bool = False,
        n_features: int = 6,
    ):
        """Construct `DeepIceFullBias`.

        Args:
            hidden_dim: The latent feature dimension.
            mlp_ratio: Mlp expansion ratio of FourierEncoder and Transformer.
            seq_length: The base feature dimension.
            depth: The depth of the (formerly plain) transformer stack.
            head_size: The size of the attention heads.
            depth_rel: The depth of the relative transformer stack.
            scaled_emb: Whether to scale the sinusoidal positional
                embeddings.
            n_features: The number of features in the input data.
        """
        super().__init__(
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            seq_length=seq_length,
            depth=depth,
            head_size=head_size,
            depth_rel=depth_rel,
            n_rel=depth_rel,
            scaled_emb=scaled_emb,
            include_dynedge=False,
            dynedge_args=None,
            n_features=n_features,
            use_nested_attention=False,
            vanilla_only=False,
            compile_blocks=False,
            qk_norm=False,
            spacetime_rope=False,
            rope_per_axis=False,
            rel_attn_bias=True,
            rel_activation_bias=True,
        )
        self.blocks = nn.ModuleList(
            [
                Block_rel(
                    input_dim=hidden_dim,
                    num_heads=hidden_dim // head_size,
                    mlp_ratio=mlp_ratio,
                    use_attn_bias=True,
                    use_activation_bias=True,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        x0, mask, seq_length = array_to_sequence(
            data.x, data.batch, padding_value=0
        )
        assert mask is not None
        x = self.fourier_ext(x0, seq_length)
        assert self.rel_pos is not None
        rel_pos_bias = self.rel_pos(x0)

        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        for blk in self.sandwich:
            x = blk(x, attn_mask, rel_pos_bias)

        batch_size = mask.shape[0]
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
        rel_pos_bias = nn.functional.pad(rel_pos_bias, (0, 0, 1, 0, 1, 0))
        for blk in self.blocks:
            x = blk(x, attn_mask, rel_pos_bias)

        return x[:, 0]
