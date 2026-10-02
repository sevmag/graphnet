"""Attention and transformer blocks used in graphnet models."""

from typing import Optional, Union, TYPE_CHECKING

import torch
import torch.nn as nn

# `import torch` does not load this submodule, and the tiled attention
# addresses it as `torch.utils.checkpoint.checkpoint`.
import torch.utils.checkpoint
from torch.functional import Tensor
from torch.nn.functional import linear
from pytorch_lightning import LightningModule

if TYPE_CHECKING:
    from graphnet.models.components.embedding import (
        SpacetimeEncoder,
        SpacetimeEncoderEPJC,
    )

    # Either encoder supplies the bias; they differ in what they let the
    # caller configure, not in the interface used here.
    SpacetimeEncoderLike = Union[SpacetimeEncoder, SpacetimeEncoderEPJC]


class DropPath(LightningModule):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(
        self,
        drop_prob: float = 0.0,
    ):
        """Construct `DropPath`.

        Args:
            drop_prob: Probability of dropping a path during training.
                If 0.0, no paths are dropped. Defaults to None.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self) -> str:
        """Return extra representation of the module."""
        return "p={}".format(self.drop_prob)


class Mlp(LightningModule):
    """Multi-Layer Perceptron (MLP) module."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: nn.Module = nn.GELU,
        dropout_prob: float = 0.0,
    ):
        """Construct `Mlp`.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features. Defaults to None.
                If None, it is set to the value of `in_features`.
            out_features: Number of output features. Defaults to None.
                If None, it is set to the value of `in_features`.
            activation: Activation layer. Defaults to `nn.GELU`.
            dropout_prob: Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        if in_features <= 0:
            raise ValueError(
                f"in_features must be greater than 0, got in_features "
                f"{in_features} instead"
            )
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.input_projection = nn.Linear(in_features, hidden_features)
        self.activation = activation()
        self.output_projection = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.output_projection(x)
        x = self.dropout(x)
        return x


class Attention_rel(LightningModule):
    """Attention mechanism with relative position bias."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
    ):
        """Construct 'Attention_rel'.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: the number of attention heads to use (default: 8)
            qkv_bias: whether to add bias to the query, key, and value
                projections. Defaults to False.
            qk_scale: a scaling factor that multiplies the dot product of query
                and key vectors. Defaults to None. If None, computed as
                :math: `head_dim^(-1/2)`.
            attn_drop: the dropout probability for the attention weights.
                Defaults to 0.0.
            proj_drop: the dropout probability for the output of the attention
                module. Defaults to 0.0.
            attn_head_dim: the feature dimensionality of each attention head.
                Defaults to None. If None, computed as `dim // num_heads`.
        """
        if input_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"dim and num_heads must be greater than 0,"
                f" got input_dim={input_dim} and num_heads={num_heads} instead"
            )

        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim or input_dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.proj_q = nn.Linear(input_dim, all_head_dim, bias=False)
        self.proj_k = nn.Linear(input_dim, all_head_dim, bias=False)
        self.proj_v = nn.Linear(input_dim, all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        rel_pos_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        batch_size, event_length, _ = q.shape

        q = linear(input=q, weight=self.proj_q.weight, bias=self.q_bias)
        q = q.reshape(batch_size, event_length, self.num_heads, -1).permute(
            0, 2, 1, 3
        )
        k = linear(input=k, weight=self.proj_k.weight, bias=None)
        k = k.reshape(batch_size, k.shape[1], self.num_heads, -1).permute(
            0, 2, 1, 3
        )
        v = linear(input=v, weight=self.proj_v.weight, bias=self.v_bias)
        v = v.reshape(batch_size, v.shape[1], self.num_heads, -1).permute(
            0, 2, 1, 3
        )

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if rel_pos_bias is not None:
            bias = torch.einsum("bhic,bijc->bhij", q, rel_pos_bias)
            attn = attn + bias
        if key_padding_mask is not None:
            assert (
                key_padding_mask.dtype == torch.float32
                or key_padding_mask.dtype == torch.float16
            ), "incorrect mask dtype"
            bias = torch.min(
                key_padding_mask[:, None, :], key_padding_mask[:, :, None]
            )
            bias[
                torch.max(
                    key_padding_mask[:, None, :], key_padding_mask[:, :, None]
                )
                < 0
            ] = 0
            attn = attn + bias.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        if rel_pos_bias is not None:
            x = x + torch.einsum("bhij,bijc->bihc", attn, rel_pos_bias)
        x = x.reshape(batch_size, event_length, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_tiled(
        self,
        x: Tensor,
        rel_pos_encoder: "SpacetimeEncoderLike",
        coords: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        q_tile: int = 64,
        use_checkpoint: bool = False,
    ) -> Tensor:
        """Query-tiled self-attention with an on-the-fly relative bias.

        Numerically identical to `forward` called with a `rel_pos_bias`
        precomputed as `rel_pos_encoder(coords)`,  but the relative bias for
        each block of `q_tile` query rows is built from
        `rel_pos_encoder.forward_tiled` and consumed immediately, so the
        `[B, L, L, H]` bias tensor is never fully materialised.
        Each query row still attends to all keys with a full-row softmax, so
        the per-row arithmetic (and hence the result) matches `forward`
        exactly, while peak memory for the bias drops from `O(L^2 * H)` to
        `O(q_tile * L * H)`.

        Args:
            x: Input tensor of shape `[B, L, input_dim]`.
            rel_pos_encoder: The spacetime encoder whose `forward_tiled`
                supplies the relative bias for a query tile.
            coords: Raw coordinates `[B, L, >=4]` (positions 0:3, time
                3) fed to `rel_pos_encoder`.
            key_padding_mask: Float mask `[B, L]` (0 valid, -inf pad),
                as used by `forward`.
            q_tile: Number of query rows processed per tile.
            use_checkpoint: Recompute each tile in the backward pass so the
                per-tile intermediates are not retained across tiles (needed to
                realise the memory saving during training); otherwise autograd
                keeps every tile's tensors. Costs roughly one extra forward.

        Returns:
            Tensor of shape `[B, L, input_dim]`, identical to `forward`.
        """
        batch_size, event_length, _ = x.shape
        num_heads = self.num_heads
        head_dim = self.proj_q.weight.shape[0] // num_heads

        def to_heads(
            t: Tensor, weight: Tensor, bias: Optional[Tensor]
        ) -> Tensor:
            return (
                linear(t, weight, bias)
                .reshape(batch_size, event_length, num_heads, head_dim)
                .permute(0, 2, 1, 3)
            )

        q = to_heads(x, self.proj_q.weight, self.q_bias) * self.scale
        k = to_heads(x, self.proj_k.weight, None)
        v = to_heads(x, self.proj_v.weight, self.v_bias)

        pair_bias = None
        if key_padding_mask is not None:
            assert (
                key_padding_mask.dtype == torch.float32
                or key_padding_mask.dtype == torch.float16
            ), "incorrect mask dtype"
            pair_bias = torch.min(
                key_padding_mask[:, None, :], key_padding_mask[:, :, None]
            )
            pair_bias[
                torch.max(
                    key_padding_mask[:, None, :], key_padding_mask[:, :, None]
                )
                < 0
            ] = 0

        def compute_tile(
            q_slice: Tensor,
            k_all: Tensor,
            v_all: Tensor,
            coords_all: Tensor,
            pair: Optional[Tensor],
            start: int,
            end: int,
        ) -> Tensor:
            rel = rel_pos_encoder.forward_tiled(coords_all, start, end)
            scores = q_slice @ k_all.transpose(-2, -1)
            scores = scores + torch.einsum("bhic,bijc->bhij", q_slice, rel)
            if pair is not None:
                scores = scores + pair[:, start:end].unsqueeze(1)
            attn = self.attn_drop(scores.softmax(dim=-1))
            out = (attn @ v_all).transpose(1, 2)
            return out + torch.einsum("bhij,bijc->bihc", attn, rel)

        out = x.new_zeros(batch_size, event_length, num_heads, head_dim)
        for start in range(0, event_length, q_tile):
            end = min(start + q_tile, event_length)
            q_slice = q[:, :, start:end]
            if use_checkpoint and torch.is_grad_enabled():
                tile_out = torch.utils.checkpoint.checkpoint(
                    compute_tile,
                    q_slice,
                    k,
                    v,
                    coords,
                    pair_bias,
                    start,
                    end,
                    use_reentrant=False,
                )
            else:
                tile_out = compute_tile(
                    q_slice, k, v, coords, pair_bias, start, end
                )
            out[:, start:end] = tile_out

        out = out.reshape(batch_size, event_length, -1)
        return self.proj_drop(self.proj(out))


class Block_rel(LightningModule):
    """Implementation of BEiTv2 Block."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        activation: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        attn_head_dim: Optional[int] = None,
    ):
        """Construct 'Block_rel'.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: Number of attention heads to use in the `Attention_rel`
            layer.
            mlp_ratio: Ratio of the hidden size of the feedforward network to
                the input size in the `Mlp` layer.
            qkv_bias: Whether or not to include bias terms in the query, key,
                and value matrices in the `Attention_rel` layer.
            qk_scale: Scaling factor for the dot product of the query and key
                matrices in the `Attention_rel` layer.
            dropout: Dropout probability to use in the `Mlp` layer.
            attn_drop: Dropout probability to use in the `Attention_rel` layer.
            drop_path: Probability of applying drop path regularization to the
                output of the layer.
            init_values: Initial value to use for the `gamma_1` and `gamma_2`
                parameters if not `None`.
            activation: Activation function to use in the `Mlp` layer.
            norm_layer: Normalization layer to use.
            attn_head_dim: Dimension of the attention head outputs in the
                `Attention_rel` layer.
        """
        super().__init__()
        self.norm1 = norm_layer(input_dim)
        self.attn = Attention_rel(
            input_dim,
            num_heads,
            attn_drop=attn_drop,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_head_dim=attn_head_dim,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(input_dim)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=input_dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            dropout_prob=dropout,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones(input_dim), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones(input_dim), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        rel_pos_bias: Optional[Tensor] = None,
        kv: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        if self.gamma_1 is None:
            xn = self.norm1(x)
            kv = xn if kv is None else self.norm1(kv)
            x = x + self.drop_path(
                self.attn(
                    xn,
                    kv,
                    kv,
                    rel_pos_bias=rel_pos_bias,
                    key_padding_mask=key_padding_mask,
                )
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            kv = xn if kv is None else self.norm1(kv)
            x = x + self.drop_path(
                self.gamma_1
                * self.drop_path(
                    self.attn(
                        xn,
                        kv,
                        kv,
                        rel_pos_bias=rel_pos_bias,
                        key_padding_mask=key_padding_mask,
                    )
                )
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

    def forward_tiled(
        self,
        x: Tensor,
        rel_pos_encoder: "SpacetimeEncoderLike",
        coords: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        q_tile: int = 64,
        use_checkpoint: bool = False,
    ) -> Tensor:
        """Forward pass with the relative bias computed query-tiled.

        Numerically identical to `forward` called with a `rel_pos_bias`
        precomputed as `rel_pos_encoder(coords)`, but the relative bias is
        built one query-tile at a time so the `[B, L, L, H]` tensor is never
        fully materialised.

        Args:
            x: Input tensor of shape `[B, L, input_dim]`.
            rel_pos_encoder: The spacetime encoder providing the relative
                bias via its `forward_tiled` method.
            coords: Raw coordinates `[B, L, >=4]` (positions 0:3, time
                3) fed to `rel_pos_encoder`.
            key_padding_mask: Float mask `[B, L]` (0 valid, -inf pad).
            q_tile: Number of query rows processed per tile.
            use_checkpoint: Recompute each tile in the backward pass so the
                per-tile intermediates are not retained (memory saving during
                training, at the cost of extra compute).

        Returns:
            Tensor of shape `[B, L, input_dim]`.
        """
        xn = self.norm1(x)
        attn = self.attn.forward_tiled(
            xn,
            rel_pos_encoder,
            coords,
            key_padding_mask=key_padding_mask,
            q_tile=q_tile,
            use_checkpoint=use_checkpoint,
        )
        if self.gamma_1 is None:
            x = x + self.drop_path(attn)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.drop_path(attn))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Block(LightningModule):
    """Transformer block."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        activation: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        """Construct 'Block'.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: Number of attention heads to use in the
                `MultiheadAttention` layer.
            mlp_ratio: Ratio of the hidden size of the feedforward network to
                the input size in the `Mlp` layer.
            dropout: Dropout probability to use in the `Mlp` layer.
            attn_drop: Dropout probability to use in the `MultiheadAttention`
                layer.
            drop_path: Probability of applying drop path regularization to the
                output of the layer.
            init_values: Initial value to use for the `gamma_1` and `gamma_2`
                parameters if not `None`.
            activation: Activation function to use in the `Mlp` layer.
            norm_layer: Normalization layer to use.
        """
        super().__init__()
        self.norm1 = norm_layer(input_dim)
        self.attn = nn.MultiheadAttention(
            input_dim, num_heads, dropout=attn_drop, batch_first=True
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(input_dim)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=input_dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            dropout_prob=dropout,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((input_dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((input_dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        if self.gamma_1 is None:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.attn(
                    xn,
                    xn,
                    xn,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )[0]
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.gamma_1
                * self.attn(
                    xn,
                    xn,
                    xn,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )[0]
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
