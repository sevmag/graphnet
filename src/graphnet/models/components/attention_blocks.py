"""Attention and transformer blocks used in graphnet models."""

from typing import Callable, Optional, Tuple, TYPE_CHECKING, Union

import torch
import torch.nn as nn
from torch.functional import Tensor
from torch.nn.functional import linear, scaled_dot_product_attention
from torch.utils.checkpoint import checkpoint
from pytorch_lightning import LightningModule

if TYPE_CHECKING:
    from graphnet.models.components.embedding import (
        SpacetimeEncoder,
        SpacetimeEncoderEPJC,
    )

    # Either encoder supplies the bias; they differ in what they let the
    # caller configure, not in the interface used here.
    SpacetimeEncoderLike = Union[SpacetimeEncoder, SpacetimeEncoderEPJC]


def apply_spacetime_rope(
    t: Tensor, cos: Tensor, sin: Tensor, num_heads: int, head_dim: int
) -> Tensor:
    """Rotate per-head 2D subspaces of `t` by per-token angles (RoPE).

    `t` is a dense `[N, num_heads * head_dim]` projection buffer. `cos`/`sin`
    are per-token rotation angles, either `[N, head_dim // 2]` (shared
    across heads) or `[N, num_heads, head_dim // 2]` (per-head, as in the
    nD-RoPE construction where every head carries its own rotation of the
    wave-vector set). Rotations preserve norms and make the q.k dot product
    depend on the coordinates only through per-token angle *differences*,
    i.e. relative spacetime geometry, at zero extra attention cost.
    """
    th = t.unflatten(-1, [num_heads, head_dim])
    half = head_dim // 2
    t1, t2 = th[..., :half], th[..., half:]
    c = cos.to(t.dtype)
    s = sin.to(t.dtype)
    if c.ndim == 2:
        c = c.unsqueeze(-2)
        s = s.unsqueeze(-2)
    return torch.cat([t1 * c - t2 * s, t1 * s + t2 * c], dim=-1).flatten(-2)


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
        use_attn_bias: bool = True,
        use_activation_bias: bool = True,
        alibi: bool = False,
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
            use_attn_bias: inject the relative-position bias into the
                pre-softmax attention logits (`<q_i, R_ij>`). Defaults to True.
            use_activation_bias: inject the relative-position bias into the
                post-softmax output activations (`sum_j P_ij R_ij`). Defaults
                to True.
            alibi: read `rel_pos_bias` as a scalar `[B, L, L]` distance and
                add it to the logits scaled by a learned per-head slope,
                ALiBi-style, instead of contracting a `[B, L, L, C]` feature
                against the query. The bias is then independent of the query
                content, which is what lets it be recomputed inside a fused
                attention kernel rather than materialised. Implies no
                activation bias -- there is no per-pair vector to add.
        """
        if input_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"dim and num_heads must be greater than 0,"
                f" got input_dim={input_dim} and num_heads={num_heads} instead"
            )

        super().__init__()
        self.num_heads = num_heads
        self.use_attn_bias = use_attn_bias
        self.use_activation_bias = use_activation_bias and not alibi
        if alibi:
            # A geometric ladder, as in ALiBi: heads span strong to
            # negligible causal priors, so the mechanism cannot dominate every
            # head at once before training has moved the slopes.
            self.alibi_gamma: Optional[nn.Parameter] = nn.Parameter(
                2.0 ** -torch.arange(num_heads, dtype=torch.float32)
            )
        else:
            self.alibi_gamma = None
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
        if rel_pos_bias is not None and self.use_attn_bias:
            if self.alibi_gamma is not None:
                bias = self.alibi_gamma.view(1, -1, 1, 1) * rel_pos_bias[
                    :, None
                ].to(attn.dtype)
            else:
                bias = torch.einsum("bhic,bijc->bhij", q, rel_pos_bias)
            attn = attn + bias
        if key_padding_mask is not None:
            assert (
                key_padding_mask.is_floating_point()
            ), "key_padding_mask must be additive, i.e. a float mask"
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
        if rel_pos_bias is not None and self.use_activation_bias:
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
                key_padding_mask.is_floating_point()
            ), "key_padding_mask must be additive, i.e. a float mask"
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
        use_attn_bias: bool = True,
        use_activation_bias: bool = True,
        alibi: bool = False,
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
            use_attn_bias: Inject the relative-position bias into the
                pre-softmax attention logits. Defaults to True.
            use_activation_bias: Inject the relative-position bias into the
                post-softmax output activations. Defaults to True.
            alibi: Read `rel_pos_bias` as a scalar distance scaled by a
                learned per-head slope rather than as a per-pair feature
                contracted against the query. See `Attention_rel`.
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
            use_attn_bias=use_attn_bias,
            use_activation_bias=use_activation_bias,
            alibi=alibi,
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

    def forward_flash(
        self,
        x: Tensor,
        feats: Tensor,
        seqlens: Tensor,
        spacetime: "LightningModule",
        use_bias: bool = True,
    ) -> Tensor:
        """`forward` with the fused spacetime-bias attention kernel.

        Semantically the eager path with `rel_pos_bias =
        spacetime(feats)` (or None when `use_bias` is False), but the
        per-pair tensor is never materialised: the kernel recomputes the
        interval features in-tile and routes dW/db to the same
        `spacetime.projection` parameters. Padding rows of the output are
        exactly zero rather than eager's garbage; nothing downstream
        reads them either way.
        """
        # Heavy optional dependency (triton, GPU-only): imported on first
        # use so CPU-only environments can still import this module.
        from flash_spacetime import (
            attention_rel_oracle_inputs,
            flash_spacetime_attention,
            merge_heads,
        )

        xn = self.norm1(x)
        q, k, v = attention_rel_oracle_inputs(self.attn, xn)
        out = flash_spacetime_attention(
            q,
            k,
            v,
            feats,
            spacetime.projection.weight,
            spacetime.projection.bias,
            seqlens,
            scale=self.attn.scale,
            use_attn_bias=use_bias and self.attn.use_attn_bias,
            use_activation_bias=use_bias and self.attn.use_activation_bias,
        )
        out = self.attn.proj_drop(self.attn.proj(merge_heads(out)))
        if self.gamma_1 is None:
            x = x + self.drop_path(out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.drop_path(out))
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
        qk_norm: bool = False,
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
            qk_norm: Apply a per-head RMSNorm to queries and keys before
                attention. Bounds the attention-logit scale, preventing the
                unbounded QK-weight growth / attention-entropy collapse that
                destabilizes training without a relative attention bias.
                Only implemented for the jagged path (`forward_jagged`);
                the padded path raises, as `nn.MultiheadAttention`'s fused
                forward does not expose q/k.
        """
        super().__init__()
        self.norm1 = norm_layer(input_dim)
        self.attn = nn.MultiheadAttention(
            input_dim, num_heads, dropout=attn_drop, batch_first=True
        )
        if qk_norm:
            self.q_norm: Optional[nn.Module] = nn.RMSNorm(self.attn.head_dim)
            self.k_norm: Optional[nn.Module] = nn.RMSNorm(self.attn.head_dim)
        else:
            self.q_norm = self.k_norm = None
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

    def _nested_self_attention(
        self,
        v: Tensor,
        offsets: Tensor,
        min_seqlen: int,
        max_seqlen: int,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
    ) -> Tensor:
        """Self-attention over a jagged sequence held as dense values.

        `v` is the `[total_tokens, D]` value buffer of a jagged
        `NestedTensor` and `offsets` marks the event boundaries. The same
        projection->attention->projection as `self.attn` is computed
        (reusing its weights) through `scaled_dot_product_attention`, which
        dispatches to fused variable-length (flash) kernels on a jagged
        tensor. Only the attention itself touches the `NestedTensor`; the
        projections and head reshapes stay on the dense buffer, since eager
        jagged tensors lack `layer_norm`/`unflatten` under `inference_mode`
        (the eval/predict path) and would otherwise raise there.
        """
        attn = self.attn
        qkv = linear(v, attn.in_proj_weight, attn.in_proj_bias)
        q, k, val = qkv.chunk(3, dim=-1)
        if self.q_norm is not None and self.k_norm is not None:
            # On the dense buffer: eager jagged tensors lack the norm ops.
            heads = [attn.num_heads, attn.head_dim]
            q = self.q_norm(q.unflatten(-1, heads)).flatten(-2)
            k = self.k_norm(k.unflatten(-1, heads)).flatten(-2)
        if rope_cos is not None and rope_sin is not None:
            # After the norm: rotations preserve the normalized RMS.
            q = apply_spacetime_rope(
                q, rope_cos, rope_sin, attn.num_heads, attn.head_dim
            )
            k = apply_spacetime_rope(
                k, rope_cos, rope_sin, attn.num_heads, attn.head_dim
            )

        def to_heads(t: Tensor) -> Tensor:
            # dense [N, D] -> jagged [B, num_heads, S*, head_dim]
            t = t.contiguous().unflatten(-1, [attn.num_heads, attn.head_dim])
            t = torch.nested.nested_tensor_from_jagged(
                t, offsets, min_seqlen=min_seqlen, max_seqlen=max_seqlen
            )
            return t.transpose(1, 2)

        out = scaled_dot_product_attention(
            to_heads(q),
            to_heads(k),
            to_heads(val),
            dropout_p=attn.dropout if self.training else 0.0,
        )
        out = (
            out.transpose(1, 2)
            .values()
            .reshape(-1, attn.num_heads * attn.head_dim)
        )
        return linear(out, attn.out_proj.weight, attn.out_proj.bias)

    def forward_jagged(
        self,
        v: Tensor,
        offsets: Tensor,
        min_seqlen: int,
        max_seqlen: int,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
    ) -> Tensor:
        """Block forward on the dense values of a jagged sequence.

        Normalisation, the MLP and the residual adds are per-token, so they
        run on the dense value buffer; only attention is aware of the event
        boundaries carried by `offsets`.
        """
        attn_out = self._nested_self_attention(
            self.norm1(v),
            offsets,
            min_seqlen,
            max_seqlen,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        if self.gamma_1 is None:
            v = v + self.drop_path(attn_out)
            v = v + self.drop_path(self.mlp(self.norm2(v)))
        else:
            v = v + self.drop_path(self.gamma_1 * attn_out)
            v = v + self.drop_path(self.gamma_2 * self.mlp(self.norm2(v)))
        return v

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        if x.is_nested:
            # Masks encode padding, which jagged tensors represent
            # structurally; a mask here would be silently ignored.
            assert attn_mask is None and key_padding_mask is None
            out = self.forward_jagged(
                x.values(),
                x.offsets(),
                x._get_min_seqlen(),
                x._get_max_seqlen(),
            )
            return torch.nested.nested_tensor_from_jagged(
                out,
                x.offsets(),
                min_seqlen=x._get_min_seqlen(),
                max_seqlen=x._get_max_seqlen(),
            )
        if self.q_norm is not None:
            raise NotImplementedError(
                "qk_norm is only implemented for the jagged path; "
                "`nn.MultiheadAttention`'s fused forward does not "
                "expose q/k."
            )
        xn = self.norm1(x)
        attn_out = self.attn(
            xn,
            xn,
            xn,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        if self.gamma_1 is None:
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * attn_out)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
