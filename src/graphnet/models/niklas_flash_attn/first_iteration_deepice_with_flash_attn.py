#important: need to install flash attention; follow guide on https://github.com/Dao-AILab/flash-attention
#for me I needed to first conda install nvcc,
#in detail the process that worked for me is this:
#0. assuming you start from a fresh environment that is a clone of base from miniforge3 installation
#1. pip install graphnet with pytorch and CU 12.1, see https://graphnet-team.github.io/graphnet/installation/install.html
#   might have to tweak setup.py and set axkward >= 1.8, at least I had to
#2. conda install nvcc like this: conda install nvidia::cuda-nvcc, see https://anaconda.org/nvidia/cuda-nvcc
#3. follow flash-attn installation on https://github.com/Dao-AILab/flash-attention:
#   first pip install ninja
#   then pip install flash-attn --no-build-isolation

from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_func
from flash_attn.modules.mha import MHA

import torch
from torch.functional import Tensor
from torch.nn.functional import linear

from einops import rearrange

from typing import Set, List, Optional, Dict, Any

from pytorch_lightning import LightningModule

from torch_geometric.data import Data
from torch_geometric.nn.pool import (
    knn_graph,
)
from torch_geometric.nn.pool.select.topk import topk

from graphnet.models import StandardModel
from graphnet.models.graphs import KNNGraph
from graphnet.models.detector import IceCube86
from graphnet.models.gnn.gnn import GNN
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.task.reconstruction import (
    DirectionReconstructionWithKappa,
    EnergyReconstruction,
)
from graphnet.models.components.layers import DropPath
from graphnet.models.components.embedding import (
    SinusoidalPosEmb
)

from graphnet.training.loss_functions import LogCoshLoss
from graphnet.training.loss_functions import VonMisesFisher3DLoss

from graphnet.data import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.dataset import ParquetDataset
from graphnet.data.utilities.sqlite_utilities import query_database

from graphnet.training.labels import Direction

#region ### Block for (modified) DeepIce auxiliaries ###

def my_sub_sample(
    data, 
    max_length, 
    columns = [0, 1, 2], 
    nb_nearest=8, 
    hlc_pos=6
):
    x = data.x
    btch = data.batch
    x = x.view(-1, 1) if x.dim() == 1 else x
    score = x[:,hlc_pos-1]

    node_index = topk(score, max_length, btch)
    edge_ind = knn_graph(x=x[:, columns][node_index], k=nb_nearest, batch=btch[node_index])
    return node_index , edge_ind

class flashv2_FourierEncoder(LightningModule):
    """Fourier encoder module.

    This module incorporates sinusoidal positional embeddings and auxiliary
    embeddings to process input sequences and produce meaningful
    representations. The module assumes that the input data is in the format of
    (x, y, z, time, charge, auxiliary), the first four features being
    mandatory.
    """

    def __init__(
        self,
        seq_length: int = 128,
        mlp_dim: Optional[int] = None,
        output_dim: int = 384,
        scaled: bool = False,
        n_features: int = 6,
    ):
        """Construct `FourierEncoder`.

        Args:
            seq_length: Dimensionality of the base sinusoidal positional
                embeddings.
            mlp_dim (Optional): Size of hidden, latent space of MLP. If not
                given, `mlp_dim` is set automatically as multiples of
                `seq_length` (in consistent with the 2nd place solution),
                depending on `n_features`.
            output_dim: Dimension of the output (I.e. number of columns).
            scaled: Whether or not to scale the embeddings.
            n_features: The number of features in the input data.
        """
        super().__init__()

        self.sin_emb = SinusoidalPosEmb(dim=seq_length, scaled=scaled)
        self.sin_emb2 = SinusoidalPosEmb(dim=seq_length // 2, scaled=scaled)

        if n_features < 4:
            raise ValueError(
                f"At least x, y, z and time of the DOM are required. Got only "
                f"{n_features} features."
            )
        elif n_features >= 6:
            self.aux_emb = torch.nn.Embedding(2, seq_length // 2)
            hidden_dim = 6 * seq_length
        else:
            hidden_dim = int((n_features + 0.5) * seq_length)

        if mlp_dim is None:
            mlp_dim = hidden_dim

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, mlp_dim, dtype=torch.bfloat16),
            torch.nn.LayerNorm(mlp_dim, dtype=torch.bfloat16, eps=1e-05, elementwise_affine=False),
            torch.nn.GELU(),
            torch.nn.Linear(mlp_dim, output_dim, dtype=torch.bfloat16),
        )

        self.n_features = n_features

    def forward(
        self,
        x: Tensor,
        seq_length: Tensor,
    ) -> Tensor:
        """Forward pass."""
        length = torch.log10(seq_length.to(dtype=x.dtype))
        embeddings = [self.sin_emb(4096 * x[:, :3]).flatten(-2)]  # Position

        if self.n_features >= 5:
            embeddings.append(self.sin_emb(1024 * x[:, 4]))  # Charge

        embeddings.append(self.sin_emb(4096 * x[:, 3]))  # Time

        if self.n_features >= 6:
            embeddings.append(self.aux_emb(x[:, 5].long()))  # Auxiliary

        len_emb = self.sin_emb2(length)
        embeddings.append(
            torch.cat([len_emb[i].expand(seq_length[i],-1) for i in range(seq_length.shape[0])], dim=0)
        )  # Length; potentially problematic runtime

        x = torch.cat(embeddings, -1).to(dtype=torch.bfloat16)
        x = self.mlp(x)

        return x
    
class dev_flashv2_SpacetimeEncoder(LightningModule):
    """Spacetime encoder module."""

    def __init__(
        self,
        seq_length: int = 32,
        hidden_dim: int = 384,
    ):
        """Construct `SpacetimeEncoder`.

        This module calculates space-time interval between each pair of events
        and generates sinusoidal positional embeddings to be added to input
        sequences.
        This deviates quite a bit from the original SpaceTimeEncoder of DeepIce.
        In particular it does not compute embeddings of padded sequences but instead iterartes over examples in the batch
        and individually generates embeddings on those. That procedure slows down the process and is incompatible with composition
        with a compression method.

        Args:
            seq_length: Dimensionality of the sinusoidal positional embeddings.
            hidden_dim: should match hidden_dim of the Transformer it is used in; to unify the sizes of each example 
                after spacetime embedding
        """
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(dim=seq_length)
        self.projection = torch.nn.Linear(seq_length, hidden_dim)

    def forward(
        self,
        x: Tensor,
        cu_seqs: Tensor,
    ) -> Tensor:
        """Forward pass."""
        #deviating quite a bit from original spacetime encoder: 
        #calculate first the four_distance for each sequence and pass them individually through sinusoidal embedding and projection
        #project to hidden_dim of model and sum for each token in sequence
        #this does not work like the original spacetime encoding but still introduces somehow information on 4 distances between tokens
        #this will then be added to q and to the softmax output
        #also this might be wasteful in time management and space, however at least regarding memory space the original would also be wasteful
        four_d_l = []
        for i in range(1, cu_seqs.shape[0]):
            pos = x[cu_seqs[i-1]:cu_seqs[i], :3]
            time = x[cu_seqs[i-1]:cu_seqs[i], 3]
            spacetime_interval = (pos[:, None] - pos[None, :]).pow(2).sum(
                -1
            ) - ((time[:, None] - time[None, :]) * (3e4 / 500 * 3e-1)).pow(2)
            four_distance = torch.sign(spacetime_interval) * torch.sqrt(
                torch.abs(spacetime_interval)
            )
            sin_emb = self.sin_emb(1024 * four_distance.clip(-4, 4))
            proj_emb = self.projection(sin_emb).sum(dim=1)
            four_d_l.append(proj_emb)
        rel_attn = torch.cat(four_d_l).to(dtype=torch.bfloat16)

        return rel_attn

class flash_Mlp(LightningModule):
    """Multi-Layer Perceptron (MLP) module."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: torch.nn.Module = torch.nn.GELU,
        dropout_prob: float = 0.0,
    ):
        """Construct `Mlp`.

        This is mostly analogous to the Mlp for DeepIce other than the dtypes being chnaged to torch.float16

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
        self.input_projection = torch.nn.Linear(in_features, hidden_features, dtype=torch.bfloat16)
        self.activation = activation()
        self.output_projection = torch.nn.Linear(hidden_features, out_features, dtype=torch.bfloat16)
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.output_projection(x)
        x = self.dropout(x)
        return x

class flashv2_attention(LightningModule):
    """Attention mechanism without relative position bias."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        softm_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """Construct flashv2_attention.

        In the forward pass you find an unused argument: rel_pos_bias. This is just for flexibly changing between using and not using
        that bias in flashv2_block.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: the number of attention heads to use (default: 8)
            qkv_bias: whether to add bias to the query, key, and value
                projections. Defaults to False.
            softm_scale: a scaling factor that multiplies before to the attention matrix before softmax calc.
                Is passed to the flash_attn function and if None uses flash_attn's standard value
            attn_drop: the dropout probability for the attention weights.
                Defaults to 0.0.
            proj_drop: the dropout probability for the output of the attention
                module. Defaults to 0.0.
        """
        if input_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"dim and num_heads must be greater than 0,"
                f" got input_dim={input_dim} and num_heads={num_heads} instead"
            )

        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.all_head_dim = self.head_dim * self.num_heads
        self.softm_scale = softm_scale

        self.proj_qkv = torch.nn.Linear(input_dim, 3*self.all_head_dim, bias=False, dtype=torch.bfloat16)
        if qkv_bias:
            self.qkv_bias = torch.nn.Parameter(torch.zeros(self.all_head_dim))
        else:
            self.qkv_bias = None

        self.attn_drop = attn_drop
        self.proj = torch.nn.Linear(self.all_head_dim, input_dim, dtype=torch.bfloat16)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(
        self,
        x: Tensor,
        cu_seqs: Tensor,
        max_seq: int,
        rel_pos_bias: Optional[Tensor] = None, #only here so I can pass the None rel_pos_bias in versions where I don't want that bias
    ) -> Tensor:
        """Forward pass."""

        qkv = linear(input=x, weight=self.proj_qkv.weight, bias=self.qkv_bias)
        qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)

        out = flash_attn_varlen_qkvpacked_func(qkv=qkv,
                                               cu_seqlens=cu_seqs,
                                               max_seqlen=max_seq,
                                               dropout_p=self.attn_drop,
                                               softmax_scale=self.softm_scale)
        out = out.reshape(-1, self.all_head_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    
class flashv2_attention_unpacked(LightningModule):
    """Attention mechanism with relative position bias."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        softm_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """Construct 'flashv2_attention_unpacked'.

        Version of flash_attn with rel_pos_bias. Implementation of rel_pos_bias is quite different from original DeepIce
        due to differences in required data shapes. This version is not generally recommended as it is slower and rel_pos_bias
        loses meaning with a compression before the Transformer.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: the number of attention heads to use (default: 8)
            qkv_bias: whether to add bias to the query, key, and value
                projections. Defaults to False.
            softm_scale: a scaling factor that multiplies before to the attention matrix before softmax calc.
                Is passed to the flash_attn function and if None uses flash_attn's standard value
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
        self.head_dim = input_dim // num_heads
        self.all_head_dim = self.head_dim * self.num_heads
        self.softm_scale = softm_scale

        self.proj_q = torch.nn.Linear(input_dim, self.all_head_dim, bias=False, dtype=torch.bfloat16)
        self.proj_k = torch.nn.Linear(input_dim, self.all_head_dim, bias=False, dtype=torch.bfloat16)
        self.proj_v = torch.nn.Linear(input_dim, self.all_head_dim, bias=False, dtype=torch.bfloat16)
        if qkv_bias:
            self.q_bias = torch.nn.Parameter(torch.zeros(self.all_head_dim))
            self.v_bias = torch.nn.Parameter(torch.zeros(self.all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = attn_drop
        self.proj = torch.nn.Linear(self.all_head_dim, input_dim, dtype=torch.bfloat16)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(
        self,
        x: Tensor,
        cu_seqs: Tensor,
        max_seq: int,
        rel_pos_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""

        q = linear(input=x, weight=self.proj_q.weight, bias=self.q_bias)
        if rel_pos_bias is not None:
            q = q + rel_pos_bias
        k = linear(input=x, weight=self.proj_k.weight, bias=None)
        v = linear(input=x, weight=self.proj_v.weight, bias=self.v_bias)
        q=q.reshape(q.shape[0], self.num_heads, self.head_dim)
        k=k.reshape(k.shape[0], self.num_heads, self.head_dim)
        v=v.reshape(v.shape[0], self.num_heads, self.head_dim)

        out = flash_attn_varlen_func(q=q,
                                     k=k,
                                     v=v,
                                     cu_seqlens_q=cu_seqs,
                                     cu_seqlens_k=cu_seqs,
                                     max_seqlen_q=max_seq,
                                     max_seqlen_k=max_seq,
                                     dropout_p=self.attn_drop,
                                     softmax_scale=self.softm_scale)
        out = out.reshape(-1, self.all_head_dim)
        if rel_pos_bias is not None:
            out = out + rel_pos_bias
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class flashv2_block(LightningModule):
    """Mimic attention_rel block in DeepIce, i.e. first Transformer block"""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        softm_scale: Optional[float] = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        activation: torch.nn.Module = torch.nn.GELU,
        norm_layer: torch.nn.Module = torch.nn.LayerNorm,
        have_rel_bias: bool = False,
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
            softm_scale: Scaling factor for the attention matrix in the attention blocks
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
            have_rel_bias: Boolean to choose whether to use a version of the rel_pos_bias from the original DeepIce. 
                Effectively chooses between flashv2_attention and flashv2_attention_unpacked.
        """
        super().__init__()
        self.norm1 = norm_layer(input_dim, dtype=torch.bfloat16, eps=1e-05, elementwise_affine=False)
        if have_rel_bias:
            self.attn = flashv2_attention_unpacked(
                input_dim,
                num_heads,
                attn_drop=attn_drop,
                qkv_bias=qkv_bias,
                softm_scale=softm_scale,
            )
        else:
            self.attn = flashv2_attention(
                input_dim,
                num_heads,
                attn_drop=attn_drop,
                qkv_bias=qkv_bias,
                softm_scale=softm_scale,
            )

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
        )
        self.norm2 = norm_layer(input_dim, dtype=torch.bfloat16, eps=1e-05, elementwise_affine=False)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = flash_Mlp(
            in_features=input_dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            dropout_prob=dropout,
        )

        if init_values is not None:
            self.gamma_1 = torch.nn.Parameter(
                init_values * torch.ones(input_dim).to(device=self.device), requires_grad=True
            ).to(device=self.device, dtype=torch.bfloat16)
            self.gamma_2 = torch.nn.Parameter(
                init_values * torch.ones(input_dim).to(device=self.device), requires_grad=True
            ).to(device=self.device, dtype=torch.bfloat16)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(
        self,
        x: Tensor,
        cu_seqs: Tensor,
        max_seq: int,
        rel_pos_bias: Optional[Tensor] = None,


    ) -> Tensor:
        """Forward pass."""
        if self.gamma_1 is None:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.attn(
                    xn,
                    cu_seqs=cu_seqs,
                    max_seq=max_seq,
                    rel_pos_bias=rel_pos_bias
                )
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.gamma_1.to(device=self.device)
                * self.drop_path(
                    self.attn(
                        xn,
                        cu_seqs=cu_seqs,
                        max_seq=max_seq,
                        rel_pos_bias=rel_pos_bias
                    )
                )
            )
            x = x + self.drop_path(self.gamma_2.to(device=self.device) * self.mlp(self.norm2(x)))
        return x
    
class flashMHA_block(LightningModule):
    """Mimic MHA block in DeepIce, i.e. second Transformer block"""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        softm_scale: Optional[float] = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        activation: torch.nn.Module = torch.nn.GELU,
        norm_layer: torch.nn.Module = torch.nn.LayerNorm,
    ):
        """Construct 'Block_rel'.

        Implements flash_attn's MHA module. Most of the arguments pertain to everything but theís module and as of now
        tweaking the actual attention mechanism can be done by adjusting the parameters passed to the MHA in the class initialization
        by hand.

        Args:
            input_dim: Dimension of the input tensor.
            num_heads: Number of attention heads to use in the `Attention_rel`
            layer.
            mlp_ratio: Ratio of the hidden size of the feedforward network to
                the input size in the `Mlp` layer.
            softm_scale: Scaling factor for the dot product of the query and key
                matrices within flash_attn's MHA.
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
        self.norm1 = norm_layer(input_dim, dtype=torch.bfloat16, eps=1e-05, elementwise_affine=False)
        self.attn = MHA(embed_dim=input_dim,
                        num_heads=num_heads,
                        use_flash_attn=True,
                        softmax_scale=softm_scale,
                        dropout=attn_drop,
                        dtype=torch.bfloat16)
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
        )
        self.norm2 = norm_layer(input_dim, dtype=torch.bfloat16, eps=1e-05, elementwise_affine=False)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = flash_Mlp(
            in_features=input_dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            dropout_prob=dropout,
        )

        if init_values is not None:
            self.gamma_1 = torch.nn.Parameter(
                init_values * torch.ones(input_dim).to(device=self.device), requires_grad=True
            ).to(device=self.device, dtype=torch.bfloat16)
            self.gamma_2 = torch.nn.Parameter(
                init_values * torch.ones(input_dim).to(device=self.device), requires_grad=True
            ).to(device=self.device, dtype=torch.bfloat16)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(
        self,
        x: Tensor,
        cu_seqs: Tensor,
        max_seq: int,


    ) -> Tensor:
        """Forward pass."""
        if self.gamma_1 is None:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.attn(
                    xn,
                    cu_seqlens=cu_seqs,
                    max_seqlen=max_seq,
                )
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            x = x + self.drop_path(
                self.gamma_1.to(device=self.device)
                * self.drop_path(
                    self.attn(
                        xn,
                        cu_seqlens=cu_seqs,
                        max_seqlen=max_seq,
                    )
                )
            )
            x = x + self.drop_path(self.gamma_2.to(device=self.device) * self.mlp(self.norm2(x)))
        return x

#endregion

#region ### Block with DeepIce Modifications ###

class Theseus_DeepIce(GNN):
    """DeepIce model."""

    def __init__(
        self,
        hidden_dim: int = 384,
        mlp_ratio: int = 4,
        seq_length: int = 2560,
        depth: int = 12,
        head_size: int = 32,
        depth_rel: int = 4,
        n_rel: int = 1,
        scaled_emb: bool = False,
        include_dynedge: bool = False,
        dynedge_args: Optional[Dict[str, Any]] = None,
        n_features: int = 6,
        have_rel_bias: bool = False,
    ):
        """Construct `Theseus_DeepIce`.

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
            have_rel_bias: choose whether to use rel_pos_bias or not.
                False is recommended for training with a compression method
        """
        super().__init__(seq_length, hidden_dim)
        fourier_out_dim = hidden_dim // 2 if include_dynedge else hidden_dim
        self.fourier_ext = flashv2_FourierEncoder(
            seq_length=seq_length,
            mlp_dim=None,
            output_dim=fourier_out_dim,
            scaled=scaled_emb,
            n_features=n_features,
        )
        self.rel_pos = dev_flashv2_SpacetimeEncoder(head_size, hidden_dim=hidden_dim)
        self.sandwich = torch.nn.ModuleList(
            [
                flashv2_block(
                    input_dim=hidden_dim, num_heads=hidden_dim // head_size, have_rel_bias=have_rel_bias
                )
                for _ in range(depth_rel)
            ]
        )
        self.cls_token = torch.nn.Linear(hidden_dim, 1, bias=False, dtype=torch.bfloat16)
        self.blocks = torch.nn.ModuleList(
            [
                flashMHA_block(
                    input_dim=hidden_dim,
                    num_heads=hidden_dim // head_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=0.0 * (i / (depth - 1)),
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

        self.seq_len = seq_length

        self.have_rel = have_rel_bias

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_tocken should not be subject to weight decay during training."""
        return {"cls_token"}

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""

        #just my subsampling part
        node_ind, _ = my_sub_sample(data=data,
                                    max_length=self.seq_len,
                                    columns=[0,1,2],
                                    nb_nearest=8,
                                    hlc_pos=6)

        subsampled = data.x[node_ind]
        batch_v_subsam = data.batch[node_ind]

        #calculate necessary variables for the attention functions: 
        #cu_seqs=cumulative seq. lengths, used to index seqeunces in flash-attn
        #max_seq=max seq. length in batch, also necessary for flash-attn
        _,seq_lengths = torch.unique_consecutive(batch_v_subsam, return_counts=True)
        batch_size = seq_lengths.shape[0]
        cu_seqs = torch.nn.functional.pad(seq_lengths.cumsum(0), pad=(1,0), value=0).to(torch.int32)
        max_seq = seq_lengths.max().to(device=self.device, dtype=torch.int32).item()

        #modified version of Fourier Encoder; works mainly the same but takes and returns different data layout
        x = self.fourier_ext(subsampled, seq_lengths).to(dtype=torch.bfloat16)

        #calculate rel_pos_bias with modified method; strongly deviates from DeepIce's rel_pos_bias calc
        if self.have_rel:
            rel_pos_bias = self.rel_pos(subsampled, cu_seqs)
        else:
            rel_pos_bias = None

        if self.include_dynedge:
            #test dynedge inclusion at later point!!!
            graph = self.dyn_edge(data)
            x = torch.cat([x, graph], 1)

        #first attention block
        for i, blk in enumerate(self.sandwich):
            x = blk(x=x,
                    cu_seqs=cu_seqs,
                    max_seq=max_seq,
                    rel_pos_bias=rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None

        #insert cls_token
        cls_token = self.cls_token.weight.expand(
           batch_size, -1
        )
        #create empty tensor and append cls_tokens at the beginning of the individual sequences; fill the rest with original x
        emp = torch.empty((x.shape[0]+batch_size, x.shape[1])).to(device=self.device, dtype=torch.bfloat16)
        cls_ind = cu_seqs + torch.arange(batch_size+1).to(device=self.device, dtype=torch.int32)
        ran = torch.arange(x.shape[0]+batch_size).to(device=self.device)
        emp[cls_ind[:-1],:] = cls_token
        emp[ran[~torch.isin(ran,cls_ind[:-1]).to(device=self.device)],:] = x
        x=emp

        #second attention block
        for blk in self.blocks:
            x = blk(x=x,
                    cu_seqs=cls_ind,
                    max_seq=max_seq+1,)

        return x[cls_ind[:-1], :].to(dtype=torch.float32)

#endregion
def main(
    path: str='/scratch/users/mbranden/sim_files/dev_northern_tracks_muon_labels_v3_part_1.db',
    gpus: Optional[List[int]]=None,
    max_epochs=100,
    early_stopping_patience=20,
    num_workers=30,
    batch_size=300,
) -> None:


    features = ['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'hlc']
    truths = ['energy', 'azimuth', 'zenith', 'position_x', 'position_y', 'position_z']

    config: Dict[str, Any] = {
        "path": path,
        "num_workers": num_workers,
        "early_stopping_patience": early_stopping_patience,
        "batch_size": batch_size,
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
        "dataset_reference": SQLiteDataset
        if path.endswith(".db")
        else ParquetDataset,}

    

    graph_definition = KNNGraph(
    detector=IceCube86(),
    nb_nearest_neighbours=8,
    input_feature_names = features,
    )

    query = "select distinct event_no from truth limit 1000000"
    out = query_database(database=path, query=query)
    sel = out['event_no'].tolist()
    
    #defining the data
    dm = GraphNeTDataModule(
        dataset_reference=config["dataset_reference"],
        dataset_args={
            "path": config["path"],
            "pulsemaps" : "InIceDSTPulses",
            "truth_table" : 'truth',
            "features": features,
            "truth":truths,
            "graph_definition" : graph_definition,
            "labels": {'direction': Direction()},   
        },
        selection=sel,
        train_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
        },
        test_selection=sel,
        test_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
        },
        train_val_split= [0.8, 0.20]
    )
    training_dataloader = dm.train_dataloader
    validation_dataloader = dm.val_dataloader

    #data format: [n_total, 6] with n_total=sum(seq_length[i]) with in range(batch_size)
    #first step in transformer subsamples to specified seq_length, so at most data is of shape [batch_size*seq_length, 6]
    #before being passed to the Fourier Encoder
    
    #define tranformer backbone
    transformer = Theseus_DeepIce(seq_length = 2560,
                                  n_features = 6,
                                  have_rel_bias = True)


    #define task; choose appropriate one
    task = EnergyReconstruction(
        target_labels=['energy'],
        hidden_size=transformer.nb_outputs,
        transform_prediction_and_target = lambda x: torch.log10(x),
        loss_function=LogCoshLoss()
    )

    task = DirectionReconstructionWithKappa(
        hidden_size=transformer.nb_outputs,
        target_labels=['direction'],
        loss_function=VonMisesFisher3DLoss(),
    )


    #compose model
    model = StandardModel(
        graph_definition = graph_definition,
        backbone = transformer,
        tasks = task,
        optimizer_kwargs = {'eps': 1e-05},
    )


    
    model.fit(
        training_dataloader,
        validation_dataloader,
        early_stopping_patience=config["early_stopping_patience"],
        distribution_strategy = 'auto',
        logger = None,
        **config["fit"],
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
    )

    

if __name__ == "__main__":
    #note: you have to necessarily pass a gpus argument because flash-attn strictly requires gpus
    main(gpus=[3])