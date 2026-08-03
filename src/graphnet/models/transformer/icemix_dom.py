"""DeepIce variant whose tokens carry no absolute spacetime.

Pulses are tokenized from DOM identity and charge only; coordinates and
times reach the model exclusively through the rotary spacetime attention
of `DeepIceRope`, making predictions invariant to a global time offset by
construction and replacing hand-scaled coordinate sinusoids with a
learned per-sensor code.
"""

import torch
import torch.nn as nn
from torch.functional import Tensor

from typing import Optional

from pytorch_lightning import LightningModule

from graphnet.models.components.embedding import SinusoidalPosEmb
from graphnet.models.transformer.icemix_rope import DeepIceRope


class DOMChargeEncoder(LightningModule):
    """Tokenizer from DOM identity and charge only.

    Drop-in replacement for `FourierEncoder` (same call signature and output
    shape) whose tokens carry no absolute position or time: each pulse is
    represented by a learned embedding of the DOM that recorded it, its
    charge, and the event length. Pairwise geometry must reach attention
    through a relative encoding such as spacetime RoPE.

    Pulses are mapped to DOMs by nearest-neighbour lookup against a fixed
    sensor table in the coordinate frame the model receives. Sensor
    positions are discrete and not augmented, so on real pulses the lookup
    is exact; it fails loud if any pulse sits farther from every sensor
    than half the minimum inter-sensor spacing. Padded inputs are the
    exception: all-zero padding rows take the embedding of whatever sensor
    is nearest the origin, which is harmless because downstream attention
    masks remove them (the same contract under which `FourierEncoder`
    embeds padding rows).

    Assumes the NuBench feature order (x, y, z, charge, t): charge is read
    from column 3.
    """

    def __init__(
        self,
        sensor_table_path: str,
        seq_length: int = 128,
        output_dim: int = 384,
        mlp_dim: Optional[int] = None,
        scaled: bool = False,
    ):
        """Construct `DOMChargeEncoder`.

        Args:
            sensor_table_path: Path to a ``torch.save``d dict with
                ``positions`` ([n_sensors, 3] float32, in the coordinate
                frame the model receives) and ``min_spacing`` (smallest
                inter-sensor distance in that frame, bounding the
                nearest-neighbour match radius).
            seq_length: Dimensionality of the charge sinusoidal embedding
                (the event-length embedding uses ``seq_length // 2``).
            output_dim: Dimension of the output (and of the per-DOM
                embedding).
            mlp_dim (Optional): Size of hidden, latent space of MLP. If not
                given, matches the concatenated embedding width.
            scaled: Whether or not to scale the sinusoidal embeddings.
        """
        super().__init__()
        table = torch.load(sensor_table_path, map_location="cpu")
        positions = table["positions"].to(torch.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"sensor table must be [n_sensors, 3], got "
                f"{tuple(positions.shape)}"
            )
        # Fixed geometry, so not part of the state_dict; the table file is
        # referenced by the model config and re-read on construction.
        self.register_buffer("sensor_positions", positions, persistent=False)
        self.match_radius = float(table["min_spacing"]) / 2.0

        self.dom_emb = nn.Embedding(positions.shape[0], output_dim)
        self.sin_emb = SinusoidalPosEmb(dim=seq_length, scaled=scaled)
        self.sin_emb2 = SinusoidalPosEmb(dim=seq_length // 2, scaled=scaled)

        hidden_dim = output_dim + seq_length + seq_length // 2
        if mlp_dim is None:
            mlp_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )

    def _dom_indices(self, pos: Tensor, strict: bool) -> Tensor:
        """Nearest sensor index per pulse position [T, 3].

        Chunked so the [chunk, n_sensors] distance matrix stays small at
        any batch size. ``strict`` enforces the exact-match radius; it is
        off only for padded inputs, whose all-zero padding rows are not
        pulses.
        """
        table = self.sensor_positions
        indices = []
        for chunk in pos.split(16384):
            min_dist, idx = torch.cdist(chunk.to(table.dtype), table).min(
                dim=1
            )
            if strict and (min_dist > self.match_radius).any():
                n_bad = int((min_dist > self.match_radius).sum())
                raise ValueError(
                    f"{n_bad} pulses match no sensor within "
                    f"{self.match_radius:.4g} (max residual "
                    f"{min_dist.max().item():.4g}); the sensor table does "
                    "not correspond to this detector/coordinate frame"
                )
            indices.append(idx)
        return torch.cat(indices)

    def forward(self, x: Tensor, seq_length: Tensor) -> Tensor:
        """Forward pass; accepts padded [B, L, D] or jagged `NestedTensor`."""
        if x.is_nested:
            v = x.values()
            length = torch.log10(seq_length.to(dtype=v.dtype))
            batch_idx = torch.repeat_interleave(
                torch.arange(seq_length.numel(), device=v.device), seq_length
            )
            out = self.mlp(
                torch.cat(
                    [
                        self.dom_emb(self._dom_indices(v[:, :3], strict=True)),
                        self.sin_emb(1024 * v[:, 3]),  # Charge
                        self.sin_emb2(length)[batch_idx],  # Length
                    ],
                    -1,
                )
            )
            return torch.nested.nested_tensor_from_jagged(
                out,
                x.offsets(),
                min_seqlen=x._get_min_seqlen(),
                max_seqlen=x._get_max_seqlen(),
            )

        batch_size, max_len, _ = x.shape
        length = torch.log10(seq_length.to(dtype=x.dtype))
        dom = self.dom_emb(
            self._dom_indices(x.reshape(-1, x.shape[-1])[:, :3], strict=False)
        ).view(batch_size, max_len, -1)
        embeddings = [
            dom,
            self.sin_emb(1024 * x[:, :, 3]),  # Charge
            self.sin_emb2(length).unsqueeze(1).expand(-1, max_len, -1),
        ]
        return self.mlp(torch.cat(embeddings, -1))


class DeepIceDOM(DeepIceRope):
    """`DeepIceRope` tokenized by `DOMChargeEncoder`, not `FourierEncoder`.

    Identical transformer to `DeepIceRope` (same blocks, depths, rotary
    encoding and jagged execution); only the pulse tokenizer differs. With
    absolute coordinates absent from the tokens, the rotary encoding is
    the model's only source of pairwise geometry.
    """

    def __init__(
        self,
        sensor_table_path: str,
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
        """Construct `DeepIceDOM`.

        Args:
            sensor_table_path: Sensor-position table for `DOMChargeEncoder`
                (see its docstring).
            hidden_dim: The latent feature dimension (also the per-DOM
                embedding dimension).
            mlp_ratio: Mlp expansion ratio of the tokenizer and Transformer.
            seq_length: The base feature dimension.
            depth: The depth of the transformer.
            head_size: The size of the attention heads.
            depth_rel: The number of blocks standing in for `DeepIce`'s
                relative-attention sandwich (see `DeepIceRope`).
            scaled_emb: Whether to scale the sinusoidal positional
                embeddings.
            n_features: The number of features in the input data. Must be 5
                (NuBench order x, y, z, charge, t): the tokenizer reads
                charge from column 3.
            rope_per_axis: See `DeepIceRope`.
            compile_blocks: See `DeepIceRope`.
        """
        if n_features != 5:
            raise ValueError(
                "DeepIceDOM assumes the NuBench feature order "
                "(x, y, z, charge, t) and reads charge from column 3; got "
                f"n_features={n_features}."
            )
        super().__init__(
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            seq_length=seq_length,
            depth=depth,
            head_size=head_size,
            depth_rel=depth_rel,
            scaled_emb=scaled_emb,
            n_features=n_features,
            rope_per_axis=rope_per_axis,
            compile_blocks=compile_blocks,
        )
        self.fourier_ext = DOMChargeEncoder(
            sensor_table_path=sensor_table_path,
            seq_length=seq_length,
            output_dim=hidden_dim,
            scaled=scaled_emb,
        )
