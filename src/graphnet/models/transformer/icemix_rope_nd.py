"""DeepIce with nD-RoPE spacetime attention (arXiv:2606.12146).

`DeepIceRope` follows the common axis-decomposed construction: the head
dimension is split into four groups and each group is rotated by one
coordinate, so every rotation plane is tied to a coordinate axis. nD-RoPE
argues that this fragments a displacement into independent 1D components
and biases the representation toward axis-aligned directions, and replaces
it with a single unified phase

    phi(x) = <omega, x>,   omega in R^n,  x in R^n

where the wave vectors `omega` are the centroid-to-vertex directions of a
regular simplex, repeated over a geometric ladder of scales and rotated
independently per attention head. The simplex conditions

    sum_i omega_i = 0,  ||omega_i|| = r,  <omega_i, omega_j> = -r^2/n

give `sum_i omega_i omega_i^T = ((n+1)/n) r^2 I_n`: the second-order
directional response is identical for every direction in space, which is
the paper's isotropy criterion. Everything downstream is unchanged — the
phases still enter as cosine/sine pairs through the same block rotation,
so attention itself is untouched.

Two adaptations are needed for this detector and are configurable:

- **Commensurate axes.** Isotropy is a statement about the space the
  simplex lives in. The four spacetime coordinates arrive in unrelated
  units (normalised detector position vs. normalised time), so an
  isotropic simplex in raw coordinates would not be isotropic in any
  physical sense. `axis_scales` maps the axes onto a common footing before
  the projection; the default is the geometric-mean frequency of each
  axis's measured pulse-pair band (the same measurements behind
  `DeepIceRope`'s per-axis ladder), which makes a typical pulse-pair
  displacement produce a phase of order one on every axis.
- **Head dimension.** The construction wants `head_size` divisible by
  `2 * (n + 1)` = 10, so that `S = head_size / 10` scales each use all
  `n + 1` wave vectors. At the production `head_size = 48` this gives 4
  full scales (20 of 24 rotation planes); the remaining planes are left
  unrotated and carry content only. `head_size` 40 or 60 divides exactly.

The paper's economy bound on the frequency base, `theta <= exp(head_size /
(2 * (n + 1) * n))`, is tight here: at `head_size = 48` and `n = 4` it
gives `theta <~ 3.3`, i.e. the whole ladder spans well under a decade.
That is a genuine property of putting a 4D isotropic ladder inside a
48-dimensional head, not a tuning oversight, and `rope_base` defaults to
that bound.
"""

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from graphnet.models.transformer.icemix_rope import DeepIceRope

# Geometric-mean frequency of each axis's measured pulse-pair band on the
# hexagon detector (x, y, z, t); see `DeepIceRope`'s per-axis bands. Used to
# put the four coordinates on a common scale before the simplex projection.
HEXAGON_AXIS_SCALES = (2.029, 2.753, 33.62, 9233.4)


def regular_simplex_directions(n: int) -> Tensor:
    """Build unit centroid-to-vertex directions of a simplex in R^n.

    Returns `[n + 1, n]` rows satisfying `sum_i w_i = 0`, `||w_i|| = 1`
    and `<w_i, w_j> = -1/n` for `i != j` (Algorithm 1, steps 1-4: centre
    the vertices of the standard simplex in R^(n+1), drop the null
    direction through its right singular vectors, then normalise).
    """
    centred = torch.eye(n + 1, dtype=torch.float64) - 1.0 / (n + 1)
    # The centring matrix has rank n; the first n right singular vectors
    # span the hyperplane the simplex lives in.
    _, _, vh = torch.linalg.svd(centred)
    basis = vh[:n].T  # [n + 1, n]
    directions = centred @ basis
    return directions / directions.norm(dim=1, keepdim=True)


def random_rotations(n: int, count: int, generator: torch.Generator) -> Tensor:
    """`count` independent rotations from SO(n), as `[count, n, n]`.

    QR of a Gaussian matrix gives a Haar-uniform orthogonal matrix once
    the sign ambiguity of the factorisation is fixed; a reflection (det
    = -1) is turned into a rotation by flipping one axis.
    """
    out = torch.empty(count, n, n, dtype=torch.float64)
    for i in range(count):
        a = torch.randn(n, n, generator=generator, dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        out[i] = q
    return out


class DeepIceRopeND(DeepIceRope):
    """`DeepIceRope` with nD-RoPE wave vectors instead of per-axis bands.

    Identical transformer, tokenizer, and jagged execution; only the
    construction of the rotation angles differs (see the module
    docstring).
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
        n_features: int = 5,
        compile_blocks: bool = False,
        rope_base: Optional[float] = None,
        axis_scales: Sequence[float] = HEXAGON_AXIS_SCALES,
        rope_seed: int = 42,
    ):
        """Construct `DeepIceRopeND`.

        Args:
            hidden_dim: The latent feature dimension.
            mlp_ratio: Mlp expansion ratio of FourierEncoder and Transformer.
            seq_length: The base feature dimension.
            depth: The depth of the transformer.
            head_size: The size of the attention heads. Divisibility by
                `2 * (n_dims + 1)` = 10 uses every rotation plane; at other
                sizes the remainder is left unrotated (see the module
                docstring).
            depth_rel: The number of blocks standing in for `DeepIce`'s
                relative-attention sandwich.
            scaled_emb: Whether to scale the sinusoidal positional
                embeddings.
            n_features: The number of features in the input data. At least
                5, in the NuBench order (x, y, z, charge, t).
            compile_blocks: Wrap the transformer block stack in
                `torch.compile`.
            rope_base: Frequency base of the scale ladder. Defaults to the
                paper's economy bound `exp(head_size / (2 * (n + 1) * n))`
                for this head size.
            axis_scales: Per-axis (x, y, z, t) multipliers applied before
                the simplex projection, putting the coordinates on a common
                scale. Defaults to the hexagon pulse-pair bands.
            rope_seed: Seed for the per-head rotations. The sampled wave
                vectors are stored in the state dict, so a checkpoint is
                exact regardless of this value; the seed only makes a fresh
                model reproducible.
        """
        super().__init__(
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            seq_length=seq_length,
            depth=depth,
            head_size=head_size,
            depth_rel=depth_rel,
            scaled_emb=scaled_emb,
            n_features=n_features,
            rope_per_axis=True,
            compile_blocks=compile_blocks,
        )
        n_dims = 4
        n_heads = hidden_dim // head_size
        half = head_size // 2
        n_wave = n_dims + 1
        n_scales = half // n_wave
        if n_scales < 1:
            raise ValueError(
                f"head_size {head_size} leaves no room for {n_wave} wave "
                f"vectors ({half} rotation planes)"
            )
        if rope_base is None:
            rope_base = math.exp(head_size / (2 * n_wave * n_dims))
        if rope_base <= 1.0:
            raise ValueError(f"rope_base must exceed 1, got {rope_base}")
        if len(axis_scales) != n_dims:
            raise ValueError(
                f"axis_scales needs {n_dims} entries, got {len(axis_scales)}"
            )

        generator = torch.Generator().manual_seed(rope_seed)
        directions = regular_simplex_directions(n_dims)
        rotations = random_rotations(n_dims, n_heads, generator)
        # Per head: rotate the whole simplex, keeping its geometry but
        # removing any shared preferred direction across heads.
        wave = torch.einsum("md,hed->hme", directions, rotations)
        scales = torch.tensor(
            [rope_base ** (-s / n_scales) for s in range(n_scales)],
            dtype=torch.float64,
        )

        # Persistent: the rotations are sampled, so a checkpoint has to
        # carry them to reproduce its own model.
        self.register_buffer("rope_wave", wave, persistent=True)
        self.register_buffer("rope_scales", scales, persistent=True)
        self.register_buffer(
            "rope_axis_scales",
            torch.tensor(axis_scales, dtype=torch.float64),
            persistent=True,
        )
        self.rope_planes = n_scales * n_wave
        self.rope_half = half
        self.rope_base = rope_base

    def _rope_angles(
        self, features: Tensor, batch_idx: Tensor, batch_size: int
    ) -> Tuple[Tensor, Tensor]:
        """Per-token, per-head rotation angles for the jagged stream.

        Coordinates are (x, y, z, t) — with the NuBench feature order the
        time column is index 4, not 3 (index 3 is charge). Returns
        `[tokens, heads, head_size // 2]` cosine/sine pairs; the cls slot of
        every event and any rotation plane beyond `rope_planes` keep
        cos = 1 / sin = 0, the identity rotation.
        """
        # The phase is accumulated in float64 and only the bounded cosine
        # and sine are cast down. Detector time carries a large axis scale,
        # so absolute phases reach 1e4 radians or more, where a float32
        # argument's rounding error alone is enough to corrupt the phase
        # *difference* that makes the encoding relative.
        coords = features[:, [0, 1, 2, 4]].double() * self.rope_axis_scales
        # <omega, x> for every head and wave vector, then the scale ladder.
        projected = torch.einsum("pd,hmd->phm", coords, self.rope_wave)
        angles = (projected.unsqueeze(-1) * self.rope_scales).flatten(-2)

        n_pulses, n_heads, _ = angles.shape
        planes = self.rope_half
        shape = (n_pulses + batch_size, n_heads, planes)
        rope_cos = features.new_ones(shape)
        rope_sin = features.new_zeros(shape)
        pos = torch.arange(n_pulses, device=angles.device) + batch_idx + 1
        rope_cos[pos, :, : self.rope_planes] = torch.cos(angles).to(
            features.dtype
        )
        rope_sin[pos, :, : self.rope_planes] = torch.sin(angles).to(
            features.dtype
        )
        return rope_cos, rope_sin
