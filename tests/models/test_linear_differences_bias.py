"""Tests for the pair bias that runs inside fused attention."""

from typing import Any, Dict, List, Optional

import pytest

import torch
from torch import Tensor
from torch.nn.functional import linear
from torch_geometric.data import Batch, Data

from graphnet.models.components.attention_blocks import Block
from graphnet.models.components.embedding import LinearDifferencesBias
from graphnet.models.transformer import DeepIceLinearBias

HEADS, HEAD_DIM = 2, 16
LENGTHS = [5, 9, 1]


def _modules(directions: object = False) -> tuple:
    torch.manual_seed(0)
    block = Block(HEADS * HEAD_DIM, HEADS, qk_norm=True).double().eval()
    ladder: Dict[str, Any] = (
        dict(direction_scale=2.5, direction_n_freq=40.0)
        if directions == "own ladder"
        else {}
    )
    bias = LinearDifferencesBias(
        HEAD_DIM,
        seq_length=8,
        time_scale=3.0,
        directions=bool(directions),
        **ladder,
    )
    bias = bias.double()
    with torch.no_grad():
        block.q_norm.weight.uniform_(0.5, 1.5)
        block.k_norm.weight.uniform_(0.5, 1.5)
        bias.projection.weight.normal_(std=0.3)
        bias.projection.bias.normal_(std=0.3)
        bias.log_speed.fill_(-0.2)
    # The reference embeds each pair's differences directly, in float64.
    bias.sin_emb.phase_dtype = None
    bias.direction_emb.phase_dtype = None
    return block, bias


def _reference(
    block: Block,
    bias: LinearDifferencesBias,
    x: Tensor,
    coords: Tensor,
    pointing: Optional[Tensor] = None,
) -> Tensor:
    """`Attention_rel` arithmetic with the bias built for every pair."""
    attn = block.attn
    q, k, v = linear(x, attn.in_proj_weight, attn.in_proj_bias).chunk(3, -1)
    assert block.q_norm is not None and block.k_norm is not None
    q = block.q_norm(q.unflatten(-1, [HEADS, HEAD_DIM]))
    k = block.k_norm(k.unflatten(-1, [HEADS, HEAD_DIM]))
    v = v.unflatten(-1, [HEADS, HEAD_DIM])
    speed = bias.time_scale * torch.exp(bias.log_speed)
    scalars = [coords[:, :3], speed * coords[:, 3:]]
    if pointing is not None:
        scalars.append(0.5 * pointing)
    per_step = torch.cat(scalars, -1)
    outputs, start = [], 0
    for n in LENGTHS:
        sl = slice(start, start + n)
        start += n
        difference = per_step[sl, None] - per_step[None, sl]  # [n, n, 4]
        embedded = torch.cat(
            [bias.sin_emb(bias.scale * difference[..., a]) for a in range(4)]
            + [
                bias.direction_emb(bias.direction_scale * difference[..., a])
                for a in range(4, difference.shape[-1])
            ],
            -1,
        )
        pair = bias.projection(embedded)  # [n, n, C]
        qs = q[sl] * HEAD_DIM**-0.5
        logits = torch.einsum("ihc,jhc->hij", qs, k[sl])
        logits = logits + torch.einsum("ihc,ijc->hij", qs, pair)
        p = logits.softmax(-1)
        out = torch.einsum("hij,jhc->ihc", p, v[sl])
        out = out + torch.einsum("hij,ijc->ihc", p, pair)
        outputs.append(out.reshape(n, -1))
    return linear(torch.cat(outputs), attn.out_proj.weight, attn.out_proj.bias)


@pytest.mark.parametrize("directions", [False, True, "own ladder"])
def test_attention_matches_the_bias_built_per_pair(directions: object) -> None:
    """Widened attention adds exactly the per-pair bias, both sides."""
    block, bias = _modules(directions)
    generator = torch.Generator().manual_seed(1)
    n = sum(LENGTHS)
    x = torch.randn(
        n, HEADS * HEAD_DIM, generator=generator, dtype=torch.double
    )
    coords = torch.randn(n, 4, generator=generator, dtype=torch.double) * 0.3
    pointing = None
    if directions:
        pointing = torch.randn(n, 3, generator=generator, dtype=torch.double)
        pointing = pointing / pointing.norm(dim=-1, keepdim=True)
    offsets = torch.tensor([0] + LENGTHS).cumsum(0)
    sin, cos = bias.phases(coords, pointing)
    with torch.no_grad():
        got = block._nested_self_attention(
            x, offsets, min(LENGTHS), max(LENGTHS), pair_bias=(bias, sin, cos)
        )
        want = _reference(block, bias, x, coords, pointing)
    torch.testing.assert_close(got, want, atol=1e-9, rtol=1e-9)


def test_bias_depends_only_on_differences() -> None:
    """Moving every step by the same amount leaves attention unchanged."""
    block, bias = _modules()
    n = sum(LENGTHS)
    x = torch.randn(n, HEADS * HEAD_DIM, dtype=torch.double)
    coords = torch.randn(n, 4, dtype=torch.double) * 0.3
    offsets = torch.tensor([0] + LENGTHS).cumsum(0)

    def attend(c: Tensor) -> Tensor:
        sin, cos = bias.phases(c)
        return block._nested_self_attention(
            x, offsets, min(LENGTHS), max(LENGTHS), pair_bias=(bias, sin, cos)
        )

    with torch.no_grad():
        shifted = coords + torch.tensor([0.4, -0.7, 0.2, 0.05])
        torch.testing.assert_close(attend(shifted), attend(coords))


def _batch(lengths: List[int]) -> Batch:
    generator = torch.Generator().manual_seed(2)
    return Batch.from_data_list(
        [Data(x=torch.randn(n, 6, generator=generator) * 0.3) for n in lengths]
    )


def test_model_reads_each_event_alone() -> None:
    """An event's vector is the same alone as inside a batch."""
    torch.manual_seed(0)
    model = DeepIceLinearBias(
        hidden_dim=64, seq_length=32, depth=2, head_size=16, qk_norm=True
    ).eval()
    batch = _batch([17, 30, 1])
    with torch.no_grad():
        together = model(batch)
        assert together.shape == (3, 64)
        assert torch.isfinite(together).all()
        for i, event in enumerate(batch.to_data_list()):
            alone = model(Batch.from_data_list([event]))
            torch.testing.assert_close(
                alone[0], together[i], atol=1e-5, rtol=1e-4
            )


def test_model_embeds_named_sensor_directions() -> None:
    """With direction features the bias widens by three embedded scalars."""
    names = ["x", "y", "z", "t", "q", "dx", "dy", "dz"]
    kwargs: Dict[str, Any] = dict(
        hidden_dim=64,
        seq_length=16,
        depth=1,
        head_size=16,
        fourier_schema={name: 1024.0 for name in names},
        input_feature_names=names,
        coordinate_features=["x", "y", "z", "t"],
    )
    plain = DeepIceLinearBias(**kwargs)
    pointing = DeepIceLinearBias(
        direction_features=["dx", "dy", "dz"], **kwargs
    )
    assert pointing.pair_bias.n_factors == plain.pair_bias.n_factors * 7 // 4
    batch = Batch.from_data_list([Data(x=torch.randn(n, 8)) for n in (4, 9)])
    with torch.no_grad():
        assert torch.isfinite(pointing(batch)).all()
