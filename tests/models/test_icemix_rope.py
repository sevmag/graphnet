"""Tests for the options of the rotary DeepIce models."""

from typing import Any, Dict, List

import pytest
import torch
from torch_geometric.data import Batch, Data

from graphnet.models.transformer import DeepIceRope, DeepIceRopeND

# A head size both constructions accept: divisible by 8 and by 10.
BOTH = pytest.mark.parametrize(
    "model_class, sizes",
    [
        (DeepIceRope, dict(hidden_dim=64, head_size=16)),
        (DeepIceRopeND, dict(hidden_dim=80, head_size=40)),
    ],
)

HIDDEN_DIM = 64


def _batch(lengths: List[int] = [17, 30, 9]) -> Batch:
    """Build a batch of variable-length synthetic events."""
    generator = torch.Generator().manual_seed(1)
    return Batch.from_data_list(
        [Data(x=torch.randn(n, 5, generator=generator) * 0.3) for n in lengths]
    )


def _rope(model_class: type = DeepIceRope, **overrides: Any) -> DeepIceRope:
    kwargs: Dict[str, Any] = dict(
        hidden_dim=HIDDEN_DIM, seq_length=32, depth=3, head_size=16
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return model_class(**kwargs).eval()


def test_depth_counts_every_block() -> None:
    """`depth` alone sets the size of the model."""
    model = _rope()
    assert len(model.blocks) == 3
    assert len(model.sandwich) == 0


def test_depth_rel_only_changes_the_state_dict_layout() -> None:
    """Blocks named by `depth_rel` are ordinary leading blocks."""
    split = _rope(depth=2, depth_rel=1)
    assert [len(split.sandwich), len(split.blocks)] == [1, 2]

    renamed = {}
    for key, value in split.state_dict().items():
        part, _, rest = key.partition(".")
        if part == "sandwich":
            key = f"blocks.{rest}"
        elif part == "blocks":
            index, _, tail = rest.partition(".")
            key = f"blocks.{int(index) + 1}.{tail}"
        renamed[key] = value
    joined = _rope(depth=3)
    joined.load_state_dict(renamed)

    batch = _batch()
    with torch.no_grad():
        torch.testing.assert_close(joined(batch), split(batch))


@BOTH
def test_qk_norm_reaches_every_block(
    model_class: type, sizes: Dict[str, int]
) -> None:
    """Unset, no block normalises; set, each one does."""
    plain = _rope(model_class, **sizes)
    assert all(block.q_norm is None for block in plain.blocks)

    normed = _rope(model_class, qk_norm=True, **sizes)
    assert all(
        block.q_norm is not None and block.k_norm is not None
        for block in normed.blocks
    )
    with torch.no_grad():
        out = normed(_batch())
    assert out.shape == (3, sizes["hidden_dim"])
    assert torch.isfinite(out).all()
