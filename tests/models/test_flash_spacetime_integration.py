"""Model-level parity: DeepIce with the fused kernel vs the eager path.

Two weight-identical DeepIce instances — one with `use_flash_spacetime`
— must produce matching per-event cls outputs and parameter gradients.
Padding rows inside the sandwich differ by design (the kernel zeroes
them, eager leaves finite garbage), and downstream masking provably
keeps them out of every valid output, which this test confirms at the
whole-model level. Comparisons allow a few output-scale ulps per dtype
(separately compiled pipelines; the fp32 bias path carries the
documented trig-argument floor).
"""

from typing import List, Tuple

import pytest
import torch
from torch_geometric.data import Data

from graphnet.models.transformer.icemix import DeepIce

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + Triton"
)


def _make_batch(lengths: List[int], seed: int) -> Data:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    total = sum(lengths)
    x = torch.randn(total, 5, generator=gen, dtype=torch.float32)
    batch_idx = torch.repeat_interleave(
        torch.arange(len(lengths)), torch.tensor(lengths)
    )
    data = Data(x=x.cuda(), batch=batch_idx.cuda())
    data.num_graphs = len(lengths)
    return data


def _make_models(seed: int) -> Tuple[DeepIce, DeepIce]:
    torch.manual_seed(seed)
    eager = DeepIce(
        hidden_dim=96,
        seq_length=64,
        depth=2,
        head_size=48,
        depth_rel=2,
        n_rel=1,
        n_features=5,
    ).cuda()
    flash = DeepIce(
        hidden_dim=96,
        seq_length=64,
        depth=2,
        head_size=48,
        depth_rel=2,
        n_rel=1,
        n_features=5,
        use_flash_spacetime=True,
    ).cuda()
    flash.load_state_dict(eager.state_dict())
    return eager, flash


@pytest.mark.parametrize("lengths", [[13], [7, 40, 1], [64, 17, 33, 2]])
def test_deepice_forward_parity(lengths: List[int]) -> None:
    """Per-event outputs match between eager and flash sandwiches (fp32)."""
    eager, flash = _make_models(3)
    data = _make_batch(lengths, 11)
    with torch.no_grad():
        out_e = eager(data)
        out_f = flash(data)
    assert torch.isfinite(out_f).all()
    torch.testing.assert_close(out_f, out_e, atol=5e-3, rtol=1e-3)


@pytest.mark.parametrize("lengths", [[7, 40, 1], [64, 17, 33, 2]])
def test_deepice_backward_parity(lengths: List[int]) -> None:
    """Parameter gradients match, including SpacetimeEncoder projection."""
    eager, flash = _make_models(5)
    data = _make_batch(lengths, 13)
    torch.manual_seed(99)
    weights = torch.randn(len(lengths), 96, device="cuda")

    grads = {}
    for name, model in (("eager", eager), ("flash", flash)):
        model.zero_grad(set_to_none=True)
        out = model(data)
        (out * weights).sum().backward()
        grads[name] = {
            n: p.grad.clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }

    assert set(grads["eager"]) == set(grads["flash"])
    assert any("rel_pos.projection" in n for n in grads["flash"])
    for name in grads["eager"]:
        torch.testing.assert_close(
            grads["flash"][name],
            grads["eager"][name],
            atol=5e-2,
            rtol=5e-3,
            msg=lambda m, name=name: f"grad {name}: {m}",
        )


def test_deepice_bf16_autocast_parity() -> None:
    """The production training configuration: bf16-mixed autocast."""
    eager, flash = _make_models(7)
    data = _make_batch([40, 9, 63], 17)
    with torch.autocast("cuda", torch.bfloat16):
        out_e = eager(data)
        out_f = flash(data)
    assert torch.isfinite(out_f).all()
    torch.testing.assert_close(
        out_f.float(), out_e.float(), atol=5e-2, rtol=2e-2
    )
