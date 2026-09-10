"""Varlen (packed, uncapped) tests for fused spacetime-bias attention.

The packed path consumes a jagged NestedTensor's values layout directly
(q/k/v [T, H, D], feats [T, F], cu_seqlens token offsets) — no padding
exists anywhere, so event lengths are unbounded. Verification:

1. packed == padded on the same events (same kernel arithmetic, different
   addressing) — near-bitwise;
2. packed forward/backward against the fp64 oracle under the same bounds
   as the padded suite;
3. bitwise determinism;
4. an uncapped smoke at NuBench-tail lengths (thousands of pulses) with a
   memory ceiling no padded eager path could meet.
"""

from typing import Dict, Tuple

import pytest
import torch

from graphnet.models.components.flash_spacetime import (
    float_padding_mask,
    spacetime_attention_reference,
    valid_row_mask,
)

if torch.cuda.is_available():
    from graphnet.models.components.flash_spacetime_triton import (
        flash_spacetime_attention,
        flash_spacetime_attention_varlen,
    )

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + Triton"
)

LENGTH_MIXES = [
    [7],
    [1, 1, 1],
    [17, 64, 3],
    [200, 33, 1, 128],
    [331, 800, 65],
]
FLAGS = [(True, True), (True, False), (False, True), (False, False)]


def _packed_case(
    lengths: list, heads: int, dim: int, seed: int, dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    total = sum(lengths)
    case = {
        "q": torch.randn(
            total, heads, dim, generator=gen, dtype=torch.float64
        ),
        "k": torch.randn(
            total, heads, dim, generator=gen, dtype=torch.float64
        ),
        "v": torch.randn(
            total, heads, dim, generator=gen, dtype=torch.float64
        ),
        "feats": torch.randn(total, 5, generator=gen, dtype=torch.float64),
        "weight": torch.randn(dim, dim, generator=gen, dtype=torch.float64)
        / dim**0.5,
        "bias": torch.randn(dim, generator=gen, dtype=torch.float64),
    }
    case = {k: t.cuda() for k, t in case.items()}
    case["cu"] = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0)), device="cuda"
    )
    case["lengths"] = lengths
    case["dtype"] = dtype
    return case


def _pad(case: Dict, key: str) -> torch.Tensor:
    """Packed [T, ...] -> padded [B, H, L, D] (or [B, L, F] for feats)."""
    lengths = case["lengths"]
    length = max(lengths)
    chunks = torch.split(case[key], lengths)
    if key == "feats":
        out = case[key].new_zeros(len(lengths), length, case[key].shape[-1])
        for i, c in enumerate(chunks):
            out[i, : c.shape[0]] = c
        return out
    out = case[key].new_zeros(
        len(lengths), length, case[key].shape[1], case[key].shape[2]
    )
    for i, c in enumerate(chunks):
        out[i, : c.shape[0]] = c
    return out.permute(0, 2, 1, 3).contiguous()


def _unpad(padded: torch.Tensor, lengths: list) -> torch.Tensor:
    """[B, H, L, D] -> packed [T, H, D] on valid rows."""
    rows = [
        padded[i, :, :n_rows].permute(1, 0, 2)
        for i, n_rows in enumerate(lengths)
    ]
    return torch.cat(rows, 0)


@pytest.mark.parametrize("lengths", LENGTH_MIXES)
@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_varlen_matches_padded(
    lengths: list, flags: Tuple[bool, bool], dtype: torch.dtype
) -> None:
    """Packed and padded paths agree on identical events (near-bitwise)."""
    case = _packed_case(
        lengths, 4, 48, abs(hash((tuple(lengths), flags))) % 2**31, dtype
    )
    seqlens = torch.tensor(lengths, device="cuda")

    out_v = flash_spacetime_attention_varlen(
        case["q"].to(dtype),
        case["k"].to(dtype),
        case["v"].to(dtype),
        case["feats"],
        case["weight"].to(dtype),
        case["bias"].to(dtype),
        case["cu"],
        use_attn_bias=flags[0],
        use_activation_bias=flags[1],
    )
    out_p = flash_spacetime_attention(
        _pad(case, "q").to(dtype),
        _pad(case, "k").to(dtype),
        _pad(case, "v").to(dtype),
        _pad(case, "feats"),
        case["weight"].to(dtype),
        case["bias"].to(dtype),
        seqlens,
        use_attn_bias=flags[0],
        use_activation_bias=flags[1],
    )
    # The two layouts run identical tile arithmetic, but the compiler
    # schedules each specialization independently, so low-precision
    # outputs may differ by a few output-scale ulps.
    if dtype is torch.bfloat16:
        atol, rtol = 2e-2, 1e-2
    else:
        atol, rtol = 1e-5, 1e-4
    torch.testing.assert_close(
        out_v, _unpad(out_p, lengths), atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("lengths", LENGTH_MIXES[2:])
@pytest.mark.parametrize("flags", FLAGS[:2])
def test_varlen_matches_oracle(
    lengths: list, flags: Tuple[bool, bool]
) -> None:
    """Packed fwd+bwd against fp64 reference on the equivalent padded case."""
    dtype = torch.float32
    case = _packed_case(
        lengths, 4, 48, abs(hash(("o", tuple(lengths), flags))) % 2**31, dtype
    )
    valid = valid_row_mask(
        len(lengths),
        max(lengths),
        torch.tensor(lengths, device="cuda"),
    )
    mask = float_padding_mask(valid, torch.float32)

    grads = {}
    outs = {}
    for path in ("kernel", "ref64"):
        cdtype = torch.float64 if path == "ref64" else dtype
        q = case["q"].to(cdtype).requires_grad_(True)
        k = case["k"].to(cdtype).requires_grad_(True)
        v = case["v"].to(cdtype).requires_grad_(True)
        w = case["weight"].to(cdtype).requires_grad_(True)
        b = case["bias"].to(cdtype).requires_grad_(True)
        if path == "kernel":
            out = flash_spacetime_attention_varlen(
                q,
                k,
                v,
                case["feats"],
                w,
                b,
                case["cu"],
                use_attn_bias=flags[0],
                use_activation_bias=flags[1],
            )
        else:
            qp = _pad({**case, "q": q}, "q")
            kp = _pad({**case, "k": k}, "k")
            vp = _pad({**case, "v": v}, "v")
            outp = spacetime_attention_reference(
                qp,
                kp,
                vp,
                _pad(case, "feats").to(torch.float32),
                w,
                b,
                key_padding_mask=mask,
                use_attn_bias=flags[0],
                use_activation_bias=flags[1],
            )
            out = _unpad(outp, case["lengths"])
        outs[path] = out.detach().double()
        (out.double().pow(2).sum() * 0.5).backward()
        grads[path] = {
            "q": q.grad,
            "k": k.grad,
            "v": v.grad,
            "W": w.grad,
            "b": b.grad,
        }

    torch.testing.assert_close(
        outs["kernel"], outs["ref64"], atol=5e-3, rtol=1e-3
    )
    for key in ("q", "k", "v", "W", "b"):
        kg, rg = grads["kernel"][key], grads["ref64"][key]
        assert kg is not None and rg is not None
        torch.testing.assert_close(
            kg.double(),
            rg.double(),
            atol=5e-2,
            rtol=5e-3,
            msg=lambda m, key=key: f"grad {key}: {m}",
        )


def test_varlen_deterministic() -> None:
    """Two identical packed fwd+bwd runs are bitwise identical."""
    case = _packed_case([64, 17, 200], 16, 48, 99, torch.float32)
    results = []
    for _ in range(2):
        q = case["q"].float().requires_grad_(True)
        w = case["weight"].float().requires_grad_(True)
        out = flash_spacetime_attention_varlen(
            q,
            case["k"].float(),
            case["v"].float(),
            case["feats"],
            w,
            case["bias"].float(),
            case["cu"],
        )
        out.sum().backward()
        results.append((out.detach().clone(), q.grad.clone(), w.grad.clone()))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    assert torch.equal(results[0][2], results[1][2])


def test_uncapped_lengths_smoke() -> None:
    """Events far beyond any padded cap run within a tight memory budget.

    A 6k-pulse event's eager R alone would need 6000^2 * 48 * 4 bytes =
    6.9 GiB; the packed op must stay under half a GiB of transient
    allocations for the whole fwd+bwd.
    """
    lengths = [6000, 137, 1503]
    case = _packed_case(lengths, 16, 48, 7, torch.bfloat16)
    q = case["q"].bfloat16().requires_grad_(True)
    w = case["weight"].bfloat16().requires_grad_(True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = flash_spacetime_attention_varlen(
        q,
        case["k"].bfloat16(),
        case["v"].bfloat16(),
        case["feats"],
        w,
        case["bias"].bfloat16(),
        case["cu"],
    )
    out.sum().backward()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30

    assert torch.isfinite(out).all()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    assert peak < 0.5, f"packed fwd+bwd peak {peak:.2f} GiB"
