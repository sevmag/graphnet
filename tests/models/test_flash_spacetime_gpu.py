"""GPU parity tests: Triton kernel vs the proven fp64 reference oracle.

Stage (b) of the verification chain (stage (a), reference == real modules,
lives in test_flash_spacetime.py). Equality standard on valid rows,
elementwise:

    |kernel[dtype] - ref[fp64]| <= 2 * |ref[dtype] - ref[fp64]| + atol[dtype]

i.e. the kernel is never meaningfully further from the exact answer than
the eager path itself is (the reference IS the eager path, proven in stage
(a)). Padding rows are asserted exactly zero (kernel contract; eager leaves
garbage there that nothing downstream reads).

The kernel currently supports C = D = 48 (the production head size) with
three unrolled 16-channel chunks; the C-width generalisation lands with the
autotune pass. Backward routes through the reference fallback until the
dedicated Triton backward replaces it — these tests pin the public op's
gradients either way.
"""

from typing import Dict, Optional, Tuple

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
        flash_spacetime_forward,
    )

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA + Triton"
)

# fp32 atol depends on whether E is in play: the sinusoid arguments reach
# |theta| <= 1024 * 4 rad, where a single-ulp argument difference (ulp(4096)
# ~ 4.9e-4) moves sin by the same amount, and the projection amplifies the
# per-channel spread to ~3e-3. The kernel's FMA-contracted interval
# arithmetic and torch's step-rounded pipeline are both correct fp32
# evaluations that differ at that scale, so 1e-6 is unachievable for any
# E-dependent output; the no-bias configuration keeps the strict floor.
ATOL = {torch.float32: 1e-6, torch.bfloat16: 1e-2}
ATOL_FP32_WITH_E = 5e-3
# Machine epsilon per comparison dtype for the scale-aware elementwise
# floor (bf16 has an 8-bit mantissa: rounding sites legitimately differ
# between two correct pipelines by a few output-scale ulps).
ULP = {torch.bfloat16: 2.0**-8, torch.float32: 2.0**-23}


def _atol(dtype: torch.dtype, flags: Tuple[bool, bool]) -> float:
    if dtype is torch.float32 and (flags[0] or flags[1]):
        return ATOL_FP32_WITH_E
    return ATOL[dtype]


# (batch, length, heads, head_dim): straddles BLOCK_M=16 / BLOCK_N=32
# boundaries on both sides, plus the production shape.
SHAPES = [
    (2, 1, 4, 48),
    (2, 3, 4, 48),
    (2, 17, 4, 48),
    (3, 32, 16, 48),
    (2, 33, 16, 48),
    (1, 64, 16, 48),
    (2, 65, 1, 48),
    (2, 127, 16, 48),
    (2, 128, 4, 48),
    (2, 200, 16, 48),
    (1, 800, 16, 48),
]
FLAGS = [(True, True), (True, False), (False, True), (False, False)]


def _case(
    shape: Tuple[int, int, int, int],
    masked: bool,
    seed: int,
) -> Dict[str, torch.Tensor]:
    batch, length, heads, dim = shape
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*size: int) -> torch.Tensor:
        return torch.randn(*size, generator=gen, dtype=torch.float64)

    case = {
        "q": rand(batch, heads, length, dim),
        "k": rand(batch, heads, length, dim),
        "v": rand(batch, heads, length, dim),
        "feats": rand(batch, length, 5),
        "weight": rand(dim, dim) / dim**0.5,
        "bias": rand(dim),
    }
    if masked and length > 1:
        seqlens = torch.randint(1, length + 1, (batch,), generator=gen)
        seqlens[0] = length
        if batch > 1:
            seqlens[-1] = 1
    else:
        seqlens = torch.full((batch,), length, dtype=torch.long)
    case["seqlens"] = seqlens
    return {k: t.cuda() for k, t in case.items()}


def _reference(
    case: Dict[str, torch.Tensor],
    flags: Tuple[bool, bool],
    dtype: torch.dtype,
) -> torch.Tensor:
    valid = valid_row_mask(
        case["q"].shape[0], case["q"].shape[2], case["seqlens"]
    )
    mask = float_padding_mask(valid, torch.float32)
    return spacetime_attention_reference(
        case["q"].to(dtype),
        case["k"].to(dtype),
        case["v"].to(dtype),
        case["feats"].to(torch.float32 if dtype != torch.float64 else dtype),
        case["weight"].to(dtype),
        case["bias"].to(dtype),
        key_padding_mask=mask,
        use_attn_bias=flags[0],
        use_activation_bias=flags[1],
    )


def _assert_2x_rule(
    kernel_out: torch.Tensor,
    dtype_ref: torch.Tensor,
    fp64_ref: torch.Tensor,
    valid: torch.Tensor,
    atol: float,
    what: str,
    reduction: bool = False,
) -> None:
    kernel_err = (kernel_out.double() - fp64_ref)[valid].abs()
    eager_err = (dtype_ref.double() - fp64_ref)[valid].abs()
    # The elementwise floor is scale-aware for low-precision dtypes: the
    # kernel rounds at different sites than eager (it keeps S and P in
    # fp32 and rounds once), so at any single element it may sit a few
    # output-scale ulps away while being at least as accurate overall —
    # which the aggregate assertion below enforces.
    ulp = ULP.get(kernel_out.dtype, 0.0)
    floor = atol + 4.0 * ulp * fp64_ref[valid].abs().clamp(min=1.0)
    ref_err = eager_err
    if reduction:
        # Global-reduction outputs (dW, db sum B*H*L^2 terms): kernel and
        # eager error patterns are independent random walks, so an element
        # where eager landed lucky must not impose a bound below eager's
        # typical error. The aggregate mean assertion below still forbids
        # the kernel being worse overall.
        ref_err = torch.maximum(eager_err, eager_err.median())
    bound = 2.0 * ref_err + floor
    bad = kernel_err > bound
    assert not bad.any(), (
        f"{what}: {int(bad.sum())} elements exceed the 2x-eager bound; "
        f"worst kernel err {kernel_err.max():.3e} vs bound "
        f"{bound[kernel_err.argmax()]:.3e}"
    )
    assert kernel_err.mean() <= 1.5 * eager_err.mean() + atol, (
        f"{what}: kernel mean error {kernel_err.mean():.3e} exceeds "
        f"1.5x eager mean {eager_err.mean():.3e}"
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_matches_oracle(
    shape: Tuple[int, int, int, int],
    flags: Tuple[bool, bool],
    masked: bool,
    dtype: torch.dtype,
) -> None:
    """Kernel forward obeys the 2x-eager rule against the fp64 oracle."""
    case = _case(shape, masked, seed=abs(hash((shape, flags, masked))) % 2**31)
    valid = valid_row_mask(shape[0], shape[1], case["seqlens"])

    out, _ = flash_spacetime_forward(
        case["q"].to(dtype),
        case["k"].to(dtype),
        case["v"].to(dtype),
        case["feats"],
        case["weight"].to(dtype),
        case["bias"].to(dtype),
        case["seqlens"],
        use_attn_bias=flags[0],
        use_activation_bias=flags[1],
    )

    assert torch.isfinite(out).all()
    pad = ~valid
    if pad.any():
        assert (
            out.permute(0, 2, 1, 3)[pad] == 0
        ).all(), "padding rows must be exactly zero"

    fp64 = _reference(case, flags, torch.float64)
    in_dtype = _reference(case, flags, dtype)
    vmask = valid[:, None, :, None].expand_as(out)
    _assert_2x_rule(out, in_dtype, fp64, vmask, _atol(dtype, flags), "forward")


@pytest.mark.parametrize("shape", [SHAPES[2], SHAPES[4], SHAPES[9]])
@pytest.mark.parametrize("flags", FLAGS[:2])
def test_forward_deterministic(
    shape: Tuple[int, int, int, int], flags: Tuple[bool, bool]
) -> None:
    """Two identical calls produce bitwise-identical output and LSE."""
    case = _case(shape, True, seed=1234)
    args = (
        case["q"].float(),
        case["k"].float(),
        case["v"].float(),
        case["feats"],
        case["weight"].float(),
        case["bias"].float(),
        case["seqlens"],
    )
    out1, lse1 = flash_spacetime_forward(
        *args, use_attn_bias=flags[0], use_activation_bias=flags[1]
    )
    out2, lse2 = flash_spacetime_forward(
        *args, use_attn_bias=flags[0], use_activation_bias=flags[1]
    )
    assert torch.equal(out1, out2)
    assert torch.equal(lse1, lse2)


@pytest.mark.parametrize("shape", [SHAPES[2], SHAPES[5], SHAPES[9]])
@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("masked", [False, True])
def test_backward_matches_oracle(
    shape: Tuple[int, int, int, int],
    flags: Tuple[bool, bool],
    masked: bool,
) -> None:
    """Public-op gradients obey the 2x-eager rule (fp32)."""
    dtype = torch.float32
    case = _case(shape, masked, seed=abs(hash(("bwd", shape, flags))) % 2**31)
    valid = valid_row_mask(shape[0], shape[1], case["seqlens"])
    gen = torch.Generator(device="cpu").manual_seed(7)
    grad_w = torch.randn(
        *case["q"].permute(0, 2, 1, 3).shape,
        generator=gen,
        dtype=torch.float64,
    ).cuda()

    grads: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
    for path in ("kernel", "ref32", "ref64"):
        cdtype = torch.float64 if path == "ref64" else dtype
        q = case["q"].to(cdtype).requires_grad_(True)
        k = case["k"].to(cdtype).requires_grad_(True)
        v = case["v"].to(cdtype).requires_grad_(True)
        w = case["weight"].to(cdtype).requires_grad_(True)
        b = case["bias"].to(cdtype).requires_grad_(True)
        if path == "kernel":
            out = flash_spacetime_attention(
                q,
                k,
                v,
                case["feats"],
                w,
                b,
                case["seqlens"],
                use_attn_bias=flags[0],
                use_activation_bias=flags[1],
            ).permute(0, 2, 1, 3)
        else:
            mask = float_padding_mask(valid, torch.float32)
            out = spacetime_attention_reference(
                q,
                k,
                v,
                case["feats"].to(
                    torch.float32 if cdtype != torch.float64 else cdtype
                ),
                w,
                b,
                key_padding_mask=mask,
                use_attn_bias=flags[0],
                use_activation_bias=flags[1],
            ).permute(0, 2, 1, 3)
        loss = (out * grad_w.to(out.dtype) * valid[:, :, None, None]).sum()
        loss.backward()
        grads[path] = {
            "q": q.grad,
            "k": k.grad,
            "v": v.grad,
            "W": w.grad,
            "b": b.grad,
        }

    for key in ("q", "k", "v", "W", "b"):
        kg, r32, r64 = (grads[p][key] for p in ("kernel", "ref32", "ref64"))
        if key in ("W", "b") and not (flags[0] or flags[1]):
            # With both biases off the projection never participates, so
            # no path produces a gradient for it.
            assert kg is None and r32 is None and r64 is None
            continue
        assert kg is not None and r32 is not None and r64 is not None
        full = torch.ones_like(r64, dtype=torch.bool)
        _assert_2x_rule(
            kg,
            r32,
            r64,
            full,
            _atol(dtype, flags),
            f"grad {key}",
            # Every gradient is an L-fold (or B*H*L^2-fold) reduction, so
            # the exchangeable-error-locations argument applies to all of
            # them; the aggregate assertion still forbids being worse
            # overall.
            reduction=True,
        )


def test_feats_requires_grad_raises() -> None:
    """Coordinates are detector data; a grad request must fail loud."""
    case = _case(SHAPES[2], False, seed=5)
    feats = case["feats"].requires_grad_(True)
    with pytest.raises(ValueError, match="feats"):
        flash_spacetime_attention(
            case["q"].float(),
            case["k"].float(),
            case["v"].float(),
            feats,
            case["weight"].float(),
            case["bias"].float(),
            case["seqlens"],
        )
