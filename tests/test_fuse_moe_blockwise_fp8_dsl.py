# Copyright (C) 2026 Tencent.

"""Qwen-style 128x128 float32-block FP8 fused MoE on the SM120 CuTe DSL path."""

import os
import sys
from pathlib import Path

_build = list(Path(__file__).parent.glob("../build/lib.*/"))
if _build:
    sys.path.insert(0, os.path.realpath(_build[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

import hpc
from test_fuse_moe_blockwise import naive_fuse_moe_blockwise_fp8
from test_fuse_moe_w4a8_mxfp4_mxfp8_dsl import _is_sm120
from utils import allclose


def _dsl_available():
    try:
        from hpc import dsl_available
    except ImportError:
        return False
    return dsl_available()[0]


pytestmark = pytest.mark.skipif(
    not _is_sm120() or not _dsl_available(),
    reason="the CuTe DSL FP8 MoE requires an SM120 GPU and nvidia-cutlass-dsl",
)


def _make_case(num_tokens, num_topk, num_experts, hidden_size, intermediate_size, seed):
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.float8_e4m3fn
    topk_ids = torch.multinomial(
        torch.ones((num_tokens, num_experts), dtype=torch.float, device=device),
        num_topk,
        replacement=False,
    ).to(torch.int32)
    topk_ids, _ = torch.sort(topk_ids, dim=1)
    topk_scale = torch.rand((num_tokens, num_topk), dtype=torch.float, device=device)
    topk_scale = topk_scale / topk_scale.sum(dim=1, keepdim=True)
    x = (torch.randn((num_tokens, hidden_size), dtype=torch.float, device=device) / 100).to(dtype)
    x_scale = torch.randn((num_tokens, hidden_size // 128), dtype=torch.float, device=device).abs()
    x_scale = x_scale.clamp(min=1e-4)
    gate_up_weight = (
        torch.randn(
            (num_experts, intermediate_size * 2, hidden_size),
            dtype=torch.float,
            device=device,
        )
        / 100
    ).to(dtype)
    gate_up_weight_scale = (
        torch.randn(
            (num_experts, intermediate_size * 2 // 128, hidden_size // 128),
            dtype=torch.float,
            device=device,
        )
        .abs()
        .clamp(min=1e-4)
    )
    down_weight = (
        torch.randn(
            (num_experts, hidden_size, intermediate_size),
            dtype=torch.float,
            device=device,
        )
        / 100
    ).to(dtype)
    down_weight_scale = (
        torch.randn(
            (num_experts, hidden_size // 128, intermediate_size // 128),
            dtype=torch.float,
            device=device,
        )
        .abs()
        .clamp(min=1e-4)
    )
    return {
        "x": x,
        "x_scale": x_scale,
        "gate_up_weight": gate_up_weight,
        "gate_up_weight_scale": gate_up_weight_scale,
        "down_weight": down_weight,
        "down_weight_scale": down_weight_scale,
        "topk_ids": topk_ids,
        "topk_scale": topk_scale,
    }


@pytest.mark.parametrize("num_tokens", [8, 32])
@pytest.mark.parametrize("hidden_size,intermediate_size", [(256, 128), (256, 256)])
@pytest.mark.parametrize("has_shared_output", [False, True])
def test_fuse_moe_blockwise_fp8_dsl(num_tokens, hidden_size, intermediate_size, has_shared_output):
    num_topk = 2
    num_experts = 8
    case = _make_case(num_tokens, num_topk, num_experts, hidden_size, intermediate_size, seed=7)
    shared_output = None
    if has_shared_output:
        shared_output = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")

    got = hpc.fuse_moe_blockwise_fp8_dsl(
        case["x"],
        case["x_scale"],
        case["gate_up_weight"],
        case["gate_up_weight_scale"],
        case["down_weight"],
        case["down_weight_scale"],
        case["topk_ids"],
        case["topk_scale"],
        0,
        num_experts,
        shared_output,
        mx_mma=False,
    )
    ref = naive_fuse_moe_blockwise_fp8(
        case["x"],
        case["x_scale"],
        case["gate_up_weight"],
        case["gate_up_weight_scale"],
        case["down_weight"],
        case["down_weight_scale"],
        case["topk_ids"],
        case["topk_scale"],
        0,
        num_experts,
        shared_output,
    )
    torch.cuda.synchronize()
    assert allclose(ref.to(torch.float32), got.to(torch.float32), rtol=0.05, atol=0.05)


def test_qwen_like_shapes_fp8_dsl():
    """Qwen3.8-Flash-Next TEP4: hidden=2560, intermediate=640."""
    case = _make_case(16, 4, 8, 2560, 640, seed=11)
    got = hpc.fuse_moe_blockwise_fp8_dsl(
        case["x"],
        case["x_scale"],
        case["gate_up_weight"],
        case["gate_up_weight_scale"],
        case["down_weight"],
        case["down_weight_scale"],
        case["topk_ids"],
        case["topk_scale"],
        0,
        8,
        mx_mma=False,
    )
    ref = naive_fuse_moe_blockwise_fp8(
        case["x"],
        case["x_scale"],
        case["gate_up_weight"],
        case["gate_up_weight_scale"],
        case["down_weight"],
        case["down_weight_scale"],
        case["topk_ids"],
        case["topk_scale"],
        0,
        8,
    )
    torch.cuda.synchronize()
    assert allclose(ref.to(torch.float32), got.to(torch.float32), rtol=0.05, atol=0.05)


def test_fp8_dsl_tall_tile():
    """Enough rows per expert that routing picks the 64-row GEMM tile."""
    case = _make_case(32, 4, 4, 256, 128, seed=13)
    got = hpc.fuse_moe_blockwise_fp8_dsl(
        case["x"],
        case["x_scale"],
        case["gate_up_weight"],
        case["gate_up_weight_scale"],
        case["down_weight"],
        case["down_weight_scale"],
        case["topk_ids"],
        case["topk_scale"],
        0,
        4,
        mx_mma=False,
    )
    ref = naive_fuse_moe_blockwise_fp8(
        case["x"],
        case["x_scale"],
        case["gate_up_weight"],
        case["gate_up_weight_scale"],
        case["down_weight"],
        case["down_weight_scale"],
        case["topk_ids"],
        case["topk_scale"],
        0,
        4,
    )
    torch.cuda.synchronize()
    assert allclose(ref.to(torch.float32), got.to(torch.float32), rtol=0.05, atol=0.05)


def _make_mx_case(num_tokens, num_topk, num_experts, hidden_size, intermediate_size, seed):
    """A case the serve path's contract actually admits.

    ``_make_case`` stores randn/100 as E4M3, which underflows bfloat16 after
    two GEMMs, so the 128-block factors are scaled up to bring the dequantized
    values back into range. The activation scales are then rounded to powers of
    two, which is what the MX path requires: it reads them as UE8M0 without
    rescaling the E4M3 data, so a scale the data was not divided by would
    multiply that block by up to 2x.
    """
    case = _make_case(num_tokens, num_topk, num_experts, hidden_size, intermediate_size, seed)
    case["x_scale"] = torch.exp2(torch.ceil(torch.log2(case["x_scale"] * 50)))
    case["gate_up_weight_scale"] = case["gate_up_weight_scale"] * 50
    case["down_weight_scale"] = case["down_weight_scale"] * 50
    return case


def _mx_relative_error(case, num_experts):
    """Relative L2 of the MX path against the exact 128-block reference.

    The reference runs first: ``mx_mma=True`` requantizes the weights in place,
    and comparing against a reference computed from the requantized weights
    would hide however much that step costs.
    """
    ref = naive_fuse_moe_blockwise_fp8(
        case["x"],
        case["x_scale"],
        case["gate_up_weight"],
        case["gate_up_weight_scale"],
        case["down_weight"],
        case["down_weight_scale"],
        case["topk_ids"],
        case["topk_scale"],
        0,
        num_experts,
    ).float()
    got = hpc.fuse_moe_blockwise_fp8_dsl(
        case["x"],
        case["x_scale"],
        case["gate_up_weight"],
        case["gate_up_weight_scale"],
        case["down_weight"],
        case["down_weight_scale"],
        case["topk_ids"],
        case["topk_scale"],
        0,
        num_experts,
        mx_mma=True,
    ).float()
    torch.cuda.synchronize()
    assert torch.isfinite(got).all()
    assert got.abs().mean() > 0
    return ((got - ref).norm() / ref.norm()).item()


# The MX recipe -- UE8M0 weight requant and 32-column intermediate quantization
# -- is not bit-identical to the exact 128-block float32 path, but it is the
# same arithmetic to within FP8 rounding, and the error it carries is the same
# order as the quantization itself. Measured at 0.06 on Qwen TEP4 shapes. A
# threshold loose enough to pass an activation scale the kernel cannot honour
# (0.83 there) would not be testing anything.
MX_MAX_RELATIVE_ERROR = 0.15


def test_fp8_dsl_mx_mma():
    """Serve path: UE8M0 MMA, same kernel as DeepSeek V4 fused GEMM."""
    case = _make_mx_case(8, 2, 4, 256, 128, seed=17)
    assert _mx_relative_error(case, 4) < MX_MAX_RELATIVE_ERROR


def test_fp8_dsl_mx_mma_prefill_tile_m():
    """Prefill: enough rows/expert that routing picks the 64-row GEMM tile."""
    # 64 tokens × 4 topk / 4 experts = 64 rows/expert → tile_m=64.
    case = _make_mx_case(64, 4, 4, 256, 128, seed=19)
    assert _mx_relative_error(case, 4) < MX_MAX_RELATIVE_ERROR


def test_fp8_dsl_mx_mma_qwen_shapes():
    """The serve path at the layer shape it was written for."""
    case = _make_mx_case(64, 4, 8, 2560, 640, seed=23)
    assert _mx_relative_error(case, 8) < MX_MAX_RELATIVE_ERROR


def test_fp8_dsl_mx_mma_prefill_tile_pair():
    """Prefill: enough rows/expert for the 128-wide slab and K=64 on both GEMMs."""
    # 1024 tokens × 4 topk / 8 experts = 512 rows/expert, past PREFILL_FUSED_ROWS.
    case = _make_mx_case(1024, 4, 8, 2560, 640, seed=47)
    assert _mx_relative_error(case, 8) < MX_MAX_RELATIVE_ERROR


def test_fp8_dsl_mx_mma_weights_prepared_at_load_time():
    """What a server does: pack once after loading, pass packed scales every call.

    The operator refuses float32 scales precisely so that this is the only
    shape a captured graph can be replaying.
    """
    from hpc import prepare_fp8_mx_mma_weights

    case = _make_mx_case(8, 2, 4, 256, 128, seed=31)
    ref = naive_fuse_moe_blockwise_fp8(
        case["x"],
        case["x_scale"],
        case["gate_up_weight"],
        case["gate_up_weight_scale"],
        case["down_weight"],
        case["down_weight_scale"],
        case["topk_ids"],
        case["topk_scale"],
        0,
        4,
    ).float()

    _, packed_13 = prepare_fp8_mx_mma_weights(case["gate_up_weight"], case["gate_up_weight_scale"])
    _, packed_2 = prepare_fp8_mx_mma_weights(case["down_weight"], case["down_weight_scale"])
    assert packed_13.dtype == torch.uint8 and packed_13.dim() == 2
    assert packed_2.dtype == torch.uint8 and packed_2.dim() == 2

    got = hpc.fuse_moe_blockwise_fp8_dsl(
        case["x"],
        case["x_scale"],
        case["gate_up_weight"],
        packed_13,
        case["down_weight"],
        packed_2,
        case["topk_ids"],
        case["topk_scale"],
        0,
        4,
        mx_mma=True,
    ).float()
    torch.cuda.synchronize()
    assert ((got - ref).norm() / ref.norm()).item() < MX_MAX_RELATIVE_ERROR


def test_prepare_fp8_mx_mma_weights_rejects_non_contiguous_scale():
    """Requant writes through the scale, so a copy would strand the caller's."""
    from hpc import prepare_fp8_mx_mma_weights

    case = _make_mx_case(8, 2, 4, 256, 128, seed=37)
    scale = case["gate_up_weight_scale"]
    # Same shape and values, laid out so that writing through it would land in
    # a copy rather than in the caller's tensor.
    strided = scale.transpose(1, 2).contiguous().transpose(1, 2)
    assert strided.shape == scale.shape and not strided.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        prepare_fp8_mx_mma_weights(case["gate_up_weight"], strided)


def test_fp8_dsl_mx_mma_rejects_non_ue8m0_activation_scale():
    """Activation scales the kernel cannot honour have to be an error.

    Nothing about the output says they were wrong -- it stays finite and the
    right shape while carrying an order of magnitude more error -- and what
    produces them is a quantizer several layers away.
    """
    from hpc.cute.fuse_moe import _pipeline

    case = _make_mx_case(8, 2, 4, 256, 128, seed=29)
    # 3 * 2^k is never a power of two, whatever k the original scale had.
    case["x_scale"] = case["x_scale"] * 3.0
    _pipeline._ue8m0_checked.clear()
    try:
        with pytest.raises(ValueError, match="powers of two"):
            hpc.fuse_moe_blockwise_fp8_dsl(
                case["x"],
                case["x_scale"],
                case["gate_up_weight"],
                case["gate_up_weight_scale"],
                case["down_weight"],
                case["down_weight_scale"],
                case["topk_ids"],
                case["topk_scale"],
                0,
                4,
                mx_mma=True,
            )
    finally:
        _pipeline._ue8m0_checked.clear()


def test_fp8_dsl_matches_under_torch_compile():
    """The traced path has to agree with the eager one.

    Eager calls skip ``torch.library.custom_op`` -- the boundary costs ~88us a
    call, which is milliseconds across 48 layers -- and go straight to the
    pipeline. A tracer still has to see the operator, so the two branches are
    different code and both have to give the same answer.
    """
    from hpc import prepare_fp8_mx_mma_weights

    case = _make_mx_case(8, 2, 4, 256, 128, seed=37)
    # What a server hands in: packed once at load time, never float32 per call.
    _, gate_up_packed = prepare_fp8_mx_mma_weights(
        case["gate_up_weight"], case["gate_up_weight_scale"]
    )
    _, down_packed = prepare_fp8_mx_mma_weights(case["down_weight"], case["down_weight_scale"])

    def moe(x, x_scale):
        return hpc.fuse_moe_blockwise_fp8_dsl(
            x,
            x_scale,
            case["gate_up_weight"],
            gate_up_packed,
            case["down_weight"],
            down_packed,
            case["topk_ids"],
            case["topk_scale"],
            0,
            4,
            mx_mma=True,
        )

    eager = moe(case["x"], case["x_scale"])
    compiled = torch.compile(moe, fullgraph=True, dynamic=False)(case["x"], case["x_scale"])
    torch.cuda.synchronize()
    assert torch.equal(eager, compiled)


def test_fp8_dsl_unprepared_weights_under_compile_say_so():
    """Tracing an unprepared call names the fix instead of breaking the graph.

    The lazy requantization is an in-place mutation, so it cannot go into a
    traced graph at all. The capture check that would otherwise catch it is
    itself untraceable, so the tracer has to be turned away before it.
    """
    case = _make_mx_case(8, 2, 4, 256, 128, seed=43)

    def moe(x, x_scale):
        return hpc.fuse_moe_blockwise_fp8_dsl(
            x,
            x_scale,
            case["gate_up_weight"],
            case["gate_up_weight_scale"],
            case["down_weight"],
            case["down_weight_scale"],
            case["topk_ids"],
            case["topk_scale"],
            0,
            4,
            mx_mma=True,
        )

    with pytest.raises(RuntimeError, match="prepare_fp8_mx_mma_weights at load time"):
        torch.compile(moe, fullgraph=True, dynamic=False)(case["x"], case["x_scale"])


def test_fp8_dsl_survives_moving_buffers():
    """Pointer objects are memoized on the address, so addresses must be re-read.

    A cached pointer for a freed address is still just that address, but a call
    whose tensors moved has to pick up the new ones. Interleave two sets of
    activations and force reallocation between calls.
    """
    from hpc import prepare_fp8_mx_mma_weights

    case = _make_mx_case(8, 2, 4, 256, 128, seed=41)
    prepare_fp8_mx_mma_weights(case["gate_up_weight"], case["gate_up_weight_scale"])
    prepare_fp8_mx_mma_weights(case["down_weight"], case["down_weight_scale"])

    def moe(x, x_scale):
        return hpc.fuse_moe_blockwise_fp8_dsl(
            x,
            x_scale,
            case["gate_up_weight"],
            case["gate_up_weight_scale"],
            case["down_weight"],
            case["down_weight_scale"],
            case["topk_ids"],
            case["topk_scale"],
            0,
            4,
            mx_mma=True,
        )

    want = moe(case["x"], case["x_scale"])
    other_x = case["x"].clone()
    other_scale = case["x_scale"].clone()
    for _ in range(4):
        # A different address for the same values, then back again, with the
        # allocator churning in between.
        assert torch.equal(moe(other_x, other_scale), want)
        torch.cuda.empty_cache()
        scratch = torch.empty(1 << 22, dtype=torch.uint8, device="cuda")
        del scratch
        assert torch.equal(moe(case["x"], case["x_scale"]), want)
