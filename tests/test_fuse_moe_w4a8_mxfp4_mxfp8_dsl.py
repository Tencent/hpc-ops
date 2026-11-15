# Copyright (C) 2026 Tencent.

"""Two-layer validation of the CuTe DSL SM120 W4A8 MXFP4 x MXFP8 fused MoE.

Every case is checked against the pure PyTorch reference (loose tolerance,
because MXFP4/MXFP8 quantization dominates the error).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import pytest
import torch
import torch.nn.functional as F

import hpc
import hpc.cute.fuse_moe
from utils import allclose


def _is_sm120():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _dsl_available():
    try:
        from hpc import dsl_available
    except ImportError:
        return False
    return dsl_available()[0]


pytestmark = pytest.mark.skipif(
    not _is_sm120() or not _dsl_available(),
    reason="the CuTe DSL fused MoE requires an SM120 GPU and nvidia-cutlass-dsl",
)

# The blockwise DSL path sums an expert's contribution into the output with a
# bfloat16 atomic in the second GEMM's epilogue. The top-k terms largely cancel,
# so the rounding is a fraction of the largest term in a row rather than of the
# element it lands on, and the comparison has to be in those units.
SUM_ULPS = 4
BF16_ULP = 2**-8

_E2M1_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _e2m1_levels(device):
    return torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=device)


def _encode_ue8m0_scale(max_abs, max_normal):
    requested = torch.clamp(max_abs / max_normal, min=2.0**-127, max=2.0**127)
    exponent = torch.ceil(torch.log2(requested))
    bits = torch.clamp(exponent + 127, min=0, max=254).to(torch.uint8)
    scale = torch.pow(2.0, bits.float() - 127)
    return bits, scale


def _pack_e2m1(normalized):
    levels = _e2m1_levels(normalized.device)
    magnitude_code = (normalized.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)
    sign_code = (normalized < 0).to(torch.int64) << 3
    code = (magnitude_code | sign_code).to(torch.uint8)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    dequantized = levels[magnitude_code] * torch.where(normalized < 0, -1.0, 1.0)
    return packed.contiguous(), dequantized


def _quantize_mxfp8(x):
    assert x.shape[-1] % 32 == 0
    blocks = x.float().reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    scale_bits, scale = _encode_ue8m0_scale(blocks.abs().amax(dim=-1), 448.0)
    quantized = (blocks / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    dequantized = quantized.float() * scale.unsqueeze(-1)
    return quantized.reshape_as(x), scale_bits, dequantized.reshape_as(x).float()


def _quantize_mxfp4(weight):
    assert weight.shape[-1] % 32 == 0
    blocks = weight.float().reshape(*weight.shape[:-1], weight.shape[-1] // 32, 32)
    scale_bits, scale = _encode_ue8m0_scale(blocks.abs().amax(dim=-1), 6.0)
    packed, normalized_dequantized = _pack_e2m1(blocks / scale.unsqueeze(-1))
    dequantized = normalized_dequantized * scale.unsqueeze(-1)
    packed = packed.reshape(*weight.shape[:-1], weight.shape[-1] // 2)
    return packed.contiguous(), scale_bits.contiguous(), dequantized.reshape_as(weight)


def _swiglu(gate, up, swiglu_limit):
    if swiglu_limit is not None:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate) * up


def _reference_blockwise_moe(
    x,
    gate_up_weight,
    down_weight,
    topk_ids,
    topk_scale,
    shared_output=None,
    swiglu_limit=None,
):
    num_tokens, num_topk = topk_ids.shape
    hidden_size = x.shape[1]
    output = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=x.device)

    for token in range(num_tokens):
        for selected in range(num_topk):
            expert = int(topk_ids[token, selected])
            gate_up = torch.mv(gate_up_weight[expert], x[token])
            gate_up = gate_up.to(torch.bfloat16).float()
            gate, up = gate_up.chunk(2)
            _, _, down_input_dequantized = _quantize_mxfp8(
                _swiglu(gate, up, swiglu_limit).unsqueeze(0)
            )
            down = torch.mv(down_weight[expert], down_input_dequantized[0])
            down = down.to(torch.bfloat16).float()
            output[token] += down * topk_scale[token, selected]

    if shared_output is not None:
        output += shared_output.float()
    return output.to(torch.bfloat16)


def _make_routing(num_tokens, num_topk, num_experts, device):
    topk_ids = torch.stack(
        [
            (torch.arange(num_topk, device=device) + token) % num_experts
            for token in range(num_tokens)
        ]
    ).to(torch.int32)
    topk_scale = torch.rand((num_tokens, num_topk), dtype=torch.float32, device=device)
    topk_scale /= topk_scale.sum(dim=1, keepdim=True)
    return topk_ids, topk_scale


def _make_case(
    num_tokens, num_topk, num_experts, hidden_size, intermediate_size, seed, expert_pool=None
):
    """``expert_pool`` routes every token into the first few experts.

    The tile height is chosen from how many rows an expert gets on average, and
    a uniform routing over ``num_experts`` is close enough to that average that
    no expert is far off it. Drawing from a smaller pool leaves the average --
    and so the choice -- alone while giving the experts that are live many times
    more rows than it.
    """
    torch.manual_seed(seed)
    device = "cuda"
    x, x_scale, x_dequantized = _quantize_mxfp8(
        torch.randn((num_tokens, hidden_size), device=device)
    )
    gate_up_weight, gate_up_scale, gate_up_dequantized = _quantize_mxfp4(
        torch.randn((num_experts, 2 * intermediate_size, hidden_size), device=device)
    )
    down_weight, down_scale, down_dequantized = _quantize_mxfp4(
        torch.randn((num_experts, hidden_size, intermediate_size), device=device)
    )
    topk_ids, topk_scale = _make_routing(num_tokens, num_topk, expert_pool or num_experts, device)
    return {
        "x": x,
        "x_scale": x_scale,
        "x_dequantized": x_dequantized,
        "gate_up_weight": gate_up_weight,
        "gate_up_scale": gate_up_scale,
        "gate_up_dequantized": gate_up_dequantized,
        "down_weight": down_weight,
        "down_scale": down_scale,
        "down_dequantized": down_dequantized,
        "topk_ids": topk_ids,
        "topk_scale": topk_scale,
    }


def _close_with_summed_rounding(reference, actual, rtol, atol):
    """``allclose`` with room for the top-k sum being rounded per contribution.

    The extra term is per row rather than per element, because that is where the
    rounding comes from: the sum's error follows the terms, not the result.
    """
    slack = reference.abs().amax(dim=-1, keepdim=True) * BF16_ULP * SUM_ULPS
    return bool(((actual - reference).abs() <= atol + rtol * reference.abs() + slack).all())


def _run(case, entry, shared_output=None, output=None, swiglu_limit=None):
    return entry(
        case["x"],
        case["x_scale"],
        case["gate_up_weight"],
        case["gate_up_scale"],
        case["down_weight"],
        case["down_scale"],
        case["topk_ids"],
        case["topk_scale"],
        rank_ep=0,
        num_expert_total=case["gate_up_weight"].shape[0],
        shared_output=shared_output,
        output=output,
        swiglu_limit=swiglu_limit,
    )


@pytest.fixture
def fuse_reduce(request, monkeypatch):
    """Pin which way the down projection finishes the top-k sum.

    Left alone it follows the routed rows' size against L2, and every shape
    small enough to test in-process lands on the same side of that, so the
    fused epilogue would never run here.
    """
    monkeypatch.setattr(hpc.cute.fuse_moe._pipeline, "_fuse_reduce", lambda *_: request.param)
    return request.param


@pytest.mark.parametrize("fuse_reduce", [False, True], indirect=True)
@pytest.mark.parametrize(
    "num_tokens,num_topk,num_experts,hidden_size,intermediate_size",
    [
        (5, 2, 3, 128, 128),
        (64, 2, 8, 256, 128),
        (200, 4, 4, 256, 256),
        (1, 1, 2, 128, 128),
    ],
)
def test_blockwise_dsl_matches_reference(
    fuse_reduce, num_tokens, num_topk, num_experts, hidden_size, intermediate_size
):
    case = _make_case(num_tokens, num_topk, num_experts, hidden_size, intermediate_size, seed=2027)

    expected = _reference_blockwise_moe(
        case["x_dequantized"],
        case["gate_up_dequantized"],
        case["down_dequantized"],
        case["topk_ids"],
        case["topk_scale"],
    )
    dsl = _run(case, hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl)

    assert _close_with_summed_rounding(expected.float(), dsl.float(), rtol=0.12, atol=0.25)


@pytest.mark.parametrize("fuse_reduce", [False, True], indirect=True)
def test_blockwise_dsl_expert_spanning_short_tiles(fuse_reduce):
    """An expert whose rows do not fit in the tile height picked for the launch.

    Twelve rows an expert on average buys the short tile; routing all of them
    into eight experts then gives each live one a hundred, or four tiles. That
    pairing is what the shapes above miss, and it is the one where the fused
    epilogue walks more rows in a pass than the tile is tall: nothing stops the
    threads holding the overhang from storing over the next tile's rows except
    the row count being clamped to the tile.
    """
    case = _make_case(200, 4, 64, 256, 256, seed=2027, expert_pool=8)

    expected = _reference_blockwise_moe(
        case["x_dequantized"],
        case["gate_up_dequantized"],
        case["down_dequantized"],
        case["topk_ids"],
        case["topk_scale"],
    )
    dsl = _run(case, hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl)

    assert _close_with_summed_rounding(expected.float(), dsl.float(), rtol=0.12, atol=0.25)


@pytest.mark.parametrize("fuse_reduce", [False, True], indirect=True)
def test_blockwise_dsl_rejects_too_many_routes_either_way(fuse_reduce):
    """The top-k bound belongs to the reduce kernel, which only some batches reach.

    The fused epilogue sums the routes itself and has no such bound, so left to
    the kernels this configuration would raise for small batches and quietly run
    for large ones. Which is why the entry checks it.
    """
    case = _make_case(4, hpc.cute.fuse_moe._reduce.MAX_TOPK + 1, 200, 128, 128, seed=13)

    with pytest.raises(ValueError, match="experts per token"):
        _run(case, hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl)


@pytest.mark.parametrize("swiglu_limit", [None, 0.5])
def test_blockwise_dsl_swiglu_limit(swiglu_limit):
    case = _make_case(32, 2, 4, 128, 128, seed=97)

    expected = _reference_blockwise_moe(
        case["x_dequantized"],
        case["gate_up_dequantized"],
        case["down_dequantized"],
        case["topk_ids"],
        case["topk_scale"],
        swiglu_limit=swiglu_limit,
    )
    dsl = _run(case, hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl, swiglu_limit=swiglu_limit)

    assert _close_with_summed_rounding(expected.float(), dsl.float(), rtol=0.12, atol=0.25)


@pytest.mark.parametrize("fuse_reduce", [False, True], indirect=True)
def test_blockwise_dsl_shared_output_and_preallocated(fuse_reduce):
    # Both paths pick up the shared output, but at different points: the reduce
    # adds it as it sums, the fused epilogue accumulates on top of it.
    case = _make_case(16, 2, 4, 128, 128, seed=11)
    num_tokens, hidden_size = case["x"].shape
    shared_output = torch.randn((num_tokens, hidden_size), device=case["x"].device).to(
        torch.bfloat16
    )
    preallocated = torch.empty(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device=case["x"].device
    )

    expected = _reference_blockwise_moe(
        case["x_dequantized"],
        case["gate_up_dequantized"],
        case["down_dequantized"],
        case["topk_ids"],
        case["topk_scale"],
        shared_output=shared_output,
    )
    dsl = _run(
        case,
        hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl,
        shared_output=shared_output,
        output=preallocated,
    )

    assert dsl.data_ptr() == preallocated.data_ptr()
    assert _close_with_summed_rounding(expected.float(), dsl.float(), rtol=0.12, atol=0.25)


def test_blockwise_dsl_accepts_packed_weight_scale():
    case = _make_case(24, 2, 3, 128, 256, seed=5)
    intermediate_size = case["down_weight"].shape[2] * 2
    hidden_size = case["x"].shape[1]

    logical = _run(case, hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl)

    packed = dict(case)
    packed["gate_up_scale"] = hpc.pack_mxfp_weight_scale_sm120(
        case["gate_up_scale"], 2 * intermediate_size, hidden_size
    )
    packed["down_scale"] = hpc.pack_mxfp_weight_scale_sm120(
        case["down_scale"], hidden_size, intermediate_size
    )
    assert torch.equal(logical, _run(packed, hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl))


def _quantize_mxfp8_like_gather(x):
    """MXFP8 with the gather's rounding, bit for bit: the block maximum over 448,
    rounded up to a power of two, as the scale."""
    from hpc.cute._quant import FP8_MAX, fp32_to_ue8m0, ue8m0_to_fp32

    blocks = x.float().reshape(x.shape[0], -1, 32)
    bits = fp32_to_ue8m0(blocks.abs().amax(dim=-1) / FP8_MAX)
    quantized = (blocks * (1.0 / ue8m0_to_fp32(bits)).unsqueeze(-1)).to(torch.float8_e4m3fn)
    return quantized.reshape_as(x), bits


# Pinned to the unfused reduce: the fused one sums with atomics, in no fixed
# order, so only the unfused output can be compared bit for bit.
@pytest.mark.parametrize("fuse_reduce", [False], indirect=True)
@pytest.mark.parametrize(
    "num_tokens,num_topk,num_experts,hidden_size,intermediate_size",
    [
        (5, 2, 3, 384, 128),
        (3, 2, 3, 2304, 128),
        (64, 2, 8, 256, 128),
        (200, 4, 4, 128, 256),
    ],
)
def test_blockwise_dsl_bf16_input_matches_prequantized(
    fuse_reduce, num_tokens, num_topk, num_experts, hidden_size, intermediate_size
):
    """BF16 activations quantized in the gather match the same activations
    quantized beforehand.

    384 columns leave the last pass over a row with half the warp past its end;
    2304 take a full pass and then one with only its first chunk in range. The zero block and the block whose maximum is exactly ``448 * 2^-7`` are the
    two places a scale rounding could disagree.
    """
    case = _make_case(num_tokens, num_topk, num_experts, hidden_size, intermediate_size, seed=31)
    torch.manual_seed(31)
    x = (torch.randn((num_tokens, hidden_size), device="cuda") * 3).to(torch.bfloat16)
    x[0, :32] = 0
    x[-1, 32:64] = 0.25
    x[-1, 40] = 3.5
    x_q, x_scale = _quantize_mxfp8_like_gather(x)

    prequantized = _run(
        dict(case, x=x_q, x_scale=x_scale), hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl
    )
    in_gather = _run(dict(case, x=x, x_scale=None), hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl)

    assert torch.equal(prequantized, in_gather)


@pytest.mark.parametrize("fuse_reduce", [False], indirect=True)
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize(
    "num_tokens,num_topk,num_experts,rank_ep,hidden_size",
    [
        (5, 2, 3, 0, 384),
        (100, 6, 32, 0, 256),
        (200, 4, 64, 0, 256),
        (200, 4, 64, 1, 256),
        (20, 3, 8, 1, 128),
        (200, 4, 4, 0, 256),
        (2, 6, 16, 0, 128),
    ],
)
def test_blockwise_dsl_gathered_a_matches_copied(
    fuse_reduce, monkeypatch, bf16, num_tokens, num_topk, num_experts, rank_ep, hidden_size
):
    """The up projection reading A rows by token matches reading a per-route copy.

    Up to 128 tokens routing has already placed every route and the gather only
    reads the rows back; past that the gather allocates them. With ``rank_ep``
    half the routes go to the other rank's experts and must be left out. 200
    rows an expert take the 64-row tile; two tokens leave every tile with one
    routed row, so nearly all of its gathers are skipped.
    """
    num_expert_local = num_experts // (2 if rank_ep else 1)
    case = _make_case(num_tokens, num_topk, num_experts, hidden_size, 128, seed=41)
    for name in ("gate_up_weight", "gate_up_scale", "down_weight", "down_scale"):
        start = rank_ep * num_expert_local
        case[name] = case[name][start : start + num_expert_local].contiguous()
    x = (torch.randn((num_tokens, hidden_size), device="cuda") * 2).to(torch.bfloat16)
    if bf16:
        case = dict(case, x=x, x_scale=None)
    else:
        x_q, x_scale = _quantize_mxfp8_like_gather(x)
        case = dict(case, x=x_q, x_scale=x_scale)

    def run():
        return hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl(
            case["x"],
            case["x_scale"],
            case["gate_up_weight"],
            case["gate_up_scale"],
            case["down_weight"],
            case["down_scale"],
            case["topk_ids"],
            case["topk_scale"],
            rank_ep=rank_ep,
            num_expert_total=num_experts,
        )

    pipeline = hpc.cute.fuse_moe._pipeline
    real_gather_a = pipeline._gather_a
    taken = []
    monkeypatch.setattr(
        pipeline, "_gather_a", lambda *args: taken.append(real_gather_a(*args)) or taken[-1]
    )
    gathered = run()
    assert taken == [True]
    monkeypatch.setattr(pipeline, "_gather_a", lambda *_: False)
    copied = run()

    assert torch.equal(gathered, copied)


@pytest.mark.parametrize("tile_n", [64, 128])
@pytest.mark.parametrize("fuse_reduce", [False, True], indirect=True)
def test_blockwise_dsl_thin_gemm2_matches_fat(fuse_reduce, tile_n, monkeypatch):
    """Unfused GEMM2 on a 64-row tile: thin CTAs match the full warp count.

    Four experts and 200 tokens put enough rows on each expert to pick the tall
    tile. Both GEMM2 widths split across the thin form's two N warps. Through
    the separate reduce the two launches agree exactly; the scatter epilogue's
    atomics only promise the sum up to its rounding.
    """
    case = _make_case(200, 4, 4, 256, 256, seed=2027)
    grouped_gemm = hpc.cute.fuse_moe._grouped_gemm
    real_use_thin = grouped_gemm._use_thin
    taken = []

    def recording_use_thin(fused, tile_m, b_fp8=False):
        result = real_use_thin(fused, tile_m, b_fp8)
        taken.append(result)
        return result

    monkeypatch.setattr(grouped_gemm, "_tile_n_for", lambda *_args: tile_n)
    monkeypatch.setattr(grouped_gemm, "_use_thin", recording_use_thin)
    thin_out = _run(case, hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl)
    assert any(taken)

    monkeypatch.setattr(grouped_gemm, "_use_thin", lambda *_args, **_kwargs: False)
    fat_out = _run(case, hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl)
    if fuse_reduce:
        assert _close_with_summed_rounding(fat_out.float(), thin_out.float(), rtol=0, atol=0)
    else:
        assert torch.equal(thin_out, fat_out)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(fuse_swiglu=True),
        dict(a_gather=True),
        dict(tile_m=32),
        dict(tile_k=64),
    ],
)
def test_thin_gemm2_rejects_other_shapes(kwargs):
    grouped_gemm = hpc.cute.fuse_moe._grouped_gemm
    kwargs = dict(dict(tile_m=64, tile_k=128), **kwargs)
    with pytest.raises(ValueError, match="thin form"):
        grouped_gemm._build(1, 128, thin=True, **kwargs)


def test_blockwise_dsl_bf16_input_rejects_x_scale():
    case = _make_case(4, 2, 2, 128, 128, seed=3)
    x = torch.randn(case["x"].shape, device="cuda").to(torch.bfloat16)

    with pytest.raises(ValueError, match="x_scale must be None"):
        _run(dict(case, x=x), hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl)


def test_blockwise_dsl_registers_custom_op():
    from hpc.cute.fuse_moe import register

    register()
    assert hasattr(torch.ops.hpc, "fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl")

    case = _make_case(8, 2, 2, 128, 128, seed=3)
    output = torch.empty(case["x"].shape, dtype=torch.bfloat16, device=case["x"].device)
    torch.ops.hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl(
        case["x"],
        case["x_scale"],
        case["gate_up_weight"],
        case["gate_up_scale"],
        case["down_weight"],
        case["down_scale"],
        case["topk_ids"],
        case["topk_scale"],
        None,
        0,
        2,
        output,
        -1.0,
    )
    assert torch.isfinite(output.float()).all()

    output.fill_(float("nan"))
    torch.ops.hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl(
        case["x_dequantized"].to(torch.bfloat16),
        None,
        case["gate_up_weight"],
        case["gate_up_scale"],
        case["down_weight"],
        case["down_scale"],
        case["topk_ids"],
        case["topk_scale"],
        None,
        0,
        2,
        output,
        -1.0,
    )
    assert torch.isfinite(output.float()).all()
