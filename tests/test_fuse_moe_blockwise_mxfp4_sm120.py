# Copyright (C) 2026 Tencent.

"""SM120 fused MoE with MXFP4 weights and MXFP8 activations."""

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import pytest
import torch
import torch.nn.functional as F

import hpc
from hpc.fuse_moe import _FP8_MAX, _fp32_to_ue8m0, _ue8m0_to_fp32

SCALE_VECTOR = 32
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
    blocks = x.float().reshape(*x.shape[:-1], x.shape[-1] // SCALE_VECTOR, SCALE_VECTOR)
    scale_bits, scale = _encode_ue8m0_scale(blocks.abs().amax(dim=-1), 448.0)
    quantized = (blocks / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    dequantized = quantized.float() * scale.unsqueeze(-1)
    return quantized.reshape_as(x), scale_bits, dequantized.reshape_as(x)


def _quantize_mxfp4(weight):
    blocks = weight.float().reshape(
        *weight.shape[:-1], weight.shape[-1] // SCALE_VECTOR, SCALE_VECTOR
    )
    scale_bits, scale = _encode_ue8m0_scale(blocks.abs().amax(dim=-1), 6.0)
    packed, normalized = _pack_e2m1(blocks / scale.unsqueeze(-1))
    dequantized = normalized * scale.unsqueeze(-1)
    packed = packed.reshape(*weight.shape[:-1], weight.shape[-1] // 2)
    return packed.contiguous(), scale_bits.contiguous(), dequantized.reshape_as(weight)


# The down GEMM sums an expert's contribution into the output with a bfloat16
# atomic. The top-k terms largely cancel, so the rounding is a fraction of the
# largest term in a row rather than of the element it lands on, and the
# comparison has to be in those units.
SUM_ULPS = 4
BF16_ULP = 2**-8


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


def _single_route(topk_scale):
    """Routing whose only nonzero weight is the first route. Adding an exact
    zero leaves a BF16 sum unchanged, so the output no longer depends on the order
    of the atomic adds."""
    scale = torch.zeros_like(topk_scale)
    scale[:, 0] = topk_scale[:, 0]
    return scale


def _close_with_summed_rounding(reference, actual, rtol, atol):
    """``allclose`` with room for the top-k sum being rounded per contribution.

    The extra term is per row rather than per element, because that is where the
    rounding comes from: the sum's error follows the terms, not the result.
    """
    slack = reference.abs().amax(dim=-1, keepdim=True) * BF16_ULP * SUM_ULPS
    return bool(((actual - reference).abs() <= atol + rtol * reference.abs() + slack).all())


def _quantize_mxfp8_like_gather(x):
    """MXFP8 with the gather's rounding, bit for bit: the block maximum over 448,
    rounded up to a power of two, as the scale."""
    blocks = x.float().reshape(x.shape[0], -1, 32)
    bits = _fp32_to_ue8m0(blocks.abs().amax(dim=-1) / _FP8_MAX)
    quantized = (blocks * (1.0 / _ue8m0_to_fp32(bits)).unsqueeze(-1)).to(torch.float8_e4m3fn)
    return quantized.reshape_as(x), bits


def _shifted(tensor, offset_bytes):
    """A contiguous copy of ``tensor`` whose storage starts ``offset_bytes`` past a
    16-byte boundary, so it keeps its shape and dtype but breaks TMA alignment."""
    elem = tensor.element_size()
    assert offset_bytes % elem == 0
    pad = offset_bytes // elem
    buffer = torch.empty(tensor.numel() + pad, dtype=tensor.dtype, device=tensor.device)
    view = buffer[pad:].view(tensor.shape)
    view.copy_(tensor)
    assert view.is_contiguous() and view.data_ptr() % 16 != 0
    return view


def check_misaligned_weights_rejected(entry, x, weights, routing, num_experts):
    """Each of the four weight tensors, shifted off a 16-byte boundary, must be
    rejected with a clear error; the aligned originals must still run."""
    names = ("gate_up_weight", "gate_up_weight_scale", "down_weight", "down_weight_scale")
    expected = entry(x, None, *weights, *routing, num_expert_total=num_experts)
    for index, name in enumerate(names):
        # int32 scales can only shift by whole elements, so 4 bytes; byte weights by 1.
        offset = weights[index].element_size()
        shifted = list(weights)
        shifted[index] = _shifted(weights[index], offset)
        with pytest.raises(RuntimeError, match="16-byte aligned"):
            entry(x, None, *shifted, *routing, num_expert_total=num_experts)
    torch.testing.assert_close(
        entry(x, None, *weights, *routing, num_expert_total=num_experts), expected, rtol=0, atol=0
    )


def _is_sm120():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


pytestmark = pytest.mark.skipif(not _is_sm120(), reason="requires an SM120 GPU")


def _close_with_atomic_sum(reference, actual, num_terms, rtol, atol):
    """``_close_with_summed_rounding`` for an output built from ``num_terms`` BF16 atomic adds.

    Every add rounds the running sum, in an order that changes from run to run, so
    the error random-walks and the slack grows with the square root of the count.
    """
    ulps = SUM_ULPS * max(1.0, math.sqrt(num_terms / 4))
    slack = reference.abs().amax(dim=-1, keepdim=True) * BF16_ULP * ulps
    return bool(((actual - reference).abs() <= atol + rtol * reference.abs() + slack).all())


@pytest.mark.parametrize(
    "num_tokens,num_topk,num_experts,hidden_size,intermediate_size",
    [
        (1, 1, 8, 128, 128),
        (1, 8, 8, 128, 128),
        (5, 6, 8, 128, 128),
        (64, 8, 16, 256, 128),
        (200, 10, 16, 256, 256),
        (130, 32, 48, 128, 128),
        # More routes per token than a warp has lanes: each lane takes several.
        (1, 40, 48, 128, 128),
        (130, 40, 48, 128, 128),
        (5, 128, 128, 128, 128),
    ],
)
def test_matches_reference(num_tokens, num_topk, num_experts, hidden_size, intermediate_size):
    torch.manual_seed(307)
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    _, _, x_dequantized = _quantize_mxfp8(x)
    gate_weight, gate_scale, gate_dequantized = _quantize_mxfp4(
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, device="cuda")
    )
    down_weight, down_scale, down_dequantized = _quantize_mxfp4(
        torch.randn(num_experts, hidden_size, intermediate_size, device="cuda")
    )
    topk_ids, topk_scale = _make_routing(num_tokens, num_topk, num_experts, "cuda")
    gate_packed = hpc.prepare_mxfp4_moe_weights_sm120(
        gate_scale, 2 * intermediate_size, hidden_size
    )
    down_packed = hpc.prepare_mxfp4_moe_weights_sm120(down_scale, hidden_size, intermediate_size)
    shared = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(shared)

    expected = _reference_blockwise_moe(
        x_dequantized,
        gate_dequantized,
        down_dequantized,
        topk_ids,
        topk_scale,
        shared,
    )
    actual = hpc.fuse_moe_blockwise_mxfp4_sm120(
        x,
        None,
        gate_weight,
        gate_packed,
        down_weight,
        down_packed,
        topk_ids,
        topk_scale,
        shared_output=shared,
        output=output,
        num_expert_total=num_experts,
    )
    assert actual.data_ptr() == output.data_ptr()
    assert _close_with_atomic_sum(
        expected.float(), actual.float(), num_topk + 1, rtol=0.12, atol=0.25
    )


def test_ep_and_cuda_graph_replay():
    torch.manual_seed(401)
    num_tokens, num_topk = 16, 8
    num_experts, hidden_size, intermediate_size = 4, 128, 128
    rank_ep, num_expert_total = 1, 16
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    _, _, x_dequantized = _quantize_mxfp8(x)
    gate_weight, gate_scale, gate_dequantized = _quantize_mxfp4(
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, device="cuda")
    )
    down_weight, down_scale, down_dequantized = _quantize_mxfp4(
        torch.randn(num_experts, hidden_size, intermediate_size, device="cuda")
    )
    topk_ids = torch.randint(
        num_expert_total, (num_tokens, num_topk), dtype=torch.int32, device="cuda"
    )
    topk_scale = torch.rand(num_tokens, num_topk, dtype=torch.float32, device="cuda")
    gate_packed = hpc.prepare_mxfp4_moe_weights_sm120(
        gate_scale, 2 * intermediate_size, hidden_size
    )
    down_packed = hpc.prepare_mxfp4_moe_weights_sm120(down_scale, hidden_size, intermediate_size)
    output = torch.empty_like(x)

    expected = torch.zeros_like(x, dtype=torch.float32)
    for token in range(num_tokens):
        for selected in range(num_topk):
            global_expert = int(topk_ids[token, selected])
            if not rank_ep * num_experts <= global_expert < (rank_ep + 1) * num_experts:
                continue
            expert = global_expert - rank_ep * num_experts
            gate_up = torch.mv(gate_dequantized[expert], x_dequantized[token]).bfloat16().float()
            gate, up = gate_up.chunk(2)
            _, _, down_input = _quantize_mxfp8((F.silu(gate) * up).unsqueeze(0))
            down = torch.mv(down_dequantized[expert], down_input[0]).bfloat16().float()
            expected[token] += down * topk_scale[token, selected]

    def run():
        return hpc.fuse_moe_blockwise_mxfp4_sm120(
            x,
            None,
            gate_weight,
            gate_packed,
            down_weight,
            down_packed,
            topk_ids,
            topk_scale,
            output=output,
            rank_ep=rank_ep,
            num_expert_total=num_expert_total,
            swiglu_limit=-1.0,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    assert _close_with_summed_rounding(expected, output.float(), rtol=0.12, atol=0.25)


def test_prepared_weight_scale_layout():
    torch.manual_seed(457)
    num_tokens, num_experts, hidden_size, intermediate_size = 37, 8, 256, 384
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    gate_weight, gate_scale, _ = _quantize_mxfp4(
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, device="cuda")
    )
    down_weight, down_scale, _ = _quantize_mxfp4(
        torch.randn(num_experts, hidden_size, intermediate_size, device="cuda")
    )
    # Only the first route carries weight, so both runs add into the output in the same order.
    topk_ids, topk_scale = _make_routing(num_tokens, 8, num_experts, "cuda")
    topk_scale = _single_route(topk_scale)
    gate_rows, gate_cols = 2 * intermediate_size, hidden_size
    down_rows, down_cols = hidden_size, intermediate_size
    packed_gate = hpc.prepare_mxfp4_moe_weights_sm120(gate_scale, gate_rows, gate_cols)
    packed_down = hpc.prepare_mxfp4_moe_weights_sm120(down_scale, down_rows, down_cols)
    assert packed_gate.dtype == torch.int32
    assert packed_down.dtype == torch.int32
    assert packed_gate.numel() == num_experts * ((gate_rows + 3) // 4 * 4) * (
        (gate_cols + 127) // 128
    )
    assert packed_down.numel() == num_experts * ((down_rows + 3) // 4 * 4) * (
        (down_cols + 127) // 128
    )

    def run():
        return hpc.fuse_moe_blockwise_mxfp4_sm120(
            x,
            None,
            gate_weight,
            packed_gate,
            down_weight,
            packed_down,
            topk_ids,
            topk_scale,
            num_expert_total=num_experts,
        )

    torch.testing.assert_close(run(), run(), rtol=0, atol=0)


def _one_route_case(num_tokens, num_experts, hidden_size, intermediate_size, seed):
    torch.manual_seed(seed)
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    gate_weight, gate_scale, _ = _quantize_mxfp4(
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, device="cuda")
    )
    down_weight, down_scale, _ = _quantize_mxfp4(
        torch.randn(num_experts, hidden_size, intermediate_size, device="cuda")
    )
    # Only the first route carries weight, so every run adds into the output in the same order.
    topk_ids, topk_scale = _make_routing(num_tokens, 8, num_experts, "cuda")
    topk_scale = _single_route(topk_scale)
    gate_scale = hpc.prepare_mxfp4_moe_weights_sm120(gate_scale, 2 * intermediate_size, hidden_size)
    down_scale = hpc.prepare_mxfp4_moe_weights_sm120(down_scale, hidden_size, intermediate_size)
    return x, (gate_weight, gate_scale, down_weight, down_scale, topk_ids, topk_scale)


@pytest.mark.parametrize("num_tokens", [1, 37, 200])
def test_unaligned_bf16_input(num_tokens):
    hidden_size = 256
    x, args = _one_route_case(num_tokens, 8, hidden_size, 128, seed=487)
    # One element in, so the input is not 16-byte aligned.
    buffer = torch.empty(num_tokens * hidden_size + 1, dtype=torch.bfloat16, device="cuda")
    unaligned = buffer[1:].view(num_tokens, hidden_size)
    unaligned.copy_(x)

    def run(inp):
        return hpc.fuse_moe_blockwise_mxfp4_sm120(inp, None, *args, num_expert_total=8)

    torch.testing.assert_close(run(unaligned), run(x), rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [16383, 16384])
def test_output_zeroing_paths(num_tokens):
    # From 16384 tokens the up GEMM zeroes the output instead of routing.
    hidden_size = 256
    x, args = _one_route_case(num_tokens, 8, hidden_size, 128, seed=509)
    zeros = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    expected = hpc.fuse_moe_blockwise_mxfp4_sm120(
        x, None, *args, shared_output=zeros, num_expert_total=8
    )
    output = torch.full_like(zeros, 777.0)

    def run():
        return hpc.fuse_moe_blockwise_mxfp4_sm120(x, None, *args, output=output, num_expert_total=8)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _compile_cases():
    """Inductor-compiled callers of the functional and out paths. Each returns
    what the caller reads afterwards."""

    def functional(args, x, output):
        return hpc.fuse_moe_blockwise_mxfp4_sm120(x, None, *args, num_expert_total=8)

    def out_returned(args, x, output):
        return hpc.fuse_moe_blockwise_mxfp4_sm120(x, None, *args, num_expert_total=8, output=output)

    def out_ignored(args, x, output):
        hpc.fuse_moe_blockwise_mxfp4_sm120(x, None, *args, num_expert_total=8, output=output)
        return output

    def out_then_add(args, x, output):
        # A consumer of the buffer inside the graph must see the written values.
        hpc.fuse_moe_blockwise_mxfp4_sm120(x, None, *args, num_expert_total=8, output=output)
        return output + 1

    return [functional, out_returned, out_ignored, out_then_add]


@pytest.mark.parametrize("unaligned_output", [False, True])
@pytest.mark.parametrize("case", _compile_cases(), ids=lambda f: f.__name__)
def test_torch_compile_inductor(case, unaligned_output):
    num_tokens, hidden_size = 37, 256
    x, args = _one_route_case(num_tokens, 8, hidden_size, 128, seed=491)

    def make_output():
        if not unaligned_output:
            return torch.full((num_tokens, hidden_size), 777.0, dtype=torch.bfloat16, device="cuda")
        buffer = torch.full(
            (num_tokens * hidden_size + 1,), 777.0, dtype=torch.bfloat16, device="cuda"
        )
        return buffer[1:].view(num_tokens, hidden_size)

    eager_output = make_output()
    expected = case(args, x, eager_output).clone()
    compiled_output = make_output()
    compiled = torch.compile(case, fullgraph=True, backend="inductor")
    actual = compiled(args, x, compiled_output)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The out path must leave the result in the caller's buffer.
    if case.__name__ != "functional":
        torch.testing.assert_close(compiled_output, eager_output, rtol=0, atol=0)


@pytest.mark.parametrize("unaligned_output", [False, True])
def test_opcheck(unaligned_output):
    num_tokens, hidden_size = 37, 256
    x, (gate_w, gate_s, down_w, down_s, ids, scale) = _one_route_case(
        num_tokens, 8, hidden_size, 128, seed=521
    )
    shared = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    tail = (gate_w, gate_s, down_w, down_s, ids, scale, shared, 0, 8, -1.0)
    torch.library.opcheck(torch.ops.hpc.fuse_moe_mx.default, (x, None, *tail))
    if unaligned_output:
        buffer = torch.empty(num_tokens * hidden_size + 1, dtype=torch.bfloat16, device="cuda")
        output = buffer[1:].view(num_tokens, hidden_size)
    else:
        output = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    torch.library.opcheck(torch.ops.hpc.fuse_moe_mx_out.default, (output, x, None, *tail))


@pytest.mark.parametrize("num_tokens", [5, 200])
def test_swiglu_clamp(num_tokens):
    torch.manual_seed(547)
    num_topk, num_experts, hidden_size, intermediate_size = 8, 8, 256, 128
    swiglu_limit = 2.0
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    _, _, x_dequantized = _quantize_mxfp8(x)
    gate_weight, gate_scale, gate_dequantized = _quantize_mxfp4(
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, device="cuda")
    )
    down_weight, down_scale, down_dequantized = _quantize_mxfp4(
        torch.randn(num_experts, hidden_size, intermediate_size, device="cuda")
    )
    gate_packed = hpc.prepare_mxfp4_moe_weights_sm120(
        gate_scale, 2 * intermediate_size, hidden_size
    )
    down_packed = hpc.prepare_mxfp4_moe_weights_sm120(down_scale, hidden_size, intermediate_size)
    topk_ids, topk_scale = _make_routing(num_tokens, num_topk, num_experts, "cuda")

    # Gate and up are sums of 256 unit products, so most of them lie well past the limit.
    gate_up = torch.einsum("th,eoh->teo", x_dequantized, gate_dequantized)
    assert (gate_up.abs() > swiglu_limit).float().mean() > 0.5

    expected = _reference_blockwise_moe(
        x_dequantized,
        gate_dequantized,
        down_dequantized,
        topk_ids,
        topk_scale,
        swiglu_limit=swiglu_limit,
    )
    unclamped = _reference_blockwise_moe(
        x_dequantized, gate_dequantized, down_dequantized, topk_ids, topk_scale
    )
    actual = hpc.fuse_moe_blockwise_mxfp4_sm120(
        x,
        None,
        gate_weight,
        gate_packed,
        down_weight,
        down_packed,
        topk_ids,
        topk_scale,
        num_expert_total=num_experts,
        swiglu_limit=swiglu_limit,
    )
    assert _close_with_atomic_sum(expected.float(), actual.float(), num_topk, rtol=0.12, atol=0.25)
    # The clamp must matter: the unclamped reference is far off.
    assert not _close_with_atomic_sum(
        unclamped.float(), actual.float(), num_topk, rtol=0.12, atol=0.25
    )


@pytest.mark.parametrize("num_tokens", [1, 200])
def test_unaligned_output_and_shared(num_tokens):
    torch.manual_seed(503)
    num_topk, num_experts, hidden_size, intermediate_size = 8, 8, 256, 128
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    gate_weight, gate_scale, _ = _quantize_mxfp4(
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, device="cuda")
    )
    down_weight, down_scale, _ = _quantize_mxfp4(
        torch.randn(num_experts, hidden_size, intermediate_size, device="cuda")
    )
    gate_packed = hpc.prepare_mxfp4_moe_weights_sm120(
        gate_scale, 2 * intermediate_size, hidden_size
    )
    down_packed = hpc.prepare_mxfp4_moe_weights_sm120(down_scale, hidden_size, intermediate_size)
    topk_ids, topk_scale = _make_routing(num_tokens, num_topk, num_experts, "cuda")
    shared = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

    def run(output, shared_output):
        return hpc.fuse_moe_blockwise_mxfp4_sm120(
            x,
            None,
            gate_weight,
            gate_packed,
            down_weight,
            down_packed,
            topk_ids,
            topk_scale,
            shared_output=shared_output,
            output=output,
            num_expert_total=num_experts,
        )

    expected = run(None, shared)
    # One element in, so neither tensor is 16-byte aligned.
    size = num_tokens * hidden_size
    buffer = torch.full((2 * size + 1,), 777.0, dtype=torch.bfloat16, device="cuda")
    output = buffer[1 : 1 + size].view(num_tokens, hidden_size)
    shared_unaligned = buffer[1 + size :].view(num_tokens, hidden_size)
    shared_unaligned.copy_(shared)
    actual = run(output, shared_unaligned)
    assert actual.data_ptr() == output.data_ptr()
    # The BF16 atomics add the routes in a different order on every call.
    diff = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert diff < 1e-2


@pytest.mark.parametrize(
    "num_tokens,num_topk,num_experts,hidden_size,intermediate_size,unaligned",
    [
        (1, 8, 8, 128, 128, False),
        (1, 8, 32, 256, 128, True),
        (5, 6, 8, 128, 128, False),
        (200, 8, 8, 256, 256, False),
        (200, 10, 16, 256, 256, True),
        (130, 32, 48, 128, 128, False),
        (130, 40, 48, 128, 128, False),
        (37, 40, 48, 256, 128, True),
    ],
)
def test_input_formats_agree(
    num_tokens, num_topk, num_experts, hidden_size, intermediate_size, unaligned
):
    torch.manual_seed(601)
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    x_fp8, x_scale = _quantize_mxfp8_like_gather(x)
    x_dequantized = (
        x_fp8.float().view(num_tokens, -1, 32) * _ue8m0_to_fp32(x_scale).unsqueeze(-1)
    ).view(num_tokens, hidden_size)
    if unaligned:
        buffer = torch.empty(x_fp8.numel() + 1, dtype=torch.float8_e4m3fn, device="cuda")
        buffer[1:].copy_(x_fp8.reshape(-1))
        x_fp8 = buffer[1:].view_as(x_fp8)
        scale_buffer = torch.empty(x_scale.numel() + 1, dtype=torch.uint8, device="cuda")
        scale_buffer[1:].copy_(x_scale.reshape(-1))
        x_scale = scale_buffer[1:].view_as(x_scale)
    gate_weight, gate_scale, gate_dequantized = _quantize_mxfp4(
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, device="cuda")
    )
    down_weight, down_scale, down_dequantized = _quantize_mxfp4(
        torch.randn(num_experts, hidden_size, intermediate_size, device="cuda")
    )
    gate_packed = hpc.prepare_mxfp4_moe_weights_sm120(
        gate_scale, 2 * intermediate_size, hidden_size
    )
    down_packed = hpc.prepare_mxfp4_moe_weights_sm120(down_scale, hidden_size, intermediate_size)
    topk_ids, topk_scale = _make_routing(num_tokens, num_topk, num_experts, "cuda")
    shared = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

    def run(activations, activation_scale, ids, scale):
        return hpc.fuse_moe_blockwise_mxfp4_sm120(
            activations,
            activation_scale,
            gate_weight,
            gate_packed,
            down_weight,
            down_packed,
            ids,
            scale,
            shared_output=shared,
            num_expert_total=num_experts,
        )

    # The routes are summed with BF16 atomics in no fixed order, and with many
    # routes that order alone moves the result by about 1%. Only the first route
    # carries weight, so the two inputs must then agree bit for bit.
    one_route = (topk_ids, _single_route(topk_scale))
    torch.testing.assert_close(
        run(x_fp8, x_scale, *one_route), run(x, None, *one_route), rtol=0, atol=0
    )

    expected = _reference_blockwise_moe(
        x_dequantized, gate_dequantized, down_dequantized, topk_ids, topk_scale, shared
    )
    for activations, activation_scale in ((x, None), (x_fp8, x_scale)):
        actual = run(activations, activation_scale, topk_ids, topk_scale)
        assert actual.dtype == torch.bfloat16
        assert _close_with_atomic_sum(
            expected.float(), actual.float(), num_topk + 1, rtol=0.12, atol=0.25
        )


@pytest.mark.parametrize("num_tokens", [1, 37])
def test_misaligned_weights_rejected(num_tokens):
    x, args = _one_route_case(num_tokens, 8, 256, 128, seed=557)
    check_misaligned_weights_rejected(
        hpc.fuse_moe_blockwise_mxfp4_sm120, x, args[:4], args[4:], num_experts=8
    )
