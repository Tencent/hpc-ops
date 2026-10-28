# Copyright (C) 2026 Tencent.

"""SM120 fused MoE with 128-block FP8 weights."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import pytest
import torch
import torch.nn.functional as F

import hpc
from hpc.fuse_moe import _FP8_MAX as FP8_MAX
from hpc.fuse_moe import _fp32_to_ue8m0 as fp32_to_ue8m0
from hpc.fuse_moe import _ue8m0_to_fp32 as ue8m0_to_fp32
from test_fuse_moe_blockwise_mxfp4_sm120 import (
    _close_with_atomic_sum,
    _close_with_summed_rounding,
    _make_routing,
    _quantize_mxfp8,
    _quantize_mxfp8_like_gather,
    _reference_blockwise_moe,
    _single_route,
    check_misaligned_weights_rejected,
)


def _is_sm120():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


pytestmark = pytest.mark.skipif(not _is_sm120(), reason="requires an SM120 GPU")


def _quantize_fp8_block128(weight):
    experts, rows, cols = weight.shape
    block = 128
    tiled = weight.float().reshape(experts, rows // block, block, cols // block, block)
    scale = (tiled.abs().amax(dim=(2, 4)) / FP8_MAX).clamp(min=2.0**-127)
    quantized = (tiled / scale[:, :, None, :, None]).to(torch.float8_e4m3fn)
    return quantized.reshape(experts, rows, cols).contiguous(), scale.contiguous()


def _dequant_block128(fp8, scale):
    return fp8.float() * scale.repeat_interleave(128, 1).repeat_interleave(128, 2)


def _quantize_activation_block128(x):
    tokens, hidden = x.shape
    blocks = x.float().reshape(tokens, hidden // 128, 128)
    scale = ue8m0_to_fp32(fp32_to_ue8m0(blocks.abs().amax(dim=-1) / FP8_MAX))
    x_fp8 = (x.float() / scale.repeat_interleave(128, dim=1)).to(torch.float8_e4m3fn)
    return x_fp8.contiguous(), scale.contiguous()


def _fp8_weights(num_experts, hidden, intermediate):
    gate_w, gate_s = _quantize_fp8_block128(
        torch.randn(num_experts, 2 * intermediate, hidden, device="cuda")
    )
    down_w, down_s = _quantize_fp8_block128(
        torch.randn(num_experts, hidden, intermediate, device="cuda")
    )
    gate_packed = hpc.prepare_fp8_moe_weights_sm120(gate_w, gate_s)
    down_packed = hpc.prepare_fp8_moe_weights_sm120(down_w, down_s)
    return (
        gate_w,
        gate_s,
        gate_packed,
        _dequant_block128(gate_w, gate_s),
        down_w,
        down_s,
        down_packed,
        _dequant_block128(down_w, down_s),
    )


def _mantissa_clear(scale):
    return bool(((scale.view(torch.int32) & 0x7FFFFF) == 0).all())


def _one_route_case(num_tokens, num_experts, hidden_size, intermediate_size, seed):
    torch.manual_seed(seed)
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    gate_w, _, gate_packed, _, down_w, _, down_packed, _ = _fp8_weights(
        num_experts, hidden_size, intermediate_size
    )
    # Only the first route carries weight, so every run adds into the output in the same order.
    topk_ids, topk_scale = _make_routing(num_tokens, 8, num_experts, "cuda")
    topk_scale = _single_route(topk_scale)
    return x, (gate_w, gate_packed, down_w, down_packed, topk_ids, topk_scale)


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
    x_fp8, x_scale = _quantize_activation_block128(x)
    x_dequantized = x_fp8.float() * x_scale.repeat_interleave(128, 1)
    gate_w, _, gate_packed, gate_deq, down_w, _, down_packed, down_deq = _fp8_weights(
        num_experts, hidden_size, intermediate_size
    )
    topk_ids, topk_scale = _make_routing(num_tokens, num_topk, num_experts, "cuda")
    shared = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(shared)

    expected = _reference_blockwise_moe(
        x_dequantized, gate_deq, down_deq, topk_ids, topk_scale, shared
    )
    actual = hpc.fuse_moe_blockwise_fp8_sm120(
        x_fp8,
        x_scale,
        gate_w,
        gate_packed,
        down_w,
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
    x_fp8, x_scale_f32 = _quantize_activation_block128(x)
    x_scale_u8 = fp32_to_ue8m0(x_scale_f32).repeat_interleave(4, dim=1).contiguous()
    if unaligned:
        buffer = torch.empty(x_fp8.numel() + 1, dtype=torch.float8_e4m3fn, device="cuda")
        buffer[1:].copy_(x_fp8.reshape(-1))
        x_fp8 = buffer[1:].view_as(x_fp8)
        scale_buffer = torch.empty(x_scale_u8.numel() + 1, dtype=torch.uint8, device="cuda")
        scale_buffer[1:].copy_(x_scale_u8.reshape(-1))
        x_scale_u8 = scale_buffer[1:].view_as(x_scale_u8)
    gate_w, _, gate_packed, gate_deq, down_w, _, down_packed, down_deq = _fp8_weights(
        num_experts, hidden_size, intermediate_size
    )
    topk_ids, topk_scale = _make_routing(num_tokens, num_topk, num_experts, "cuda")
    shared = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

    def run(activations, activation_scale, ids, scale):
        return hpc.fuse_moe_blockwise_fp8_sm120(
            activations,
            activation_scale,
            gate_w,
            gate_packed,
            down_w,
            down_packed,
            ids,
            scale,
            shared_output=shared,
            num_expert_total=num_experts,
        )

    # Only the first route carries weight, so the two scale layouts must agree.
    one_route = (topk_ids, _single_route(topk_scale))
    torch.testing.assert_close(
        run(x_fp8, x_scale_f32, *one_route), run(x_fp8, x_scale_u8, *one_route), rtol=0, atol=0
    )

    gathered, gathered_bits = _quantize_mxfp8_like_gather(x)
    x_dequantized = (
        gathered.float().view(num_tokens, -1, 32) * ue8m0_to_fp32(gathered_bits).unsqueeze(-1)
    ).view(num_tokens, hidden_size)
    expected = _reference_blockwise_moe(
        x_dequantized, gate_deq, down_deq, topk_ids, topk_scale, shared
    )
    actual = run(x, None, topk_ids, topk_scale)
    assert _close_with_atomic_sum(
        expected.float(), actual.float(), num_topk + 1, rtol=0.12, atol=0.25
    )


def test_ep_and_cuda_graph_replay():
    torch.manual_seed(401)
    num_tokens, num_topk = 16, 8
    num_experts, hidden_size, intermediate_size = 4, 128, 128
    rank_ep, num_expert_total = 1, 16
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    x_fp8, x_scale = _quantize_activation_block128(x)
    x_dequantized = x_fp8.float() * x_scale.repeat_interleave(128, 1)
    gate_w, _, gate_packed, gate_deq, down_w, _, down_packed, down_deq = _fp8_weights(
        num_experts, hidden_size, intermediate_size
    )
    topk_ids = torch.randint(
        num_expert_total, (num_tokens, num_topk), dtype=torch.int32, device="cuda"
    )
    topk_scale = torch.rand(num_tokens, num_topk, dtype=torch.float32, device="cuda")
    output = torch.empty_like(x)

    expected = torch.zeros_like(x, dtype=torch.float32)
    for token in range(num_tokens):
        for selected in range(num_topk):
            global_expert = int(topk_ids[token, selected])
            if not rank_ep * num_experts <= global_expert < (rank_ep + 1) * num_experts:
                continue
            expert = global_expert - rank_ep * num_experts
            gate_up = torch.mv(gate_deq[expert], x_dequantized[token]).bfloat16().float()
            gate, up = gate_up.chunk(2)
            _, _, down_input = _quantize_mxfp8((F.silu(gate) * up).unsqueeze(0))
            down = torch.mv(down_deq[expert], down_input[0]).bfloat16().float()
            expected[token] += down * topk_scale[token, selected]

    def run():
        return hpc.fuse_moe_blockwise_fp8_sm120(
            x_fp8,
            x_scale,
            gate_w,
            gate_packed,
            down_w,
            down_packed,
            topk_ids,
            topk_scale,
            output=output,
            rank_ep=rank_ep,
            num_expert_total=num_expert_total,
            swiglu_limit=-1.0,
        )

    eager = run().clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    assert _close_with_summed_rounding(expected, eager.float(), rtol=0.12, atol=0.25)
    assert _close_with_summed_rounding(expected, output.float(), rtol=0.12, atol=0.25)


def test_prepacked_weight_scale():
    torch.manual_seed(457)
    num_tokens, num_experts, hidden, intermediate = 8, 8, 128, 128
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    gate_w, gate_s = _quantize_fp8_block128(
        torch.randn(num_experts, 2 * intermediate, hidden, device="cuda")
    )
    down_w, down_s = _quantize_fp8_block128(
        torch.randn(num_experts, hidden, intermediate, device="cuda")
    )
    packed_gate = hpc.prepare_fp8_moe_weights_sm120(gate_w, gate_s)
    packed_down = hpc.prepare_fp8_moe_weights_sm120(down_w, down_s)
    assert _mantissa_clear(gate_s) and _mantissa_clear(down_s)
    gate_w_clone = gate_w.clone()
    gate_s_clone = gate_s.clone()
    down_w_clone = down_w.clone()
    down_s_clone = down_s.clone()
    repacked_gate = hpc.prepare_fp8_moe_weights_sm120(gate_w_clone, gate_s_clone)
    repacked_down = hpc.prepare_fp8_moe_weights_sm120(down_w_clone, down_s_clone)
    assert torch.equal(repacked_gate, packed_gate)
    assert torch.equal(repacked_down, packed_down)
    assert torch.equal(gate_w_clone.view(torch.uint8), gate_w.view(torch.uint8))
    assert torch.equal(down_w_clone.view(torch.uint8), down_w.view(torch.uint8))
    topk_ids, topk_scale = _make_routing(num_tokens, 8, num_experts, "cuda")
    topk_scale = _single_route(topk_scale)

    def run():
        return hpc.fuse_moe_blockwise_fp8_sm120(
            x,
            None,
            gate_w,
            packed_gate,
            down_w,
            packed_down,
            topk_ids,
            topk_scale,
            num_expert_total=num_experts,
        )

    torch.testing.assert_close(run(), run(), rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [1, 37, 200])
def test_unaligned_bf16_input(num_tokens):
    hidden_size = 256
    x, args = _one_route_case(num_tokens, 8, hidden_size, 128, seed=487)
    # One element in, so the input is not 16-byte aligned.
    buffer = torch.empty(num_tokens * hidden_size + 1, dtype=torch.bfloat16, device="cuda")
    unaligned = buffer[1:].view(num_tokens, hidden_size)
    unaligned.copy_(x)

    def run(inp):
        return hpc.fuse_moe_blockwise_fp8_sm120(inp, None, *args, num_expert_total=8)

    torch.testing.assert_close(run(unaligned), run(x), rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [16383, 16384])
def test_output_zeroing_paths(num_tokens):
    # From 16384 tokens the up GEMM zeroes the output instead of routing.
    hidden_size = 256
    x, args = _one_route_case(num_tokens, 8, hidden_size, 128, seed=509)
    zeros = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    expected = hpc.fuse_moe_blockwise_fp8_sm120(
        x, None, *args, shared_output=zeros, num_expert_total=8
    )
    output = torch.full_like(zeros, 777.0)

    def run():
        return hpc.fuse_moe_blockwise_fp8_sm120(x, None, *args, output=output, num_expert_total=8)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("output_kind", ["none", "aligned", "unaligned"])
def test_torch_compile_inductor(output_kind):
    num_tokens, hidden_size = 37, 256
    x, args = _one_route_case(num_tokens, 8, hidden_size, 128, seed=491)

    def make_output():
        if output_kind == "none":
            return None
        if output_kind == "aligned":
            return torch.full((num_tokens, hidden_size), 777.0, dtype=torch.bfloat16, device="cuda")
        buffer = torch.full(
            (num_tokens * hidden_size + 1,), 777.0, dtype=torch.bfloat16, device="cuda"
        )
        return buffer[1:].view(num_tokens, hidden_size)

    def run(inp, output):
        result = hpc.fuse_moe_blockwise_fp8_sm120(
            inp, None, *args, num_expert_total=8, output=output
        )
        return result if output is None else output

    expected = run(x, make_output()).clone()
    output = make_output()
    actual = torch.compile(run, fullgraph=True, backend="inductor")(x, output)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if output is not None:
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [1, 200])
def test_unaligned_output_and_shared(num_tokens):
    torch.manual_seed(503)
    num_topk, num_experts, hidden_size, intermediate_size = 8, 8, 256, 128
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    gate_w, _, gate_packed, _, down_w, _, down_packed, _ = _fp8_weights(
        num_experts, hidden_size, intermediate_size
    )
    topk_ids, topk_scale = _make_routing(num_tokens, num_topk, num_experts, "cuda")
    shared = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

    def run(output, shared_output):
        return hpc.fuse_moe_blockwise_fp8_sm120(
            x,
            None,
            gate_w,
            gate_packed,
            down_w,
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


@pytest.mark.parametrize("num_tokens", [1, 37])
def test_misaligned_weights_rejected(num_tokens):
    x, args = _one_route_case(num_tokens, 8, 256, 128, seed=563)
    check_misaligned_weights_rejected(
        hpc.fuse_moe_blockwise_fp8_sm120, x, args[:4], args[4:], num_experts=8
    )
