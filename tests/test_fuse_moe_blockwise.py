import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import math
from pathlib import Path
from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

import hpc
from utils import allclose

# Set random seed for reproducibility
torch.manual_seed(41)
torch.cuda.manual_seed(41)


def naive_gather_expert_inputs(x, x_scale, topk_ids, num_expert, rank_ep):
    num_tokens, num_topk = topk_ids.shape
    num_tokens, hidden_size = x.shape

    unique_values, num_tokens_per_expert_partial = torch.unique(
        topk_ids.flatten(), return_counts=True, sorted=True
    )
    start_expert = rank_ep * num_expert
    end_expert = (rank_ep + 1) * num_expert
    mask = (unique_values >= start_expert) & (unique_values < end_expert)
    unique_values = unique_values[mask]
    num_tokens_per_expert_partial = num_tokens_per_expert_partial[mask]
    num_tokens_per_expert = torch.full([num_expert], 0, dtype=torch.int32, device="cuda")

    for i in range(unique_values.numel()):
        num_tokens_per_expert[unique_values[i] - start_expert] = num_tokens_per_expert_partial[i]

    cu_num_tokens_per_expert = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), num_tokens_per_expert]),
        dim=0,
    ).to(torch.int32)

    y = torch.zeros((num_tokens * num_topk, hidden_size), dtype=x.dtype, device=x.device)
    y_scale = torch.zeros(
        (num_tokens * num_topk, x_scale.size(1)), dtype=torch.float32, device=x_scale.device
    )
    token_pos = torch.zeros((num_tokens, num_topk), dtype=torch.int32, device=x.device)
    token_pos.fill_(-1)

    # reset
    num_tokens_per_expert.fill_(0)

    for idx, iexpert in enumerate(topk_ids.flatten()):
        itoken = idx // num_topk
        icol = idx % num_topk
        if iexpert >= start_expert and iexpert < end_expert:
            pos = (
                cu_num_tokens_per_expert[iexpert - start_expert]
                + num_tokens_per_expert[iexpert - start_expert]
            )
            y[pos] = x[itoken]
            y_scale[pos] = x_scale[itoken]
            token_pos[itoken, icol] = pos.item()
            num_tokens_per_expert[iexpert - start_expert] += 1

    return (
        y,
        y_scale,
        token_pos,
        num_tokens_per_expert,
        cu_num_tokens_per_expert,
        unique_values,
    )


def naive_group_gemm(x, w, num_tokens_per_expert, cu_num_tokens_per_expert, xscale, wscale):

    m, k = x.shape
    num_group, n, _ = w.shape

    m_pergroup = m // num_group

    y = torch.zeros((m, n), dtype=torch.bfloat16, device=x.device)

    for i in range(num_group):
        start_idx = int(cu_num_tokens_per_expert[i].item())
        end_idx = int(start_idx + num_tokens_per_expert[i].item())
        if num_tokens_per_expert[i].item() == 0:
            continue

        x_group = x[start_idx:end_idx]  # (m, k)
        w_group = w[i]  # (n, k)

        m = x_group.size(0)
        n = w_group.size(0)
        k = x_group.size(1)

        x_scale_group = xscale[start_idx:end_idx]  # (m, k / 128)
        w_scale_group = wscale[i]  # (n/128, 64)
        w_scale_group = w_scale_group[:, : k // 128]

        assert x_group.size(1) == w_group.size(1)
        assert w_scale_group.size(0) == w_group.size(0) // 128
        assert w_scale_group.size(1) == w_group.size(1) // 128

        output = torch.zeros((m, n), dtype=torch.float32, device=x_group.device)

        num_tile_n = n // 128
        num_tile_k = k // 128

        x_chunks = torch.chunk(x_group, num_tile_k, dim=1)
        w_chunks = torch.chunk(w_group, num_tile_k, dim=1)

        for n in range(num_tile_n):
            tmp = torch.zeros((m, 128), dtype=torch.float32, device=x_group.device)
            for k in range(num_tile_k):
                x_i = x_chunks[k]  # （m, 128)
                x_scale_i = x_scale_group[:, k]  # (m, 1)
                w_scale_i = w_scale_group[n, k]  # (1)
                assert x_scale_i.size(0) == x_i.size(0)
                assert w_scale_i.numel() == 1

                w_chunk = w_chunks[k]  # (n, 128)
                w_i = w_chunk[n * 128 : (n + 1) * 128].t()  # (128, 128)

                scale_a = torch.tensor([1.0], dtype=torch.float32, device="cuda")
                scale_b = torch.tensor([1.0], dtype=torch.float32, device="cuda")

                gemm_output = torch._scaled_mm(x_i, w_i, scale_a, scale_b, out_dtype=torch.float32)
                tmp += gemm_output * x_scale_i.unsqueeze(1) * w_scale_i
            output[:, n * 128 : (n + 1) * 128] = tmp

        y[start_idx:end_idx] = output.to(torch.bfloat16)
    return y


def naive_act_mul_and_blockwise_quant(gate_up_out):

    def _quantize_blockwise_fp8(
        tensor: torch.Tensor,
        block_size: int = 128,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        compute_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert tensor.dim() == 2, "Tensor must be 2D"

        batch_size, num_features = tensor.shape
        num_blocks = (num_features + block_size - 1) // block_size

        fp8_max = 448.0

        quantized = torch.empty(batch_size, num_features, device=tensor.device, dtype=fp8_dtype)

        scales = torch.empty(batch_size, num_blocks, device=tensor.device, dtype=compute_dtype)

        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = min(start + block_size, num_features)

            if start >= end:
                continue

            block = tensor[:, start:end]  # (batch, block_features)

            block_abs_max = block.abs().amax(dim=1, keepdim=True)  # (batch, 1)

            scale = block_abs_max / fp8_max
            inv_scale = 1.0 / (scale + 1e-8)

            # save scale
            scales[:, block_idx] = scale.squeeze(1)

            # quant
            quantized_block = (block * inv_scale).to(fp8_dtype)
            quantized[:, start:end] = quantized_block

        return quantized, scales

    def silu(x):
        return x / (1 + (-x).exp())

    # silu and mul
    gate_up = gate_up_out.float()
    gate, up = torch.chunk(gate_up.float(), 2, dim=1)
    out = silu(gate) * up

    # block wise quant
    outfp8, out_scale = _quantize_blockwise_fp8(out)
    return outfp8, out_scale


def naive_reduce(x_bf16, topk_pos, topk_scale, shared_output=None):
    num_tokens, num_topk = topk_pos.shape
    total_num_tokens, hidden_size = x_bf16.shape

    y_bf16 = torch.zeros((num_tokens, hidden_size), dtype=torch.bfloat16, device=x_bf16.device)
    for i in range(num_tokens):
        acc = torch.zeros((1, hidden_size), dtype=torch.float, device="cuda")
        cur_topk_pos = topk_pos[i]
        cur_topk_scale = topk_scale[i]
        for j, pos in enumerate(cur_topk_pos):
            if pos >= 0:
                acc += x_bf16[pos].float() * cur_topk_scale[j].float()
        if shared_output is not None:
            acc += shared_output[i].float()
        y_bf16[i] = acc.to(torch.bfloat16)
    return y_bf16


def naive_fuse_moe_blockwise_fp8(
    x,
    x_scale,
    gate_up_weight,
    gate_up_weight_scale,
    down_weight,
    down_weight_scale,
    topk_ids,
    topk_scale,
    rank_ep,
    num_expert,
    shared_output=None,
):
    num_expert_local = gate_up_weight.size(0)
    # count_and_gather
    (
        gate_up_input,
        gate_up_input_scale,
        topk_pos,
        num_tokens_per_expert,
        cu_num_tokens_per_expert,
        expert_ids,
    ) = naive_gather_expert_inputs(x, x_scale, topk_ids, num_expert_local, rank_ep)

    # gate_up_proj
    gate_up_output = naive_group_gemm(
        gate_up_input,
        gate_up_weight,
        num_tokens_per_expert,
        cu_num_tokens_per_expert,
        gate_up_input_scale,
        gate_up_weight_scale,
    )

    # act_and_mul
    down_input, down_input_scale = naive_act_mul_and_blockwise_quant(gate_up_output)

    # down_proj
    down_output = naive_group_gemm(
        down_input,
        down_weight,
        num_tokens_per_expert,
        cu_num_tokens_per_expert,
        down_input_scale,
        down_weight_scale,
    )

    # reduce
    y = naive_reduce(down_output, topk_pos, topk_scale, shared_output)

    return y


@pytest.mark.parametrize("num_tokens", [128])
@pytest.mark.parametrize("num_topk", [8])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("intermediate_size", [512, 256])
@pytest.mark.parametrize("num_expert", [128])
@pytest.mark.parametrize("rank_ep", [0, 1])
@pytest.mark.parametrize("size_ep", [1, 4, 8])
@pytest.mark.parametrize("has_shared_output", [False, True])
def test_fuse_moe_blockwise_fp8(
    num_tokens,
    num_topk,
    hidden_size,
    intermediate_size,
    num_expert,
    rank_ep,
    size_ep,
    has_shared_output,
):
    dtype = torch.float8_e4m3fn

    topk_ids = torch.randint(
        0, num_expert, (num_tokens, num_topk), dtype=torch.int32, device="cuda"
    )
    topk_ids, _ = torch.sort(topk_ids, dim=1)
    topk_scale = torch.randn((num_tokens, num_topk), dtype=torch.float, device="cuda") / num_topk

    x = (torch.randn((num_tokens, hidden_size), dtype=torch.float, device="cuda") / 100).to(dtype)
    x_scale = torch.randn((num_tokens, hidden_size // 128), dtype=torch.float, device="cuda")
    gate_up_weight = torch.randn(
        (num_expert // size_ep, intermediate_size * 2, hidden_size),
        dtype=torch.float,
        device="cuda",
    ).to(dtype)
    gate_up_weight_scale = torch.randn(
        (num_expert // size_ep, intermediate_size * 2 // 128, (hidden_size // 128 + 3) // 4 * 4),
        dtype=torch.float,
        device="cuda",
    )
    down_weight = torch.randn(
        (num_expert // size_ep, hidden_size, intermediate_size),
        dtype=torch.float,
        device="cuda",
    ).to(dtype)
    down_weight_scale = torch.randn(
        (num_expert // size_ep, hidden_size // 128, (intermediate_size // 128 + 3) // 4 * 4),
        dtype=torch.float,
        device="cuda",
    )
    if has_shared_output:
        shared_output = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    else:
        shared_output = None

    my = hpc.fuse_moe_blockwise_fp8(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert,
        shared_output,
    )
    gt = naive_fuse_moe_blockwise_fp8(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert,
        shared_output,
    )

    torch.cuda.synchronize()

    assert allclose(gt.to(torch.float32), my.to(torch.float32), rtol=0.01, atol=0.01)
