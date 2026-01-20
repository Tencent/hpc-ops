import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import math
from pathlib import Path

import pytest
import torch

import hpc
from utils import allclose

# Set random seed for reproducibility
torch.manual_seed(41)
torch.cuda.manual_seed(41)


def naive_gather_expert_inputs(x, topk_ids, num_expert, rank_ep):
    num_tokens, num_topk = topk_ids.shape
    num_seq, hidden_size = x.shape

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
            token_pos[itoken, icol] = pos.item()
            num_tokens_per_expert[iexpert - start_expert] += 1

    return (
        y,
        token_pos,
        num_tokens_per_expert,
        cu_num_tokens_per_expert,
        unique_values,
    )


def naive_group_gemm(x, w, cu_seqlens, scale, expert_ids):

    m, k = x.shape
    num_group, n, _ = w.shape

    y = torch.zeros((m, n), dtype=torch.bfloat16, device=x.device)

    start_idx = 0
    for i in range(num_group):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()

        x_group = x[start_idx:end_idx]
        w_group = w[i]

        y_group = torch._scaled_mm(
            x_group,
            w_group.t(),
            scale_a=scale[i],
            scale_b=torch.ones((1), dtype=torch.float, device="cuda"),
            bias=None,
            out_dtype=torch.bfloat16,
        )
        y[start_idx:end_idx] = y_group
    return y


def naive_act_mul_and_quant(gate_up, scale):

    def silu(x):
        return x / (1 + (-x).exp())

    gate, up = torch.chunk(gate_up.float(), 2, dim=1)
    out = (silu(gate).to(torch.bfloat16) * up.to(torch.bfloat16)).float() * scale
    outfp8 = out.to(torch.float8_e4m3fn)
    return outfp8


def naive_reduce(x_bf16, topk_pos, topk_scale, shared_output=None):
    num_seq, num_topk = topk_pos.shape
    total_num_seq, hidden_size = x_bf16.shape

    y_bf16 = torch.zeros((num_seq, hidden_size), dtype=torch.bfloat16, device=x_bf16.device)
    for i in range(num_seq):
        y_bf16[i] = torch.sum(x_bf16[topk_pos[i]] * topk_scale[i].unsqueeze(1), dim=0)
        if shared_output is not None:
            y_bf16[i] += shared_output[i]

    return y_bf16


def naive_fuse_moe_pertensor_fp8(
    x,
    gate_up_weight,
    down_weight,
    gate_up_scale,
    down_scale,
    act_and_mul_scale,
    topk_ids,
    topk_scale,
    rank_ep,
    shared_output=None,
):
    num_expert = gate_up_weight.size(0)
    # count_and_gather
    gate_up_input, topk_pos, seqlens, cu_seqlens, expert_ids = naive_gather_expert_inputs(
        x, topk_ids, num_expert, rank_ep
    )

    # gate_up_proj
    gate_up_output = naive_group_gemm(
        gate_up_input, gate_up_weight, cu_seqlens, gate_up_scale, expert_ids
    )

    # act_and_mul
    down_input = naive_act_mul_and_quant(gate_up_output, act_and_mul_scale)

    # down_proj
    down_output = naive_group_gemm(down_input, down_weight, cu_seqlens, down_scale, expert_ids)

    # reduce
    y = naive_reduce(down_output, topk_pos, topk_scale, shared_output)

    return y


@pytest.mark.parametrize("num_seq", [128])
@pytest.mark.parametrize("num_topk", [8])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("intermediate_size", [512])
@pytest.mark.parametrize("num_expert", [128])
@pytest.mark.parametrize("rank_ep", [0, 1])
@pytest.mark.parametrize("size_ep", [1, 4, 8])
@pytest.mark.parametrize("has_shared_output", [False, True])
def test_fuse_moe_pertensor_fp8(
    num_seq,
    num_topk,
    hidden_size,
    intermediate_size,
    num_expert,
    rank_ep,
    size_ep,
    has_shared_output,
):
    dtype = torch.float8_e4m3fn

    topk_ids = torch.randint(0, num_expert, (num_seq, num_topk), dtype=torch.int32, device="cuda")
    topk_ids, _ = torch.sort(topk_ids, dim=1)

    x = (torch.randn((num_seq, hidden_size), dtype=torch.float, device="cuda") / 100).to(dtype)
    gate_up_weight = torch.randn(
        (num_expert // size_ep, intermediate_size * 2, hidden_size),
        dtype=torch.float,
        device="cuda",
    ).to(dtype)
    down_weight = torch.randn(
        (num_expert // size_ep, hidden_size, intermediate_size),
        dtype=torch.float,
        device="cuda",
    ).to(dtype)
    gate_up_scale = torch.randn((num_expert // size_ep), dtype=torch.float, device="cuda")
    down_scale = torch.randn((num_expert // size_ep), dtype=torch.float, device="cuda")
    act_and_mul_scale = torch.randn((1), dtype=torch.float, device="cuda")
    topk_scale = torch.randn((num_seq, num_topk), dtype=torch.float, device="cuda") / num_topk
    if has_shared_output:
        shared_output = torch.randn((num_seq, hidden_size), dtype=torch.bfloat16, device="cuda")
    else:
        shared_output = None

    my = hpc.fuse_moe_pertensor_fp8(
        x,
        gate_up_weight,
        down_weight,
        gate_up_scale,
        down_scale,
        act_and_mul_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert // size_ep,
        shared_output=shared_output,
    )
    gt = naive_fuse_moe_pertensor_fp8(
        x,
        gate_up_weight,
        down_weight,
        gate_up_scale,
        down_scale,
        act_and_mul_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        shared_output,
    )

    assert allclose(gt.to(torch.float32), my.to(torch.float32), rtol=0.08, atol=0.1)
