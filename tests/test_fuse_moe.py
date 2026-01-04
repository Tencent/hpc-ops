import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
import pytest
from pathlib import Path
from utils import allclose


def naive_gather_expert_inputs(x, topk_ids):
    num_tokens, num_topk = topk_ids.shape
    num_seq, hidden_size = x.shape

    unique_values, seqlens = torch.unique(topk_ids.flatten(), return_counts=True, sorted=True)
    cu_seqlens = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens]), dim=0
    ).to(torch.int32)

    topk_ids_flat = topk_ids.reshape(-1)

    sorted_expert_ids, sort_indices = torch.sort(topk_ids_flat, dim=-1)

    sort_token_indices = sort_indices // num_topk
    token_pos, _ = torch.sort(
        torch.argsort(sort_token_indices).reshape(num_tokens, num_topk), dim=1
    )

    y = x[sort_token_indices]
    return y, token_pos, seqlens, cu_seqlens, unique_values


def naive_group_gemm(x, w, cu_seqlens, scale, expert_ids):

    m, k = x.shape
    _, n, _ = w.shape
    num_group = len(cu_seqlens) - 1

    y = torch.zeros((m, n), dtype=torch.bfloat16, device=x.device)

    start_idx = 0
    for i in range(num_group):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()

        x_group = x[start_idx:end_idx]
        w_group = w[expert_ids[i]]  # w[i]

        y_group = torch._scaled_mm(
            x_group,
            w_group.t(),
            scale_a=scale[expert_ids[i]],
            scale_b=torch.ones((1), dtype=torch.float, device="cuda"),
            bias=None,
            out_dtype=torch.bfloat16,
        )
        y[start_idx:end_idx] = y_group

        start_idx = end_idx

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


def naive_fuse_moe(
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
    # count_and_gather
    gate_up_input, topk_pos, seqlens, cu_seqlens, expert_ids = naive_gather_expert_inputs(
        x, topk_ids
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


@pytest.mark.parametrize("num_seq", [8, 128])
@pytest.mark.parametrize("num_topk", [8])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("intermediate_size", [512])
@pytest.mark.parametrize("num_expert", [128])
@pytest.mark.parametrize("rank_ep", [0])
@pytest.mark.parametrize("has_shared_output", [False, True])
def test_fuse_moe(
    num_seq, num_topk, hidden_size, intermediate_size, num_expert, rank_ep, has_shared_output
):
    torch.manual_seed(0)
    dtype = torch.float8_e4m3fn

    topk_ids = torch.randint(0, num_expert, (num_seq, num_topk), dtype=torch.int32, device="cuda")
    topk_ids, _ = torch.sort(topk_ids, dim=1)
    print(topk_ids)

    x = (torch.randn((num_seq, hidden_size), dtype=torch.float, device="cuda") / 100).to(dtype)
    gate_up_weight = torch.randn(
        (num_expert, intermediate_size * 2, hidden_size), dtype=torch.float, device="cuda"
    ).to(dtype)
    down_weight = torch.randn(
        (num_expert, hidden_size, intermediate_size),
        dtype=torch.float,
        device="cuda",
    ).to(dtype)
    gate_up_scale = torch.randn((num_expert), dtype=torch.float, device="cuda")
    down_scale = torch.randn((num_expert), dtype=torch.float, device="cuda")
    act_and_mul_scale = torch.randn((1), dtype=torch.float, device="cuda")
    topk_scale = torch.randn((num_seq, num_topk), dtype=torch.float, device="cuda") / num_topk
    if has_shared_output:
        shared_output = torch.randn((num_seq, hidden_size), dtype=torch.bfloat16, device="cuda")
    else:
        shared_output = None
    for _ in range(1):
        my = hpc.fuse_moe(
            x,
            gate_up_weight,
            down_weight,
            gate_up_scale,
            down_scale,
            act_and_mul_scale,
            topk_ids,
            topk_scale,
            rank_ep,
            num_expert,
            shared_output=shared_output,
        )
        gt = naive_fuse_moe(
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

        # torch.cuda.synchronize()

    print("gt")
    print(gt)

    print("my")
    print(my)

    abs_diff = torch.abs(gt - my)
    vals, idxs = torch.topk(abs_diff.view(-1), 10)
    idxs = [torch.unravel_index(idx, gt.shape) for idx in idxs]

    for i, idx in enumerate(idxs):
        cpu_idx = tuple(tensor.cpu().item() for tensor in idx)
        print(
            "{:+.4f} vs {:+.4f} with diff = {:.4f}, @ {}".format(gt[idx], my[idx], vals[i], cpu_idx)
        )

    assert gt.device == my.device
    assert gt.shape == my.shape
    assert allclose(gt.to(torch.float32), my.to(torch.float32), rtol=0.08, atol=0.1)


file_available = os.path.exists("/cfs_cloud_code/theocheng/fused_moe_topk")


@pytest.mark.skipif(not file_available, reason="fused_moe_topk files does not exists!!!")
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("intermediate_size", [4096])
@pytest.mark.parametrize("num_expert", [128])
@pytest.mark.parametrize("rank_ep", [0])
def test_fuse_moe_realdata(hidden_size, intermediate_size, num_expert, rank_ep):
    dtype = torch.float8_e4m3fn

    path = Path("/cfs_cloud_code/theocheng/fused_moe_topk")

    for ifile, file_path in enumerate(path.rglob("*")):
        if ifile < 1:
            topk_ids = torch.load(file_path)
            topk_ids, _ = torch.sort(topk_ids, dim=1)
            print("ifile:", ifile, " file_path:", file_path)

            num_seq = topk_ids.size(0)
            num_topk = topk_ids.size(1)
            print(num_seq, num_topk)

            x = (torch.randn((num_seq, hidden_size), dtype=torch.float, device="cuda") / 1000).to(
                dtype
            )
            gate_up_weight = torch.randn(
                (num_expert, intermediate_size * 2, hidden_size), dtype=torch.float, device="cuda"
            ).to(dtype)
            down_weight = torch.randn(
                (num_expert, hidden_size, intermediate_size),
                dtype=torch.float,
                device="cuda",
            ).to(dtype)
            gate_up_scale = torch.randn((num_expert), dtype=torch.float, device="cuda")
            down_scale = torch.randn((num_expert), dtype=torch.float, device="cuda")
            act_and_mul_scale = torch.randn((1), dtype=torch.float, device="cuda")
            topk_scale = (
                torch.randn((num_seq, num_topk), dtype=torch.float, device="cuda") / num_topk
            )

            for _ in range(1):
                unique_values, counts = torch.unique(
                    topk_ids.flatten(), return_counts=True, sorted=True
                )
                if counts.size(0) == num_expert:
                    gt = naive_fuse_moe(
                        x,
                        gate_up_weight,
                        down_weight,
                        gate_up_scale,
                        down_scale,
                        act_and_mul_scale,
                        topk_ids,
                        topk_scale,
                        rank_ep,
                    )
                    my = hpc.fuse_moe(
                        x,
                        gate_up_weight,
                        down_weight,
                        gate_up_scale,
                        down_scale,
                        act_and_mul_scale,
                        topk_ids,
                        topk_scale,
                        rank_ep,
                        num_expert,
                    )

                    torch.cuda.synchronize()

                    print("gt")
                    print(gt)

                    print("my")
                    print(my)

                    abs_diff = torch.abs(gt - my)
                    vals, idxs = torch.topk(abs_diff.view(-1), 10)
                    idxs = [torch.unravel_index(idx, gt.shape) for idx in idxs]

                    for i, idx in enumerate(idxs):
                        cpu_idx = tuple(tensor.cpu().item() for tensor in idx)
                        print(
                            "{:+.4f} vs {:+.4f} with diff = {:.4f}, @ {}".format(
                                gt[idx], my[idx], vals[i], cpu_idx
                            )
                        )

                    assert allclose(gt.to(torch.float32), my.to(torch.float), rtol=0.08, atol=0.1)
