import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import math

import pytest
import torch

import hpc
from utils import allclose

# Set random seed for reproducibility
torch.manual_seed(41)
torch.cuda.manual_seed(41)


def naive_group_gemm(x, w, seqlens, cu_seqlens, xscale, wscale):

    m, k = x.shape
    num_group, n, _ = w.shape

    m_pergroup = m // num_group

    y = torch.zeros((m, n), dtype=torch.bfloat16, device=x.device)

    xscale = (xscale.repeat_interleave(128, dim=0).permute(1, 0).reshape((num_group, -1, k)))[
        :, :m_pergroup, :
    ].reshape(-1, k)
    wscale = wscale.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)[:, :, :k]
    x = (x.to(torch.bfloat16) * xscale).to(torch.bfloat16)
    w = (w.to(torch.bfloat16) * wscale).to(torch.bfloat16)

    for i in range(num_group):
        start_idx = int(cu_seqlens[i].item())
        end_idx = int(start_idx + seqlens[i].item())  # cu_seqlens[i + 1].item()
        if seqlens[i].item() == 0:
            continue

        x_group = x[start_idx:end_idx]
        w_group = w[i]

        y[start_idx:end_idx] = x_group @ w_group.t()

    return y


@pytest.mark.parametrize("num_group", [128])
@pytest.mark.parametrize("actual_m", [30])
@pytest.mark.parametrize("m", [32])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("k", [4096])
def test_group_gemm1(num_group, actual_m, m, n, k):
    dtype = torch.float8_e4m3fn

    seqlens = torch.full((num_group,), actual_m, dtype=torch.int32, device="cuda")

    total_seq = torch.sum(seqlens)
    total_seq_pad = m * num_group
    mean_seq = int(total_seq / num_group)
    x = (torch.randn((total_seq, k), dtype=torch.float, device="cuda") / 10).to(dtype)
    w = (torch.randn((num_group, n, k), dtype=torch.float, device="cuda") / 10).to(dtype)
    xscale = torch.randn((k // 128, total_seq_pad), dtype=torch.float, device="cuda")
    wscale = torch.randn((num_group, n // 128, k // 128), dtype=torch.float, device="cuda")

    cu_seqlens = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens]), dim=0
    ).to(torch.int32)

    gt = naive_group_gemm(x, w, seqlens, cu_seqlens, xscale, wscale)
    my = hpc.group_gemm_blockwise_fp8(
        x, w, seqlens, cu_seqlens, xscale, wscale, num_seq_per_group_avg=mean_seq
    )

    abs_diff = torch.abs(gt - my)
    vals, idxs = torch.topk(abs_diff.view(-1), 20)
    idxs = [torch.unravel_index(idx, gt.shape) for idx in idxs]

    assert gt.device == my.device
    assert gt.dtype == my.dtype
    assert gt.shape == my.shape
    assert torch.allclose(my.to(torch.float), gt.to(torch.float), rtol=0.08, atol=0.1)


@pytest.mark.parametrize("num_group", [256])
@pytest.mark.parametrize("actual_m", [30])
@pytest.mark.parametrize("m", [1280])
@pytest.mark.parametrize("k", [4096])
def test_reformat_x_scale(num_group, actual_m, m, k):
    total_seq_pad = m * num_group
    xscale = torch.rand((total_seq_pad, k // 128), dtype=torch.float, device="cuda")

    # ref
    seqlens = torch.full((num_group,), actual_m, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.cumsum(
        torch.tensor(
            [0]
            + [
                m,
            ]
            * num_group,
            device="cuda",
        ),
        dim=0,
    ).to(torch.int32)
    total_seq = torch.sum(seqlens)
    mean_seq = int(total_seq / num_group)

    def get_tilem(avg_bs):
        if avg_bs <= 16:
            return 16
        elif avg_bs <= 32:
            return 32
        else:
            return 64

    tilem = get_tilem(mean_seq)
    assert m % tilem == 0, "m must be divided by TileM"
    # transpose, pad to align TileM and compact arrangement for x scale per group
    ref_new_x_scale = torch.zeros((k // 128, total_seq_pad), dtype=torch.float, device="cuda")
    current_m = 0
    for i in range(num_group):
        seqlen_pad = (seqlens[i] + tilem - 1) // tilem * tilem
        ref_new_x_scale[:, current_m : current_m + seqlens[i]] = xscale[
            cu_seqlens[i] : cu_seqlens[i] + seqlens[i], :
        ].t()
        current_m += seqlen_pad

    # real
    real_new_x_scale = torch.zeros((k // 128, total_seq_pad), dtype=torch.float, device="cuda")
    real_new_x_scale = hpc.reformat_x_scale(xscale, seqlens, cu_seqlens, mean_seq, real_new_x_scale)

    # check
    ref_valid_list = []
    real_valid_list = []
    current_m = 0
    for i in range(num_group):
        seqlen_pad = (seqlens[i] + tilem - 1) // tilem * tilem
        ref_valid_list.append(ref_new_x_scale[:, current_m : current_m + seqlens[i]])
        real_valid_list.append(real_new_x_scale[:, current_m : current_m + seqlens[i]])
        current_m += seqlen_pad

    ref_valid_x_scale = torch.cat(ref_valid_list, dim=1)
    real_valid_x_scale = torch.cat(real_valid_list, dim=1)

    assert allclose(ref_valid_x_scale, real_valid_x_scale, rtol=1e-5, atol=1e-5)


def naive_deepep_input_format_group_gemm_blockwise_fp8(x, w, seqlens, cu_seqlens, xscale, wscale):
    m, k = x.shape
    num_group, n, _ = w.shape
    m_pergroup = m // num_group
    y = torch.zeros((m, n), dtype=torch.bfloat16, device=x.device)
    xscale = xscale.repeat_interleave(128, dim=1)
    wscale = wscale.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)[:, :, :k]
    x = (x.to(torch.bfloat16) * xscale).to(torch.bfloat16)
    w = (w.to(torch.bfloat16) * wscale).to(torch.bfloat16)

    for i in range(num_group):
        start_idx = int(cu_seqlens[i].item())
        end_idx = int(start_idx + seqlens[i].item())
        if seqlens[i].item() == 0:
            continue

        x_group = x[start_idx:end_idx]
        w_group = w[i]

        y[start_idx:end_idx] = x_group @ w_group.t()

    return y


@pytest.mark.parametrize("num_group", [256])
@pytest.mark.parametrize("actual_m", [30])
@pytest.mark.parametrize("m", [1280])
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("k", [4096])
def test_deepep_input_fromat_group_gemm_blockwise_fp8(num_group, actual_m, m, n, k):
    dtype = torch.float8_e4m3fn

    seqlens = torch.full((num_group,), actual_m, dtype=torch.int32, device="cuda")

    total_seq = torch.sum(seqlens)
    mean_seq = int(total_seq / num_group)
    total_seq_pad = m * num_group
    x = (torch.randn((total_seq_pad, k), dtype=torch.float, device="cuda") / 10).to(dtype)
    w = (torch.randn((num_group, n, k), dtype=torch.float, device="cuda") / 10).to(dtype)
    # need to be transpose, pad to align TileM and compact arrangement
    xscale = torch.randn((total_seq_pad, k // 128), dtype=torch.float, device="cuda")
    wscale = torch.randn((num_group, n // 128, k // 128), dtype=torch.float, device="cuda")
    cu_seqlens = torch.cumsum(
        torch.tensor(
            [0]
            + [
                m,
            ]
            * num_group,
            device="cuda",
        ),
        dim=0,
    ).to(torch.int32)

    gt = naive_deepep_input_format_group_gemm_blockwise_fp8(
        x, w, seqlens, cu_seqlens, xscale, wscale
    )
    new_x_scale = torch.zeros((k // 128, total_seq_pad), dtype=torch.float, device="cuda")
    new_x_scale = hpc.reformat_x_scale(xscale, seqlens, cu_seqlens, mean_seq, new_x_scale)
    my = hpc.group_gemm_blockwise_fp8(
        x, w, seqlens, cu_seqlens, new_x_scale, wscale, num_seq_per_group_avg=mean_seq
    )

    # check
    gt = gt.reshape(num_group, m, n)
    my = my.reshape(num_group, m, n)
    valid_gt = torch.cat([gt[i, :num] for i, num in enumerate(seqlens)], dim=0)
    valid_my = torch.cat([my[i, :num] for i, num in enumerate(seqlens)], dim=0)

    assert valid_gt.device == valid_my.device
    assert valid_gt.dtype == valid_my.dtype
    assert valid_gt.shape == valid_my.shape
    assert allclose(valid_gt.to(torch.float), valid_my.to(torch.float), rtol=0.01, atol=0.05)
