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


def naive_group_gemm_pertensor_fp8(x, w, seqlens, cu_seqlens, scale):

    m, k = x.shape
    num_group, n, _ = w.shape

    y = torch.zeros((m, n), dtype=torch.bfloat16, device=x.device)

    start_idx = 0
    for i in range(num_group):
        start_idx = int(cu_seqlens[i].item())
        end_idx = int(start_idx + seqlens[i].item())  # cu_seqlens[i + 1].item()
        if seqlens[i].item() == 0:
            continue

        x_group = x[start_idx:end_idx]
        w_group = w[i]

        y_group = torch._scaled_mm(
            x_group, w_group.t(), scale_a=scale, scale_b=scale, bias=None, out_dtype=torch.bfloat16
        )
        y[start_idx:end_idx] = y_group

        start_idx = end_idx

    return y


@pytest.mark.parametrize("num_group", [8])
@pytest.mark.parametrize("actual_m", [8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("m", [512])
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("k", [7168])
def test_group_gemm_pertensor_fp8(num_group, actual_m, m, n, k):
    dtype = torch.float8_e4m3fn

    seqlens = torch.full((num_group,), actual_m, dtype=torch.int32, device="cuda")

    total_seq = torch.sum(seqlens)
    mean_seq = int(total_seq / num_group)
    x = torch.randn((total_seq, k), dtype=torch.float, device="cuda").to(dtype)
    w = torch.randn((num_group, n, k), dtype=torch.float, device="cuda").to(dtype)
    scale = torch.tensor(1.0, dtype=torch.float, device="cuda")
    scale_hpc = torch.full((num_group,), 1.0, dtype=torch.float, device="cuda")

    cu_seqlens = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens]), dim=0
    ).to(torch.int32)

    gt = naive_group_gemm_pertensor_fp8(x, w, seqlens, cu_seqlens, scale)
    my = hpc.group_gemm_pertensor_fp8(
        x, w, seqlens, cu_seqlens, scale_hpc, num_seq_per_group_avg=mean_seq
    )

    abs_diff = torch.abs(gt - my)
    vals, idxs = torch.topk(abs_diff.view(-1), 10)
    idxs = [torch.unravel_index(idx, gt.shape) for idx in idxs]

    assert allclose(gt.to(torch.float32), my.to(torch.float32), rtol=0.08, atol=0.01)
