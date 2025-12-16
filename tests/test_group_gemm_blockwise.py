import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
import pytest


def naive_group_gemm(x, w, seqlens, cu_seqlens, xscale, wscale):

    m, k = x.shape
    num_group, n, _ = w.shape

    m_pergroup = m // num_group

    y = torch.zeros((m, n), dtype=torch.bfloat16, device=x.device)

    xscale = (xscale.repeat_interleave(128, dim=0).permute(1, 0).reshape((num_group, -1, k)))[
        :, :m_pergroup, :
    ].reshape(-1, k)
    print(xscale.shape)
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
# @pytest.mark.parametrize("actual_m", [16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("actual_m", [30])
@pytest.mark.parametrize("m", [32])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("k", [4096])
def test_group_gemm1(num_group, actual_m, m, n, k):
    torch.cuda.manual_seed(10086)
    dtype = torch.float8_e4m3fn

    seqlens = torch.full((num_group,), actual_m, dtype=torch.int32, device="cuda")

    total_seq = torch.sum(seqlens)
    total_seq_pad = m * num_group
    mean_seq = int(total_seq / num_group)
    print(total_seq, mean_seq)
    x = (torch.randn((total_seq, k), dtype=torch.float, device="cuda") / 10).to(dtype)
    w = (torch.randn((num_group, n, k), dtype=torch.float, device="cuda") / 10).to(dtype)
    xscale = torch.randn((k // 128, total_seq_pad), dtype=torch.float, device="cuda")
    wscale = torch.randn((num_group, n // 128, 64), dtype=torch.float, device="cuda")

    cu_seqlens = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens]), dim=0
    ).to(torch.int32)
    print(seqlens)
    print(cu_seqlens)

    for _ in range(10):
        gt = naive_group_gemm(x, w, seqlens, cu_seqlens, xscale, wscale)
        my = hpc.group_gemm_blockwise_fp8(
            x, w, seqlens, cu_seqlens, xscale, wscale, num_seq_per_group_avg=mean_seq
        )
        my1 = hpc.group_gemm_fp8(x, w, seqlens, cu_seqlens, xscale, num_seq_per_group_avg=mean_seq)

    print("gt")
    print(gt[:5, -10:])

    print("my")
    print(my[:5, -10:])

    abs_diff = torch.abs(gt - my)
    vals, idxs = torch.topk(abs_diff.view(-1), 20)
    idxs = [torch.unravel_index(idx, gt.shape) for idx in idxs]

    for i, idx in enumerate(idxs):
        cpu_idx = tuple(tensor.cpu().item() for tensor in idx)
        print(
            "{:+.4f} vs {:+.4f} with diff = {:.4f}, @ {}".format(gt[idx], my[idx], vals[i], cpu_idx)
        )

    assert gt.device == my.device
    assert gt.dtype == my.dtype
    assert gt.shape == my.shape
    assert torch.allclose(my.to(torch.float), gt.to(torch.float), rtol=0.08, atol=0.1)
