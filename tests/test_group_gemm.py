import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
import pytest


def naive_group_gemm(x, w, cu_seqlens, scale):

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
            x_group, w_group.t(), scale_a=scale, scale_b=scale, bias=None, out_dtype=torch.bfloat16
        )
        y[start_idx:end_idx] = y_group

        start_idx = end_idx

    return y


@pytest.mark.parametrize("num_group", [16])
@pytest.mark.parametrize("m", [10])
@pytest.mark.parametrize("n", [8192])
@pytest.mark.parametrize("k", [4096])
def test_group_gemm1(num_group, m, n, k):
    torch.cuda.manual_seed(10086)
    dtype = torch.float8_e4m3fn

    # seqlens = torch.tensor([57, 59,  4, 11,  6,  9,  3,  6,  6,  6,  2,  6,  3,  6,  9,  6], device='cuda:0', dtype=torch.int32)
    seqlens = torch.full((num_group,), m, dtype=torch.int32, device="cuda")

    total_seq = torch.sum(seqlens)
    print(total_seq)
    x = torch.randn((total_seq, k), dtype=torch.float, device="cuda").to(dtype)
    w = torch.randn((num_group, n, k), dtype=torch.float, device="cuda").to(dtype)
    scale = torch.tensor(1.0, dtype=torch.float, device="cuda")
    scale_hpc = torch.full((num_group,), 1.0, dtype=torch.float, device="cuda")

    cu_seqlens = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens]), dim=0
    ).to(torch.int32)
    print(seqlens)

    for _ in range(1):
        gt = naive_group_gemm(x, w, cu_seqlens, scale)
        my = hpc.group_gemm_fp8(x, w, seqlens, cu_seqlens, scale_hpc)

    print("gt")
    print(gt[:10, 15])

    print("my")
    print(my[:10, 15])

    abs_diff = torch.abs(gt - my)
    vals, idxs = torch.topk(abs_diff.view(-1), 10)
    idxs = [torch.unravel_index(idx, gt.shape) for idx in idxs]

    for i, idx in enumerate(idxs):
        cpu_idx = tuple(tensor.cpu().item() for tensor in idx)
        print(
            "{:+.4f} vs {:+.4f} with diff = {:.4f}, @ {}".format(gt[idx], my[idx], vals[i], cpu_idx)
        )

    assert gt.device == my.device
    assert gt.dtype == my.dtype
    assert gt.shape == my.shape
    assert torch.allclose(my.to(torch.float), gt.to(torch.float), rtol=0.08, atol=0.01)


file_available = os.path.exists("/cfs_cloud_code/theocheng/hpc-ops/group_seqlens.pt")


@pytest.mark.skipif(not file_available, reason="group_seqlens.pt does not exists!!!")
@pytest.mark.parametrize("num_group", [16])
@pytest.mark.parametrize("n", [8192])
@pytest.mark.parametrize("k", [4096])
def test_group_gemm2(num_group, n, k):
    torch.cuda.manual_seed(10086)
    dtype = torch.float8_e4m3fn

    group_seqlens = torch.load("/cfs_cloud_code/theocheng/hpc-ops/group_seqlens.pt")
    print(group_seqlens.shape)

    seqlens_all = group_seqlens.flatten().reshape(-1, num_group)

    for i in range(len(seqlens_all)):
        if i == 0:
            seqlens = seqlens_all[i]
            print("i:", i, " seqlens:", seqlens)

            total_seq = torch.sum(seqlens)
            x = torch.randn((total_seq, k), dtype=torch.float, device="cuda").to(dtype)
            w = torch.randn((num_group, n, k), dtype=torch.float, device="cuda").to(dtype)
            scale = torch.tensor(1.0, dtype=torch.float, device="cuda")
            scale_hpc = torch.full((num_group,), 1.0, dtype=torch.float, device="cuda")

            cu_seqlens = torch.cumsum(
                torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens]), dim=0
            ).to(torch.int32)

            torch.cuda.synchronize()

            for _ in range(1):
                gt = naive_group_gemm(x, w, cu_seqlens, scale)
                my = hpc.group_gemm_fp8(x, w, seqlens, cu_seqlens, scale_hpc)
                torch.cuda.synchronize()

            print("gt")
            print(gt[:10, 15])

            print("my")
            print(my[:10, 15])

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

            assert gt.device == my.device
            assert gt.dtype == my.dtype
            assert gt.shape == my.shape
            assert torch.allclose(my.to(torch.float), gt.to(torch.float), rtol=0.08, atol=0.01)
