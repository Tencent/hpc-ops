import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import torch
import pytest
import hpc
from utils import allclose

torch.manual_seed(41)
torch.cuda.manual_seed(41)


def naive_group_gemm_bf16(x, w, seqlens, cu_seqlens):
    m, k = x.shape
    num_group, n, _ = w.shape

    y = torch.zeros((m, n), dtype=torch.bfloat16, device=x.device)

    for i in range(num_group):
        start_idx = int(cu_seqlens[i].item())
        end_idx = int(cu_seqlens[i + 1].item())

        if seqlens[i].item() == 0:
            continue

        x_group = x[start_idx:end_idx]  # [M_i, K]
        w_group = w[i]  # [N, K]

        # y_group = x_group @ w_group.T
        y_group = torch.matmul(x_group, w_group.t())

        y[start_idx:end_idx] = y_group

    return y


@pytest.mark.parametrize("num_group", [8])
@pytest.mark.parametrize("actual_m", [8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("m", [512])
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("k", [7168])
def test_group_gemm_bf16(num_group, actual_m, m, n, k):
    dtype = torch.bfloat16

    seqlens = torch.full((num_group,), actual_m, dtype=torch.int32, device="cuda")
    total_seq = torch.sum(seqlens)
    mean_seq = int(total_seq / num_group)

    x = torch.randn((total_seq, k), dtype=dtype, device="cuda")
    w = torch.randn((num_group, n, k), dtype=dtype, device="cuda")

    cu_seqlens = torch.zeros(num_group + 1, dtype=torch.int32, device="cuda")
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    gt = naive_group_gemm_bf16(x, w, seqlens, cu_seqlens)

    my = hpc.group_gemm_bf16(x, w, seqlens, cu_seqlens, num_seq_per_group_avg=mean_seq)

    assert allclose(gt.to(torch.float32), my.to(torch.float32), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_group_gemm_bf16(8, 128, 512, 4096, 7168)
