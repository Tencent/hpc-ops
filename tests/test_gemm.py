import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
from utils import allclose


import pytest


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 8, reason="skip non sm90")
@pytest.mark.parametrize("m", [512])
@pytest.mark.parametrize("n", [7168])
@pytest.mark.parametrize("k", [7168])
def test_gemm_fp8(m, n, k):
    dtype = torch.float8_e4m3fn

    x = torch.randn((m, k), dtype=torch.float, device="cuda").to(dtype)
    w = torch.randn((n, k), dtype=torch.float, device="cuda").to(dtype)
    scale = torch.tensor(1.0, dtype=torch.float, device="cuda")

    for _ in range(11):
        gt = torch._scaled_mm(
            x, w.t(), scale_a=scale, scale_b=scale, bias=None, out_dtype=torch.bfloat16
        )
        my = hpc.gemm(x, w)

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

    assert allclose(gt.to(torch.float32), my.to(torch.float32), rtol=0.08, atol=0.01)


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 10, reason="skip non sm100")
@pytest.mark.parametrize("m", [256])
@pytest.mark.parametrize("n", [192])
@pytest.mark.parametrize("k", [4096])
def test_gemm_bf16(m, n, k):
    dtype = torch.bfloat16

    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    x = torch.randn((m, k), dtype=torch.float, device="cuda").to(dtype)
    w = torch.randn((n, k), dtype=torch.float, device="cuda").to(dtype)

    gt = torch.matmul(x, w.t())
    my = hpc.gemm(x, w)

    print("gt")
    print(gt)

    print("my")
    print(my)

    assert allclose(gt.to(torch.float32), my.to(torch.float32), rtol=0.08, atol=0.01)
