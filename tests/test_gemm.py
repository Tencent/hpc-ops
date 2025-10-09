import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math


def test_gemm():

    m = 512
    n = 2432
    n = 7168
    k = 7168
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

    assert gt.device == my.device
    assert gt.dtype == my.dtype
    assert gt.shape == my.shape
    assert torch.allclose(my.to(torch.float), gt.to(torch.float), rtol=0.08, atol=0.01)
