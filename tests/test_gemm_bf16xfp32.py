import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import pytest
from utils import allclose


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 9, reason="skip on non sm90!")
@pytest.mark.parametrize(
    "m",
    [1, 6, 16, 32, 48, 64, 96, 144, 208, 304, 416, 624, 832, 1024, 2048, 4096, 12303, 32768],
)
@pytest.mark.parametrize("n", [192, 512, 1024, 2048])
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("use_fp32_output", [True, False])
@pytest.mark.parametrize("use_split_flag", [True, False])
def test_gemm_bf16xfp32_sm90(m, n, k, use_fp32_output, use_split_flag):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    dtype = torch.bfloat16

    x = torch.randn((m, k), dtype=torch.float, device="cuda").to(dtype)
    w = torch.randn((n, k), dtype=torch.float, device="cuda")

    scale = 1 / 256
    w_high = w.to(torch.bfloat16)
    w_low = ((w - w_high.float()) / scale).to(torch.bfloat16)

    split_flag = None
    if use_split_flag:
        split_flag = hpc.get_gemm_bf16xfp32_workspace(n)

    gt = torch.matmul(x.float(), w.t())

    my = hpc.gemm_bf16xfp32(x, w_high, w_low, scale, use_fp32_output, True, split_flag)

    if use_split_flag:
        assert (split_flag == 0).all()

    assert allclose(gt, my.float(), rtol=0.08, atol=0.01)
