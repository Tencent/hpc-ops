import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
import pytest
import torch.nn.functional as F
from utils import allclose

has_deep_gemm = True
try:
    import deep_gemm
except:
    has_deep_gemm = False


def align_to(m, target):
    return (m + target - 1) // target * target


@pytest.mark.parametrize("m", [9614])
@pytest.mark.parametrize("n", [13824])
@pytest.mark.parametrize("k", [5120])
@pytest.mark.skipif(not has_deep_gemm, reason="This test need install deepgemm")
def test_gemm_blockwise(m, n, k):
    dtype = torch.float8_e4m3fn

    x = torch.randn((m, k), dtype=torch.float, device="cuda").to(dtype)
    w = torch.randn((n, k), dtype=torch.float, device="cuda").to(dtype)
    x_scale = torch.rand((m, k // 128), dtype=torch.float, device="cuda")
    w_scale = torch.rand((n // 128, k // 128), dtype=torch.float, device="cuda")
    bias = torch.rand((n), dtype=torch.bfloat16, device="cuda")
    output = torch.randn((m, n), dtype=torch.bfloat16, device="cuda")

    for i in range(10):
        # deep_gemm + element_wise add
        deep_gemm.fp8_gemm_nt(
            a=(x, x_scale), b=(w, w_scale), d=output, c=None, disable_ue8m0_cast=False, recipe=None
        )
        output = output + bias

        # hpc
        my = hpc.gemm_blockwise(x, w, x_scale, w_scale, bias.to(torch.float32))

        assert allclose(output, my, atol=0.01, rtol=0.01)
