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


@pytest.mark.skipif(not has_deep_gemm, reason="This test need install deepgemm")
# @pytest.mark.parametrize("m", [9614])
# @pytest.mark.parametrize("n", [13824])
# @pytest.mark.parametrize("k", [5120])
@pytest.mark.parametrize("m", [1])
@pytest.mark.parametrize("n", [512])
@pytest.mark.parametrize("k", [5120])
def test_gemm_blockwise_with_bias(m, n, k):
    dtype = torch.float8_e4m3fn

    x = torch.randn((m, k), dtype=torch.float, device="cuda").to(dtype)
    w = torch.randn((n, k), dtype=torch.float, device="cuda").to(dtype)
    x_scale = torch.rand((m, k // 128), dtype=torch.float, device="cuda")
    w_scale = torch.rand((n // 128, k // 128), dtype=torch.float, device="cuda")
    bias = torch.rand((n), dtype=torch.float32, device="cuda")
    output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    for i in range(10):
        # deep_gemm + element_wise add
        deep_gemm.fp8_gemm_nt(
            a=(x, x_scale), b=(w, w_scale), d=output, c=None, disable_ue8m0_cast=False, recipe=None
        )
        output = (output + bias).to(torch.bfloat16)

        # hpc
        my = hpc.gemm_blockwise(x, w, x_scale, w_scale, True, bias)

        assert allclose(output, my, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not has_deep_gemm, reason="This test need install deepgemm")
@pytest.mark.parametrize("m", [1])
@pytest.mark.parametrize("k", [7168])
@pytest.mark.parametrize("n", [2112])
def test_gemm_blockwise(m, n, k):
    dtype = torch.float8_e4m3fn

    x = torch.randn((m, k), dtype=torch.float, device="cuda").to(dtype)
    w = torch.randn((n, k), dtype=torch.float, device="cuda").to(dtype)
    x_scale = torch.rand((m, k // 128), dtype=torch.float, device="cuda")
    w_scale = torch.rand(((n + 127) // 128, k // 128), dtype=torch.float, device="cuda")
    output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    # w_scale.size(1) must aligned to 4
    ws_pad = (w_scale.size(1) + 3) // 4 * 4
    w_scale_pad = w_scale.clone()
    pad_size = ws_pad - w_scale.size(1)
    w_scale_pad = F.pad(w_scale_pad, (0, pad_size, 0, 0))

    for i in range(20):
        # deep_gemm
        deep_gemm.fp8_gemm_nt(
            a=(x, x_scale), b=(w, w_scale), d=output, c=None, disable_ue8m0_cast=False, recipe=None
        )

        # hpc
        my = hpc.gemm_blockwise(x, w, x_scale, w_scale_pad)

        # assert allclose(output.to(torch.float), my[1] + my[0], atol=0.01, rtol=0.01)
    print(output)
    print(my)
    assert allclose(output, my, atol=0.01, rtol=0.01)
