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


def simple_naive_blockwise_gemm(x, w, x_scale, w_scale, bias):
    x_scale = x_scale.repeat_interleave(128, dim=1)
    w_scale = w_scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    x = (x.to(torch.bfloat16) * x_scale).to(torch.bfloat16)
    w = (w.to(torch.bfloat16) * w_scale).to(torch.bfloat16)

    y = x @ w.t()
    y += bias

    return y


def naive_blockwise_gemm(x, w, x_scale, w_scale, bias):
    m, k = x.shape
    n, _ = w.shape
    num_tile_k, m_pad = x_scale.shape
    num_tile_n, _ = w_scale.shape
    num_tile_m = m // 128 + 1
    assert num_tile_n == n // 128
    assert num_tile_k == k // 128

    result = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
    ones = torch.tensor(1.0, dtype=torch.float, device="cuda")

    for tm in range(num_tile_m):
        for tn in range(num_tile_n):
            start = tm * 128
            end = min(m, (tm + 1) * 128)
            out = torch.zeros((end - start, 128), dtype=torch.float, device="cuda")
            for tk in range(num_tile_k):
                tiled_x = x[start:end, tk * 128 : (tk + 1) * 128]
                tiled_w = w[tn * 128 : (tn + 1) * 128, tk * 128 : (tk + 1) * 128]
                tiled_x_scale = x_scale[tk, start:end]
                tiled_w_scale = w_scale[tn, tk]
                tile_scale = tiled_x_scale * tiled_w_scale
                tmp = torch._scaled_mm(
                    tiled_x,
                    tiled_w.t(),
                    scale_a=ones,
                    scale_b=ones,
                    bias=None,
                    out_dtype=torch.float,
                )
                tmp = tmp * tile_scale.unsqueeze(1)
                out = out + tmp
            out = out + bias[tn * 128 : (tn + 1) * 128].unsqueeze(0)
            result[start:end, tn * 128 : (tn + 1) * 128] = out.to(torch.bfloat16)
    return result


@pytest.mark.parametrize("m", [9614])
@pytest.mark.parametrize("n", [5120, 13824])
@pytest.mark.parametrize("k", [5120, 13824])
def test_gemm_blockwise_with_transpose(m, n, k):
    assert k // 128 % 4 == 0
    dtype = torch.float8_e4m3fn

    x = torch.randn((m, k), dtype=torch.float, device="cuda").to(dtype)
    w = torch.randn((n, k), dtype=torch.float, device="cuda").to(dtype)
    x_scale = torch.ones((m, k // 128), dtype=torch.float, device="cuda")
    w_scale = torch.ones((n // 128, k // 128), dtype=torch.float, device="cuda")
    bias = torch.randn((n), dtype=torch.float, device="cuda")

    m_pad = (m + 3) // 4 * 4
    x_scale_t = x_scale.clone()
    pad_size = m_pad - m
    x_scale_t = F.pad(x_scale_t, (0, 0, 0, pad_size))
    x_scale_t = x_scale_t.t().contiguous()  # (m, k // 128) -> (k // 128, m_pad)

    my = hpc.gemm_blockwise(x, w, x_scale, w_scale, bias)
    gt = hpc.gemm_blockwise(x, w, x_scale_t, w_scale, bias)
    assert allclose(gt, my, atol=32, rtol=0.01)


@pytest.mark.parametrize("m", [4, 8, 9614])
@pytest.mark.parametrize("n", [5120, 13824])
@pytest.mark.parametrize("k", [5120, 13824])
def test_gemm_blockwise(m, n, k):
    assert k // 128 % 4 == 0
    dtype = torch.float8_e4m3fn

    x = torch.randn((m, k), dtype=torch.float, device="cuda").to(dtype)
    w = torch.randn((n, k), dtype=torch.float, device="cuda").to(dtype)
    x_scale = torch.rand((m, k // 128), dtype=torch.float, device="cuda")
    w_scale = torch.rand((n // 128, k // 128), dtype=torch.float, device="cuda")
    bias = torch.randn((n), dtype=torch.float, device="cuda")

    my = hpc.gemm_blockwise(x, w, x_scale, w_scale, bias)
    gt = simple_naive_blockwise_gemm(x, w, x_scale, w_scale, bias)
    assert allclose(gt, my, atol=32, rtol=0.01)
