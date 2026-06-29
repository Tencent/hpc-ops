import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import pytest
import torch.nn.functional as F
from utils import allclose


def simple_naive_blockwise_gemm(x, w, x_scale, w_scale):
    x_scale = x_scale.repeat_interleave(128, dim=1)
    w_scale = w_scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    x = (x.to(torch.bfloat16) * x_scale).to(torch.bfloat16)
    w = (w.to(torch.bfloat16) * w_scale[0 : w.size(0), :]).to(torch.bfloat16)

    y = x @ w.t()
    return y


LL_SHAPES = [
    (1536, 2048),
    (2048, 768),
    (1536, 3200),
    (3200, 768),
    (2560, 2048),
    (2048, 2048),
    (768, 2048),
    (2048, 384),
    (768, 3200),
    (3200, 384),
]


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("n,k", LL_SHAPES)
def test_gemm_blockwise_low_latency(m, n, k):
    dtype = torch.float8_e4m3fn

    x = torch.randn((m, k), dtype=torch.float, device="cuda").to(dtype)
    w = torch.randn((n, k), dtype=torch.float, device="cuda").to(dtype)
    x_scale = torch.rand((m, k // 128), dtype=torch.float, device="cuda")
    w_scale = torch.rand(((n + 127) // 128, k // 128), dtype=torch.float, device="cuda")

    ws_pad = (w_scale.size(1) + 3) // 4 * 4
    w_scale_pad = w_scale.clone()
    pad_size = ws_pad - w_scale.size(1)
    w_scale_pad = F.pad(w_scale_pad, (0, pad_size, 0, 0))

    my = hpc.gemm_blockwise(x, w, x_scale, w_scale_pad, True, None)
    gt = simple_naive_blockwise_gemm(x, w, x_scale, w_scale)
    assert allclose(gt, my, atol=0.5, rtol=0.01)
