import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math

import numpy as np


def reference_torch_rms_norm_with_scale(x, weight, scale, eps):
    rms = torch.rsqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    x_normalized = x * rms
    if weight is not None:
        x_normalized = x_normalized * weight.float()
    inv_scale = 1.0 / scale
    return (x_normalized * inv_scale).to(torch.float8_e4m3fn).to(torch.bfloat16)


# torch impl for rms_norm
def reference_torch_rms_norm(x, weight, eps):
    rms = torch.rsqrt(torch.mean(x.float().pow(2), dim=-1, keepdim=True) + eps)
    x_normalized = x * rms
    if weight is not None:
        x_normalized = x_normalized * weight.float()
    return x_normalized


@pytest.mark.parametrize("batch_size", [1, 2, 4, 5, 8, 14, 16, 17, 32, 64])
@pytest.mark.parametrize("hidden_states", [5120, 320])
@pytest.mark.parametrize("scale", [2.5])
@pytest.mark.parametrize("is_moe", [False, True])
def test_fused_rms_norm_with_scale(batch_size, hidden_states, scale, is_moe):
    torch.manual_seed(0)
    rms_norm_weight = torch.rand((1, hidden_states), dtype=torch.bfloat16).cuda()
    x = torch.randn(batch_size, hidden_states, dtype=torch.bfloat16).cuda()
    if is_moe:
        scale_tensor = torch.tensor([scale, 2 * scale], dtype=torch.float32).cuda()
    else:
        scale_tensor = torch.tensor([scale], dtype=torch.float32).cuda()
    eps = 1e-6

    if not rms_norm_weight.is_contiguous():
        rms_norm_weight = rms_norm_weight.contiguous()

    gt = reference_torch_rms_norm_with_scale(x, rms_norm_weight, scale_tensor[0], eps)
    if is_moe:
        gt_2 = reference_torch_rms_norm_with_scale(x, rms_norm_weight, scale_tensor[1], eps)
    else:
        gt_2 = gt

    gt_fp32 = reference_torch_rms_norm(x, rms_norm_weight, eps)
    output = hpc.normalization.fused_rms_norm_with_scale(
        x, rms_norm_weight, scale=scale_tensor, eps=eps, is_moe=is_moe
    )

    if is_moe:
        y_fp32, y_fp8, y_fp8_2 = output
    else:
        y_fp32, y_fp8, y_fp8_2 = gt_fp32, output, output

    assert y_fp8.dtype == torch.float8_e4m3fn
    assert y_fp8_2.dtype == torch.float8_e4m3fn
    assert y_fp32.dtype == torch.float32
    assert torch.allclose(y_fp32, gt_fp32)
    assert torch.allclose(y_fp8_2.to(torch.bfloat16), gt_2)
    assert torch.allclose(y_fp8.to(torch.bfloat16), gt)
