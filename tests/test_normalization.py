import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
from utils import allclose


def reference_torch_rmsnorm_with_scale(x, weight, scale, eps):
    rms = torch.rsqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    x_normalized = x * rms
    if weight is not None:
        x_normalized = x_normalized * weight.float()
    inv_scale = 1.0 / scale
    return (x_normalized * inv_scale).to(torch.float8_e4m3fn).to(torch.bfloat16)


# torch impl for rmsnorm
def reference_torch_rmsnorm(x, weight, eps):
    rms = torch.rsqrt(torch.mean(x.float().pow(2), dim=-1, keepdim=True) + eps)
    x_normalized = x * rms
    if weight is not None:
        x_normalized = x_normalized * weight.float()
    return x_normalized


@pytest.mark.parametrize("batch_size", [1, 2, 4, 5, 8, 14, 16, 17, 32, 64])
@pytest.mark.parametrize("hidden_states", [5120, 320, 4096])
@pytest.mark.parametrize("scale", [2.5])
@pytest.mark.parametrize("is_moe", [False, True])
def test_fused_rmsnorm_with_scale(batch_size, hidden_states, scale, is_moe):
    torch.manual_seed(0)
    rmsnorm_weight = torch.rand((1, hidden_states), dtype=torch.bfloat16).cuda()
    x = torch.randn(batch_size, hidden_states, dtype=torch.bfloat16).cuda()
    if is_moe:
        scale_tensor = torch.tensor([scale, 2 * scale], dtype=torch.float32).cuda()
    else:
        scale_tensor = torch.tensor([scale], dtype=torch.float32).cuda()
    eps = 1e-6

    if not rmsnorm_weight.is_contiguous():
        rmsnorm_weight = rmsnorm_weight.contiguous()

    gt = reference_torch_rmsnorm_with_scale(x, rmsnorm_weight, scale_tensor[0], eps)
    if is_moe:
        gt_2 = reference_torch_rmsnorm_with_scale(x, rmsnorm_weight, scale_tensor[1], eps)
    else:
        gt_2 = gt

    gt_fp32 = reference_torch_rmsnorm(x, rmsnorm_weight, eps)
    output = hpc.normalization.fused_rmsnorm_with_scale(
        x, rmsnorm_weight, scale=scale_tensor, eps=eps, is_moe=is_moe
    )

    if is_moe:
        y_fp32, y_fp8, y_fp8_2 = output
    else:
        y_fp32, y_fp8, y_fp8_2 = gt_fp32, output, output

    assert allclose(gt_fp32, y_fp32)
    assert allclose(gt_2, y_fp8_2.to(torch.bfloat16), atol=0.15, rtol=0.0125)
    assert allclose(gt, y_fp8.to(torch.bfloat16), atol=0.15, rtol=0.0125)
