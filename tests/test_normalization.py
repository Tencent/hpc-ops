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


@pytest.mark.parametrize("is_moe", [False, True])
def test_fused_rmsnorm_with_scale_fake(is_moe):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty((4, 320), dtype=torch.bfloat16, device="cuda")
        weight = torch.empty((320,), dtype=torch.bfloat16, device="cuda")
        scale = torch.empty((2 if is_moe else 1,), dtype=torch.float32, device="cuda")
        outputs = torch.ops.hpc.fused_rmsnorm_with_scale(x, weight, scale, 1e-6, is_moe)
        result = hpc.normalization.fused_rmsnorm_with_scale(
            x, weight, scale=scale, eps=1e-6, is_moe=is_moe
        )

    for tensors in (outputs, result):
        assert len(tensors) == 3
        for output, dtype in zip(
            tensors, (torch.float8_e4m3fn, torch.float32, torch.float8_e4m3fn)
        ):
            assert output.shape == x.shape
            assert output.device == x.device
            assert output.dtype == dtype


@pytest.mark.parametrize("is_moe", [False, True])
def test_fused_rmsnorm_with_scale_compiled(is_moe):
    x = torch.randn((2, 320), dtype=torch.bfloat16, device="cuda")
    weight = torch.rand((320,), dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([2.5, 5.0] if is_moe else [2.5], device="cuda")

    def run(x, weight, scale):
        return hpc.normalization.fused_rmsnorm_with_scale(
            x, weight, scale=scale, eps=1e-6, is_moe=is_moe
        )

    expected = run(x, weight, scale)
    actual = torch.compile(run, backend="eager", fullgraph=True)(x, weight, scale)
    assert len(actual) == len(expected) == 3
    for a, b in zip(actual if is_moe else actual[:1], expected if is_moe else expected[:1]):
        torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0)


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
    y_fp8, y_fp32, y_fp8_2 = hpc.normalization.fused_rmsnorm_with_scale(
        x, rmsnorm_weight, scale=scale_tensor, eps=eps, is_moe=is_moe
    )

    if is_moe:
        assert allclose(gt_fp32, y_fp32)
        assert allclose(gt_2, y_fp8_2.to(torch.bfloat16), atol=0.15, rtol=0.0125)
    assert allclose(gt, y_fp8.to(torch.bfloat16), atol=0.15, rtol=0.0125)
