import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
from utils import allclose


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
@pytest.mark.parametrize("hidden_states", [5120, 320, 4096])
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

    assert allclose(gt_fp32, y_fp32)
    assert allclose(gt_2, y_fp8_2.to(torch.bfloat16), atol=0.15, rtol=0.0125)
    assert allclose(gt, y_fp8.to(torch.bfloat16), atol=0.15, rtol=0.0125)


def torch_rmsnorm(x, weight, eps):
    dtype = x.dtype
    x = x.to(torch.float)
    weight = weight.to(torch.float)
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    if weight is not None:
        return (weight * x).to(dtype)
    else:
        return x.to(dtype)


def quantize_blockwise_fp8(tensor: torch.Tensor, block_size: int = 128, eps: float = 1e-6):
    assert tensor.dim() == 2, "Tensor must be 2D"
    batch_size, num_features = tensor.shape
    num_blocks = (num_features + block_size - 1) // block_size

    fp8_max = 448.0
    quantized = torch.empty(
        batch_size, num_features, device=tensor.device, dtype=torch.float8_e4m3fn
    )
    scales = torch.empty(batch_size, num_blocks, device=tensor.device, dtype=torch.float32)

    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, num_features)
        if start >= end:
            continue
        block = tensor[:, start:end]  # (batch, block_features)
        block_abs_max = block.abs().amax(dim=1, keepdim=True)  # (batch, 1)

        scale = block_abs_max / fp8_max
        inv_scale = 1.0 / (scale + eps)

        # save scale
        scales[:, block_idx] = scale.squeeze(1)
        # quant
        quantized_block = (block * inv_scale).to(torch.float8_e4m3fn)
        quantized[:, start:end] = quantized_block
    return quantized, scales


def torch_fused_rmsnorm_blockwise_quant(x, rmsnorm_weight, eps, with_blockwise_quant):
    out = torch_rmsnorm(x, rmsnorm_weight, eps)
    if with_blockwise_quant:
        return quantize_blockwise_fp8(out)
    else:
        return out, None


def apply_rope(q: torch.Tensor, cos_sin_cache: torch.Tensor) -> torch.Tensor:
    """
    q: (num_tokens, nheads, dim)
    cos_sin_cache: (num_tokens, dim)
        format: [cos0, sin0, cos1, sin1, ...]
    """
    dtype = q.dtype
    q = q.to(torch.float)
    cos_sin_cache = cos_sin_cache.to(torch.float)

    num_tokens, nheads, dim = q.shape

    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]

    cos = cos_sin_cache[..., 0::2]
    sin = cos_sin_cache[..., 1::2]

    # import pdb; pdb.set_trace()
    q_even_rot = q_even * cos.unsqueeze(1) - q_odd * sin.unsqueeze(1)
    q_odd_rot = q_even * sin.unsqueeze(1) + q_odd * cos.unsqueeze(1)

    result = torch.zeros_like(q)
    result[..., 0::2] = q_even_rot
    result[..., 1::2] = q_odd_rot

    return result.to(dtype)


@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("hidden_states", [128, 512, 1024, 4096])
@pytest.mark.parametrize("with_blockwise_quant", [False, True])
def test_fused_rmsnorm_blockwise_quant(batch_size, hidden_states, with_blockwise_quant):
    torch.manual_seed(0)

    x = torch.randn(batch_size, hidden_states, dtype=torch.bfloat16, device="cuda")
    rmsnorm_weight = torch.randn((1, hidden_states), dtype=torch.bfloat16, device="cuda")

    gt, gt_scale = torch_fused_rmsnorm_blockwise_quant(
        x, rmsnorm_weight, eps=1e-6, with_blockwise_quant=with_blockwise_quant
    )
    my, my_scale = hpc.fused_rmsnorm_blockwise_quant(
        x, rmsnorm_weight, eps=1e-6, with_blockwise_quant=with_blockwise_quant
    )
    if with_blockwise_quant:
        assert allclose(gt_scale, my_scale, atol=0.001, rtol=0.001)
        assert allclose(gt, my, atol=32, rtol=0.01)
    else:
        assert my_scale is None
        assert allclose(gt, my, atol=0.0001, rtol=0.01)


@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("rope_dim", [64])
@pytest.mark.parametrize("q_heads", [8])
@pytest.mark.parametrize("k_heads", [1])
def test_fused_rmsnorm_rope(batch_size, dim, rope_dim, q_heads, k_heads):
    torch.manual_seed(0)

    q = torch.randn(batch_size, q_heads, dim, dtype=torch.bfloat16, device="cuda")
    q_weight = torch.randn((1, dim), dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch_size, k_heads, dim, dtype=torch.bfloat16, device="cuda")
    k_weight = torch.randn((1, dim), dtype=torch.bfloat16, device="cuda")

    positions = torch.randint(
        low=0, high=4096, size=(batch_size,), dtype=torch.int64, device="cuda"
    )
    cos_sin_cache = torch.rand((4096, rope_dim), dtype=torch.bfloat16, device="cuda")

    my_q, my_k = hpc.fused_rmsnorm_rope(
        positions, q, q_weight, k, k_weight, cos_sin_cache, 1e-6
    )  # (batch, num_heads, dim)

    gt_q = torch_rmsnorm(q.reshape(-1, q.size(-1)), q_weight, eps=1e-6).reshape(
        batch_size, q_heads, dim
    )
    gt_k = torch_rmsnorm(k.reshape(-1, q.size(-1)), k_weight, eps=1e-6).reshape(
        batch_size, k_heads, dim
    )

    gt_q[..., -rope_dim:] = apply_rope(
        gt_q.reshape(batch_size, q_heads, -1)[..., -rope_dim:], cos_sin_cache[positions]
    )

    gt_k[..., -rope_dim:] = apply_rope(
        gt_k.reshape(batch_size, k_heads, -1)[..., -rope_dim:], cos_sin_cache[positions]
    )

    assert allclose(gt_q, my_q, atol=0.01, rtol=0.01)
    assert allclose(gt_k, my_k, atol=0.01, rtol=0.01)
