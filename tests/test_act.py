import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import pytest
from utils import allclose
from typing import Tuple


def _act_mul_and_quant(gate_up, scale):

    def silu(x):
        return x / (1 + (-x).exp())

    gate, up = torch.chunk(gate_up.float(), 2, dim=1)
    out = (silu(gate).to(torch.bfloat16) * up.to(torch.bfloat16)).to(torch.float32) * scale
    outfp8 = out.to(torch.float8_e4m3fn)
    return outfp8


def gt_masked_act_mul_and_quant(gate_up, scale, num_per_expert):

    def silu(x):
        return x / (1 + (-x).exp())

    gate, up = torch.chunk(gate_up.float(), 2, dim=1)
    out = silu(gate) * up * scale
    outfp8 = out.to(torch.float8_e4m3fn)
    return outfp8


def gt_vllm_silu_and_quant(x, scale):
    from vllm import _custom_ops as ops

    b = x.shape[0]
    h = x.shape[1]

    y1 = torch.empty(b, h // 2, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(y1).to(torch.float8_e4m3fn)

    torch.ops._C.silu_and_mul(y1, x)
    torch.ops._C.static_scaled_fp8_quant(y, y1, scale)
    return y


def _quantize_blockwise_fp8(
    tensor: torch.Tensor,
    block_size: int = 128,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    compute_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert tensor.dim() == 2, "Tensor must be 2D"

    batch_size, num_features = tensor.shape
    num_blocks = (num_features + block_size - 1) // block_size

    if fp8_dtype == torch.float8_e4m3fn:
        fp8_max = 448.0
    elif fp8_dtype == torch.float8_e5m2:
        fp8_max = 57344.0
    else:
        fp8_max = 448.0

    quantized = torch.empty(batch_size, num_features, device=tensor.device, dtype=fp8_dtype)

    scales = torch.empty(batch_size, num_blocks, device=tensor.device, dtype=compute_dtype)

    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, num_features)

        if start >= end:
            continue

        block = tensor[:, start:end]  # (batch, block_features)

        block_abs_max = block.abs().amax(dim=1, keepdim=True)  # (batch, 1)

        scale = block_abs_max / fp8_max
        inv_scale = 1.0 / (scale + 1e-8)

        # save scale
        scales[:, block_idx] = scale.squeeze(1)

        # quant
        quantized_block = (block * inv_scale).to(fp8_dtype)
        quantized[:, start:end] = quantized_block

    return quantized, scales


def _act_mul_and_blockwise_quant(gate_up):

    def silu(x):
        return x / (1 + (-x).exp())

    # silu and mul
    gate_up = gate_up.float()
    gate, up = torch.chunk(gate_up.float(), 2, dim=1)
    out = silu(gate) * up

    # block wise quant
    outfp8, out_scale = _quantize_blockwise_fp8(out)

    # m-major --> k-major
    new_out_scale = out_scale.T.contiguous().T
    return outfp8, new_out_scale


@pytest.mark.parametrize("num_batch", [64, 62 * 1024, 128 * 1024])
@pytest.mark.parametrize("intermediate_size", [2128, 512, 4608])
@pytest.mark.parametrize("use_output", [True, False])
def test_act_mul_and_quant(num_batch, intermediate_size, use_output):
    gate_up_out = torch.randn(
        (num_batch, intermediate_size * 2), dtype=torch.bfloat16, device="cuda"
    )
    scale = torch.tensor([1.24], dtype=torch.float32, device="cuda")

    if use_output:
        out = torch.empty(num_batch, intermediate_size, dtype=torch.float8_e4m3fn, device="cuda")
        hpc.act_mul_and_quant(gate_up_out, scale, output=out)
    else:
        out = hpc.act_mul_and_quant(gate_up_out, scale)

    gt = _act_mul_and_quant(gate_up_out, scale)

    assert allclose(gt.to(torch.float32), out.to(torch.float32))


@pytest.mark.parametrize("num_expert", [32])
@pytest.mark.parametrize("num_max_tokens_per_expert", [336])
@pytest.mark.parametrize("num_intermediate_size", [2048])
def test_masked_act_mul_and_quant(num_expert, num_max_tokens_per_expert, num_intermediate_size):

    num_tokens = num_expert * num_max_tokens_per_expert

    gate_up_out = torch.randn(
        (num_tokens, num_intermediate_size * 2), dtype=torch.bfloat16, device="cuda"
    )

    scale = torch.randn((1, num_intermediate_size), dtype=torch.bfloat16, device="cuda")

    num_per_expert = torch.tensor(
        [
            4,
            7,
            7,
            2,
            1,
            9,
            28,
            5,
            9,
            9,
            14,
            8,
            9,
            10,
            7,
            37,
            13,
            13,
            1,
            9,
            9,
            14,
            13,
            28,
            8,
            10,
            13,
            19,
            0,
            2,
            22,
            3,
        ],
        dtype=torch.int32,
        device="cuda",
    )

    my = hpc.masked_act_mul_and_quant(gate_up_out, scale, num_per_expert)
    gt = gt_masked_act_mul_and_quant(gate_up_out, scale, num_per_expert)

    assert gt.device == my.device
    assert gt.dtype == my.dtype
    assert gt.shape == my.shape

    # zero out the unmasked data
    idx = torch.arange(num_tokens, device="cuda")
    iexpert = idx // num_max_tokens_per_expert
    itoken = idx % num_max_tokens_per_expert

    keep = itoken < num_per_expert[iexpert]

    gt = gt.to(torch.float)
    my = my.to(torch.float)

    gt[~keep] = 0.0
    my[~keep] = 0.0

    print(gt)
    print(my)

    assert allclose(gt.to(torch.float32), my.to(torch.float32), atol=0.15, rtol=0.0125)


@pytest.mark.parametrize("num_batch", [29, 64, 62 * 1024, 128 * 1024])
@pytest.mark.parametrize("intermediate_size", [4096, 512, 1024])
def test_act_mul_and_blokcwise_quant(num_batch, intermediate_size):
    gate_up_out = torch.randn(
        (num_batch, intermediate_size * 2), dtype=torch.bfloat16, device="cuda"
    )

    my, my_scale = hpc.act_mul_and_blockwise_quant(gate_up_out)

    gt, gt_scale = _act_mul_and_blockwise_quant(gate_up_out)

    assert my_scale.stride() == gt_scale.stride()
    assert allclose(gt_scale, my_scale)
    assert allclose(gt.to(torch.float32), my.to(torch.float32), atol=32, rtol=0.01)
