import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

from typing import Tuple

import pytest
import torch

import hpc
from utils import allclose

# Set random seed for reproducibility
torch.manual_seed(41)
torch.cuda.manual_seed(41)


def _act_mul_and_quant(gate_up, scale):

    def silu(x):
        return x / (1 + (-x).exp())

    gate, up = torch.chunk(gate_up.float(), 2, dim=1)
    out = (silu(gate).to(torch.bfloat16) * up.to(torch.bfloat16)).to(torch.float32) * scale
    outfp8 = out.to(torch.float8_e4m3fn)
    return outfp8


def ref_masked_act_mul_and_quant(gate_up, scale, num_per_expert):

    def silu(x):
        return x / (1 + (-x).exp())

    gate, up = torch.chunk(gate_up.float(), 2, dim=1)
    out = silu(gate) * up * scale
    outfp8 = out.to(torch.float8_e4m3fn)
    return outfp8


@pytest.mark.parametrize("num_batch", [128 * 1024])
@pytest.mark.parametrize("intermediate_size", [4608])
@pytest.mark.parametrize("use_output", [True, False])
def test_act_mul_and_quant(num_batch, intermediate_size, use_output):
    gate_up_out = torch.randn(
        (num_batch, intermediate_size * 2), dtype=torch.bfloat16, device="cuda"
    )
    scale = torch.rand((1,), dtype=torch.float32, device="cuda") + 1.0

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

    scale = torch.randn((1,), dtype=torch.float32, device="cuda")

    num_per_expert = torch.randint(
        1, num_max_tokens_per_expert, (num_expert,), dtype=torch.int32, device="cuda"
    )

    output = torch.empty((num_tokens, num_intermediate_size), device="cuda").to(
        dtype=torch.float8_e4m3fn
    )
    my = hpc.masked_act_mul_and_quant(gate_up_out, scale, num_per_expert, output=output)
    gt = ref_masked_act_mul_and_quant(gate_up_out, scale, num_per_expert)

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

    assert allclose(gt.to(torch.float32), my.to(torch.float32), atol=0.15, rtol=0.0125)


def ref_masked_act_mul_and_blockwise_quant(gate_up, num_per_expert):

    def silu(x):
        return x / (1 + (-x).exp())

    gate, up = torch.chunk(gate_up.float(), 2, dim=1)
    out = silu(gate) * up
    scale = out.abs().reshape(out.size(0), -1, 128).max(dim=-1).values / 448
    scale_cp = scale.repeat_interleave(128, dim=-1)
    outfp8 = (out / (scale_cp + 1e-8)).to(torch.float8_e4m3fn)
    return outfp8, scale


@pytest.mark.parametrize("num_expert", [32])
@pytest.mark.parametrize("num_max_tokens_per_expert", [336])
@pytest.mark.parametrize("num_intermediate_size", [2048])
def test_masked_act_mul_and_blockwise_quant(
    num_expert, num_max_tokens_per_expert, num_intermediate_size
):

    num_tokens = num_expert * num_max_tokens_per_expert

    gate_up_out = torch.randn(
        (num_tokens, num_intermediate_size * 2), dtype=torch.bfloat16, device="cuda"
    )

    num_per_expert = torch.randint(
        0, num_max_tokens_per_expert, (num_expert,), dtype=torch.int32, device="cuda"
    )

    gt, gt_scale = ref_masked_act_mul_and_blockwise_quant(gate_up_out, num_per_expert)
    output = torch.empty((num_tokens, num_intermediate_size), device="cuda").to(
        dtype=torch.float8_e4m3fn
    )
    output_scale = torch.empty(
        (num_tokens, num_intermediate_size // 128), device="cuda", dtype=torch.float32
    )
    my, my_scale = hpc.masked_act_mul_and_blockwise_quant(
        gate_up_out, num_per_expert, output=output, output_scale=output_scale
    )

    assert gt.device == my.device
    assert gt.dtype == my.dtype
    assert gt.shape == my.shape
    assert gt_scale.device == my_scale.device
    assert gt_scale.dtype == my_scale.dtype
    assert gt_scale.shape == my_scale.shape

    # zero out the unmasked data
    idx = torch.arange(num_tokens, device="cuda")
    iexpert = idx // num_max_tokens_per_expert
    itoken = idx % num_max_tokens_per_expert

    keep = itoken < num_per_expert[iexpert]

    gt = gt.to(torch.float)
    my = my.to(torch.float)

    gt[~keep] = 0.0
    my[~keep] = 0.0
    gt_scale[~keep] = 0.0
    my_scale[~keep] = 0.0

    assert allclose(gt.to(torch.float32), my.to(torch.float32), atol=32, rtol=0.0125)
    assert allclose(
        gt_scale.to(torch.float32),
        my_scale.to(torch.float32),
        atol=0.15,
        rtol=0.0125,
    )
