import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import pytest


def _act_mul_and_quant(gate_up, scale):

    def silu(x):
        # return torch.nn.functional.silu(x)
        return x / (1 + (-x).exp())

    gate, up = torch.chunk(gate_up.float(), 2, dim=1)
    out = silu(gate) * up * scale
    outfp8 = out.to(torch.float8_e4m3fn)
    return outfp8


@pytest.mark.parametrize("num_batch", [64, 62 * 1024, 128 * 1024])
def test_act_mul_and_quant(num_batch):

    intermediate_size = 17024 // 8
    gate_up_out = torch.randn(
        (num_batch, intermediate_size * 2), dtype=torch.bfloat16, device="cuda"
    )
    scale = torch.tensor([1.24], dtype=torch.float32, device="cuda")

    out = hpc.act_mul_and_quant(gate_up_out, scale)
    gt = _act_mul_and_quant(gate_up_out, scale)

    assert torch.allclose(out.to(torch.float), gt.to(torch.float))
    assert gt.device == out.device
    assert gt.dtype == out.dtype
    assert gt.shape == out.shape
