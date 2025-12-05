import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
from utils import calculate_errors, errors_to_string, allclose


def reference_scale3(x, scale_tensor, scale2_tensor, is_moe):
    x_fp32 = x.to(torch.float32)
    x_fp8 = (x_fp32 * scale_tensor[0]).to(torch.float8_e4m3fn)
    x_fp8_scale2 = None
    if is_moe:
        x_fp8_scale2 = (x_fp32 * scale2_tensor[0]).to(torch.float8_e4m3fn)
    return x_fp8, x_fp8_scale2, x_fp32


@pytest.mark.parametrize("batch_size", [1, 2, 4, 5, 8, 14, 16, 17, 32, 64])
@pytest.mark.parametrize("hidden_states", [4096])
@pytest.mark.parametrize("scale", [0.6])
@pytest.mark.parametrize("is_moe", [True, False])
def test_scale3(batch_size, hidden_states, scale, is_moe):
    torch.manual_seed(0)

    x = torch.randn(batch_size, hidden_states, dtype=torch.bfloat16).cuda()
    scale_tensor = torch.tensor([scale], dtype=torch.float32).cuda()
    scale2_tensor = torch.tensor([2 * scale], dtype=torch.float32).cuda()

    gt = reference_scale3(x, scale_tensor, scale2_tensor, is_moe=is_moe)
    my = hpc.scale3(x, scale_tensor, scale2_tensor, is_moe=is_moe)

    torch.cuda.synchronize()
    my_fp8, my_fp8_scale2, my_fp32 = my
    gt_fp8, gt_fp8_scale2, gt_fp32 = gt
    assert my_fp8.dtype == torch.float8_e4m3fn
    assert allclose(my_fp8.to(torch.bfloat16), gt_fp8.to(torch.bfloat16), atol=0.15, rtol=0.0125)
    if is_moe:
        assert my_fp32.dtype == torch.float32
        assert allclose(my_fp32, gt_fp32)
        assert allclose(
            my_fp8_scale2.to(torch.bfloat16),
            gt_fp8_scale2.to(torch.bfloat16),
            atol=0.15,
            rtol=0.00125,
        )
