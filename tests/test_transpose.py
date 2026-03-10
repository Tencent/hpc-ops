import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import pytest
from utils import allclose
import torch.nn.functional as F


@pytest.mark.parametrize("m", [1, 2, 16, 128])
@pytest.mark.parametrize("k", [128, 256, 1536, 2048, 7168])
def test_transpose(m, k):
    x_scale = torch.randn((m, k // 128), device="cuda", dtype=torch.float32)

    m_pad = (m + 3) // 4 * 4
    x_scale_t = x_scale.clone()
    pad_size = m_pad - m
    x_scale_t = F.pad(x_scale_t, (0, 0, 0, pad_size))  # (left, right, up, down)
    gt = x_scale_t.t().contiguous()  # (m, k // 128) -> (k // 128, m_pad)

    my = hpc.pad_and_transpose(x_scale)

    assert allclose(gt[:, 0:m], my[:, 0:m])
