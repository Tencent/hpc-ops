import sys
import os
from pathlib import Path

sys.path.append(
    os.path.realpath(list(Path(__file__).parent.glob('../build/lib.*/'))[0]))

import hpc
import torch


def test_add():
  a = torch.randn(3, 5, device='cuda')
  b = torch.randn(3, 5, device='cuda')

  gt = a + b
  c = hpc.ops.add(a, b)

  assert torch.allclose(c, gt)
  assert c.device == a.device
  assert c.dtype == a.dtype
  assert c.shape == a.shape
