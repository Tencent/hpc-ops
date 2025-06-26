import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '../build/lib.linux-x86_64-cpython-310/')))

import hpc
import torch


def test_add():
  a = torch.randn(3, 5, device='cuda')
  b = torch.randn(3, 5, device='cuda')

  gt = a + b
  c = hpc.add(a, b)
  torch.allclose(c, gt)

  assert torch.allclose(c, gt)
  assert c.device == a.device
  assert c.dtype == a.dtype
  assert c.shape == a.shape
