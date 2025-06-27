import sys
import os
from pathlib import Path

sys.path.insert(
    0, os.path.realpath(list(Path(__file__).parent.glob('../build/lib.*/'))[0]))

import hpc
import torch


def test_cast_fp16():
  a = torch.randn(3, 5, device='cuda')
  b = hpc.ops.cast(a, torch.float16)

  assert b.device == a.device
  assert b.dtype == torch.float16


def test_cast_fp8_e4m3():
  a = torch.randn(3, 5, device='cuda')
  b = hpc.ops.cast(a, torch.float8_e4m3fn)

  print(a)
  print(b)

  assert b.device == a.device
  assert b.dtype == torch.float8_e4m3fn


def test_cast_fp8_e5m2():
  a = torch.randn(3, 5, device='cuda')
  b = hpc.ops.cast(a, torch.float8_e5m2)

  print(a)
  print(b)

  assert b.device == a.device
  assert b.dtype == torch.float8_e5m2


def test_cast_fp8_e8m0():
  a = torch.randn(3, 5, device='cuda')
  b = hpc.ops.cast(a, torch.float8_e8m0fnu)

  print(a)
  print(b)

  assert b.device == a.device
  assert b.dtype == torch.float8_e8m0fnu
