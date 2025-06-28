import sys
import os
from pathlib import Path

sys.path.insert(
    0, os.path.realpath(list(Path(__file__).parent.glob('../build/lib.*/'))[0]))

import hpc
import torch


def _act_mul_and_quant(gate_up, scale):
  def silu(x):
    return torch.nn.functional.silu(x)
    # return x / (1 + (-x).exp())

  gate, up = torch.chunk(gate_up.float(), 2, dim=1)
  out = silu(gate) * up * scale
  outfp8 = out.to(torch.float8_e4m3fn)
  return outfp8

def test_act_mul_and_quant():

  intermediate_size = 17024 // 8 
  gate_up_out = torch.randn((64, intermediate_size * 2), dtype=torch.bfloat16, device='cuda') * 0 + 1.
  scale = torch.tensor([1.24], dtype=torch.float32, device='cuda')

  out = hpc.act_mul_and_quant(gate_up_out, scale)
  gt = _act_mul_and_quant(gate_up_out, scale)

  print("\n")
  print(out)
  print(out.shape)
  print(gt)
  print(gt.shape)

  assert torch.allclose(out.to(torch.float), gt.to(torch.float))
  assert gt.device == out.device
  assert gt.dtype == out.dtype
  assert gt.shape == out.shape
