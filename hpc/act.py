import torch
from torch import Tensor


def act_mul_and_quant(gate_up: Tensor, scale: Tensor) -> Tensor:
  """Applies activation, multiplication, and quantization to the gate_up projection.

  Specifically:
  1. Splits the `gate_up` tensor into gate (first half) and up (second half)
  2. Applies activation (typically SiLU) to the gate portion
  3. Computes element-wise multiplication: activated_gate × up
  4. Scales the result using the first element of `scale`
  5. Quantizes the output to fp8_e4m3 format

  Executes via a custom high-performance GPU kernel.

  Args:
    gate_up: Concatenated gate and up projections.
        Shape: [N, 2*C] (N = batch size, C = hidden dimension)
        Dtype: bfloat16
    scale: Quantization scale factor.
        Only the first tensor element is used.
        Dtype: float32

  Returns:
    Quantized output tensor.
        Shape: [N, C]
        Dtype: fp8_e4m3
  """
  return torch.ops.hpc.act_mul_and_quant(gate_up, scale)
