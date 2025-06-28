import torch
from torch import Tensor


def act_mul_and_quant(gate_up: Tensor, scale: Tensor) -> Tensor:
  """Performs act(left) * right * scale for gate_up output.

    Executes the operation in a custom GPU kernel for optimized performance.

    Args:
        gate_up: gate_up projection result with size[N, C * 2], dtype = bfloat16 
        scale: per tensor quantization scale, only using the first element, dtype = float

    Returns:
        Output tensor with output dtype = fp8_e4m3 
    """
  return torch.ops.hpc.act_mul_and_quant(gate_up, scale)
