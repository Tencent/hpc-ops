import torch
from torch import Tensor


def cast(a: Tensor, dtype) -> Tensor:
  """Converts tensor data type using GPU kernel.

    Executes type conversion in a custom GPU kernel for optimized performance.

    Args:
        a: Input tensor to convert
        dtype: Target data type (e.g. torch.float16, torch.int32, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu)

    Returns:
        New tensor with same values as input but converted to specified dtype
    """
  return torch.ops.hpc.cast(a, dtype)
