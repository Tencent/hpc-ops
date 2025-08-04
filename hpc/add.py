import torch
from torch import Tensor


def add(a: Tensor, b: Tensor) -> Tensor:
    """Performs element-wise addition on GPU.

    Executes the operation in a custom GPU kernel for optimized performance.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Output tensor containing element-wise sum of inputs
    """
    return torch.ops.hpc.add(a, b)
