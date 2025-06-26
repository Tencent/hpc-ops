import torch
from torch import Tensor

__all__ = ["add", "cast"]


def add(a: Tensor, b: Tensor) -> Tensor:
  """Performs a + b in gpu kernel"""
  return torch.ops.hpc.add(a, b)


def cast(a: Tensor, dtype) -> Tensor:
  """Performs type cast in gpu kernel"""
  return torch.ops.hpc.cast(a, dtype)
