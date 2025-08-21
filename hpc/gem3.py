import torch
from torch import Tensor


def gem3(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    return torch.ops.hpc.gem3(q, k, v)
