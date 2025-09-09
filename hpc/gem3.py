import torch
from torch import Tensor


def gem3(q: Tensor, k: Tensor, v: Tensor, qscale: Tensor, kscale: Tensor) -> Tensor:
    return torch.ops.hpc.gem3(q, k, v, qscale, kscale)


def gemm(x: Tensor, weight: Tensor) -> Tensor:
    return torch.ops.hpc.gemm(x, weight)
