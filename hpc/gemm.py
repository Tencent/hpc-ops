import torch
from torch import Tensor


def pad_and_transpose(x: Tensor) -> Tensor:
    return torch.ops.hpc.pad_and_transpose(x)


def gemm_blockwise(
    x: Tensor, weight: Tensor, x_scale: Tensor, w_scale: Tensor, bias: Tensor
) -> Tensor:
    """Performs block wise quant GEMM operation with FP8 precision.

    The block size must be 128. you can refer the test 'tests/test_gemm_blockwise.py' and 'tests/test_gemm_benchmark.py'.
    Args:
        x: Input activation tensor
            Shape: [m, k]
            Dtype: fp8
        weight: Weight tensor
            Shape: [n, k]
            Dtype: fp8
        x_scale: Scaling factor for x tensor
            Shape: [k // 128, m_pad],  (m_pad = (m + 127) // 128 * 128)
            Dtype: fp32
            The origin x_scale shape is [m, k // 128], should pad to [m_pad, k // 128],
            and transpose it, this should be handle in last quant kernel
        weight_scale: Scaling factor for weight tensor
            Shape: [n // 128, 128]
            Dtype: fp23
            The origin weight_scale shape is [n // 128, k // 128], should pad to [n // 128, 128],
            so the k <= 16384
        bias: bias tensor
            Shape: [n]
            Dtype: fp23
    Returns:
        Tensor: Output tensor after matrix multiplication
            Shape: [m, n]
            Dtype: bfloat16

    """
    return torch.ops.hpc.gemm_blockwise(x, weight, x_scale, w_scale, bias)
