import torch
from torch import Tensor
from typing import Tuple


def group_gemm_fp8(
    x: Tensor,
    weight: Tensor,
    seqlens: Tensor,
    cu_seqlens: Tensor,
    y_scale: Tensor,
    num_seq_per_group_avg: int = 32,
    output: Tensor = None,
    tma_desc: Tensor = None,
) -> Tensor:
    """Performs group GEMM operation with FP8 precision.

    This function executes multiple matrix multiplications in a group manner
    using FP8 precision for improved performance.

    Args:
        x: Input activation tensor
            Shape: [total_seq, hidden_size]
            Dtype: fp8
        weight: Weight tensor for group matrix multiplication
            Shape: [num_group, output_dim, hidden_size]
            Dtype: fp8
        seqlens: Sequence lengths for each group
            Shape: [num_group]
            Dtype: int32
        cu_seqlens: Cumulative sequence lengths indicating start indices in input tensor
            Shape: [num_group + 1]
            Dtype: int32
        y_scale: Scaling factor for FP8 quantization
            Shape: [num_group]
            Dtype: float32

    Returns:
        Tensor: Output tensor after group matrix multiplication
            Shape: [total_seq, output_dim]
            Dtype: bfloat16

    Raises:
        RuntimeError: If the input tensors have incompatible shapes or types,
            or if the CUDA kernel execution fails.

    Note:
        - All input tensors must be on CUDA device

    """
    return torch.ops.hpc.group_gemm_fp8(
        x, weight, seqlens, cu_seqlens, y_scale, num_seq_per_group_avg, output, tma_desc
    )


def group_gemm_blockwise_fp8(
    x: Tensor,
    weight: Tensor,
    seqlens: Tensor,
    cu_seqlens: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    num_seq_per_group_avg: int = 32,
    output: Tensor = None,
    tma_desc: Tensor = None,
) -> Tensor:
    """Performs group GEMM operation with FP8 precision.

    This function executes multiple matrix multiplications in a group manner
    using FP8 precision for improved performance.

    Args:
        x: Input activation tensor
            Shape: [total_seq, hidden_size]
            Dtype: fp8
        weight: Weight tensor for group matrix multiplication
            Shape: [num_group, output_dim, hidden_size]
            Dtype: fp8
        seqlens: Sequence lengths for each group
            Shape: [num_group]
            Dtype: int32
        cu_seqlens: Cumulative sequence lengths indicating start indices in input tensor
            Shape: [num_group + 1]
            Dtype: int32
        x_scale: Scaling factor for x FP8 quantization
            Shape: [hidden_size // 128, total_seq_pad]
            Dtype: float32
        w_scale: Scaling factor for weight FP8 quantization
            Shape: [num_group, output_dim // 128, 64]
            Dtype: float32

    Returns:
        Tensor: Output tensor after group matrix multiplication
            Shape: [total_seq, output_dim]
            Dtype: bfloat16

    Raises:
        RuntimeError: If the input tensors have incompatible shapes or types,
            or if the CUDA kernel execution fails.

    Note:
        - All input tensors must be on CUDA device
        - The length of x_scale for each group must be aligned to multiple of 16/32/64 according to num_seq_per_group_avg
        - Only support hidden_size <= 8192

    """
    return torch.ops.hpc.group_gemm_blockwise_fp8(
        x, weight, seqlens, cu_seqlens, x_scale, w_scale, num_seq_per_group_avg, output, tma_desc
    )


@torch.library.register_fake("hpc::group_gemm_fp8")
def group_gemm_fp8_fake(
    x, weight, seqlens, cu_seqlens, y_scale, num_seq_per_group_avg, output, tma_des
):
    return torch.empty((x.shape[0], weight.shape[1]), dtype=torch.bfloat16)


@torch.library.register_fake("hpc::group_gemm_blockwise_fp8")
def group_gemm_blockwise_fp8_fake(
    x, weight, seqlens, cu_seqlens, x_scale, w_scale, num_seq_per_group_avg, output, tma_des
):
    return torch.empty((x.shape[0], weight.shape[1]), dtype=torch.bfloat16)
