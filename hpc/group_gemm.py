import torch
from torch import Tensor
from typing import Tuple, Optional


def reformat_x_scale(
    x_scale: Tensor,
    seqlens: Tensor,
    cu_seqlens: Tensor,
    num_seq_per_group_avg: int,
    output: Optional[Tensor] = None,
) -> Tensor:
    """Performs transpose, pad to aligned to tile m and compact arrangement
    just for deepep format input when use fp8 blockwise group gemm
    Args:
        x_scale: Scaling factor for x FP8 quantization
            Shape: [total_seq_pad, hidden_size // 128]
            Dtype: float32

        seqlens: Sequence lengths for each group
            Shape: [num_group]
            Dtype: int32

        cu_seqlens: Cumulative sequence lengths indicating start indices in input tensor
            Shape: [num_group + 1]
            Dtype: int32

        num_seq_per_group_avg: average number seqs per group, use for get tile m

    Returns:
        Tensor: Output x scale tensor after reformat
            Shape: [hidden_size // 128, compact_total_seq_pad]
            Dtype: float32

    Raises:
        RuntimeError: If the input tensors have incompatible shapes or types,
            or if the CUDA kernel execution fails.

    Note:
        - All input tensors must be on CUDA device
        - The length of x_scale for each group must be aligned to multiple of 16/32/64 according to num_seq_per_group_avg

    """
    return torch.ops.hpc.reformat_x_scale(
        x_scale, seqlens, cu_seqlens, output, num_seq_per_group_avg
    )


def group_gemm_pertensor_fp8(
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
    return torch.ops.hpc.group_gemm_pertensor_fp8(
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
            Shape: [num_group, output_dim // 128, (hidden_size // 128 + 3) // 4 * 4]
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
        - The size of w_scale must be multiple of 4

    """
    return torch.ops.hpc.group_gemm_blockwise_fp8(
        x, weight, seqlens, cu_seqlens, x_scale, w_scale, num_seq_per_group_avg, output, tma_desc
    )


@torch.library.register_fake("hpc::group_gemm_pertensor_fp8")
def group_gemm_pertensor_fp8_fake(
    x, weight, seqlens, cu_seqlens, y_scale, num_seq_per_group_avg, output, tma_des
):
    return torch.empty((x.shape[0], weight.shape[1]), dtype=torch.bfloat16)


@torch.library.register_fake("hpc::group_gemm_blockwise_fp8")
def group_gemm_blockwise_fp8_fake(
    x, weight, seqlens, cu_seqlens, x_scale, w_scale, num_seq_per_group_avg, output, tma_des
):
    return torch.empty((x.shape[0], weight.shape[1]), dtype=torch.bfloat16)
