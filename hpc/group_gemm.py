from typing import Optional, Tuple

import torch
from torch import Tensor


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


def group_gemm_fp8(
    x: Tensor,
    weight: Tensor,
    seqlens: Tensor,
    cu_seqlens: Tensor,
    y_scale: Tensor,
    num_seq_per_group_avg: int = 32,
    output: Tensor = None,
    tma_desc: Tensor = None,
    task_map_workspace: Tensor = None,
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
        x,
        weight,
        seqlens,
        cu_seqlens,
        y_scale,
        num_seq_per_group_avg,
        output,
        tma_desc,
        task_map_workspace,
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
    task_map_workspace: Tensor = None,
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
        x,
        weight,
        seqlens,
        cu_seqlens,
        x_scale,
        w_scale,
        num_seq_per_group_avg,
        output,
        tma_desc,
        task_map_workspace,
    )


def group_gemm_groupwise_w4a8_mma_weight_reformat(
    weight: Tensor, weight_scale: Tensor, group_size: int
):
    """Perform weight and scale reformat for group gemm weight for better performance.

    more detail see: https://www.bilibili.com/video/BV1XH4y1c7JZ/?share_source=copy_web&vd_source=9fdc1699a8d12c6def84b64749925090
    """

    num_group = weight.shape[0]
    m = weight.shape[1]
    k_half = weight.shape[2]

    assert m % 64 == 0, "m must be divided by 64"
    assert k_half * 2 % 128 == 0, "k must be divided by 128"
    assert k_half * 2 // group_size == weight_scale.size(
        -1
    ), "weight and weight scale must have same k"
    assert weight.dtype == torch.int8, "weight's data type must be int8"
    assert weight_scale.dtype == torch.bfloat16, "weight's data type must be bfloat16"

    # weight reformat
    """
    1. Interleave in M mode to better store.
        Reorder each 16 rows according to [0, 2, 4, 6, 8,...,60, 62, 1, 3, 5, 7, 9,..., 61, 63]
    """
    m_blocksize = 64
    weight = weight.reshape(num_group, m // m_blocksize, m_blocksize, k_half)
    prmt_list = list(range(0, 64, 2)) + list(range(1, 64, 2))
    weight = weight[:, :, prmt_list, :]
    weight = weight.reshape(num_group, m, k_half)

    """
    2. Interleave in K mode for better load.
        View weight as 64 elements as one item. Reorder each four rows into one row.

        -----------------
        | (0, 0) | (0, 1)
        -----------------
        | (1, 0) | (1, 1)      -------------------------------------------------------------------------
        -----------------  =>  | (0, 0) | (1, 0) | (2, 0) | (3, 0) | (0, 1) | (1, 1) | (2, 1) | (3, 1) |
        | (2, 0) | (2, 1)      -------------------------------------------------------------------------
        -----------------
        | (3, 0) | (3, 1)
        -----------------
    """
    m_blocksize = 4
    k_blocksize = 64 // 2  # int8 = 2 int4
    weight = weight.reshape(
        num_group, m // m_blocksize, m_blocksize, k_half // k_blocksize, k_blocksize
    )
    weight = weight.transpose(2, 3).contiguous()
    weight = weight.reshape(num_group, m, k_half)

    """
    3. Interleave in K mode to fast data type convert.
        Reorder each 8 elements according to [0, 2, 4, 6, 1, 3, 5 ,7]
    """
    k_blocksize = 8 // 2  # int8 = 2 int4
    weight = weight.reshape(num_group, m, k_half // k_blocksize, k_blocksize)
    weight = weight.view(torch.uint8)
    low_bit_int4_weight = weight & 0x0F  # [0, 2, 4, 6]
    high_bit_int4_weight = (weight >> 4) & 0x0F  # [1, 3, 5, 7]

    out = torch.empty_like(weight)
    out[:, :, :, 0] = low_bit_int4_weight[:, :, :, 0] | (low_bit_int4_weight[:, :, :, 1] << 4)
    out[:, :, :, 1] = low_bit_int4_weight[:, :, :, 2] | (low_bit_int4_weight[:, :, :, 3] << 4)
    out[:, :, :, 2] = high_bit_int4_weight[:, :, :, 0] | (high_bit_int4_weight[:, :, :, 1] << 4)
    out[:, :, :, 3] = high_bit_int4_weight[:, :, :, 2] | (high_bit_int4_weight[:, :, :, 3] << 4)

    # weight scale reformat
    """
    1. Interleave in M mode for correspond weight
        Reorder each 16 rows according to [0, 2, 4, 6, 8,...,60, 62, 1, 3, 5, 7, 9,..., 61, 63]
    """
    k_scale = weight_scale.size(-1)
    m_blocksize = 64
    weight_scale = weight_scale.reshape(num_group, m // m_blocksize, m_blocksize, k_scale)
    weight_scale = weight_scale[:, :, prmt_list, :]
    weight_scale = weight_scale.reshape(num_group, m, k_scale)

    """
    2. Pad to 16 Byte in k mode for better load
        Because of dtype scale is bfloat16, so scale tensor in k dimension is align to 8 elements
    """
    k_scale_pad = (k_scale + 7) // 8 * 8
    # Replace `torch.zeros` with `torch.ones` to pass the Computer Sanitizer check.
    out_scale = torch.ones(
        (num_group, m, k_scale_pad), dtype=weight_scale.dtype, device=weight_scale.device
    )
    out_scale[:, :, :k_scale] = weight_scale

    return out.view(torch.int8).reshape(num_group, m, k_half).contiguous(), out_scale


def group_gemm_groupwise_w4a8_mma(
    x: Tensor,
    weight: Tensor,
    seqlens: Tensor,
    cu_seqlens: Tensor,
    y_scale: Tensor,
    group_size: int,
    output: Optional[Tensor] = None,
):
    """Performs group GEMM operation with int4 weight, fp8 activation and bf16 precision use mma 16x8x16 instructions.
    It will have better performance when m of each expert is small than 8.

    Args:
        x: Input activation tensor
            Shape: [total_seq, hidden_size]
            Dtype: fp8
        weight: Weight tensor for group matrix multiplication which reformat by group_gemm_groupwise_w4a8_mma_weight_reformat
            Shape: [num_group, output_dim, hidden_size // 2]
            Dtype: int8
        seqlens: Sequence lengths for each group
            Shape: [num_group]
            Dtype: int32
        cu_seqlens: Cumulative sequence lengths indicating start indices in input tensor
            Shape: [num_group + 1]
            Dtype: int32
        group_size: int32, group size of weight groupwise quant, only support 64/128
        y_scale: Scaling factor for activation static per tensor fp8 quantization and weight groupwise(group_size == 64/128) int4 quantization, which should be pad to 16 bytes.
            Shape: [num_group, output_dim, (hidden_size // group_size + 7) // 8 * 8]
            Dtype: bfloat16
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
    return torch.ops.hpc.group_gemm_groupwise_w4a8_mma(
        x, weight, seqlens, cu_seqlens, y_scale, group_size, output
    )


@torch.library.register_fake("hpc::group_gemm_fp8")
def group_gemm_fp8_fake(
    x,
    weight,
    seqlens,
    cu_seqlens,
    y_scale,
    num_seq_per_group_avg,
    output,
    tma_desc,
    task_map_workspace,
):
    return torch.empty((x.shape[0], weight.shape[1]), dtype=torch.bfloat16)


@torch.library.register_fake("hpc::group_gemm_blockwise_fp8")
def group_gemm_blockwise_fp8_fake(
    x,
    weight,
    seqlens,
    cu_seqlens,
    x_scale,
    w_scale,
    num_seq_per_group_avg,
    output,
    tma_desc,
    task_map_workspace,
):
    return torch.empty((x.shape[0], weight.shape[1]), dtype=torch.bfloat16)
