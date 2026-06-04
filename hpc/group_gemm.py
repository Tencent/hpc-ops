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
    assert k_half * 2 % 128 == 0 or k_half * 2 == 192, "k must be divided by 128 or k is 192"
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


def prepack_mxfp8_scale(
    sfx: Optional[Tensor],
    sfw: Optional[Tensor],
    cu_seqlens: Optional[Tensor],
    num_seq_per_group_avg: int,
) -> Tuple[Tensor, Tensor]:
    """Prepack raw MXFP8 block scales (UE8M0) into the UTCCP-friendly layout
    consumed by ``group_gemm_mxfp8``.

    Either ``sfx`` or ``sfw`` (or both) may be None to skip that side's prepack:
      - Pass ``sfw=None`` to only prepack SFX (online, every forward pass).
      - Pass ``sfx=None, cu_seqlens=None`` to only prepack SFW (offline, once at
        model load).
      - Pass both to prepack everything (legacy usage, backward compatible).

    The skipped side returns an empty tensor (numel=0).

    Args:
        sfx: Raw input-side scale, uint8 (UE8M0 raw bits), or None to skip.
            Shape: [m_total, k // 32]
        sfw: Raw weight-side scale, uint8 (UE8M0 raw bits), or None to skip.
            Shape: [num_group, n, k // 32]
        cu_seqlens: Cumulative seqlens, int32, shape [num_group + 1].
            Required when sfx is provided; can be None when only prepacking SFW.
        num_seq_per_group_avg: Average seq per group, used to derive kTileM
            and the worst-case SFX buffer size.

    Returns:
        sfx_packed: Prepacked SFX, uint8 (1-D buffer), or empty if sfx is None.
        sfw_packed: Prepacked SFW, uint8 (1-D buffer), or empty if sfw is None.
    """
    return torch.ops.hpc.prepack_mxfp8_scale(sfx, sfw, cu_seqlens, num_seq_per_group_avg)


def group_gemm_mxfp8(
    x: Tensor,
    weight: Tensor,
    sfx_packed: Tensor,
    sfw_packed: Tensor,
    seqlens: Tensor,
    cu_seqlens: Tensor,
    num_seq_per_group_avg: int,
    output: Optional[Tensor] = None,
    tma_desc: Optional[Tensor] = None,
) -> Tensor:
    """MXFP8 (block-scaled FP8) group GEMM.

    Use ``prepack_mxfp8_scale`` first to obtain prepacked SFX/SFW.

    Args:
        x: Input activation, fp8_e4m3, shape [m_total, k]
        weight: Weight tensor, fp8_e4m3, shape [num_group, n, k]
        sfx_packed: Prepacked input-side SF (uint8 buffer from ``prepack_mxfp8_scale``)
        sfw_packed: Prepacked weight-side SF (uint8 buffer from ``prepack_mxfp8_scale``)
        seqlens: Per-group sequence lengths, int32, shape [num_group]
        cu_seqlens: Cumulative seqlens, int32, shape [num_group + 1]
        num_seq_per_group_avg: Used to dispatch kTileM (must match the value passed
            to ``prepack_mxfp8_scale``).
        output: Optional output tensor, bf16, shape [m_total, n]
        tma_desc: Optional TMA descriptor workspace, shape [num_group * 2, 128]

    Returns:
        y: Output tensor, bf16, shape [m_total, n]
    """
    return torch.ops.hpc.group_gemm_mxfp8(
        x,
        weight,
        sfx_packed,
        sfw_packed,
        seqlens,
        cu_seqlens,
        num_seq_per_group_avg,
        output,
        tma_desc,
    )


def group_gemm_cp_async_mxfp8(
    x: Tensor,
    weight: Tensor,
    sfx_packed: Tensor,
    sfw_packed: Tensor,
    seqlens: Tensor,
    cu_seqlens: Tensor,
    num_seq_per_group_avg: int,
    x_row_map: Optional[Tensor] = None,
    output: Optional[Tensor] = None,
    tma_desc: Optional[Tensor] = None,
) -> Tensor:
    """MXFP8 group GEMM where A (activation) is loaded via cp.async + row_map
    indirection. When ``x_row_map`` is provided, ``x`` is treated as the raw
    un-permuted source — the kernel gathers per-row on-the-fly. This lets
    fuse_moe skip the physical gather kernel.

    SFA/SFW still use TMA + offline-prepacked layout (use ``prepack_mxfp8_scale``
    to produce them).

    Args:
        x: fp8_e4m3 activation. Shape is either [m_total, k] (no row_map) or
           [num_orig_rows, k] (with row_map; gathered logically to [m_total, k]).
        weight: fp8_e4m3 weight, [num_group, n, k]
        sfx_packed/sfw_packed: prepacked SF (use ``prepack_mxfp8_scale``)
        seqlens / cu_seqlens: per-group counters (post-permutation)
        num_seq_per_group_avg: dispatch hint
        x_row_map: optional int32 [m_total] inverse permutation (post-row → src-row)
        output, tma_desc: optional reusable buffers

    Returns:
        bf16 [m_total, n]
    """
    x_num_rows = x.shape[0] if x_row_map is None else x_row_map.shape[0]
    return torch.ops.hpc.group_gemm_cp_async_mxfp8(
        x,
        weight,
        sfx_packed,
        sfw_packed,
        seqlens,
        cu_seqlens,
        num_seq_per_group_avg,
        x_row_map,
        x.shape[0],
        output,
        tma_desc,
    )


@torch.library.register_fake("hpc::group_gemm_cp_async_mxfp8")
def group_gemm_cp_async_mxfp8_fake(
    x,
    weight,
    sfx_packed,
    sfw_packed,
    seqlens,
    cu_seqlens,
    num_seq_per_group_avg,
    x_row_map,
    x_num_rows,
    output,
    tma_desc,
):
    if x_row_map is not None:
        m_total = x_row_map.shape[0]
    else:
        m_total = x.shape[0]
    n = weight.shape[1]
    return torch.empty((m_total, n), dtype=torch.bfloat16, device=x.device)


@torch.library.register_fake("hpc::prepack_mxfp8_scale")
def prepack_mxfp8_scale_fake(sfx, sfw, cu_seqlens, num_seq_per_group_avg):
    # Derive k_sf and num_group from whichever tensor is present
    k_sf = 0
    num_group = 0
    n = 0
    m_total = 0

    if sfw is not None:
        num_group = sfw.shape[0]
        n = sfw.shape[1]
        k_sf = sfw.shape[2]
    if sfx is not None:
        m_total = sfx.shape[0]
        if k_sf == 0:
            k_sf = sfx.shape[1]
        if num_group == 0 and cu_seqlens is not None:
            num_group = cu_seqlens.shape[0] - 1

    # mirror LAUNCH_MXFP8 dispatch table in group_gemm_mxfp8_async
    kTileM = 256
    for ktm in (8, 16, 32, 48, 64, 96, 128, 160, 192, 224):
        if num_seq_per_group_avg <= ktm:
            kTileM = ktm
            break
    kSFAlignM = 128 if kTileM <= 128 else 256

    # SFX output
    if sfx is not None:
        sfx_max_tiles = m_total // kTileM + num_group
        sfx_padded_max = sfx_max_tiles * kSFAlignM
        sfx_out = torch.empty(sfx_padded_max * k_sf, dtype=torch.uint8)
    else:
        sfx_out = torch.empty(0, dtype=torch.uint8)

    # SFW output
    if sfw is not None:
        sfw_out = torch.empty(num_group * n * k_sf, dtype=torch.uint8)
    else:
        sfw_out = torch.empty(0, dtype=torch.uint8)

    return (sfx_out, sfw_out)


@torch.library.register_fake("hpc::group_gemm_mxfp8")
def group_gemm_mxfp8_fake(
    x,
    weight,
    sfx_packed,
    sfw_packed,
    seqlens,
    cu_seqlens,
    num_seq_per_group_avg,
    output,
    tma_desc,
):
    return torch.empty((x.shape[0], weight.shape[1]), dtype=torch.bfloat16)
