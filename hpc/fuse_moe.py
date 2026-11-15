# Copyright (C) 2026 Tencent.

from typing import Optional, Tuple

import torch
from torch import Tensor


def count_and_gather(
    x: Tensor,
    topk_ids: Tensor,
    num_expert: int,
    rank_ep: int,
    intermediate_size: int,
    num_seq_per_group_avg: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Sorts and aggregates token based on expert assignments for MoE layers.

    This function organizes input token according to their assigned expert IDs
    after top-k expert selection, grouping token by expert for efficient
    parallel computation in mixture-of-experts architectures.

    Args:
        x: Input token features tensor
            Shape: [num_seq, hidden_size]
            Dtype: fp8
        topk_ids: Expert assignment indices for each token
            Shape: [num_seq, num_topk]
            Dtype: int32
        num_expert: Number of experts available on the current device
            Shape: scalar
            DType: int
        rank_ep: Current device rank in expert parallelism
            Shape: scalar
            DType: int

    Returns:
        Tuple containing eight tensors:
        - output: Sorted token features grouped by expert
            Shape: [num_seq * num_topk, hidden_size]
            Dtype: fp8
        - output_for_group_gemm: allocate output for group gemm
            Shape: [num_seq * num_topk, intermediate_size]
            Dtype: bfloat16
        - topk_pos: Position indices mapping each expert output back to original sequence
            Shape: [num_seq * num_topk]
            Dtype: int32
        - seqlens: Number of token assigned to each expert
            Shape: [num_expert]
            Dtype: int32
        - cu_seqlens: Cumulative token counts for expert indexing
            Shape: [num_expert + 1]
            Dtype: int32
        - tiles: Number of tiles assigned to each expert
            Shape: [num_expert]
            Dtype: int32
        - cu_tiles: Cumulative tiles counts for expert indexing
            Shape: [num_expert + 1]
            Dtype: int32
        - tmas: Describe the tma info of output
            Shape: [num_expert * 2 * 128]
            Dtype: int8

    Raises:
        RuntimeError: If input tensors have incompatible shapes or types,
            if expert assignments exceed available expert count,
            or if buffer sizes are insufficient for output tensors.

    Note:
        - This function is designed for expert-parallel distributed training
        - All input and output tensors must reside on the same device
        - The function modifies the output buffers in-place when provided
        - Expert assignments in topk_ids should be in range [0, num_expert-1]
    """
    return torch.ops.hpc.count_and_gather(
        x, topk_ids, num_expert, rank_ep, intermediate_size, num_seq_per_group_avg
    )


def reduce(
    x: Tensor,
    topk_pos: Tensor,
    topk_scale: Tensor,
    shared_output: Tensor = None,
) -> Tensor:
    """reduce token based on expert assignments for MoE layers.

    This kernel implements the reduction (scatter-add) operation in the final stage
    of Mixture of Experts (MoE) computation. It aggregates the weighted contributions
    from selected experts back to the original sequence positions.

    Args:
        x: Input token features tensor from expert processing
            Shape: [total_num_seq, hidden_size]
            Dtype: bfloat16
        topk_pos: Position indices mapping each expert output back to original sequence
            Shape: [num_seq, num_topk]
            Dtype: int32
        topk_scale: Scaling factors for expert outputs (typically gating scores)
            Shape: [num_seq, num_topk]
            Dtype: float32
        shared_output: output for shared experts, default is None
            Shape: [num_seq, hidden_size]
            Dtype: bfloat16

    Returns:
        Tensor: Reduced output tensor containing aggregated expert contributions
            Shape: [num_seq, hidden_size]
            Dtype: bfloat16

    Raises:
        RuntimeError: If tensor shapes are incompatible or CUDA kernel execution fails.
        ValueError: If input tensors are on different devices or have unsupported dtypes.

    Note:
        - This is a performance-critical kernel optimized for MoE workloads
        - The operation is equivalent to a weighted scatter-add operation
        - Input tensors must be contiguous and on the same device
        - For best performance, hidden_size should be a multiple of 32
        - topk_pos values must be in range [0, total_num_seq-1]
    """
    return torch.ops.hpc.reduce(x, topk_pos, topk_scale, shared_output)


_CP_ASYNC_N_TP_MAX = 512


def fuse_moe(
    x: Tensor,
    gate_up_weight: Tensor,
    down_weight: Tensor,
    gate_up_scale: Tensor,
    down_scale: Tensor,
    act_and_mul_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    use_bf16_mul: bool = True,
    shared_output: Tensor = None,
    output: Tensor = None,
) -> Tensor:
    """Run per-tensor FP8 FusedMoE."""
    return torch.ops.hpc.fuse_moe(
        x,
        gate_up_weight,
        down_weight,
        gate_up_scale,
        down_scale,
        act_and_mul_scale,
        topk_ids,
        topk_scale,
        shared_output,
        rank_ep,
        num_expert_total,
        use_bf16_mul,
        output,
    )


def fuse_moe_pertensor_fp8(
    x: Tensor,
    gate_up_weight: Tensor,
    down_weight: Tensor,
    gate_up_scale: Tensor,
    down_scale: Tensor,
    act_and_mul_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    use_bf16_mul: bool = True,
    shared_output: Tensor = None,
) -> Tensor:
    """Run per-tensor FP8 FusedMoE."""

    return torch.ops.hpc.fuse_moe_pertensor_fp8(
        x,
        gate_up_weight,
        down_weight,
        gate_up_scale,
        down_scale,
        act_and_mul_scale,
        topk_ids,
        topk_scale,
        shared_output,
        rank_ep,
        num_expert_total,
        use_bf16_mul,
        None,
    )


def fuse_moe_blockwise_fp8(
    x: Tensor,
    x_scale: Tensor,
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    shared_output: Tensor = None,
) -> Tensor:
    """Run blockwise FP8 FusedMoE."""
    return torch.ops.hpc.fuse_moe_blockwise_fp8(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        shared_output,
        rank_ep,
        num_expert_total,
        None,
    )


def fuse_moe_blockwise(
    x: Tensor,
    x_scale: Tensor,
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    shared_output: Tensor = None,
    output: Tensor = None,
) -> Tensor:
    """Run blockwise FP8 FusedMoE."""
    return torch.ops.hpc.fuse_moe_blockwise(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        shared_output,
        rank_ep,
        num_expert_total,
        output,
    )


def fuse_moe_blockwise_fp8_dsl(
    x: Tensor,
    x_scale: Tensor,
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    shared_output: Optional[Tensor] = None,
    output: Optional[Tensor] = None,
    swiglu_limit: Optional[float] = None,
    mx_mma: bool = True,
) -> Tensor:
    """SM120 CuTe DSL counterpart of :func:`fuse_moe_blockwise_fp8`.

    Takes FP8 E4M3 weights with float32 scales per 128x128 block, as in
    Qwen-style FP8 checkpoints. ``mx_mma=True`` (default) rounds those scales up
    to UE8M0 and runs a native MXFP8 block-scaled GEMM; ``False`` keeps exact
    float32 128-block arithmetic.

    ``mx_mma=True`` requires ``x_scale`` to hold exact powers of two, because the
    kernel reads those scales as UE8M0 without rescaling ``x``. The first call on
    each device checks it. Weights should be passed through
    :func:`hpc.prepare_fp8_mx_mma_weights` at load time; float32
    weight scales are otherwise requantized in place on first use, which a CUDA
    graph capture cannot do.

    Requires ``nvidia-cutlass-dsl``; imported lazily because it pulls in ``cutlass``.
    """
    from .cute.fuse_moe import fuse_moe_blockwise_fp8_dsl as _impl

    return _impl(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        rank_ep=rank_ep,
        num_expert_total=num_expert_total,
        shared_output=shared_output,
        output=output,
        swiglu_limit=swiglu_limit,
        mx_mma=mx_mma,
    )


def fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl(
    x: Tensor,
    x_scale: Optional[Tensor],
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    shared_output: Optional[Tensor] = None,
    output: Optional[Tensor] = None,
    swiglu_limit: Optional[float] = None,
) -> Tensor:
    """Run the SM120 blockwise MXFP4 x MXFP8 fused MoE (CuTe DSL).

    Args:
        x: Activations, shape ``[tokens, hidden]``: MXFP8 E4M3, or BF16 to have
            the kernel quantize them to MXFP8 (one UE8M0 scale per 32 elements,
            rounded up to a power of two).
        x_scale: Raw UE8M0 activation scales, one per 32 hidden elements;
            ``None`` when ``x`` is BF16.
        gate_up_weight: Packed MXFP4 E2M1 data, shape
            ``[local_experts, 2 * intermediate, hidden // 2]``, dtype ``uint8``.
        gate_up_weight_scale: Raw UE8M0 weight scales, either logical
            ``[local_experts, 2 * intermediate, hidden // 32]`` or the packed
            layout from :func:`hpc.pack_mxfp_weight_scale_sm120`.
        down_weight: Packed MXFP4 E2M1 data, shape
            ``[local_experts, hidden, intermediate // 2]``, dtype ``uint8``.
        down_weight_scale: As ``gate_up_weight_scale``, for ``down_weight``.
        topk_ids: Global expert IDs, shape ``[tokens, topk]``, dtype ``int32``.
        topk_scale: Router weights, shape ``[tokens, topk]``, dtype ``float32``.
        rank_ep: Rank within the expert-parallel group.
        num_expert_total: Total experts across all expert-parallel ranks.
        shared_output: Optional BF16 shared-expert result to add.
        output: Optional preallocated BF16 output.
        swiglu_limit: Optional DeepSeek-style SwiGLU clamp.

    Requires ``nvidia-cutlass-dsl``; imported lazily because it pulls in ``cutlass``.
    """
    from .cute.fuse_moe import fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl as _impl

    return _impl(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        rank_ep=rank_ep,
        num_expert_total=num_expert_total,
        shared_output=shared_output,
        output=output,
        swiglu_limit=swiglu_limit,
    )


@torch.library.register_fake("hpc::count_and_gather")
def count_and_gather_fake(
    x, topk_ids, num_expert, rank_ep, intermediate_size, num_seq_per_group_avg
):
    return (
        torch.empty((topk_ids.shape[0] * topk_ids.shape[1], x.shape[1]), dtype=torch.float8_e4m3fn),
        torch.empty(
            (topk_ids.shape[0] * topk_ids.shape[1], intermediate_size), dtype=torch.bfloat16
        ),
        torch.empty((topk_ids.shape[0] * topk_ids.shape[1]), dtype=torch.int32),
        torch.empty((num_expert), dtype=torch.int32),
        torch.empty((num_expert + 1), dtype=torch.int32),
        torch.empty((num_expert), dtype=torch.int32),
        torch.empty((num_expert + 1), dtype=torch.int32),
        torch.empty((num_expert * 2 * 128), dtype=torch.int8),
    )


@torch.library.register_fake("hpc::reduce")
def reduce_fake(x, topk_pos, topk_scale, shared_output):
    return torch.empty((topk_pos.shape[0], x.shape[1]), dtype=torch.bfloat16)


@torch.library.register_fake("hpc::fuse_moe")
def fuse_moe_fake(
    x,
    gate_up_weight,
    down_weight,
    gate_up_scale,
    down_scale,
    act_and_mul_scale,
    topk_ids,
    topk_scale,
    shared_output,
    rank_ep,
    num_expert_total,
    use_bf16_mul,
    output,
):
    return (
        output
        if output is not None
        else torch.empty((x.shape[0], x.shape[1]), dtype=torch.bfloat16)
    )


@torch.library.register_fake("hpc::fuse_moe_pertensor_fp8")
def fuse_moe_pertensor_fp8_fake(
    x,
    gate_up_weight,
    down_weight,
    gate_up_scale,
    down_scale,
    act_and_mul_scale,
    topk_ids,
    topk_scale,
    shared_output,
    rank_ep,
    num_expert_total,
    use_bf16_mul,
    output,
):
    return (
        output
        if output is not None
        else torch.empty((x.shape[0], x.shape[1]), dtype=torch.bfloat16)
    )


@torch.library.register_fake("hpc::fuse_moe_blockwise")
def fuse_moe_blockwise_fake(
    x: Tensor,
    x_scale: Tensor,
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    shared_output,
    rank_ep: int,
    num_expert_total: int,
    output,
):
    return (
        output
        if output is not None
        else torch.empty((x.shape[0], x.shape[1]), dtype=torch.bfloat16)
    )


@torch.library.register_fake("hpc::fuse_moe_blockwise_fp8")
def fuse_moe_blockwise_fp8_fake(
    x: Tensor,
    x_scale: Tensor,
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    shared_output,
    rank_ep: int,
    num_expert_total: int,
    output,
):
    return (
        output
        if output is not None
        else torch.empty((x.shape[0], x.shape[1]), dtype=torch.bfloat16)
    )
