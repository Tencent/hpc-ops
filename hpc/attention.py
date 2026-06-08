from enum import Enum
from typing import Optional

import torch
from torch import Tensor


class QuantType(Enum):
    QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD = 0
    QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR = 1
    QPERTENSOR_KPERTENSOR_VPERTENSOR = 2
    QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD_QKHADAMARD = 3


def attention_prefill_bf16(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seqlens_q: Tensor,
    cu_seqlens_q: Tensor,
    max_seqlens_q: int,
    output: Tensor = None,
) -> Tensor:
    """Computes attention prefill using bfloat16 precision.

    This function performs the attention prefill computation using custom hardware
    operations optimized for bfloat16 data type. The prefill stage processes all
    input tokens simultaneously to generate the initial attention context, which
    is typically used during the first forward pass of autoregressive models.

    Args:
        q: Query tensor for attention computation
            Shape: [total_seq, num_head_q, num_dim_qk]
            Dtype: bfloat16
        k: Key tensor for attention computation
            Shape: [total_seq, num_head_kv, num_dim_qk]
            Dtype: bfloat16
        v: Value tensor for attention computation
            Shape: [total_seq, num_head_kv, num_dim_v]
            Dtype: bfloat16
        seqlens_q: num_seq_q for each batch
            Shape: [num_batch]
            Dtype: int32
        cu_seqlens_q: start_seq_q for each batch
            Shape: [num_batch + 1]
            Dtype: int32
        max_seqlens_q: max seqlens amang all batchs
            Shape: scalar
            Dtype: int

    Returns:
        Tensor: Attention output tensor in bfloat16 format on CUDA device
            Shape: [total_seq, num_head_q, num_dim_v]
            Dtype: bfloat16

    Raises:
        RuntimeError: If the shapes or dtypes do not satisfy the constraints above.

    Note:
        - All input tensors must be on CUDA device and in bfloat16 format
        - The query and key tensors must have the same embedding dimension (num_dim_qk)
        - total_seq = sum(seqlens_q[ibatch] for ibatch in range(num_batch))
    """

    return torch.ops.hpc.attention_prefill_bf16(
        q, k, v, seqlens_q, cu_seqlens_q, max_seqlens_q, output
    )


def attention_with_kvcache_prefill_bf16(
    q: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    cu_seqlens_q: Tensor,
    block_ids: Tensor,
    seqlens_kvcache: Tensor,
    max_seqlens_q: int,
    output: Tensor = None,
) -> Tensor:
    """Computes paged KV-cache attention prefill in bfloat16.

    This interface supports both NHD-contiguous and HND-backed KV cache layouts
    through tensor strides while keeping the same logical tensor shape.
    In other words, the cache tensors should still be passed as
    ``[num_blocks, block_size, num_head_kv, num_dim]``, and layout selection is
    expressed by stride metadata (for example, via
    ``transpose(1, 2).contiguous().transpose(1, 2)``).

    Args:
        q: Query tensor for attention computation
            Shape: [total_seq, num_head_q, num_dim_qk]
            Dtype: bfloat16
        kcache: Paged key cache tensor.
            Logical shape must be ``[num_blocks, block_size, num_head_kv, num_dim_qk]``.
            Both standard contiguous (NHD-like) and stride-transformed HND-backed
            views are supported.
            Unused slots in each request's last cache block should be zero-padded.
            Shape: [num_blocks, block_size, num_head_kv, num_dim_qk]
            Dtype: bfloat16
        vcache: Paged value cache tensor.
            Logical shape must be ``[num_blocks, block_size, num_head_kv, num_dim_v]``.
            Both standard contiguous (NHD-like) and stride-transformed HND-backed
            views are supported.
            Unused slots in each request's last cache block should be zero-padded.
            Shape: [num_blocks, block_size, num_head_kv, num_dim_v]
            Dtype: bfloat16
        cu_seqlens_q: start_seq_q for each batch
            Shape: [num_batch + 1]
            Dtype: int32
        block_ids: kvcache page block index tensor for get paged kvcache.
            Shape: [num_batch, max_blocks]
            Dtype: int32
        seqlens_kvcache: number tokens in kvcache before cur iteration.
            Shape: [num_batch]
            Dtype: int32
        max_seqlens_q: max seqlens amang all batchs
            Shape: scalar
            Dtype: int

    Returns:
        Tensor: Attention output tensor in bfloat16 format on CUDA device
            Shape: [total_seq, num_head_q, num_dim_v]
            Dtype: bfloat16

    Raises:
        RuntimeError: If the shapes or dtypes do not satisfy the constraints above.

    Note:
        - All input tensors must be on CUDA.
        - ``q``/``kcache`` head dimensions must match on ``num_dim_qk``.
        - ``kcache``/``vcache`` may be non-contiguous as long as logical shape is
          preserved and strides encode the desired KV layout (NHD/HND).
        - total_seq = sum(seqlens_q[ibatch] for ibatch in range(num_batch))
    """

    return torch.ops.hpc.attention_with_kvcache_prefill_bf16(
        q,
        kcache,
        vcache,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        max_seqlens_q,
        output,
    )


def attention_with_kvcache_prefill_fp8(
    q: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    qscale: Tensor,
    kscale: Tensor,
    vscale: Tensor,
    cu_seqlens_q: Tensor,
    block_ids: Tensor,
    seqlens_kvcache: Tensor,
    max_seqlens_q: int,
    quant_type: QuantType = QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
    output: Tensor = None,
) -> Tensor:
    """Computes paged KV-cache attention prefill with FP8 KV tensors.

    This interface supports both NHD-contiguous and HND-backed KV cache layouts
    through tensor strides while keeping the same logical tensor shape.
    Layout selection is represented by cache tensor strides, not by changing
    tensor rank or shape.

    Args:
        q: Query tensor for attention computation
            Shape: [total_seq, num_head_q, num_dim_qk]
            Dtype: fp8
        kcache: Paged key cache tensor.
            Logical shape must be ``[num_blocks, block_size, num_head_kv, num_dim_qk]``.
            Both standard contiguous (NHD-like) and stride-transformed HND-backed
            views are supported.
            Unused slots in each request's last cache block should be zero-padded.
            Shape: [num_blocks, block_size, num_head_kv, num_dim_qk]
            Dtype: fp8
        vcache: Paged value cache tensor.
            Logical shape must be ``[num_blocks, block_size, num_head_kv, num_dim_v]``.
            Both standard contiguous (NHD-like) and stride-transformed HND-backed
            views are supported.
            Unused slots in each request's last cache block should be zero-padded.
            Shape: [num_blocks, block_size, num_head_kv, num_dim_v]
            Dtype: fp8
        qscale: QK fp8 quant scale. Per Token Per Head Fp8 Quant.
            Shape: [num_batch, num_head_q, max_seqlens_q_pad]
            Dtype: float32
        kscale: K fp8 quant scale tensor.
            Shape: depends on `quant_type`:
                   - If `quant_type == QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR`:
                       Shape: [1]
                   - If `quant_type == QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD`:
                       Shape: [num_blocks, scale_block_size, num_head_kv, num_dim_scale]
            Dtype: float32/fp8
            For per-token/per-head K scale mode, stride-transformed layouts are
            also accepted as long as the logical shape above is preserved.
        vscale: V fp8 quant scale. Per Tensor Fp8 Quant.
            Shape: [1]
            Dtype: float32
        cu_seqlens_q: start_seq_q for each batch
            Shape: [num_batch + 1]
            Dtype: int32
        block_ids: kvcache page block index tensor for get paged kvcache.
            Shape: [num_batch, max_blocks]
            Dtype: int32
        seqlens_kvcache: number tokens in kvcache before cur iteration.
            Shape: [num_batch]
            Dtype: int32
        max_seqlens_q: max seqlens amang all batchs
            Shape: scalar
            Dtype: int
        quant_type: Type of quantization scheme for attention computation.
            Defaults to QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR.
        output: Optional output tensor to store the attention result.
            If provided, must be a bf16 tensor with appropriate shape
            (typically [total_seq, num_head_q, num_dim_v]).
            The result will be written into this tensor and the same tensor
            will be returned. If None, a new tensor is allocated and returned.
            Dtype: bf16

    Returns:
        Tensor: Attention output tensor in bfloat16 format on CUDA device
            Shape: [total_seq, num_head_q, num_dim_v]
            Dtype: bfloat16

    Raises:
        RuntimeError: If the shapes or dtypes do not satisfy the constraints above.

    Note:
        - All input tensors must be on CUDA.
        - ``kcache``/``vcache`` may be non-contiguous as long as logical shape is
          preserved and strides encode the desired KV layout (NHD/HND).
        - total_seq = sum(seqlens_q[ibatch] for ibatch in range(num_batch))
    """
    return torch.ops.hpc.attention_with_kvcache_prefill_fp8(
        q,
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        max_seqlens_q,
        quant_type.value,
        output,
    )


def attention_with_kvcache_blocksparse_prefill_fp8(
    q: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    qscale: Tensor,
    kscale: Tensor,
    vscale: Tensor,
    cu_seqlens_q: Tensor,
    block_ids: Tensor,
    seqlens_kvcache: Tensor,
    max_seqlens_q: int,
    quant_type: QuantType = QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
    block_mask: Optional[Tensor] = None,
    output: Tensor = None,
) -> Tensor:
    """Unified dense / block-sparse attention prefill with paged FP8 KV cache.

    When `block_mask` is None, this dispatches to the dense-compatible
    `kHasMask=False` path inside the unified kernel family.
    When provided, only KV tiles marked True in `block_mask` are computed.

    Recommendation: the causal diagonal tile (the last KV tile in each
    Q-tile's causal range) should be True in block_mask to avoid NaN.
    The kernel guarantees no deadlock regardless of mask content, but if
    any Q-tile has zero active tiles, softmax(all -inf) will produce NaN.

    Args:
        q: Query tensor. Shape: [total_seq, num_head_q, num_dim_qk], Dtype: fp8_e4m3
        kcache: Paged K cache.
            Logical shape: [num_blocks, block_size, num_head_kv, num_dim_qk].
            Both contiguous NHD-like and stride-transformed HND-backed layouts
            are supported.
            Dtype: fp8.
        vcache: Paged V cache.
            Logical shape: [num_blocks, block_size, num_head_kv, num_dim_v].
            Both contiguous NHD-like and stride-transformed HND-backed layouts
            are supported.
            Dtype: fp8.
        qscale: Per-token per-head Q dequant scale. Shape: [num_batch, num_head_q, max_seq_q_pad],
            Dtype: float32
        kscale: K fp8 quant scale tensor.
            Shape: depends on `quant_type`:
                   - If `quant_type == QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR`:
                       Shape: [1]
                   - If `quant_type == QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD`:
                       Shape: [num_blocks, scale_block_size, num_head_kv, num_dim_scale]
            Dtype: float32/fp8
            For per-token/per-head K scale mode, stride-transformed layouts are
            also accepted as long as the logical shape above is preserved.
        vscale: V fp8 quant scale.
            Shape: depends on `quant_type`:
                   - If `quant_type == QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR`:
                       Shape: [1] (per-tensor)
                   - If `quant_type == QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD`:
                       Shape: [num_head_kv] (per-head)
            Dtype: float32
        cu_seqlens_q: Cumulative Q lengths. Shape: [num_batch + 1], Dtype: int32
        block_ids: Page table. Shape: [num_batch, max_blocks], Dtype: int32
        seqlens_kvcache: KV cache lengths. Shape: [num_batch], Dtype: int32
        max_seqlens_q: Max Q sequence length (scalar).
        quant_type: Type of quantization scheme for attention computation.
            Defaults to QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR (legacy). Pass
            QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD for the new dense-aligned
            layout (K per-token-group per-head per-dim-group, V per-head).
        block_mask: Optional bool mask for KV tiles. True = compute, False = skip.
            Shape: [num_batch, num_head_q, max_tile_m, num_tile_kv_in_mask], Dtype: uint8.
        output: Optional pre-allocated output tensor.

    Returns:
        Tensor: Shape [total_seq, num_head_q, num_dim_v], Dtype: bfloat16.
    """
    return torch.ops.hpc.attention_with_kvcache_blocksparse_prefill_fp8(
        q,
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        max_seqlens_q,
        quant_type.value,
        block_mask,
        output,
    )


def attention_decode_bf16(
    q: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    block_ids: Tensor,
    num_seq_kvcache: Tensor,
    mtp: int = 0,
    new_kv_included: bool = False,
    splitk: bool = True,
    split_flag: Tensor = None,
    output: Tensor = None,
) -> Tensor:
    """Computes attention decode using bfloat16 precision.

    This function performs the attention decode computation using custom hardware
    operations optimized for bfloat16 data type.

    Args:
        q: Query tensor for attention computation
            Shape: [num_batch * num_seq_q, num_head_q, num_dim_qk]
            Dtype: bfloat16
        kcache: Key tensor for attention computation in paged format.
                 Constrainst the unused slots in last block of vcache for each request to be set zeros.
            Shape: [num_blocks, block_size, num_head_kv, num_dim_qk]
            Dtype: bfloat16
        vcache: Value tensor for attention computation in paged format.
                 Constrainst the unused slots in last block of vcache for each request to be set zeros.
            Shape: [num_blocks, block_size, num_head_kv, num_dim_qk]
            Dtype: bfloat16
        block_ids: kvcache page block index tensor for get paged kvcache.
            Shape: [num_batch, max_blocks]
            Dtype: int32
        num_seq_kvcache: number tokens in kvcache before cur iteration.
            Shape: [num_batch]
            Dtype: int32
        mtp: number draft tokens.
            Shape: scalar
            Dtype: int32
        new_kv_included: the seqlen in num_seq_kvcache include new kv or not.
            Shape: scalar
            Dtype: bool
        splitk: use the split k implemention or not.
            Shape: scalar
            Dtype: bool
        output: Output tensor for store output value inplace.
            Shape: [num_batch * num_seq_q, num_head_q, num_dim_qk]
            Dtype: bfloat16
    Returns:
        output: Output tensor for store output value.
            Shape: [num_batch * num_seq_q, num_head_q, num_dim_qk]
            Dtype: bfloat16

    Raises:
        RuntimeError: If the shapes or dtypes do not satisfy the constraints above.

    Note:
        - All input tensors must be on CUDA device and in bfloat16 format
        - The query and key tensors must have the same embedding dimension (num_dim_qk)
        - The batch size (num_batch) must be consistent across all input tensors
    """
    return torch.ops.hpc.attention_decode_bf16(
        q,
        kcache,
        vcache,
        block_ids,
        num_seq_kvcache,
        mtp,
        new_kv_included,
        splitk,
        split_flag,
        output,
    )


def attention_decode_fp8(
    q: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    block_ids: Tensor,
    num_seq_kvcache: Tensor,
    qscale: Tensor,
    kscale: Tensor,
    vscale: Tensor,
    mtp: int = 0,
    new_kv_included: bool = False,
    quant_type: QuantType = QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
    splitk: bool = True,
    task_map: Tensor = None,
    split_flag: Tensor = None,
    output: Tensor = None,
) -> Tensor:
    """Computes attention decode using bfloat16 precision.

    This function performs the attention decode computation using custom hardware
    operations optimized for bfloat16 data type.
    Perform fp8 attention: softmax(Q*K^T * qscale * kscale / sqrt(head_dim)) * V * vscale.

    Args:
        q: Query tensor for attention computation
            Shape: [num_batch * num_seq_q, num_head_q, num_dim_qk]
            Dtype: bfloat16
        kcache: Key tensor for attention computation in paged format.
                 Constrainst the unused slots in last block of vcache for each request to be set zeros.
            Shape: [num_blocks, block_size, num_head_kv, num_dim_qk]
            Dtype: float8_e4m3
        vcache: Value tensor for attention computation in paged format.
                 Constrainst the unused slots in last block of vcache for each request to be set zeros.
            Shape: [num_blocks, block_size, num_head_kv, num_dim_qk]
            Dtype: float8_e4m3
        qscale: Q fp8 quant scale. Per Token Per Head Fp8 Quant.
            Shape: [num_batch, num_head_q]
            Dtype: float32
        kscale: K fp8 quant scale. Per Tensor Or Per Token Fp8 Quant. When use per-token quant, it will reside in kcache.
            Shape: [1] or Same shape with kvcache, and last n rows store fp32 scale.
            Dtype: float32
        vscale: V fp8 quant scale. Per Tensor Or Per Token Fp8 Quant. When use per-token quant, it will reside in vcache.
            Shape: [1] or Same shape with kvcache, and last n rows store fp32 scale.
            Dtype: float32
        block_ids: kvcache page block index tensor for get paged kvcache.
            Shape: [num_batch, max_blocks]
            Dtype: int32
        num_seq_kvcache: number tokens in kvcache before cur iteration.
            Shape: [num_batch]
            Dtype: int32
        new_kv_included: the seqlen in num_seq_kvcache include new kv or not.
            Shape: scalar
            Dtype: bool
        mtp: number draft tokens.
            Shape: scalar
            Dtype: int32
        quant_type: select which quant type.
            Shape: scalar
            Dtype: QuantType
        splitk: use the split k implemention or not.
            Shape: scalar
            Dtype: bool
        task_map: pre scheduled task map, it can be allocated by 'get_attention_decode_task_workspace' and assign by 'assign_attention_decode_task'.
            Shape: [num_task_map_bytes]
            Dtype: int8
        output: Output tensor for store output value inplace.
            Shape: [num_batch * num_seq_q, num_head_q, num_dim_qk]
            Dtype: bfloat16
    Returns:
        output: Output tensor for store output value.
            Shape: [num_batch * num_seq_q, num_head_q, num_dim_qk]
            Dtype: bfloat16

    Raises:
        RuntimeError: If the shapes or dtypes do not satisfy the constraints above.

    Note:
        - All input tensors must be on CUDA device.
        - The query and key tensors must have the same embedding dimension (num_dim_qk)
        - The batch size (num_batch) must be consistent across all input tensors
    """
    return torch.ops.hpc.attention_decode_fp8(
        q,
        kcache,
        vcache,
        block_ids,
        num_seq_kvcache,
        qscale,
        kscale,
        vscale,
        mtp,
        new_kv_included,
        quant_type.value,
        splitk,
        task_map,
        split_flag,
        output,
    )


def get_attention_decode_task_workspace(
    max_num_batch: int, max_seqlen: int, num_head_kv: int, min_process_len: int = 512
):
    """Allocate task map for attention decode.

    On sm90 this dispatches to the dynamic flat-bin workspace (kTileN=64); on
    other archs it uses the legacy per-SM workspace (kTileN=128). The returned
    tensor layout differs between the two — always pair it with
    ``assign_attention_decode_task`` / the matching attention kernel from the
    same process, never reuse it across GPUs with different capabilities.

    Args:
        max_num_batch: max batch size decode will process
        max_seqlen: max seqlen of service support.
        num_head_kv:
        min_process_len: each sm will process at_least 'min_process_len' token.

    Returns:
        Tensor: Shape [task_map_byte_size], Dtype: int8.
    """
    # Per-task size: must match SM90DynamicTaskInfo (12 int32 = 48 B).
    kTaskInfoByteSize = 48
    kMaxCtaPerSm = 4

    num_sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    max_num_cta_count = num_sm_count * kMaxCtaPerSm

    kMinTileN = 64
    max_num_tasks = max_num_batch * num_head_kv * ((max_seqlen + kMinTileN - 1) // kMinTileN)
    max_num_tile_per_cta = max(
        (max_num_tasks + max_num_cta_count - 1) // max_num_cta_count, min_process_len // kMinTileN
    )
    max_num_tasks = max_num_tile_per_cta * max_num_cta_count
    finish_tasks = max_num_cta_count

    num_tile_per_cta_store_task = 1
    int_size = 4

    num_chunks_bytes = max_num_batch * num_head_kv * int_size
    max_num_batch_pad = (
        (num_chunks_bytes + kTaskInfoByteSize - 1) // kTaskInfoByteSize * kTaskInfoByteSize
    )
    kTaskStrideInts = kTaskInfoByteSize // int_size  # 12
    num_cta_count_pad_ints = (
        (max_num_cta_count + kTaskStrideInts - 1) // kTaskStrideInts * kTaskStrideInts
    )
    num_cta_count_pad = num_cta_count_pad_ints * int_size

    sched_need_byte_size = (
        max_num_tasks + finish_tasks + num_tile_per_cta_store_task
    ) * kTaskInfoByteSize + max_num_batch_pad
    workspace_byte_size = sched_need_byte_size + 2 * num_cta_count_pad
    workspace = torch.zeros(
        workspace_byte_size,
        dtype=torch.int8,
        device="cuda",
    )

    workspace.view(torch.int32)[2] = num_head_kv
    workspace.view(torch.int32)[3] = max_num_batch
    workspace.view(torch.int32)[4] = sched_need_byte_size

    return workspace


def assign_attention_decode_task(
    num_seq_kvcache: Tensor,
    task_map: Tensor,
    num_head_kv: int,
    mtp: int,
    new_kv_included: bool,
    min_process_len: int = 512,
) -> Tensor:
    """Populate the task_map returned by ``get_attention_decode_task_workspace``.

    Dispatches by device capability to match the allocator: sm90 → dynamic
    assigner; other archs → legacy per-SM assigner. Both impls support
    ``num_seq_kvcache`` on either CUDA or CPU; the CPU path is for tests that
    want to allclose a host reference against the CUDA output.

    Args:
        num_seq_kvcache: number tokens in kvcache before cur iteration.
            Shape: [num_batch]
            Dtype: int32
        task_map: task_map for store output
            Shape: [task_map_byte_size]
            Dtype: int8
        num_head_kv: num_head_kv.
        mtp: number draft tokens.
        new_kv_included: whether ``num_seq_kvcache`` already counts new KV.
        min_process_len: each sm will process at_least 'min_process_len' token.
    """
    if num_seq_kvcache.device.type == "cpu":
        task_map_host = torch.ops.hpc.assign_attention_decode_task(
            num_seq_kvcache, num_head_kv, mtp, new_kv_included, min_process_len, None
        )
        task_map[:8].copy_(task_map_host.reshape(-1)[:8], non_blocking=True)
        task_map[48 : task_map_host.numel()].copy_(
            task_map_host.reshape(-1)[48:], non_blocking=True
        )
        return task_map
    else:
        return torch.ops.hpc.assign_attention_decode_task(
            num_seq_kvcache, num_head_kv, mtp, new_kv_included, min_process_len, task_map
        )


def print_attention_decode_task(task_map: Tensor) -> None:
    """Pretty-print a task_map produced by the sm90 dynamic path."""
    kTaskStride = 12

    task = task_map.view(torch.int32).reshape(-1, kTaskStride).cpu()

    num_tile_per_cta_plus1 = int(task[0][0].item())
    num_total_ctas = int(task[0][1].item())
    num_tile_per_cta = num_tile_per_cta_plus1 - 1
    num_head_kv = int(task[0][2].item())
    max_num_batch = int(task[0][3].item())

    num_chunks_row_start = 1 + num_total_ctas * num_tile_per_cta_plus1
    num_chunks_flat = task[num_chunks_row_start:].reshape(-1)[: num_head_kv * max_num_batch]
    num_chunks_table = num_chunks_flat.reshape(num_head_kv, max_num_batch)

    print(
        f"\n[sm90 Dynamic Decode Attn Task Map] num_tile_per_cta={num_tile_per_cta}, "
        f"num_head_kv={num_head_kv}, max_num_batch={max_num_batch}, "
        f"num_total_ctas={num_total_ctas}"
    )
    print(f"num_chunks[ihead_kv, ibatch]:\n{num_chunks_table}\n")

    num_task_per_cta = torch.zeros(num_total_ctas, dtype=torch.int32)
    seqkv_per_cta = torch.zeros(num_total_ctas, dtype=torch.int64)
    global_task_idx = 0

    for icta in range(num_total_ctas):
        bin_start_row = 1 + icta * num_tile_per_cta_plus1
        header_row = task[bin_start_row]
        # Skip empty bins to keep the log compact.
        if int(header_row[0].item()) < 0 or int(header_row[1].item()) < 0:
            continue

        print(f"#######CTA{icta}########")
        for itask in range(num_tile_per_cta):
            row = task[bin_start_row + itask]
            ihead_kv = int(row[0].item())
            ibatch = int(row[1].item())
            if ihead_kv < 0 or ibatch < 0:
                break
            ichunk = int(row[2].item())
            iseq_start = int(row[3].item())
            num_seqkv = int(row[4].item())
            num_seqkvcache = int(row[5].item())
            num_tile_kv = int(row[6].item())
            num_tile_full = int(row[7].item())
            is_casual_chunk = int(row[8].item())
            print(
                f"task:{global_task_idx}, ihead_kv:{ihead_kv}, ibatch:{ibatch}, "
                f"ichunk:{ichunk}, iseq_start:{iseq_start}, num_seqkv:{num_seqkv}, "
                f"num_seqkvcache:{num_seqkvcache}, num_tile_kv:{num_tile_kv}, "
                f"num_tile_full:{num_tile_full}, is_casual_chunk:{is_casual_chunk}"
            )
            global_task_idx += 1
            num_task_per_cta[icta] += 1
            seqkv_per_cta[icta] += num_seqkv

    print("\nSummary (non-empty bins only):")
    for icta in range(num_total_ctas):
        if num_task_per_cta[icta] == 0:
            continue
        print(
            f"CTA:{icta}, num_tasks:{int(num_task_per_cta[icta])}, "
            f"total_seqkv:{int(seqkv_per_cta[icta])}"
        )
    empty_bins = int((num_task_per_cta == 0).sum())
    print(f"[idle] {empty_bins}/{num_total_ctas} bins were empty")


@torch.library.register_fake("hpc::attention_prefill_bf16")
def attention_prefill_bf16_fake(q, k, v, seqlens_q, cu_seqlens_q, max_seqlens_q, output):
    return torch.empty((q.size(0), q.size(1), v.size(-1)), dtype=torch.bfloat16, device=q.device)


@torch.library.register_fake("hpc::attention_with_kvcache_prefill_bf16")
def attention_with_kvcache_prefill_bf16_fake(
    q, kcache, vcache, cu_seqlens_q, block_ids, seqlens_kvcache, max_seqlens_q, output
):
    return torch.empty(
        (q.size(0), q.size(1), vcache.size(-1)), dtype=torch.bfloat16, device=q.device
    )


@torch.library.register_fake("hpc::attention_with_kvcache_prefill_fp8")
def attention_with_kvcache_prefill_fp8_fake(
    q,
    kcache,
    vcache,
    qscale,
    kscale,
    vscale,
    cu_seqlens_q,
    block_ids,
    seqlens_kvcache,
    max_seqlens_q,
    quant_type,
    output=None,
):
    return torch.empty(
        (q.size(0), q.size(1), vcache.size(-1)), dtype=torch.bfloat16, device=q.device
    )


@torch.library.register_fake("hpc::attention_with_kvcache_blocksparse_prefill_fp8")
def attention_with_kvcache_blocksparse_prefill_fp8_fake(
    q,
    kcache,
    vcache,
    qscale,
    kscale,
    vscale,
    cu_seqlens_q,
    block_ids,
    seqlens_kvcache,
    max_seqlens_q,
    quant_type,
    block_mask=None,
    output=None,
):
    return torch.empty(
        (q.size(0), q.size(1), vcache.size(-1)), dtype=torch.bfloat16, device=q.device
    )


@torch.library.register_fake("hpc::attention_decode_bf16")
def attention_decode_bf16_fake(
    q, kcache, vcache, block_ids, num_seq_kvcache, new_kv_included, splitk, output
):
    return torch.empty_like(q)


@torch.library.register_fake("hpc::attention_decode_fp8")
def attention_decode_fp8_fake(
    q,
    kcache,
    vcache,
    block_ids,
    num_seq_kvcache,
    qscale,
    kscale,
    vscale,
    mtp,
    new_kv_included,
    quant_type,
    splitk,
    task_map=None,
    split_flag=None,
    output=None,
):
    return torch.empty_like(q)
