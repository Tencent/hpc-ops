from typing import Optional

import torch
from torch import Tensor

from enum import Enum


class QuantType(Enum):
    QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD = 0
    QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR = 1
    QPERTENSOR_KPERTENSOR_VPERTENSOR = 2


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
            Dtype: bfloat16
        vcache: Value tensor for attention computation in paged format.
                 Constrainst the unused slots in last block of vcache for each request to be set zeros.
            Shape: [num_blocks, block_size, num_head_kv, num_dim_qk]
            Dtype: bfloat16
        qscale: Q fp8 quant scale. Per Token Per Head Fp8 Quant.
            Shape: [num_batch, num_head_q]
            Dtype: float32
        kscale: K fp8 quant scale. Per Tensor Fp8 Quant.
            Shape: [1]
            Dtype: float32
        vscale: V fp8 quant scale. Per Tensor Fp8 Quant.
            Shape: [1]
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


def attention_mla_with_kvcache_bf16(
    q: Tensor,
    kvcache: Tensor,
    block_ids: Tensor,
    cu_seqlens_q: Tensor,
    num_seq_kv: Tensor,
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

    return torch.ops.hpc.attention_mla_with_kvcache_bf16(
        q, kvcache, block_ids, cu_seqlens_q, num_seq_kv, output
    )


def sparse_mla_with_kvcache_bf16(
    q: Tensor,
    win_kvcache: Tensor,
    win_block_ids: Tensor,
    win_topk_ids: Tensor,
    compress_kvcache: Tensor,
    compress_block_ids: Tensor,
    compress_topk_ids: Tensor,
    cu_seqlens_q: Tensor,
    sink_weight: Tensor,
    softmax_scale: float,
    output: Tensor = None,
) -> Tensor:
    """Computes index attention prefill using bfloat16 precision.

    This function performs the index attention computation using custom hardware
    operations optimized for bfloat16 data type. The prefill stage processes all
    input tokens simultaneously to generate the initial attention context, which
    is typically used during the first forward pass of autoregressive models.

    Args:
        q: Query tensor for attention computation
            Shape: [num_total_tokens_q, num_head_q, num_dim]
            Dtype: bfloat16
        win_kvcache: Sliding window kvcache tensor for attention computation in paged format.
                 Constrainst the unused slots in last block of vcache for each request to be set zeros.
            Shape: [num_win_blocks, block_size, num_head_kv, num_dim]
            Dtype: bfloat16
        win_block_ids: Sliding window paged kvcache block index tensor for get paged kvcache.
            Shape: [num_batch, num_max_win_blocks]
            Dtype: int32
        win_topk_ids: Sliding window topk index for select kvcache
            Shape: [num_total_tokens, num_win_topk]
            Dtype: int32
        compress_kvcache: Compress 4/128 kvcache tensor for attention computation in paged format.
                 Constrainst the unused slots in last block of vcache for each request to be set zeros.
            Shape: [num_compress_blocks, block_size, num_head_kv, num_dim]
            Dtype: bfloat16
        compress_block_ids: Compress 4/128 paged kvcache block index tensor for get paged kvcache.
            Shape: [num_batch, num_max_compress_blocks]
            Dtype: int32
        compress_topk_ids: Compress 4/128 topk index for select kvcache
            Shape: [num_total_tokens_q, num_compress_topk]
            Dtype: int32
        cu_seqlens_q: start_seq_q for each batch
            Shape: [num_batch + 1]
            Dtype: int32
        sink_weight:
            Shape: [num_head_q]
            Dtype: float32
        softmax_scale:
            Shape: scalar
            Dtype: float32

    Returns:
        Tensor: Attention output tensor in bfloat16 format on CUDA device
            Shape: [num_total_tokens_q, num_head_q, num_dim]
            Dtype: bfloat16

    Raises:
        RuntimeError: If the shapes or dtypes do not satisfy the constraints above.

    Note:
        - All input tensors must be on CUDA device and in bfloat16 format
        - num_total_tokens_q = cu_seqlens_q[num_batch + 1]
    """

    return torch.ops.hpc.attention_sparse_mla_with_kvcache_bf16(
        q,
        win_kvcache,
        win_block_ids,
        win_topk_ids,
        compress_kvcache,
        compress_block_ids,
        compress_topk_ids,
        cu_seqlens_q,
        sink_weight,
        softmax_scale,
        output,
    )


def attention_blocksparse_prefill_fp8_dim192(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_kv: Tensor,
    max_seqlens_q: int,
    max_seqlens_kv: int,
    q_scale: Tensor,
    k_scale: Tensor,
    v_scale: Tensor,
    softmax_scale: Optional[float] = None,
    block_mask: Optional[Tensor] = None,
    output: Tensor = None,
) -> Tensor:
    """Dim192 dense / block-sparse attention prefill with FP8 varlen QKV.

    Supports MLA dim_qk=192, dim_v=128. When `block_mask` is None, dispatches
    to the dense-compatible `kHasMask=False` path. When provided, only KV tiles
    marked True in `block_mask` are computed.

    Args:
        q: Query tensor. Shape: [total_q, num_head_q, dim_qk], Dtype: fp8_e4m3
        k: Key tensor. Shape: [total_kv, num_head_kv, dim_qk], Dtype: fp8_e4m3
        v: Value tensor. Shape: [total_kv, num_head_kv, dim_v], Dtype: fp8_e4m3
        cu_seqlens_q: Cumulative Q lengths. Shape: [B+1], Dtype: int32
        cu_seqlens_kv: Cumulative KV lengths. Shape: [B+1], Dtype: int32
        max_seqlens_q: Max Q sequence length (scalar).
        max_seqlens_kv: Max KV sequence length (scalar).
        q_scale: Per-tensor Q dequant scale. Shape: [1], Dtype: float32.
        k_scale: Per-tensor K dequant scale. Shape: [1], Dtype: float32.
        v_scale: Per-tensor V dequant scale. Shape: [1], Dtype: float32.
        softmax_scale: Attention scale factor. Default: dim_qk ** -0.5.
        block_mask: Optional sparse mask. Shape: [B, H_q, Qb, Kb], Dtype: uint8.
        output: Optional pre-allocated output tensor.

    Returns:
        Tensor: Shape [total_q, num_head_q, dim_v], Dtype: bfloat16.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    return torch.ops.hpc.attention_blocksparse_prefill_fp8_dim192(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlens_q,
        max_seqlens_kv,
        block_mask,
        q_scale,
        k_scale,
        v_scale,
        softmax_scale,
        output,
    )


def mla_prefill_bf16(
    q: Tensor,
    kv: Tensor,
    seqlens_q: Tensor,
    cu_seqlens_q: Tensor,
    num_dim_qk: int,
    num_dim_v: int,
    max_seqlens_q: int,
    output: Tensor = None,
) -> Tensor:
    return torch.ops.hpc.mla_prefill_bf16(
        q, kv, seqlens_q, cu_seqlens_q, num_dim_qk, num_dim_v, max_seqlens_q, output
    )


def get_attention_decode_task_workspace(
    max_num_batch: int, max_seqlen: int, num_head_kv: int, min_process_len: int = 1024
):
    """Allocate task map for attention decode.
    Args:
        max_num_batch: max batch size decode will process
        max_seqlen: max seqlen of service support.
        num_head_kv:
        min_process_len: each sm will process at_least 'min_process_len' token.

    Returns:
        Tensor: Shape [task_map_byte_size], Dtype: int8.
    """

    kTileN = 128
    num_sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count

    num_sm_count = num_sm_count // num_head_kv

    task_info_byte_size = 32
    max_num_tasks = max_num_batch * (max_seqlen + kTileN - 1) // kTileN
    max_num_tiles_per_sm = max(
        (max_num_tasks + num_sm_count - 1) // num_sm_count, min_process_len // kTileN
    )
    max_num_tasks = max_num_tiles_per_sm * num_sm_count
    finish_tasks = num_sm_count
    num_tile_per_sm_store_task = 1
    int_size = 4
    max_num_batch_pad = (
        (max_num_batch * int_size + task_info_byte_size - 1)
        // task_info_byte_size
        * task_info_byte_size
    )

    num_sm_count_pad = (
        (num_sm_count + task_info_byte_size - 1) // task_info_byte_size * task_info_byte_size
    )

    sched_need_byte_size = (
        max_num_tasks + finish_tasks + num_tile_per_sm_store_task
    ) * task_info_byte_size + max_num_batch_pad
    workspace_byte_size = sched_need_byte_size + 2 * num_sm_count_pad
    workspace = torch.zeros(
        workspace_byte_size,
        dtype=torch.int8,
        device="cuda",
    )

    workspace.view(torch.int32)[1] = num_head_kv
    workspace.view(torch.int32)[2] = max_num_batch
    workspace.view(torch.int32)[3] = sched_need_byte_size

    return workspace


def assign_attention_decode_task(
    num_seq_kvcache: Tensor,
    task_map: Tensor,
    num_head_kv: int,
    mtp: int,
    new_kv_included: bool,
    min_process_len: int = 1024,
) -> Tensor:
    """Computes attention decode task map.

    Args:
        num_seq_kvcache: number tokens in kvcache before cur iteration.
            Shape: [num_batch]
            Dtype: int32
        task_map: task_map for store output
            Shape: [task_map_byte_size]
            Dtype: int8
        num_head_kv: num_head_kv.
            Shape: scalar
            Dtype: int32
        mtp: number draft tokens.
            Shape: scalar
            Dtype: int32
        new_kv_included: the seqlen in num_seq_kvcache include new kv or not.
            Shape: scalar
            Dtype: bool
        min_process_len: each sm will process at_least 'min_process_len' token.
            Shape: scalar
            Dtype: int
    """
    if num_seq_kvcache.device.type == "cpu":
        task_map_host = torch.ops.hpc.assign_attention_decode_task(
            num_seq_kvcache, num_head_kv, mtp, new_kv_included, min_process_len, None
        )
        task_map[:4].copy_(task_map_host.reshape(-1)[:4], non_blocking=True)
        task_map[32 : task_map_host.numel()].copy_(
            task_map_host.reshape(-1)[32:], non_blocking=True
        )
        return task_map
    else:
        return torch.ops.hpc.assign_attention_decode_task(
            num_seq_kvcache, num_head_kv, mtp, new_kv_included, min_process_len, task_map
        )


def print_attention_decode_task(task_map: Tensor) -> None:
    task = task_map.view(torch.int32).reshape(-1, 8).cpu()
    num_tile_per_sm = task[0][0]
    num_head_kv = task[0][1]
    num_sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count

    num_sm_count = num_sm_count // num_head_kv

    print(
        f"\n[Decode Attn Task Map]: num_tile_per_sm({task[0][0]}), num_chunks:{task[num_sm_count * num_tile_per_sm + 1:]}\n"
    )

    idx = 0
    num_task_per_sm = torch.zeros(num_sm_count)
    seqlen_per_sm = torch.zeros(num_sm_count)
    for ism in range(num_sm_count):
        print(f"#######SM{ism}########")
        for itask in range(num_tile_per_sm):
            i = ism * num_tile_per_sm + itask
            if task[i + 1][0] < 0:
                break
            print(
                f"task:{idx}, ibatch:{task[i + 1][0]}, ichunk:{task[i + 1][1]}, iseq_start:{task[i + 1][2]}, seqkv:{task[i + 1][3]}, num_seqkvcache:{task[i + 1][4]}, num_tile_kv:{task[i + 1][5]}, num_tile_full:{task[i + 1][6]}, is_casual_chunk:{task[i + 1][7]}\n"
            )
            idx += 1
            num_task_per_sm[ism] += 1
            seqlen_per_sm[ism] += task[i + 1][3]

    print(f"Summary:")
    for ism in range(num_sm_count):
        print(f"SM:{ism}, num_tasks:{num_task_per_sm[ism]}, total_seq:{seqlen_per_sm[ism]}")


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
    output,
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


@torch.library.register_fake("hpc::attention_blocksparse_prefill_fp8_dim192")
def attention_blocksparse_prefill_fp8_dim192_fake(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlens_q,
    max_seqlens_kv,
    block_mask,
    q_scale,
    k_scale,
    v_scale,
    softmax_scale,
    output,
):
    return torch.empty((q.size(0), q.size(1), v.size(-1)), dtype=torch.bfloat16, device=q.device)


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
    new_kv_included,
    splitk,
    split_flag,
    output,
):
    return torch.empty_like(q)


@torch.library.register_fake("hpc::attention_mla_with_kvcache_bf16")
def attention_mla_with_kvcache_bf16_fake(
    q: Tensor,
    kvcache: Tensor,
    block_ids: Tensor,
    cu_seqlens_q: Tensor,
    num_seq_kv: Tensor,
    output: Tensor = None,
):
    return torch.empty_like(q)


@torch.library.register_fake("hpc::attention_sparse_mla_with_kvcache_bf16")
def sparse_mla_with_kvcache_bf16_fake(
    q,
    win_kvcache,
    win_block_ids,
    win_topk_ids,
    compress_kvcache,
    compress_block_ids,
    compress_topk_ids,
    cu_seqlens_q,
    sink_weight,
    softmax_scale,
    output,
):
    return torch.empty_like(q)
