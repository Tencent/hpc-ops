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


def mla_decode_with_kvcache_bf16(
    q: Tensor,
    kvcache: Tensor,
    block_ids: Tensor,
    cu_seqlens_q: Tensor,
    num_seq_kv: Tensor,
    sink_weight: Tensor = None,
    softmax_scale: float = 0.0,
    task_tensor: Tensor = None,
    output: Tensor = None,
    splitk: bool = True,
) -> Tensor:
    """dim576 MLA decode — persistent kernel pipeline.

    Implements a three-kernel chain (get_scheduler_map → persistent_attn →
    combine). Designed to win on small-batch / kv-skewed shapes where a
    dense single-CTA-per-batch kernel underutilises the SM array.

    Constraints (raised via TORCH_CHECK if violated):

    - ``num_dim_qk == 576``  (kv_lora_rank=512 + qk_rope_head_dim=64).
    - ``num_head_q ∈ {1, 2, 4, 8, 16, 32, 64}`` (64 uses a split-M
      cooperative kernel; ≤32 uses the single-WG kernel).
    - Decode only: ``total_seq_q == num_batch`` (one q-token per batch).
    - ``num_batch ≤ 1024``.
    - ``kvcache.size(1) == 64``  (paged block size).

    Args:
        q: [num_batch, num_head_q, 576] bf16, packed [ql_nope(512) | q_pe(64)].
        kvcache: [num_blocks, 64, 576] bf16, packed [kv_c_normed(512) | k_pe(64)];
            the first 512 cols are reused as both K-latent and V-latent.
        block_ids: [num_batch, num_seq_max_blocks] int32.
        cu_seqlens_q: [num_batch + 1] int32, ``cu_seqlens_q[b+1] - cu_seqlens_q[b] == 1``.
        num_seq_kv: [num_batch] int32, per-batch KV length.
        sink_weight: optional fp32 [num_head_q]
        softmax_scale: 0.0 means use ``1 / sqrt(num_dim_qk)``.
        task_tensor: optional precomputed scheduler map from
            :func:`get_mla_scheduler_map`. When provided, skips the inline
            get_scheduler_map kernel. Multi-layer transformer decode should
            build this once per forward pass and pass it to every layer's
            attn call to amortise the launch cost.
        output: optional pre-allocated bf16 [num_batch, num_head_q, 512].
            Latent V dim is always 512 (dim576 attn writes V in latent space).
        splitk: when True (default), split each batch's KV-tile range across
            multiple SMs for higher utilization on long-KV / small-batch
            shapes — current performance-optimal path. When False, every
            non-empty batch is pinned to a single SM (no split-KV); the
            combine step then reduces over a single partial, which makes
            each batch's output bit-invariant w.r.t. its sibling batches in
            the same call. Cost: long-KV batches no longer scale beyond one
            SM. Use False only when bit-level batch-invariance is required
            and the resulting throughput hit is acceptable.

    Returns:
        Tensor: [num_batch, num_head_q, 512] bf16.
    """
    return torch.ops.hpc.mla_decode_with_kvcache_bf16(
        q,
        kvcache,
        block_ids,
        cu_seqlens_q,
        num_seq_kv,
        sink_weight,
        float(softmax_scale),
        task_tensor,
        output,
        splitk,
    )


def sparse_mla_dsa_with_kvcache_bf16(
    q: Tensor,
    kvcache: Tensor,
    block_ids: Tensor,
    topk_ids: Tensor,
    cu_seqlens_q: Tensor,
    sink_weight: Tensor = None,
    softmax_scale: float = 0.0,
    task_tensor: Tensor = None,
    output: Tensor = None,
    splitk: bool = True,
) -> Tensor:
    """Sparse dim576 MLA (V3.2-DSA-style topk) with shared K/V latent.

    Handles both decode and prefill / chunk-prefill in a single entry; the
    mode is auto-detected from ``total_seq_q`` vs ``num_phys_batch``
    (= ``cu_seqlens_q.numel() - 1``):

    - **Decode**: ``total_seq_q == num_phys_batch``. ``topk_ids`` has one row
      per physical batch, ``num_phys_batch <= 1024``.
    - **Prefill / chunk-prefill**: ``total_seq_q > num_phys_batch``.
      ``topk_ids`` has one row per query token (``[total_seq_q,
      num_max_topk]``); no 1024-batch cap. The kernel resolves
      ``iquery_token -> iphys_batch`` via ``cu_seqlens_q`` internally so the
      caller does not need to expand ``block_ids``.

    The producer warpgroup gathers KV via cp.async + ZFILL using
    ``topk_ids`` (resolved through ``block_ids`` to the paged kvcache);
    the math warpgroup masks invalid topk positions to ``-inf`` so they
    contribute nothing to softmax.

    Constraints:

    - ``num_dim_qk == 576`` (kv_lora_rank=512 + qk_rope_head_dim=64).
    - ``num_head_q ∈ {1, 2, 4, 8, 16, 32, 64}``.
    - ``kvcache.size(1) == 64`` (paged block size).
    - ``topk_ids`` int32, with ``-1`` marking invalid entries;
      ``num_max_topk`` must be a multiple of 64 and ≤ 2048.

    Args:
        q: [total_seq_q, num_head_q, 576] bf16, packed [ql_nope(512) | q_pe(64)].
        kvcache: [num_blocks, 64, 576] bf16, packed [kv_c_normed(512) | k_pe(64)].
        block_ids: [num_phys_batch, num_seq_max_blocks] int32.
        topk_ids: [num_topk_rows, num_max_topk] int32 — token-level KV indices
            (e.g. via DeepSeek-V3.2 lightning indexer). ``num_topk_rows``
            equals ``num_phys_batch`` (decode) or ``total_seq_q`` (prefill).
            ``-1`` and out-of-range entries are masked.
        cu_seqlens_q: [num_phys_batch + 1] int32.
        sink_weight: optional fp32 [num_head_q].
        softmax_scale: 0.0 → use 1/sqrt(num_dim_qk).
        task_tensor: optional pre-allocated scheduler buffer (decode mode only;
            see :func:`get_mla_scheduler_map`'s sparse mode for sizing).
            Auto-allocated if None.
        output: optional pre-allocated bf16 [total_seq_q, num_head_q, 512].
        splitk: when True (default), split each batch's KV-tile range across
            multiple SMs for higher utilization on long-KV / small-batch
            shapes — current performance-optimal path. When False, every
            non-empty batch is pinned to a single SM (no split-KV); the
            combine step then reduces over a single partial, which makes
            each batch's output bit-invariant w.r.t. its sibling batches in
            the same call. Cost: long-KV batches no longer scale beyond one
            SM. Use False only when bit-level batch-invariance is required
            and the resulting throughput hit is acceptable.

    Returns:
        Tensor: [total_seq_q, num_head_q, 512] bf16.
    """
    return torch.ops.hpc.sparse_mla_dsa_with_kvcache_bf16(
        q,
        kvcache,
        block_ids,
        topk_ids,
        cu_seqlens_q,
        sink_weight,
        float(softmax_scale),
        task_tensor,
        output,
        splitk,
    )


def get_mla_scheduler_map(
    num_seq_kv: Tensor,
    cu_seqlens_q: Tensor,
    num_actual_tokens: int,
    index_topk: int = 0,
    splitk: bool = True,
    task_tensor: Tensor = None,
) -> Tensor:
    """Build the per-forward scheduler map for the dim576 persistent MLA path.

    Two modes, picked by ``index_topk``:

    1. **Dense** (``index_topk == 0``, default) — read per-batch length
       from ``num_seq_kv[i]``. Use this for
       :func:`mla_decode_with_kvcache_bf16`.
    2. **Sparse uniform** (``index_topk > 0``) — every batch contributes
       exactly ``index_topk`` tokens; ``num_seq_kv`` is used only for its
       size (= num_batch) and device. The tensor's values are ignored, so
       any int32 size-B cuda tensor works (e.g. an uninitialised buffer).
       Use this for :func:`sparse_mla_dsa_with_kvcache_bf16`.

    **Prefill / chunk-prefill** is selected by ``num_actual_tokens``: when it
    is > 0 and differs from ``num_seq_kv.numel()`` (= ``num_phys_batch``), the
    map is built with one logical batch *per query token* (sized by
    ``num_actual_tokens = total_seq_q``, uncapped by the dense 1024 limit).
    ``index_topk`` must then be the uniform per-token ``num_max_topk``. The
    returned map carries ``task_list / cu_tasks / cu_splits``, where each
    task_list entry is 8 self-describing ints carrying the token->batch map.

    ``cu_seqlens_q`` is **always required** (unified decode/prefill): the
    scheduler resolves ``iquery_token -> iphys_batch`` by binary search in
    both modes. For decode, pass the identity map
    ``cu_seqlens_q = torch.arange(num_batch + 1, dtype=int32)`` (one q-token
    per physical batch).

    For multi-layer transformer decode, call this once per forward pass and
    pass the returned tensor to every per-layer attn call via the
    ``task_tensor=`` argument. Amortises the launch cost.

    Args:
        num_seq_kv: int32 cuda Tensor of shape [num_batch] (num_batch ≤ 1024
            in decode). Dense: per-batch KV length values.
        cu_seqlens_q: int32 cuda Tensor of shape [num_phys_batch + 1].
            Always required; for decode pass ``arange(num_batch + 1)``.
        num_actual_tokens: > 0 and != ``num_seq_kv.numel()`` selects prefill
            mode (= ``total_seq_q``). 0 (default) keeps decode-mode sizing.
        index_topk: 0 for dense; > 0 for DSA, every query token contributes
            ``index_topk`` selected KV tokens.
        splitk: must match the splitk value passed to the consumer kernel.
        task_tensor: optional pre-allocated int32 buffer to reuse across
            forwards. Must be CUDA, contiguous, int32, and at least
            ``(num_logical + num_sm) * 8 + (num_sm + 1) + (num_logical + 1)``
            ints (layout: task_list | cu_tasks | cu_splits; each task_list
            entry is 8 self-describing ints carrying the token->batch map, so
            no separate iquery_to_ibatch_map segment is needed), where
            ``num_logical`` is ``num_actual_tokens`` in prefill else
            ``num_batch``. Auto-allocated if omitted.

    Returns:
        Tensor: opaque scheduler map; pass back via ``task_tensor=``.
    """
    return torch.ops.hpc.get_mla_scheduler_map(
        num_seq_kv, cu_seqlens_q, int(num_actual_tokens), int(index_topk), splitk, task_tensor
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
    mtp,
    new_kv_included,
    quant_type,
    splitk,
    task_map=None,
    split_flag=None,
    output=None,
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


@torch.library.register_fake("hpc::mla_decode_with_kvcache_bf16")
def mla_decode_with_kvcache_bf16_fake(
    q: Tensor,
    kvcache: Tensor,
    block_ids: Tensor,
    cu_seqlens_q: Tensor,
    num_seq_kv: Tensor,
    sink_weight: Tensor = None,
    softmax_scale: float = 0.0,
    task_tensor: Tensor = None,
    output: Tensor = None,
    splitk: bool = True,
):
    out_shape = (q.size(0), q.size(1), 512)
    return q.new_empty(out_shape, dtype=torch.bfloat16)


@torch.library.register_fake("hpc::sparse_mla_dsa_with_kvcache_bf16")
def sparse_mla_dsa_with_kvcache_bf16_fake(
    q: Tensor,
    kvcache: Tensor,
    block_ids: Tensor,
    topk_ids: Tensor,
    cu_seqlens_q: Tensor,
    sink_weight: Tensor = None,
    softmax_scale: float = 0.0,
    task_tensor: Tensor = None,
    output: Tensor = None,
    splitk: bool = True,
):
    out_shape = (q.size(0), q.size(1), 512)
    return q.new_empty(out_shape, dtype=torch.bfloat16)


@torch.library.register_fake("hpc::get_mla_scheduler_map")
def get_mla_scheduler_map_fake(
    num_seq_kv: Tensor,
    cu_seqlens_q: Tensor,
    num_actual_tokens: int,
    index_topk: int = 0,
    splitk: bool = True,
    task_tensor: Tensor = None,
):
    num_sm = torch.cuda.get_device_properties(num_seq_kv.device).multi_processor_count
    n = (num_actual_tokens + num_sm) * 4 + (num_sm + 1) + (num_actual_tokens + 1)
    return num_seq_kv.new_empty((n,), dtype=torch.int32)


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
