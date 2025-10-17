import torch
from torch import Tensor


def attention_prefill_bf16(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Computes attention prefill using bfloat16 precision.

    This function performs the attention prefill computation using custom hardware
    operations optimized for bfloat16 data type. The prefill stage processes all
    input tokens simultaneously to generate the initial attention context, which
    is typically used during the first forward pass of autoregressive models.

    Args:
        q: Query tensor for attention computation
            Shape: [num_batch, num_seq_q, num_head_q, num_dim_qk]
            Dtype: bfloat16
        k: Key tensor for attention computation
            Shape: [num_batch, num_seq_kv, num_head_kv, num_dim_qk]
            Dtype: bfloat16
        v: Value tensor for attention computation
            Shape: [num_batch, num_seq_kv, num_head_kv, num_dim_v]
            Dtype: bfloat16

    Returns:
        Tensor: Attention output tensor in bfloat16 format on CUDA device
            Shape: [num_batch, num_seq_q, num_head_q, num_dim_v]
            Dtype: bfloat16

    Raises:
        RuntimeError: If the shapes or dtypes do not satisfy the constraints above.

    Note:
        - All input tensors must be on CUDA device and in bfloat16 format
        - The query and key tensors must have the same embedding dimension (num_dim_qk)
        - The batch size (num_batch) must be consistent across all input tensors
    """

    return torch.ops.hpc.attention_prefill_bf16(q, k, v)


def attention_decode_bf16(
    q: Tensor,
    kvcache: Tensor,
    block_ids: Tensor,
    cache_lens: Tensor,
    output: Tensor = None,
) -> Tensor:
    """Computes attention decode using bfloat16 precision.

    This function performs the attention decode computation using custom hardware
    operations optimized for bfloat16 data type.

    Args:
        q: Query tensor for attention computation
            Shape: [num_batch * num_seq_q, num_head_q, num_dim_qk]
            Dtype: bfloat16
        kvcache: Key/Value tensor for attention computation in paged format.
                 Constrainst the unused slots in last block of vcache for each request to be set zeros.
            Shape: [num_blocks, 2, block_size, num_head_kv, num_dim_qk]
            Dtype: bfloat16
        block_ids: kvcache page block index tensor for get paged kvcache.
            Shape: [num_batch, max_blocks]
            Dtype: int32
        cache_lens: number tokens in kvcache before cur iteration.
            Shape: [num_batch]
            Dtype: int32
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
    return torch.ops.hpc.attention_decode_bf16(q, kvcache, block_ids, cache_lens, output)
