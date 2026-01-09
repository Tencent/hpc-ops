import torch
from torch import Tensor


def mqa_indexer_logits(
    q: Tensor,
    kvcache: Tensor,
    weight: Tensor,
    block_ids: Tensor,
    cu_seqlens_q: Tensor,
    seqlens_kv: Tensor,
    ratio: int,
    max_context_len: int,
    output: Tensor = None,
) -> Tensor:
    """Computes qk logits for Query and Compressed KV Cache.

    Formula: P = (ReLU(Q @ K) * W), and output Y is P reduce along num_head_q

    Args:
        q: Query tensor for attention computation
            Shape: [total_seq, num_head_q, head_dim]
            Dtype: bfloat16
        kvcache: Compressed KV cache tensor for attention computation in paged format.
            Shape: [num_blocks, block_size, head_dim]
            Dtype: bfloat16
        weight: weight for weighted sum along num_head_q.
            Shape: [total_seq, num_head_q]
            Dtype: bfloat16
        block_ids: kvcache page block index tensor for get paged kvcache.
            Shape: [num_batch, max_blocks]
            Dtype: int32
        cu_seqlens_q: start_seq_q for each batch
            Shape: [num_batch + 1]
            Dtype: int32
        seqlens_kv: number tokens in kvcache contain the cur query.
            Shape: [num_batch]
            Dtype: int32
        ratio: Compressed ratio for KV cache.
            Shape: scalar
            Dtype: int
        max_context_len: max context len for output.
            Shape: scalar
            Dtype: int

    Returns:
        Tensor: output logits tensor in bfloat16 format on CUDA device
            Shape: [total_seq, max_context_len]
            Dtype: float32

    Raises:
        RuntimeError: If the shapes or dtypes do not satisfy the constraints above.

    Note:
        - All input tensors must be on CUDA device and in bfloat16 format
        - The query and kvcache tensors must have the same embedding dimension (head_dim)
        - total_seq = sum(seqlens_q[ibatch] for ibatch in range(num_batch))
    """

    return torch.ops.hpc.mqa_indexer_logits(
        q,
        kvcache,
        weight,
        block_ids,
        cu_seqlens_q,
        seqlens_kv,
        ratio,
        max_context_len,
        output,
    )
