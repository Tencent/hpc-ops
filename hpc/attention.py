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
