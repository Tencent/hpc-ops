from typing import Optional, Tuple

import torch
from torch import Tensor


def rope_norm_blocked_kvcache(
    key_cache: Tensor,
    value_cache: Tensor,
    qkv: Tensor,
    cos_sin: Tensor,
    num_seqlen_per_req: Tensor,
    kvcache_indices: Tensor,
    is_prefill: bool,
    use_qk_norm: bool,
    q_norm_weight: Optional[Tensor] = None,
    k_norm_weight: Optional[Tensor] = None,
    out_q: Optional[Tensor] = None,
    out_k: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Applies Rotary Position Embedding (RoPE) with blocked KV cache. Supports QK normalization.

    This function applies RoPE transformation to query and key tensors, and updates
    the KV cache pool with the transformed keys and original values. It supports both prefill and
    decode phases, and supports optional QK normalization.

    Args:
        key_cache: Key cache tensor for storing transformed key vectors.
            Shape: [num_blocks, block_size, num_kv_heads, qk_head_dim]
            Dtype: bfloat16
        value_cache: Value cache tensor for storing value vectors.
            Shape: [num_blocks, block_size, num_kv_heads, v_head_dim]
            Dtype: bfloat16
        qkv: Input QKV tensor of shape. Contains concatenated Q, K, V heads. Note that q&k share the same head dimension, k&v share the same head number.
            Shape: [num_rows, num_q_heads*qk_head_dim + num_kv_heads*qk_head_dim + num_kv_heads*v_head_dim].
            Dtype: bfloat16
        cos_sin: First half of last dimension contains cos values, second half contains sin values.
            Shape: [max_rot_position, qk_head_dim].
            Dtype: float32
        num_seqlen_per_req: Sequence length for each request.
            Shape: [num_requests].
            Dtype: int32
        kvcache_indices: Block indices in cache pool for each request.
            Shape: [num_requests, max_blocks_per_req].
            Dtype: int32
        is_prefill: Boolean flag indicating whether this is a prefill phase.
            Shape: scalar
            Dtype: bool
        use_qk_norm: Boolean flag indicating whether to apply QK normalization.
            Shape: scalar
            Dtype: bool
        q_norm_weight: Optional weight tensor for query normalization.
            Shape: [num_q_heads, qk_head_dim]
            Dtype: float32
        k_norm_weight: Optional weight tensor for key normalization.
            Shape: [num_kv_heads, qk_head_dim]
            Dtype: float32
        out_q: Optional output tensor for transformed query vectors.
            Shape: [num_rows, num_q_heads, qk_head_dim]
            Dtype: bfloat16
        out_k: Optional output tensor for transformed key vectors.
            Shape: [num_rows, num_kv_heads, qk_head_dim]
            Dtype: bfloat16

    Returns:
        A tuple of (out_q, out_k) tensors after RoPE(+ qk norm) transformation:
            - out_q: Transformed query tensor
                Shape: [num_rows, num_q_heads, qk_head_dim]
                Dtype: bfloat16
            - out_k: Transformed key tensor
                Shape: [num_rows, num_kv_heads, qk_head_dim]
                Dtype: bfloat16

    Note:
        - The function uses the neox version of RoPE transformation
        - KV cache is updated in-place with transformed K and original V
        - When out_q and out_k are provided, uses inplace operation
    """
    return torch.ops.hpc.rope_norm_blocked_kvcache(
        key_cache,
        value_cache,
        qkv,
        cos_sin,
        num_seqlen_per_req,
        kvcache_indices,
        is_prefill,
        use_qk_norm,
        q_norm_weight,
        k_norm_weight,
        out_q,
        out_k,
    )


@torch.library.register_fake("hpc::rope_norm_blocked_kvcache")
def rope_norm_blocked_kvcache_fake(
    key_cache,
    value_cache,
    qkv,
    cos_sin,
    num_seqlen_per_req,
    kvcache_indices,
    is_prefill,
    use_qk_norm,
    q_norm_weight,
    k_norm_weight,
    out_q,
    out_k,
):
    hidden_size = qkv.shape[-1]
    k_heads = key_cache.shape[-2]
    v_heads = value_cache.shape[-2]
    qk_head_dim = key_cache.shape[-1]
    v_head_dim = value_cache.shape[-1]
    q_heads = (hidden_size - k_heads * qk_head_dim - v_heads * v_head_dim) // qk_head_dim
    return (
        torch.empty(
            qkv.shape[0],
            q_heads,
            qk_head_dim,
            dtype=qkv.dtype,
            device=qkv.device,
        ),
        torch.empty(
            qkv.shape[0],
            k_heads,
            qk_head_dim,
            dtype=qkv.dtype,
            device=qkv.device,
        ),
    )
