from typing import Optional, Tuple

import torch
from torch import Tensor


def rope_norm_blocked_kvcache(
    key_cache: Tensor,
    value_cache: Tensor,
    qkv: Tensor,
    cos_sin: Tensor,
    num_seqlen_per_req: Tensor,
    q_index: Tensor,
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
        q_index: Query indices for each request.
            Shape: [num_requests + 1].
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
        q_index,
        kvcache_indices,
        is_prefill,
        use_qk_norm,
        q_norm_weight,
        k_norm_weight,
        out_q,
        out_k,
    )


def rope_norm_blocked_kvcache_w8c8_dqskv(
    key_cache: Tensor,
    value_cache: Tensor,
    qkv: Tensor,
    cos_sin: Tensor,
    num_seqlen_per_req: Tensor,
    q_index: Tensor,
    kvcache_indices: Tensor,
    is_prefill: bool,
    use_qk_norm: bool,
    max_seqlens: int,
    k_scale: Tensor,
    v_scale: Tensor,
    q_norm_weight: Optional[Tensor] = None,
    k_norm_weight: Optional[Tensor] = None,
    upper_max: Optional[float] = None,
    out_q: Optional[Tensor] = None,
    out_k: Optional[Tensor] = None,
    out_attention: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Applies Rotary Position Embedding (RoPE) with blocked KV cache. Supports QK normalization.

    This function applies RoPE transformation to query and key tensors, and updates
    the KV cache pool with the transformed keys and original values. It supports both prefill and
    decode phases, and supports optional QK normalization.

    Quantization:
    - dqskv means dynamic quantization for Q, static quantization for K and V
    - Q,K,V are quantized from bf16 to fp8
    - Q using dynamic quantization, per token and per head
    - KV using static quantization, per tensor

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
        q_index: Query indices for each request.
            Shape: [num_requests + 1].
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
        max_seqlens: Maximum sequence length for each request.
            Shape: scalar
            Dtype: int
        k_scale: Scale tensor for key quantization.
            Shape: [1]
            Dtype: float32
        v_scale: Scale tensor for value quantization.
            Shape: [1]
            Dtype: float32
        q_norm_weight: Optional weight tensor for query normalization.
            Shape: [num_q_heads, qk_head_dim]
            Dtype: float32
        k_norm_weight: Optional weight tensor for key normalization.
            Shape: [num_kv_heads, qk_head_dim]
            Dtype: float32
        upper_max: Upper bound for quantization.
            Shape: scalar
            Dtype: float
        out_q: Optional output tensor for transformed query vectors.
            Shape: [num_rows, num_q_heads, qk_head_dim]
            Dtype: bfloat16
        out_k: Optional output tensor for transformed key vectors.
            Shape: [num_rows, num_kv_heads, qk_head_dim]
            Dtype: bfloat16
        out_attention: Optional output tensor for attention output, used in this op for building TMA descriptor.
            Shape: [num_rows, num_q_heads, v_head_dim]
            Dtype: bfloat16

    Returns:
        A tuple of (out_q, out_k, q_scale, split_k_flag, out_attention, tma_tensor) tensors after RoPE(+ qk norm) transformation:
            - out_q: Transformed query tensor
                Shape: [num_rows, num_q_heads, qk_head_dim]
                Dtype: bfloat16
            - out_k: Transformed key tensor
                Shape: [num_rows, num_kv_heads, qk_head_dim]
                Dtype: bfloat16
            - q_scale: Scale tensor for query quantization. Return different scales shape for prefill and decoding.
                Shape: [num_rows, num_q_heads] for decoding, [num_requests, num_q_heads, max_seqlen] for prefill
                Dtype: float32
            - split_k_flag: Temporary tensor for split K.
                Shape: [num_requests, num_kv_heads]
                Dtype: int32
            - out_attention: Attention output tensor allocated for later attention calculation.
                Shape: [num_rows, num_q_heads, v_head_dim]
                Dtype: bfloat16
            - tma_tensor: TMA tensor for later attention calculation. 2 means TMA tensors for Q and Output, respectively.
                Shape: [num_requests, 2, 128]
                Dtype: uint8

    Note:
        - The function uses the neox version of RoPE transformation
        - KV cache is updated in-place with transformed K and original V
        - When out_q and out_k are provided, uses inplace operation
    """
    return torch.ops.hpc.rope_norm_blocked_kvcache_w8c8_dqskv(
        key_cache,
        value_cache,
        qkv,
        cos_sin,
        num_seqlen_per_req,
        q_index,
        kvcache_indices,
        is_prefill,
        use_qk_norm,
        max_seqlens,
        k_scale,
        v_scale,
        q_norm_weight,
        k_norm_weight,
        upper_max,
        out_q,
        out_k,
        out_attention,
    )


def rope_interleave(
    input: Tensor,
    cos_sin_cache: Tensor,
    cu_seqlen: Tensor,
    seqlen_kv: Tensor,
    output: Optional[Tensor] = None,
):
    return torch.ops.hpc.rope_interleave(input, cos_sin_cache, cu_seqlen, seqlen_kv, output)


@torch.library.register_fake("hpc::rope_norm_blocked_kvcache")
def rope_norm_blocked_kvcache_fake(
    key_cache,
    value_cache,
    qkv,
    cos_sin,
    num_seqlen_per_req,
    q_index,
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


@torch.library.register_fake("hpc::rope_norm_blocked_kvcache_w8c8_dqskv")
def rope_norm_blocked_kvcache_w8c8_dqskv_fake(
    key_cache,
    value_cache,
    qkv,
    cos_sin,
    num_seqlen_per_req,
    q_index,
    kvcache_indices,
    is_prefill,
    use_qk_norm,
    max_seqlens,
    k_scale,
    v_scale,
    q_norm_weight,
    k_norm_weight,
    upper_max,
    out_q,
    out_k,
    out_attention,
):
    num_rows = qkv.shape[0]
    hidden_size = qkv.shape[-1]
    k_heads = key_cache.shape[-2]
    v_heads = value_cache.shape[-2]
    qk_head_dim = key_cache.shape[-1]
    v_head_dim = value_cache.shape[-1]
    num_request = num_seqlen_per_req.shape[0]
    q_heads = (hidden_size - k_heads * qk_head_dim - v_heads * v_head_dim) // qk_head_dim
    max_seqlens_pad128 = ((max_seqlens + 127) // 128) * 128
    if is_prefill:
        return (
            torch.empty(
                num_rows,
                q_heads,
                qk_head_dim,
                dtype=qkv.dtype,
                device=qkv.device,
            ),
            torch.empty(
                num_rows,
                k_heads,
                qk_head_dim,
                dtype=qkv.dtype,
                device=qkv.device,
            ),
            torch.empty(
                num_request,
                q_heads,
                max_seqlens_pad128,
                dtype=k_scale.dtype,
                device=k_scale.device,
            ),
            torch.empty(
                num_request,
                k_heads,
                dtype=torch.int32,
                device=qkv.device,
            ),
            torch.empty(
                num_rows,
                q_heads,
                v_head_dim,
                dtype=qkv.dtype,
                device=qkv.device,
            ),
            torch.empty(
                num_request,
                2,  # TMA tensors for Q and Output, respectively
                128,  # TMA tensor size of each TMA descriptor
                dtype=torch.uint8,
                device=qkv.device,
            ),
        )
    else:
        return (
            torch.empty(
                num_rows,
                q_heads,
                qk_head_dim,
                dtype=qkv.dtype,
                device=qkv.device,
            ),
            torch.empty(
                num_rows,
                k_heads,
                qk_head_dim,
                dtype=qkv.dtype,
                device=qkv.device,
            ),
            torch.empty(
                num_rows,
                q_heads,
                dtype=k_scale.dtype,
                device=k_scale.device,
            ),
            torch.empty(
                num_rows,
                k_heads,
                dtype=torch.int32,
                device=qkv.device,
            ),
            torch.empty(
                num_rows,
                q_heads,
                v_head_dim,
                dtype=qkv.dtype,
                device=qkv.device,
            ),
            torch.empty(
                num_request,
                2,  # TMA tensors for Q and Output, respectively
                128,  # TMA tensor size of each TMA descriptor
                dtype=torch.uint8,
                device=qkv.device,
            ),
        )


@torch.library.register_fake("hpc::rope_interleave")
def rope_interleave_fake(
    input: Tensor,
    cos_sin_cache: Tensor,
    cu_seqlen: Tensor,
    seqlen_kv: Tensor,
    output: Optional[Tensor] = None,
):
    return torch.empty_like(input)
