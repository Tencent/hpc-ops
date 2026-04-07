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
    qk_norm_policy: Optional[int] = 1,
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
        qk_norm_policy: Optional policy for QK normalization. 1 means rope first, 2 means norm first.
            Shape: scalar
            Dtype: int

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
    if not use_qk_norm:
        qk_norm_policy = 0  # 1 means rope first, 2 means norm first
    return torch.ops.hpc.rope_norm_blocked_kvcache(
        key_cache,
        value_cache,
        qkv,
        cos_sin,
        num_seqlen_per_req,
        q_index,
        kvcache_indices,
        is_prefill,
        qk_norm_policy,
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
    qk_norm_policy: Optional[int] = 1,
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
        qk_norm_policy: Policy for QK normalization. 1 means rope first, 2 means norm first.
            Shape: scalar
            Dtype: int

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
    if not use_qk_norm:
        qk_norm_policy = 0  # 1 means rope first, 2 means norm first
    return torch.ops.hpc.rope_norm_blocked_kvcache_w8c8_dqskv(
        key_cache,
        value_cache,
        qkv,
        cos_sin,
        num_seqlen_per_req,
        q_index,
        kvcache_indices,
        is_prefill,
        qk_norm_policy,
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


def rope_norm_w8c8(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cos_sin: Tensor,
    num_seqlen_per_req: Tensor,
    q_index: Tensor,
    is_prefill: bool,
    max_seqlens: int,
    k_scale: Tensor,
    v_scale: Tensor,
    qk_norm_policy: int = 0,
    q_norm_weight: Optional[Tensor] = None,
    k_norm_weight: Optional[Tensor] = None,
    upper_max: Optional[float] = None,
    out_q: Optional[Tensor] = None,
    out_k: Optional[Tensor] = None,
    out_v: Optional[Tensor] = None,
    out_attention: Optional[Tensor] = None,
):
    return torch.ops.hpc.rope_norm_w8c8(
        q,
        k,
        v,
        cos_sin,
        num_seqlen_per_req,
        q_index,
        is_prefill,
        max_seqlens,
        k_scale,
        v_scale,
        qk_norm_policy,
        q_norm_weight,
        k_norm_weight,
        upper_max,
        out_q,
        out_k,
        out_v,
        out_attention,
    )


def rope_interleave(
    input: Tensor,
    cos_sin_cache: Tensor,
    position: Tensor,
    output: Optional[Tensor] = None,
):
    """Applies Rotary Position Embedding (RoPE) with blocked KV cache. Supports QK normalization.

    This function applies RoPE transformation.
    Args:
        input: Input tensor for apply rope.
            Shape: [num_tokens, num_heads, dim]
            Dtype: bfloat16
        cos_sin_cache: cos_sin_cache.
            Shape: [max_seqlen, dim]
            Dtype: float32
        position: position in request
            Shape: [num_token]
            Dtype: int64
    Returns:
        Tensor: Attention output tensor in bfloat16
            Shape: [num_tokens, num_heads, dim]
            Dtype: bfloat16
    """
    return torch.ops.hpc.rope_interleave(input, cos_sin_cache, position, output)


def rope_norm_store_kv(
    key_cache: Tensor,
    value_cache: Tensor,
    qkv: Tensor,
    cos_sin: Tensor,
    num_seqlen_per_req: Tensor,
    q_index: Tensor,
    kvcache_indices: Tensor,
    is_prefill: bool,
    q_norm_weight: Optional[Tensor] = None,
    k_norm_weight: Optional[Tensor] = None,
    out_q: Optional[Tensor] = None,
    out_k: Optional[Tensor] = None,
    out_v: Optional[Tensor] = None,
    qk_norm_policy: int = 0,
) -> Tensor:
    """Applies RoPE to Q/K, optionally applies QK RMSNorm, and writes K/V into a paged KV cache.

    This function fuses RoPE rotation, optional QK RMSNorm, and blocked KV-cache writes
    into a single CUDA kernel pass, supporting both prefill and decode modes.

    Args:
        key_cache: Paged key cache to be updated in-place.
            Shape: [num_blocks, block_size, num_kv_heads, qk_head_dim]
            Dtype: bfloat16
        value_cache: Paged value cache to be updated in-place.
            Shape: [num_blocks, block_size, num_kv_heads, v_head_dim]
            Dtype: bfloat16
        qkv: Packed Q/K/V input tensor.
            Shape: [num_rows, num_q_heads * qk_head_dim + num_kv_heads * qk_head_dim + num_kv_heads * v_head_dim]
            Dtype: bfloat16
        cos_sin: Precomputed RoPE cosine/sine table.
            Shape: [max_seq_len, qk_head_dim]
            Dtype: float32
        num_seqlen_per_req: Current total sequence length (including new tokens) for each request.
            Shape: [num_req]
            Dtype: int32
        q_index: Prefix-sum index of Q tokens across requests.
            Shape: [num_req + 1]
            Dtype: int32
        kvcache_indices: Physical block index table for paged KV cache addressing.
            Shape: [num_req, max_blocks]
            Dtype: int32
        is_prefill: Whether to run in prefill mode (True) or decode mode (False).
            Shape: scalar
            Dtype: bool
        q_norm_weight: RMSNorm weight for Q. Required when qk_norm_policy != 0.
            Shape: [qk_head_dim]
            Dtype: float32
        k_norm_weight: RMSNorm weight for K. Required when qk_norm_policy != 0.
            Shape: [qk_head_dim]
            Dtype: float32
        out_q: Optional pre-allocated output buffer for Q.
            Shape: [num_rows, num_q_heads, qk_head_dim]
            Dtype: bfloat16
        out_k: Optional output buffer for K. If provided, K is written here instead of key_cache.
            Shape: [num_rows, num_kv_heads, qk_head_dim]
            Dtype: bfloat16
        out_v: Optional output buffer for V. If provided, V is written here instead of value_cache.
            Shape: [num_rows, num_kv_heads, v_head_dim]
            Dtype: bfloat16
        qk_norm_policy: Controls whether RMSNorm is applied and its order relative to RoPE.
            Shape: scalar
            Dtype: int
            - 0: No RMSNorm.
            - 1: RoPE then RMSNorm.
            - 2: RMSNorm then RoPE.

    Returns:
        Tensor: Rotated (and optionally normalized) Q tensor.
            Shape: [num_rows, num_q_heads, qk_head_dim]
            Dtype: bfloat16

    Raises:
        RuntimeError: If the shapes or dtypes do not satisfy the constraints above.
    """
    return torch.ops.hpc.rope_norm_store_kv(
        key_cache,
        value_cache,
        qkv,
        cos_sin,
        num_seqlen_per_req,
        q_index,
        kvcache_indices,
        is_prefill,
        q_norm_weight,
        k_norm_weight,
        out_q,
        out_k,
        out_v,
        qk_norm_policy,
    )


def rope_norm_store_kv_fp8(
    key_cache: Tensor,
    value_cache: Tensor,
    qkv: Tensor,
    cos_sin: Tensor,
    num_seqlen_per_req: Tensor,
    q_index: Tensor,
    kvcache_indices: Tensor,
    is_prefill: bool,
    k_scale: Tensor,
    v_scale: Tensor,
    quant_policy: int,
    max_seqlens: int = 0,
    upper_max: Optional[float] = None,
    q_scale_inv: Optional[Tensor] = None,
    q_norm_weight: Optional[Tensor] = None,
    k_norm_weight: Optional[Tensor] = None,
    out_q: Optional[Tensor] = None,
    out_k: Optional[Tensor] = None,
    out_v: Optional[Tensor] = None,
    qk_norm_policy: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Applies RoPE to Q/K with FP8 quantization, optionally applies QK RMSNorm, and writes K/V into a paged FP8 KV cache.

    Extends rope_norm_store_kv with FP8 quantization for Q output and KV cache storage,
    supporting dynamic per-token per-head (dqskv) and static (sqskv) quantization policies.

    Args:
        key_cache: Paged key cache to be updated in-place.
            Shape: [num_blocks, block_size, num_kv_heads, qk_head_dim]
            Dtype: float8_e4m3fn
        value_cache: Paged value cache to be updated in-place.
            Shape: [num_blocks, block_size, num_kv_heads, v_head_dim]
            Dtype: float8_e4m3fn
        qkv: Packed Q/K/V input tensor.
            Shape: [num_rows, num_q_heads * qk_head_dim + num_kv_heads * qk_head_dim + num_kv_heads * v_head_dim]
            Dtype: bfloat16
        cos_sin: Precomputed RoPE cosine/sine table.
            Shape: [max_seq_len, qk_head_dim]
            Dtype: float32
        num_seqlen_per_req: Current total sequence length (including new tokens) for each request.
            Shape: [num_req]
            Dtype: int32
        q_index: Prefix-sum index of Q tokens across requests.
            Shape: [num_req + 1]
            Dtype: int32
        kvcache_indices: Physical block index table for paged KV cache addressing.
            Shape: [num_req, max_blocks]
            Dtype: int32
        is_prefill: Whether to run in prefill mode (True) or decode mode (False).
            Shape: scalar
            Dtype: bool
        k_scale: Static quantization scale for K. Per-tensor.
            Shape: [1]
            Dtype: float32
        v_scale: Static quantization scale for V. Per-tensor.
            Shape: [1]
            Dtype: float32
        quant_policy: Q quantization mode. K/V always use static scaling.
            Shape: scalar
            Dtype: int
            - 1: dqskv — dynamic per-token per-head quantization; scale computed by the kernel
                 and written to the returned q_scale tensor.
            - 2: sqskv — static quantization; uses the caller-supplied q_scale_inv.
        max_seqlens: Maximum sequence length in the batch. Used to size the q_scale allocation
            in prefill mode (padded to a multiple of 128).
            Shape: scalar
            Dtype: int
        upper_max: FP8 saturation upper bound. Defaults to FP8_MAX (~448.0).
            Shape: scalar
            Dtype: float
        q_scale_inv: Static scale reciprocal for Q. Required when quant_policy=2.
            Shape: [1]
            Dtype: float32
        q_norm_weight: RMSNorm weight for Q. Required when qk_norm_policy != 0.
            Shape: [qk_head_dim]
            Dtype: float32
        k_norm_weight: RMSNorm weight for K. Required when qk_norm_policy != 0.
            Shape: [qk_head_dim]
            Dtype: float32
        out_q: Optional pre-allocated output buffer for Q.
            Shape: [num_rows, num_q_heads, qk_head_dim]
            Dtype: float8_e4m3fn
        out_k: Optional output buffer for K. If provided, K is written here instead of key_cache.
            Shape: [num_rows, num_kv_heads, qk_head_dim]
            Dtype: float8_e4m3fn
        out_v: Optional output buffer for V. If provided, V is written here instead of value_cache.
            Shape: [num_rows, num_kv_heads, v_head_dim]
            Dtype: float8_e4m3fn
        qk_norm_policy: Controls whether RMSNorm is applied and its order relative to RoPE.
            Shape: scalar
            Dtype: int
            - 0: No RMSNorm.
            - 1: RoPE then RMSNorm.
            - 2: RMSNorm then RoPE.

    Returns:
        Tuple of:
        - out_q_fp8 (Tensor): Rotated (and optionally normalized) Q tensor quantized to FP8.
            Shape: [num_rows, num_q_heads, qk_head_dim]
            Dtype: float8_e4m3fn
        - q_scale (Tensor): Dynamic per-token per-head Q scale (dqskv only).
            Prefill shape: [num_req, num_q_heads, max_seqlens_pad128]; Decode shape: [num_rows, num_q_heads].
            Empty tensor when quant_policy=2.
            Dtype: float32
        - split_k_flag (Tensor): Per-request per-KV-head flag zeroed by the kernel, used by downstream attention.
            Shape: [num_req, num_kv_heads]
            Dtype: int32

    Raises:
        RuntimeError: If the shapes or dtypes do not satisfy the constraints above.
    """
    return torch.ops.hpc.rope_norm_store_kv_fp8(
        key_cache,
        value_cache,
        qkv,
        cos_sin,
        num_seqlen_per_req,
        q_index,
        kvcache_indices,
        is_prefill,
        k_scale,
        v_scale,
        quant_policy,
        max_seqlens,
        upper_max,
        q_scale_inv,
        q_norm_weight,
        k_norm_weight,
        out_q,
        out_k,
        out_v,
        qk_norm_policy,
    )


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
    out_q = torch.empty(
        num_rows,
        q_heads,
        qk_head_dim,
        dtype=qkv.dtype,
        device=qkv.device,
    )
    out_k = torch.empty(
        num_rows,
        k_heads,
        qk_head_dim,
        dtype=qkv.dtype,
        device=qkv.device,
    )
    q_scale = None
    if is_prefill:
        q_scale = torch.empty(
            num_request,
            q_heads,
            max_seqlens_pad128,
            dtype=k_scale.dtype,
            device=k_scale.device,
        )
    else:
        q_scale = torch.empty(
            num_rows,
            q_heads,
            dtype=k_scale.dtype,
            device=k_scale.device,
        )

    split_k_flag = torch.empty(
        num_request,
        k_heads,
        dtype=torch.int32,
        device=qkv.device,
    )
    out_attention = torch.empty(
        num_rows,
        q_heads,
        v_head_dim,
        dtype=qkv.dtype,
        device=qkv.device,
    )
    tma_tensor = torch.empty(
        num_request,
        2,  # TMA tensors for Q and Output, respectively
        128,  # TMA tensor size of each TMA descriptor
        dtype=torch.uint8,
        device=qkv.device,
    )
    return (out_q, out_k, q_scale, split_k_flag, out_attention, tma_tensor)


@torch.library.register_fake("hpc::rope_interleave")
def rope_interleave_fake(
    input: Tensor,
    cos_sin_cache: Tensor,
    position: Tensor,
    output: Optional[Tensor] = None,
):
    return torch.empty_like(input)


@torch.library.register_fake("hpc::rope_norm_store_kv")
def rope_norm_store_kv_fake(
    key_cache,
    value_cache,
    qkv,
    cos_sin,
    num_seqlen_per_req,
    q_index,
    kvcache_indices,
    is_prefill,
    q_norm_weight,
    k_norm_weight,
    out_q,
    out_k,
    out_v,
    qk_norm_policy,
):
    hidden_size = qkv.shape[-1]
    kv_heads = key_cache.shape[-2]
    qk_head_dim = key_cache.shape[-1]
    v_head_dim = value_cache.shape[-1]
    q_heads = (hidden_size - kv_heads * qk_head_dim - kv_heads * v_head_dim) // qk_head_dim
    num_rows = qkv.shape[0]
    return torch.empty(num_rows, q_heads, qk_head_dim, dtype=qkv.dtype, device=qkv.device)


@torch.library.register_fake("hpc::rope_norm_store_kv_fp8")
def rope_norm_store_kv_fp8_fake(
    key_cache,
    value_cache,
    qkv,
    cos_sin,
    num_seqlen_per_req,
    q_index,
    kvcache_indices,
    is_prefill,
    k_scale,
    v_scale,
    quant_policy,
    max_seqlens,
    upper_max,
    q_scale_inv,
    q_norm_weight,
    k_norm_weight,
    out_q,
    out_k,
    out_v,
    qk_norm_policy,
):
    num_rows = qkv.shape[0]
    qk_dim = key_cache.shape[-1]
    kv_heads = key_cache.shape[-2]
    v_dim = value_cache.shape[-1]
    num_req = num_seqlen_per_req.shape[0]
    q_heads = (qkv.shape[-1] - kv_heads * qk_dim - kv_heads * v_dim) // qk_dim

    out_q_fp8 = torch.empty(
        num_rows,
        q_heads,
        qk_dim,
        dtype=torch.float8_e4m3fn,
        device=qkv.device,
    )

    if quant_policy == 1:  # dq skv
        if is_prefill:
            aligned = ((max_seqlens + 127) // 128) * 128
            q_scale = torch.empty(
                num_req,
                q_heads,
                aligned,
                dtype=torch.float32,
                device=qkv.device,
            )
        else:
            q_scale = torch.empty(
                num_rows,
                q_heads,
                dtype=torch.float32,
                device=qkv.device,
            )
    else:
        q_scale = None

    split_k_flag = torch.empty(
        num_req,
        kv_heads,
        dtype=torch.int32,
        device=qkv.device,
    )
    return (out_q_fp8, q_scale, split_k_flag)
