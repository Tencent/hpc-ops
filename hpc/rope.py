from typing import Optional, Tuple

import torch
from torch import Tensor


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
