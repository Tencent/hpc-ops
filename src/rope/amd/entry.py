# Copyright (C) 2026 Tencent.
# Copyright 2026, The FlagOS Contributors.

import torch

from hpc._amd.utils.validation import check_bf16, check_tensor, require

from .rope import rope_norm_store_kv_async


SCHEMAS = (
    "rope_norm_store_kv(Tensor! kcache, Tensor! vcache, Tensor qkv, Tensor cos_sin, Tensor num_seqlen_per_req, Tensor q_index, Tensor kvcache_indices, bool is_prefill, Tensor? q_norm_weight, Tensor? k_norm_weight, Tensor? out_q=None, Tensor? out_k=None, Tensor? out_v=None, int qk_norm_policy=0) -> Tensor",
)


def rope_norm_store_kv(
    kcache,
    vcache,
    qkv,
    cos_sin,
    num_seqlen_per_req,
    q_index,
    kvcache_indices,
    is_prefill,
    q_norm_weight,
    k_norm_weight,
    out_q=None,
    out_k=None,
    out_v=None,
    qk_norm_policy=0,
):
    check_bf16(qkv, qkv, "qkv")
    check_tensor(cos_sin, qkv, torch.float32, "cos_sin")
    check_tensor(num_seqlen_per_req, qkv, torch.int32, "num_seqlen_per_req")
    check_tensor(q_index, qkv, torch.int32, "q_index")
    check_tensor(kvcache_indices, qkv, torch.int32, "kvcache_indices")
    require(
        qkv.ndim == 2 and cos_sin.ndim == 2 and cos_sin.shape[1] == 128,
        "invalid RoPE input dimensions; head_dim must be 128",
    )
    require(
        num_seqlen_per_req.ndim == 1 and q_index.ndim == 1 and kvcache_indices.ndim == 2,
        "invalid request metadata dimensions",
    )
    for name, cache in (("key_cache", kcache), ("value_cache", vcache)):
        check_tensor(cache, qkv, torch.bfloat16, name, contiguous=False)
        require(cache.ndim == 4, "KV caches must be 4-D")
        require(
            cache.shape[3] == 128
            and cache.stride(3) == 1
            and cache.stride(2) == 128
            and cache.stride(1) == cache.shape[2] * 128
            and cache.stride(0) >= cache.shape[1] * cache.stride(1),
            "KV cache rows must be contiguous and non-overlapping",
        )
    require(
        kcache.shape == vcache.shape and kcache.shape[1] > 0,
        "K/V cache shapes must match and block_size must be positive",
    )
    requests = num_seqlen_per_req.shape[0]
    require(
        q_index.shape == (requests + 1,)
        and kvcache_indices.shape[0] == requests
        and (requests == 0 or kvcache_indices.shape[1] > 0),
        "request metadata shapes do not match",
    )
    kv_heads = kcache.shape[2]
    require(kv_heads > 0 and qkv.shape[1] % 128 == 0, "invalid QKV width")
    q_heads = qkv.shape[1] // 128 - 2 * kv_heads
    require(
        (q_heads, kv_heads) in ((8, 1), (64, 8)),
        "supported (Q heads, KV heads) are (8,1) and (64,8)",
    )
    require(qk_norm_policy in (0, 1, 2), "qk_norm_policy must be 0, 1 or 2")
    if qk_norm_policy:
        require(
            q_norm_weight is not None and k_norm_weight is not None,
            "Q/K norm weights are required for qk_norm_policy != 0",
        )
        for weight in (q_norm_weight, k_norm_weight):
            check_tensor(weight, qkv, torch.float32, "norm weight")
            require(weight.shape == (128,), "norm weight must have shape [128]")
    rows = qkv.shape[0]
    q = (
        out_q
        if out_q is not None
        else torch.empty((rows, q_heads, 128), device=qkv.device, dtype=qkv.dtype)
    )
    for output, heads in ((q, q_heads), (out_k, kv_heads), (out_v, kv_heads)):
        if output is not None:
            check_bf16(output, qkv, "output")
            require(output.shape == (rows, heads, 128), "output has an invalid shape")
    rope_norm_store_kv_async(
        kcache,
        vcache,
        qkv,
        cos_sin,
        num_seqlen_per_req,
        q_index,
        kvcache_indices,
        q_norm_weight,
        k_norm_weight,
        q,
        out_k,
        out_v,
        qk_norm_policy,
    )
    return q


def register(library):
    implementations = {
        "rope_norm_store_kv": rope_norm_store_kv,
    }
    for schema in SCHEMAS:
        library.define(schema)
        name = schema.split("(", 1)[0]
        library.impl(name, implementations[name], "CUDA")


# TODO(amd): rope_norm_store_kv_fp8 requires an FP8 kernel.
