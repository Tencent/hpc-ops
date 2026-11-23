# Copyright (C) 2026 Tencent.

import torch
import triton
import triton.language as tl

from ._validation import check_bf16, check_tensor, require


@triton.jit
def _norm_pair(first, second, weight, offsets):
    inv = tl.rsqrt(tl.sum(first * first + second * second, 0) / 128 + 1e-6)
    first = first * inv * tl.load(weight + offsets)
    second = second * inv * tl.load(weight + offsets + 64)
    return first, second


@triton.jit
def _store_pair(destination, first, second, offsets):
    tl.store(destination + offsets, first)
    tl.store(destination + offsets + 64, second)


@triton.jit
def _rope_kernel(
    QKV,
    COS_SIN,
    LENGTHS,
    QINDEX,
    BLOCK_IDS,
    Q_WEIGHT,
    K_WEIGHT,
    Q,
    KCACHE,
    VCACHE,
    OUT_K,
    OUT_V,
    ROWS: tl.constexpr,
    REQUESTS: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    K_STRIDE: tl.constexpr,
    V_STRIDE: tl.constexpr,
    POLICY: tl.constexpr,
    SEPARATE_K: tl.constexpr,
    SEPARATE_V: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    if row >= ROWS:
        if head >= Q_HEADS:
            request = row - ROWS
            length = tl.load(LENGTHS + request)
            if length > 0:
                last = length - 1
                physical = tl.load(
                    BLOCK_IDS + request.to(tl.int64) * MAX_BLOCKS + last // BLOCK_SIZE
                )
                for start in range(0, BLOCK_SIZE * 128, 256):
                    idx = start + tl.arange(0, 256)
                    offset = ((idx // 128) * KV_HEADS + head - Q_HEADS) * 128 + idx % 128
                    mask = (idx // 128 > last % BLOCK_SIZE) & (idx < BLOCK_SIZE * 128)
                    tl.store(KCACHE + physical.to(tl.int64) * K_STRIDE + offset, 0, mask)
                    tl.store(VCACHE + physical.to(tl.int64) * V_STRIDE + offset, 0, mask)
    else:
        lo = tl.full((), 0, tl.int32)
        hi = tl.full((), REQUESTS + 1, tl.int32)
        while lo < hi:
            mid = (lo + hi) // 2
            boundary = tl.load(QINDEX + mid)
            lo = tl.where(boundary <= row, mid + 1, lo)
            hi = tl.where(boundary <= row, hi, mid)
        request = lo - 1
        if request >= 0 and request < REQUESTS:
            length = tl.load(LENGTHS + request)
            if length > 0:
                position = row + length - tl.load(QINDEX + request + 1)
                if position >= 0:
                    physical = tl.load(
                        BLOCK_IDS + request.to(tl.int64) * MAX_BLOCKS + position // BLOCK_SIZE
                    )
                    offsets = tl.arange(0, 64)
                    input_offset = row.to(tl.int64) * (Q_HEADS + 2 * KV_HEADS) * 128
                    source = QKV + input_offset + head * 128
                    first = tl.load(source + offsets).to(tl.float32)
                    second = tl.load(source + offsets + 64).to(tl.float32)
                    if POLICY == 2:
                        if head < Q_HEADS:
                            first, second = _norm_pair(first, second, Q_WEIGHT, offsets)
                        else:
                            first, second = _norm_pair(first, second, K_WEIGHT, offsets)
                    cosine = tl.load(COS_SIN + position.to(tl.int64) * 128 + offsets)
                    sine = tl.load(COS_SIN + position.to(tl.int64) * 128 + offsets + 64)
                    rotated_first = first * cosine - second * sine
                    second = second * cosine + first * sine
                    first = rotated_first
                    if POLICY == 1:
                        if head < Q_HEADS:
                            first, second = _norm_pair(first, second, Q_WEIGHT, offsets)
                        else:
                            first, second = _norm_pair(first, second, K_WEIGHT, offsets)
                    if head < Q_HEADS:
                        _store_pair(
                            Q + (row.to(tl.int64) * Q_HEADS + head) * 128, first, second, offsets
                        )
                    else:
                        kv_head = head - Q_HEADS
                        kv_offset = (position % BLOCK_SIZE * KV_HEADS + kv_head) * 128
                        if SEPARATE_K:
                            _store_pair(
                                OUT_K + (row.to(tl.int64) * KV_HEADS + kv_head) * 128,
                                first,
                                second,
                                offsets,
                            )
                        else:
                            _store_pair(
                                KCACHE + physical.to(tl.int64) * K_STRIDE + kv_offset,
                                first,
                                second,
                                offsets,
                            )
                        vi = tl.arange(0, 128)
                        value = tl.load(
                            QKV + input_offset + (Q_HEADS + KV_HEADS + kv_head) * 128 + vi
                        )
                        if SEPARATE_V:
                            tl.store(
                                OUT_V + (row.to(tl.int64) * KV_HEADS + kv_head) * 128 + vi, value
                            )
                        else:
                            tl.store(
                                VCACHE + physical.to(tl.int64) * V_STRIDE + kv_offset + vi, value
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
    if requests:
        with torch.cuda.device(qkv.device):
            _rope_kernel[(rows + requests, q_heads + kv_heads)](
                qkv,
                cos_sin,
                num_seqlen_per_req,
                q_index,
                kvcache_indices,
                q_norm_weight if qk_norm_policy else None,
                k_norm_weight if qk_norm_policy else None,
                q,
                kcache,
                vcache,
                out_k,
                out_v,
                rows,
                requests,
                q_heads,
                kv_heads,
                kcache.shape[1],
                kvcache_indices.shape[1],
                kcache.stride(0),
                vcache.stride(0),
                qk_norm_policy,
                out_k is not None,
                out_v is not None,
                num_warps=1,
                enable_fp_fusion=False,
            )
    return q


# TODO(amd): rope_norm_store_kv_fp8 requires an FP8 kernel.
