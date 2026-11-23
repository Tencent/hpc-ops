# Copyright (C) 2026 Tencent.

import triton
import triton.language as tl


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
