# Copyright (C) 2026 Tencent.

import triton
import triton.language as tl


@triton.jit
def _stem_tpd_kernel(
    X,
    Y,
    QLEN,
    KLEN,
    PROMPT,
    HEADS: tl.constexpr,
    QROWS: tl.constexpr,
    KCOLS: tl.constexpr,
    BLOCK: tl.constexpr,
    block_size,
    alpha,
    initial_blocks,
    window_size,
    medium_rate,
    medium_bias,
    large_rate,
    large_bias,
):
    row = tl.program_id(0)
    request = row // (HEADS * QROWS)
    qrow = row % QROWS
    cols = tl.arange(0, BLOCK)
    qlen = tl.load(QLEN + request)
    klen = tl.load(KLEN + request)
    qblocks = tl.cdiv(qlen, block_size)
    kblocks = tl.cdiv(klen, block_size)
    prompt = tl.cdiv(tl.load(PROMPT + request), block_size)
    offset = tl.cdiv(klen - qlen, block_size)
    live = (qrow < qblocks) & (qblocks > 0) & (kblocks > 0)
    x = tl.load(
        X + row.to(tl.int64) * KCOLS + cols,
        live & (cols < tl.minimum(kblocks, KCOLS)),
        other=float("-inf"),
    )
    bits = x.to(tl.uint16, bitcast=True).to(tl.uint32)
    finite = (bits & 0x7F80) != 0x7F80
    ordered = tl.where(bits & 0x8000 != 0, (~bits) & 0xFFFF, bits ^ 0x8000)
    ordered = tl.where(finite, ordered, 0x7F)
    k = tl.where(
        prompt < 56,
        prompt,
        tl.where(
            prompt < 160,
            (prompt * medium_rate).to(tl.int32) + medium_bias,
            (prompt * large_rate).to(tl.int32) + large_bias,
        ),
    )
    qpos = qrow + offset
    decay = prompt - k
    ratio = (qpos - k).to(tl.float32) / tl.maximum(decay - 1, 1)
    decayed = tl.floor(k + ratio * (k * alpha - k)).to(tl.int32)
    budget = tl.where((qpos < k) | (decay <= 1), k, tl.maximum(1, tl.minimum(k, decayed)))
    threshold = tl.full((), 0x80, tl.uint32)
    total_finite = tl.sum(finite.to(tl.int32), 0)
    if budget < total_finite:
        threshold = tl.full((), 0, tl.uint32)
        for bit in range(15, -1, -1):
            candidate = threshold | (1 << bit).to(tl.uint32)
            count = tl.sum((ordered >= candidate).to(tl.int32), 0)
            threshold = tl.where(count >= budget, candidate, threshold)
    diagonal = tl.minimum(qpos, kblocks - 1)
    selected = (ordered >= threshold) | (cols < initial_blocks)
    selected |= ((cols <= diagonal) & (cols > diagonal - window_size)) | (cols == diagonal)
    selected &= live & (cols < kblocks)
    tl.store(Y + row.to(tl.int64) * KCOLS + cols, selected.to(tl.uint8), cols < KCOLS)
