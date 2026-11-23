# Copyright (C) 2026 Tencent.

import triton
import triton.language as tl


@triton.jit
def _ordered(x):
    bits = x.to(tl.uint32, bitcast=True)
    key = tl.where(bits & 0x80000000 != 0, ~bits, bits ^ 0x80000000)
    return tl.where(x != x, 0xFFFFFFFF, key).to(tl.uint32)


@triton.jit
def _select_kernel(
    X,
    LENGTHS,
    LIVE,
    Y,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
    HAS_LIVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    live = tl.load(LIVE) if HAS_LIVE else 2147483647
    if row < live:
        length = tl.minimum(tl.maximum(tl.load(LENGTHS + row), 0), WIDTH) if HAS_LENGTHS else WIDTH
        cols = tl.arange(0, BLOCK)
        if length <= TOPK:
            for start in range(0, length, BLOCK):
                indices = start + cols
                tl.store(Y + row.to(tl.int64) * OUT_STRIDE + indices, indices, indices < length)
        else:
            bins = tl.arange(0, 256)
            prefix = tl.full((), 0, tl.uint32)
            prefix_mask = tl.full((), 0, tl.uint32)
            remaining = tl.full((), TOPK, tl.int32)
            for shift in range(24, -1, -8):
                histogram = tl.full((256,), 0, tl.int32)
                for start in range(0, length, BLOCK):
                    indices = start + cols
                    value = tl.load(
                        X + row.to(tl.int64) * STRIDE + indices, indices < length, other=0
                    )
                    key = _ordered(value)
                    active = (indices < length) & ((key & prefix_mask) == prefix)
                    histogram += tl.histogram(((key >> shift) & 255).to(tl.int32), 256, active)
                counts = tl.cumsum(histogram, 0, reverse=True)
                bucket = tl.max(tl.where(counts >= remaining, bins, 0), 0)
                remaining -= tl.sum(tl.where(bins > bucket, histogram, 0), 0)
                prefix |= bucket.to(tl.uint32) << shift
                prefix_mask |= tl.full((), 255, tl.uint32) << shift
            written = tl.full((), 0, tl.int32)
            ties_seen = tl.full((), 0, tl.int32)
            for start in range(0, length, BLOCK):
                indices = start + cols
                value = tl.load(X + row.to(tl.int64) * STRIDE + indices, indices < length, other=0)
                key = _ordered(value)
                ties = (indices < length) & (key == prefix)
                tie_rank = tl.cumsum(ties.to(tl.int32), 0)
                selected = (indices < length) & (
                    (key > prefix) | (ties & (ties_seen + tie_rank <= remaining))
                )
                ranks = tl.cumsum(selected.to(tl.int32), 0)
                tl.store(Y + row.to(tl.int64) * OUT_STRIDE + written + ranks - 1, indices, selected)
                written += tl.sum(selected.to(tl.int32), 0)
                ties_seen += tl.sum(ties.to(tl.int32), 0)


@triton.jit
def _histogram_kernel(
    X,
    LENGTHS,
    LIVE,
    HIST,
    STATE,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    TILES: tl.constexpr,
    TOPK: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
    row_base,
):
    tile, local = tl.program_id(0), tl.program_id(1)
    row = row_base + local
    length = tl.minimum(tl.maximum(tl.load(LENGTHS + row), 0), WIDTH)
    if (row < tl.load(LIVE)) & (length > TOPK):
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(X + row.to(tl.int64) * STRIDE + cols, cols < length, other=0)
        key = _ordered(value)
        active = cols < length
        if SHIFT != 24:
            prefix = tl.load(STATE + local * 4).to(tl.uint32)
            active &= (key >> (SHIFT + 8)) == (prefix >> (SHIFT + 8))
        histogram = tl.histogram(((key >> SHIFT) & 255).to(tl.int32), 256, active)
        tl.store(HIST + (local * TILES + tile) * 256 + tl.arange(0, 256), histogram)


@triton.jit
def _refine_kernel(
    LENGTHS,
    LIVE,
    HIST,
    STATE,
    WIDTH: tl.constexpr,
    TILES: tl.constexpr,
    TOPK: tl.constexpr,
    SHIFT: tl.constexpr,
    row_base,
):
    local = tl.program_id(0)
    row = row_base + local
    length = tl.minimum(tl.maximum(tl.load(LENGTHS + row), 0), WIDTH)
    if (row < tl.load(LIVE)) & (length > TOPK):
        bins = tl.arange(0, 256)
        histogram = tl.full((256,), 0, tl.int32)
        for start in range(0, TILES, 8):
            tiles = start + tl.arange(0, 8)
            values = tl.load(
                HIST + (local * TILES + tiles[:, None]) * 256 + bins[None, :],
                tiles[:, None] < TILES,
                other=0,
            )
            histogram += tl.sum(values, 0)
        if SHIFT == 24:
            remaining = tl.full((), TOPK, tl.int32)
            prefix = tl.full((), 0, tl.uint32)
            tl.store(STATE + local * 4 + 2, 0)
            tl.store(STATE + local * 4 + 3, 0)
        else:
            remaining = tl.load(STATE + local * 4 + 1)
            prefix = tl.load(STATE + local * 4).to(tl.uint32)
        counts = tl.cumsum(histogram, 0, reverse=True)
        bucket = tl.max(tl.where(counts >= remaining, bins, 0), 0)
        remaining -= tl.sum(tl.where(bins > bucket, histogram, 0), 0)
        prefix |= bucket.to(tl.uint32) << SHIFT
        tl.store(STATE + local * 4, prefix.to(tl.int32, bitcast=True))
        tl.store(STATE + local * 4 + 1, remaining)


@triton.jit
def _collect_kernel(
    X,
    LENGTHS,
    LIVE,
    STATE,
    Y,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    row_base,
):
    tile, local = tl.program_id(0), tl.program_id(1)
    row = row_base + local
    if row < tl.load(LIVE):
        length = tl.minimum(tl.maximum(tl.load(LENGTHS + row), 0), WIDTH)
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        if length <= TOPK:
            tl.store(Y + row.to(tl.int64) * OUT_STRIDE + cols, cols, cols < length)
        else:
            value = tl.load(X + row.to(tl.int64) * STRIDE + cols, cols < length, other=0)
            key = _ordered(value)
            threshold = tl.load(STATE + local * 4).to(tl.uint32)
            remaining = tl.load(STATE + local * 4 + 1)
            ties = (cols < length) & (key == threshold)
            tie_ranks = tl.cumsum(ties.to(tl.int32), 0)
            previous = tl.atomic_add(
                STATE + local * 4 + 3, tl.sum(ties.to(tl.int32), 0), sem="relaxed"
            )
            selected = (cols < length) & (
                (key > threshold) | (ties & (previous + tie_ranks <= remaining))
            )
            ranks = tl.cumsum(selected.to(tl.int32), 0)
            start = tl.atomic_add(
                STATE + local * 4 + 2, tl.sum(selected.to(tl.int32), 0), sem="relaxed"
            )
            tl.store(Y + row.to(tl.int64) * OUT_STRIDE + start + ranks - 1, cols, selected)
