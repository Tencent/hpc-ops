# Copyright (C) 2026 Tencent.
# Copyright 2026, The FlagOS Contributors.

import torch
import triton
import triton.language as tl


# Order-preserving uint32 key for an FP32 value, so comparing keys compares the
# floats: negatives invert, non-negatives flip the sign bit, NaN sorts above all.
@triton.jit
def _ordered(x):
    bits = x.to(tl.uint32, bitcast=True)
    key = tl.where(bits & 0x80000000 != 0, ~bits, bits ^ 0x80000000)
    return tl.where(x != x, 0xFFFFFFFF, key).to(tl.uint32)


# One workgroup per row: four 8-bit radix passes narrow the threshold key, then
# one pass writes the indices. Rows at or past LIVE stay untouched.
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
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    live = tl.load(LIVE)
    if row < live:
        length = tl.minimum(tl.maximum(tl.load(LENGTHS + row), 0), WIDTH)
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
                # bucket is the highest byte whose keys-at-or-above still cover
                # the quota; remaining is then the quota left for prefix matches.
                counts = tl.cumsum(histogram, 0, reverse=True)
                bucket = tl.max(tl.where(counts >= remaining, bins, 0), 0)
                remaining -= tl.sum(tl.where(bins > bucket, histogram, 0), 0)
                prefix |= bucket.to(tl.uint32) << shift
                prefix_mask |= tl.full((), 255, tl.uint32) << shift
            # Keys above the threshold all go in; keys equal to it fill the
            # remaining quota in column order, so exact ties cannot overflow.
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


# The same selection across many workgroups for long rows. STATE holds four
# int32 per row: threshold prefix, remaining quota, output cursor, tie cursor.
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
            # The first pass owns initialization, so caller scratch needs none.
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
            # Tiles claim their share of the tie quota and their output range
            # atomically, so which tied keys win depends on arrival order.
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


_BLOCK_SIZE = 4096
_RADIX_BINS = 256
_MAX_WORKSPACE_ROWS = 64


# Short rows need no scratch, so the minimum is the four bytes that select that
# path; longer rows get per-tile histograms for _MAX_WORKSPACE_ROWS rows at most.
def topk_filtered_workspace_size(num_rows, max_kv_len):
    if max_kv_len < 32768 or num_rows == 0:
        return 4, 4
    rows = min(num_rows, _MAX_WORKSPACE_ROWS)
    return 4, (rows * triton.cdiv(max_kv_len, _BLOCK_SIZE) * _RADIX_BINS + rows * 4) * 4


def topk_filtered_min_workspace_size(max_kv_len):
    return 4


def topk_filtered_peak_workspace_size(max_kv_len):
    return topk_filtered_workspace_size(_MAX_WORKSPACE_ROWS, max_kv_len)[1]


def topk_filtered_async(logits, ke, output, topk, num_valid_rows, workspace):
    _, recommended = topk_filtered_workspace_size(ke.numel(), logits.shape[1])
    if ke.numel():
        with torch.cuda.device(logits.device):
            if recommended > 4 and workspace.numel() >= recommended:
                width = logits.shape[1]
                tiles = triton.cdiv(width, _BLOCK_SIZE)
                pool = min(ke.numel(), _MAX_WORKSPACE_ROWS)
                storage = workspace[:recommended].view(torch.int32)
                histogram = storage[: pool * tiles * _RADIX_BINS]
                state = storage[pool * tiles * _RADIX_BINS :]
                for row_base in range(0, ke.numel(), pool):
                    rows = min(pool, ke.numel() - row_base)
                    for shift in (24, 16, 8, 0):
                        _histogram_kernel[(tiles, rows)](
                            logits,
                            ke,
                            num_valid_rows,
                            histogram,
                            state,
                            width,
                            logits.stride(0),
                            tiles,
                            topk,
                            shift,
                            _BLOCK_SIZE,
                            row_base,
                            num_warps=4,
                        )
                        _refine_kernel[(rows,)](
                            ke,
                            num_valid_rows,
                            histogram,
                            state,
                            width,
                            tiles,
                            topk,
                            shift,
                            row_base,
                            num_warps=4,
                        )
                    _collect_kernel[(tiles, rows)](
                        logits,
                        ke,
                        num_valid_rows,
                        state,
                        output,
                        width,
                        logits.stride(0),
                        output.stride(0),
                        topk,
                        _BLOCK_SIZE,
                        row_base,
                        num_warps=4,
                    )
            else:
                _select_kernel[(ke.numel(),)](
                    logits,
                    ke,
                    num_valid_rows,
                    output,
                    logits.shape[1],
                    logits.stride(0),
                    output.stride(0),
                    topk,
                    _BLOCK_SIZE,
                    num_warps=4,
                )
