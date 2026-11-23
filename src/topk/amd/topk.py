# Copyright (C) 2026 Tencent.

import torch
import triton
import triton.language as tl

from hpc._triton._validation import check_tensor, require


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


def _size(value, name):
    require(0 <= value <= 2147483647, f"{name} must fit a non-negative int32")


def workspace_size(num_rows, max_kv_len):
    _size(num_rows, "num_rows")
    _size(max_kv_len, "max_kv_len")
    if max_kv_len < 32768 or num_rows == 0:
        return 4, 4
    rows = min(num_rows, 64)
    return 4, (rows * triton.cdiv(max_kv_len, 4096) * 256 + rows * 4) * 4


def min_workspace_size(max_kv_len):
    _size(max_kv_len, "max_kv_len")
    return 4


def peak_workspace_size(max_kv_len):
    return workspace_size(64, max_kv_len)[1]


def topk_filtered(logits, ke, output, topk, num_valid_rows, counters=None, workspace=None):
    check_tensor(logits, logits, torch.float32, "logits", contiguous=False)
    require(
        logits.ndim == 2 and logits.stride(1) == 1 and logits.stride(0) >= logits.shape[1],
        "logits must have non-overlapping rows and contiguous inner stride",
    )
    check_tensor(ke, logits, torch.int32, "ke")
    check_tensor(num_valid_rows, logits, torch.int32, "num_valid_rows")
    check_tensor(output, logits, torch.int32, "output", contiguous=False)
    require(topk in (512, 2048), "topk must be 512 or 2048")
    require(ke.ndim == 1 and ke.numel() <= logits.shape[0], "invalid ke shape")
    require(num_valid_rows.numel() == 1, "num_valid_rows must contain one element")
    require(
        output.ndim == 2
        and output.shape[0] >= ke.numel()
        and output.shape[1] >= topk
        and output.stride(1) == 1
        and output.stride(0) >= topk,
        "invalid output shape or strides",
    )
    _, recommended = workspace_size(ke.numel(), logits.shape[1])
    for tensor, name in ((counters, "counters"), (workspace, "workspace")):
        if tensor is not None:
            check_tensor(tensor, logits, torch.uint8, name)
            require(
                tensor.numel() >= 4 and tensor.data_ptr() % 4 == 0,
                f"{name} must have at least four aligned bytes",
            )
    if ke.numel():
        with torch.cuda.device(logits.device):
            if workspace is None and recommended > 4:
                workspace = torch.empty(recommended, device=logits.device, dtype=torch.uint8)
            if recommended > 4 and workspace.numel() >= recommended:
                width = logits.shape[1]
                tiles = triton.cdiv(width, 4096)
                pool = min(ke.numel(), 64)
                storage = workspace[:recommended].view(torch.int32)
                histogram = storage[: pool * tiles * 256]
                state = storage[pool * tiles * 256 :]
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
                            4096,
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
                        4096,
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
                    True,
                    True,
                    4096,
                    num_warps=4,
                )
    return output
