# Copyright (C) 2026 Tencent.

import triton
import triton.language as tl


@triton.jit
def _reduce_kernel(
    X,
    POSITIONS,
    SCALES,
    SHARED,
    Y,
    HIDDEN: tl.constexpr,
    TOPK: tl.constexpr,
    INPUT_ROWS: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0, tl.float32)
    for route in range(TOPK):
        index = row.to(tl.int64) * TOPK + route
        pos = tl.load(POSITIONS + index)
        valid = (pos >= 0) & (pos < INPUT_ROWS)
        scale = tl.load(SCALES + index, valid, other=0)
        x = tl.load(
            X + pos.to(tl.int64) * HIDDEN + cols,
            valid & (cols < HIDDEN),
            other=0,
        ).to(tl.float32)
        value += x * scale
    offset = row.to(tl.int64) * HIDDEN + cols
    if HAS_SHARED:
        value += tl.load(SHARED + offset, cols < HIDDEN, other=0).to(tl.float32)
    tl.store(Y + offset, value, cols < HIDDEN)
