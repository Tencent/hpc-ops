# Copyright (C) 2026 Tencent.

import triton
import triton.language as tl
from triton.experimental.tle import language as tle


@triton.jit
def _split_channels(x, BLOCK: tl.constexpr):
    first, second = tl.split(tl.permute(tl.reshape(x, (2, 2, BLOCK)), (1, 2, 0)))
    x0, x1 = tl.split(tl.trans(first))
    x2, x3 = tl.split(tl.trans(second))
    return x0, x1, x2, x3


@triton.jit
def _project(w, x, channels, cols, HIDDEN: tl.constexpr, PROJECTION: tl.constexpr):
    weights = tl.load(
        w + PROJECTION * 4 * HIDDEN + channels[:, None] * HIDDEN + cols[None, :],
        cols[None, :] < HIDDEN,
        other=0,
    )
    return tl.sum(tl.sum(x * weights, 1), 0)


@triton.jit
def _pre_head_kernel(
    X,
    W,
    SCALE,
    BASE,
    Y,
    POST,
    RMS_WEIGHT,
    XA,
    POST_IN,
    RESIDUAL_OUT,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    HEAD: tl.constexpr,
    NORM: tl.constexpr,
    CAST_BF16: tl.constexpr,
    USE_TLE: tl.constexpr,
    FUSE_POST: tl.constexpr,
    norm_eps,
    hc_eps,
    magnitude,
    rms_eps,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    channels = tl.arange(0, 4)
    x = tl.load(
        X + row.to(tl.int64) * 4 * HIDDEN + channels[:, None] * HIDDEN + cols[None, :],
        cols[None, :] < HIDDEN,
        other=0,
    ).to(tl.float32)
    if FUSE_POST:
        xa = tl.load(XA + row.to(tl.int64) * HIDDEN + cols, cols < HIDDEN, other=0).to(tl.float32)
        gates = tl.load(POST_IN + row * 4 + channels)
        x = (gates[:, None] * xa[None, :] + x).to(tl.bfloat16)
        tl.store(
            RESIDUAL_OUT + (row.to(tl.int64) * 4 + channels[:, None]) * HIDDEN + cols[None, :],
            x,
            cols[None, :] < HIDDEN,
        )
        x = x.to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(tl.sum(x * x, 1), 0) / (4 * HIDDEN) + norm_eps)
    scale = tl.load(SCALE)
    p0 = (
        tl.sigmoid(_project(W, x, channels, cols, HIDDEN, 0) * inv_rms * scale + tl.load(BASE))
        + hc_eps
    )
    p1 = (
        tl.sigmoid(_project(W, x, channels, cols, HIDDEN, 1) * inv_rms * scale + tl.load(BASE + 1))
        + hc_eps
    )
    p2 = (
        tl.sigmoid(_project(W, x, channels, cols, HIDDEN, 2) * inv_rms * scale + tl.load(BASE + 2))
        + hc_eps
    )
    p3 = (
        tl.sigmoid(_project(W, x, channels, cols, HIDDEN, 3) * inv_rms * scale + tl.load(BASE + 3))
        + hc_eps
    )
    if not HEAD:
        for p in tl.static_range(4, 8):
            value = _project(W, x, channels, cols, HIDDEN, p) * inv_rms
            gate = magnitude * tl.sigmoid(value * tl.load(SCALE + 1) + tl.load(BASE + p)) + hc_eps
            tl.store(POST + row * 4 + p - 4, gate)
    if USE_TLE:
        x0 = tle.extract_tile(x, (0, 0), (1, BLOCK)).reshape((BLOCK,))
        x1 = tle.extract_tile(x, (1, 0), (1, BLOCK)).reshape((BLOCK,))
        x2 = tle.extract_tile(x, (2, 0), (1, BLOCK)).reshape((BLOCK,))
        x3 = tle.extract_tile(x, (3, 0), (1, BLOCK)).reshape((BLOCK,))
    else:
        x0, x1, x2, x3 = _split_channels(x, BLOCK)
    y = p0 * x0 + p1 * x1 + p2 * x2 + p3 * x3
    if NORM:
        if CAST_BF16:
            y = y.to(tl.bfloat16).to(tl.float32)
        square = tl.where(cols < HIDDEN, y * y, 0)
        y = y * tl.rsqrt(tl.sum(square, 0) / HIDDEN + rms_eps)
        y *= tl.load(RMS_WEIGHT + cols, cols < HIDDEN, other=0).to(tl.float32)
    tl.store(Y + row.to(tl.int64) * HIDDEN + cols, y, cols < HIDDEN)


@triton.jit
def _post_kernel(
    X,
    RESIDUAL,
    GATES,
    Y,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    SPLIT_CHANNELS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    channels = tl.arange(0, 4)
    offsets = (row.to(tl.int64) * 4 + channels[:, None]) * HIDDEN + cols[None, :]
    x = tl.load(X + row.to(tl.int64) * HIDDEN + cols, cols < HIDDEN, other=0).to(tl.float32)
    residual = tl.load(RESIDUAL + offsets, cols[None, :] < HIDDEN, other=0).to(tl.float32)
    if SPLIT_CHANNELS:
        r0, r1, r2, r3 = _split_channels(residual, BLOCK)
        for channel in tl.static_range(4):
            if channel == 0:
                value = r0
            elif channel == 1:
                value = r1
            elif channel == 2:
                value = r2
            else:
                value = r3
            gate = tl.load(GATES + row * 4 + channel)
            offset = (row.to(tl.int64) * 4 + channel) * HIDDEN + cols
            tl.store(Y + offset, x * gate + value, cols < HIDDEN)
    else:
        gates = tl.load(GATES + row * 4 + channels)
        tl.store(Y + offsets, x[None, :] * gates[:, None] + residual, cols[None, :] < HIDDEN)
