# Copyright (C) 2026 Tencent.
# Copyright 2026, The FlagOS Contributors.

import os

import torch
import triton
import triton.language as tl
from triton.experimental.tle import require_tle
from triton.experimental.tle import language as tle

USE_TLE = os.getenv("HPC_TRITON_USE_TLE", "0") == "1"


# Split a [4, BLOCK] tile into its four rows without a transpose in memory; the
# TLE path does the same with extract_tile.
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
    # The fused post output is rounded to BF16 before it feeds the projections,
    # so this matches running post and pre as two kernels.
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
    # One scale for all four channels: the norm runs over the whole [4, HIDDEN] row.
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
    # Projections 0-3 gate this layer's mix; 4-7 are the gates the post stage
    # will apply, scaled by magnitude. The head variant has only the first four.
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


def fuse_ihc_pre_head_async(
    x,
    w,
    hc_scale,
    hc_base,
    y,
    post,
    norm_eps,
    hc_eps,
    magnitude,
    rms_weight,
    rms_eps,
    cast_bfloat_for_norm,
    head,
    post_inputs,
):
    if USE_TLE:
        require_tle("amd", tle.extract_tile)
    batch, _, hidden = x.shape
    xa, post_in, residual_out = post_inputs if post_inputs is not None else (None, None, None)
    if batch:
        with torch.cuda.device(x.device):
            _pre_head_kernel[(batch,)](
                x,
                w,
                hc_scale,
                hc_base,
                y,
                post,
                rms_weight,
                xa,
                post_in,
                residual_out,
                hidden,
                triton.next_power_of_2(hidden),
                head,
                rms_weight is not None,
                cast_bfloat_for_norm,
                USE_TLE,
                post_inputs is not None,
                norm_eps,
                hc_eps,
                magnitude,
                rms_eps,
                num_warps=16 if hidden == 6144 and (USE_TLE or post_inputs is not None) else 8,
                enable_fp_fusion=False,
            )


def fuse_ihc_post_async(x, residual, H_post, y):
    batch, hidden = x.shape
    if batch:
        if batch <= 16:
            block, waves = 512, 8
        elif batch < 512:
            block, waves = 256, 1
        else:
            block, waves = 1024, 2
        with torch.cuda.device(x.device):
            _post_kernel[(batch, triton.cdiv(hidden, block))](
                x,
                residual,
                H_post,
                y,
                hidden,
                block,
                batch > 16,
                num_warps=waves,
                enable_fp_fusion=False,
            )
