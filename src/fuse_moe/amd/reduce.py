# Copyright (C) 2026 Tencent.

import torch
import triton
import triton.language as tl

from hpc._triton._validation import check_bf16, check_tensor, require


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


def reduce(x, topk_pos, topk_scale, shared_output=None):
    check_bf16(x, x, "x")
    check_tensor(topk_pos, x, torch.int32, "topk_pos")
    check_tensor(topk_scale, x, torch.float32, "topk_scale")
    require(
        x.ndim == 2 and topk_pos.ndim == 2 and topk_scale.shape == topk_pos.shape,
        "expected x [tokens,D], topk_pos and topk_scale [N,K]",
    )
    rows, topk = topk_pos.shape
    hidden = x.shape[1]
    require(topk <= 128, "num_topk must be <= 128")
    if shared_output is not None:
        check_bf16(shared_output, x, "shared_output")
        require(shared_output.shape == (rows, hidden), "shared_output must have shape [N,D]")
    y = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)
    if rows and hidden:
        with torch.cuda.device(x.device):
            _reduce_kernel[(rows, triton.cdiv(hidden, 1024))](
                x,
                topk_pos,
                topk_scale,
                shared_output,
                y,
                hidden,
                topk,
                x.shape[0],
                shared_output is not None,
                1024,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return y


# TODO(amd): count_and_gather and fused MoE entries require FP8 kernels.
