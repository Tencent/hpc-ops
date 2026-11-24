# Copyright (C) 2026 Tencent.

import torch
import triton
import triton.language as tl


@triton.jit
def _group_offsets(LENGTHS, OFFSETS, GROUPS: tl.constexpr, TILE: tl.constexpr, BLOCK: tl.constexpr):
    groups = tl.arange(0, BLOCK)
    lengths = tl.load(LENGTHS + groups, groups < GROUPS, other=0).to(tl.int64)
    aligned = tl.cdiv(lengths, TILE) * TILE
    offsets = tl.cumsum(aligned) - aligned
    tl.store(OFFSETS + groups, offsets, groups < GROUPS)


@triton.jit
def _reformat_kernel(
    X,
    Y,
    LENGTHS,
    CU,
    OFFSETS,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    group = tl.program_id(0)
    local_rows = tl.arange(0, 128)
    cols = tl.arange(0, BLOCK_COLS)
    length = tl.load(LENGTHS + group)
    source = tl.load(CU + group).to(tl.int64)
    destination = tl.load(OFFSETS + group)
    for start in range(0, length, 128):
        rows = start + local_rows
        valid = (rows[:, None] < length) & (cols[None, :] < COLS)
        value = tl.load(X + (source + rows[:, None]) * COLS + cols[None, :], valid, other=0)
        tl.store(Y + cols[None, :].to(tl.int64) * ROWS + destination + rows[:, None], value, valid)


def get_reformat_tile_size(num_seq_per_group_avg):
    return next((size for size in (8, 16, 32, 48) if num_seq_per_group_avg <= size), 64)


def reformat_x_scale_async(x_scale, output, seqlens, cu_seqlens, offsets, num_seq_per_group_avg):
    rows, cols = x_scale.shape
    groups = seqlens.numel()
    tile = get_reformat_tile_size(num_seq_per_group_avg)
    with torch.cuda.device(x_scale.device):
        _group_offsets[(1,)](
            seqlens, offsets, groups, tile, triton.next_power_of_2(groups), num_warps=4
        )
        _reformat_kernel[(groups,)](
            x_scale,
            output,
            seqlens,
            cu_seqlens,
            offsets,
            rows,
            cols,
            groups,
            triton.next_power_of_2(cols),
            num_warps=4,
        )
