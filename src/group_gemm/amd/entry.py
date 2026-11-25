# Copyright (C) 2026 Tencent.

import torch

from hpc._amd.utils.validation import check_tensor, require

from .reformat import get_reformat_tile_size, reformat_x_scale_async


SCHEMAS = (
    "reformat_x_scale(Tensor x_scale, Tensor seqlens, Tensor cu_seqlens, Tensor? out_x_scale, int num_seq_per_group_avg) -> (Tensor)",
)


def reformat_x_scale(x_scale, seqlens, cu_seqlens, out_x_scale, num_seq_per_group_avg):
    check_tensor(x_scale, x_scale, torch.float32, "x_scale")
    check_tensor(seqlens, x_scale, torch.int32, "seqlens")
    check_tensor(cu_seqlens, x_scale, torch.int32, "cu_seqlens")
    require(
        x_scale.ndim == 2 and x_scale.shape[1] in (16, 32, 56),
        "x_scale must have shape [M,16/32/56]",
    )
    require(
        seqlens.ndim == 1 and cu_seqlens.shape == (seqlens.numel() + 1,),
        "invalid group metadata shapes",
    )
    rows, cols = x_scale.shape
    groups = seqlens.numel()
    require(groups > 0 or rows == 0, "nonempty x_scale requires groups")
    tile = get_reformat_tile_size(num_seq_per_group_avg)
    require(
        not groups or (rows % groups == 0 and (rows // groups) % tile == 0),
        "per-group input capacity must align to the selected tile size",
    )
    output = (
        out_x_scale
        if out_x_scale is not None
        else torch.empty((cols, rows), device=x_scale.device, dtype=x_scale.dtype)
    )
    check_tensor(output, x_scale, torch.float32, "out_x_scale")
    require(output.shape == (cols, rows), "out_x_scale must have shape [N,M]")
    if rows and groups:
        offsets = torch.empty(groups, device=x_scale.device, dtype=torch.int64)
        reformat_x_scale_async(x_scale, output, seqlens, cu_seqlens, offsets, num_seq_per_group_avg)
    return output


# TODO(amd): grouped GEMM entries require FP8 and matrix-unit kernels.


def register(library):
    for schema in SCHEMAS:
        library.define(schema)
    library.impl("reformat_x_scale", reformat_x_scale, "CUDA")
