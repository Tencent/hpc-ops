# Copyright (C) 2026 Tencent.

import torch
import triton

from hpc._amd.utils.validation import check_bf16, check_tensor, require

from .reduce import _reduce_kernel


SCHEMAS = (
    "reduce(Tensor x, Tensor topk_pos, Tensor topk_scale, Tensor ? shared_output) -> (Tensor)",
)


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


def register(library):
    implementations = {
        "reduce": reduce,
    }
    for schema in SCHEMAS:
        library.define(schema)
        name = schema.split("(", 1)[0]
        library.impl(name, implementations[name], "CUDA")


# TODO(amd): count_and_gather and fused MoE entries require FP8 kernels.
