# Copyright (C) 2026 Tencent.

import torch
import triton

from hpc._amd.utils.validation import check_tensor, require

from .topk import _collect_kernel, _histogram_kernel, _refine_kernel, _select_kernel


SCHEMAS = (
    "topk_filtered(Tensor logits, Tensor ke, Tensor! output, int topk, Tensor num_valid_rows, Tensor? counters, Tensor? workspace) -> (Tensor)",
    "topk_filtered_workspace_size(int num_rows, int max_kv_len) -> (int, int)",
    "topk_filtered_min_workspace_size(int max_kv_len) -> int",
    "topk_filtered_peak_workspace_size(int max_kv_len) -> int",
)


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


def register(library):
    implementations = {
        "topk_filtered": topk_filtered,
        "topk_filtered_workspace_size": workspace_size,
        "topk_filtered_min_workspace_size": min_workspace_size,
        "topk_filtered_peak_workspace_size": peak_workspace_size,
    }
    for schema in SCHEMAS:
        library.define(schema)
        name = schema.split("(", 1)[0]
        dispatch = "CompositeExplicitAutograd" if "workspace_size" in name else "CUDA"
        library.impl(name, implementations[name], dispatch)
