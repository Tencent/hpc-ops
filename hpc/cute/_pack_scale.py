# Copyright (C) 2026 Tencent.

"""Repack logical UE8M0 weight scales into the interleaved SM120 layout."""

from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from ._compile import current_stream, dynamic_tensor, import_cutlass, require_sm120
from ._layouts import (
    SCALE_K_ALIGNMENT,
    SCALE_VECTOR_SIZE,
    ceil_div,
    scale_offset_device,
    scale_storage_size,
)

_THREADS = 256

_cache: Dict[Tuple[Any, ...], Any] = {}


def _build():
    cutlass, cute = import_cutlass()
    import cuda.bindings.driver as cuda

    @cute.kernel
    def pack_kernel(
        src: cute.Tensor,
        dst: cute.Tensor,
        rows: cutlass.Int32,
        num_k_blocks: cutlass.Int32,
        storage_per_expert: cutlass.Int32,
        k_atoms: cutlass.Int32,
        total: cutlass.Int32,
        threads: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        index = cutlass.Int32(bidx) * threads + cutlass.Int32(tidx)

        if index < total:
            block_k = index % num_k_blocks
            row_and_expert = index // num_k_blocks
            row = row_and_expert % rows
            expert = row_and_expert // rows

            offset = scale_offset_device(row, block_k * SCALE_VECTOR_SIZE, k_atoms)
            dst[cutlass.Int64(expert) * storage_per_expert + offset] = src[index]

    @cute.jit
    def pack_host(
        src: cute.Tensor,
        dst: cute.Tensor,
        rows: cutlass.Int32,
        num_k_blocks: cutlass.Int32,
        storage_per_expert: cutlass.Int32,
        k_atoms: cutlass.Int32,
        total: cutlass.Int32,
        stream: cuda.CUstream,
        threads: cutlass.Constexpr = _THREADS,
    ):
        blocks = (total + threads - 1) // threads
        pack_kernel(
            src, dst, rows, num_k_blocks, storage_per_expert, k_atoms, total, threads
        ).launch(grid=[blocks, 1, 1], block=[threads, 1, 1], stream=stream)

    return pack_host


def _compiled(*args):
    """Compile once per configuration and reuse.

    Tensor layouts are marked dynamic, so a single artifact serves every expert
    count and matrix shape; ``args`` only seeds the trace.
    """
    key = ("pack", _THREADS)
    if key not in _cache:
        _, cute = import_cutlass()
        _cache[key] = cute.compile(_build(), *args)
    return _cache[key]


def pack_mxfp_weight_scale_sm120(scale: Tensor, rows: int, cols: int) -> Tensor:
    """Pack logical UE8M0 scale factors for an SM120 block-scaled GEMM.

    Uses :func:`scale_offset_device` and the layout in ``_layouts``.

    Args:
        scale: ``uint8`` tensor of shape ``[experts, rows, cols // 32]``.
        rows: Logical weight row count, a positive multiple of 128.
        cols: Logical reduction extent, a positive multiple of 128.

    Returns:
        A ``uint8`` tensor of shape ``[experts, packed_size]``.
    """
    require_sm120(scale.device)

    if scale.dtype != torch.uint8:
        raise ValueError(f"scale must be uint8; got {scale.dtype}")
    if scale.dim() != 3:
        raise ValueError(f"scale must be 3-D [experts, rows, cols // 32]; got {tuple(scale.shape)}")
    if rows <= 0 or rows % SCALE_K_ALIGNMENT != 0:
        raise ValueError(f"rows must be a positive multiple of 128; got {rows}")
    if cols <= 0 or cols % SCALE_K_ALIGNMENT != 0:
        raise ValueError(f"cols must be a positive multiple of 128; got {cols}")

    num_k_blocks = cols // SCALE_VECTOR_SIZE
    if tuple(scale.shape[1:]) != (rows, num_k_blocks):
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match rows={rows}, cols={cols}"
        )

    scale = scale.contiguous()
    experts = scale.shape[0]
    storage_per_expert = scale_storage_size(rows, cols)
    output = torch.empty((experts, storage_per_expert), dtype=torch.uint8, device=scale.device)

    total = experts * rows * num_k_blocks
    src = dynamic_tensor(scale.reshape(-1))
    dst = dynamic_tensor(output.reshape(-1))

    cutlass, _ = import_cutlass()
    args = (
        src,
        dst,
        cutlass.Int32(rows),
        cutlass.Int32(num_k_blocks),
        cutlass.Int32(storage_per_expert),
        cutlass.Int32(ceil_div(cols, SCALE_K_ALIGNMENT)),
        cutlass.Int32(total),
        current_stream(),
    )
    _compiled(*args)(*args)
    return output
