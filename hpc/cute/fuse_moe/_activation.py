# Copyright (C) 2026 Tencent.

"""SwiGLU followed by 128-column FP8 quantization with float32 scales.

The blockwise MXFP8 form of this pass has no standalone kernel: its scale is a
reduction over the row, which the up-projection GEMM can do in its epilogue while
the tile is still in registers. See ``_grouped_gemm``.

Qwen's 128x128 float32 block scales fuse the same way once the up-projection
tile is 128 columns wide: ``grouped_block_scaled_gemm`` with ``c_scale`` in
float32. ``quantize_silu_mul_fp8_block128`` remains as the standalone pass
for callers that still have a bfloat16 gate/up tensor.
"""

from typing import Any, Dict, Tuple

from torch import Tensor

from .._compile import current_stream, import_cutlass, launcher, make_ptr, scalar
from .._quant import FP8_MAX
from ._routing import find_row_group

_cache: Dict[Tuple[Any, ...], Any] = {}


BLOCK128 = 128
BLOCK128_THREADS = 128
BLOCK128_ELEMS = 8
# 16 threads cover one 128-column block; four butterfly steps reduce them.
BLOCK128_LANES = BLOCK128 // BLOCK128_ELEMS


def _build_block128(tile_m: int):
    cutlass, cute = import_cutlass()
    import cuda.bindings.driver as cuda
    import cutlass.cute.math as cute_math

    @cute.kernel
    def block128_kernel(
        gate_up: cute.Pointer,
        output: cute.Pointer,
        output_scale: cute.Pointer,
        cu_m_tiles_ptr: cute.Pointer,
        counts_ptr: cute.Pointer,
        num_experts: cutlass.Int32,
        intermediate_size: cutlass.Int32,
        swiglu_limit: cutlass.Float32,
        threads: cutlass.Constexpr,
    ):
        cute.arch.griddepcontrol_wait()

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        cu_m_tiles = cute.make_tensor(cu_m_tiles_ptr, cute.make_layout(num_experts + 1))
        row_counts = cute.make_tensor(counts_ptr, cute.make_layout(num_experts))
        row = cutlass.Int32(bidx)
        group = find_row_group(cu_m_tiles, num_experts, row // tile_m)
        if row - cu_m_tiles[group] * tile_m < row_counts[group]:
            _, column_blocks, _ = cute.arch.grid_dim()
            vector = cutlass.Int32(bidy) * threads + cutlass.Int32(tidx)
            column = vector * BLOCK128_ELEMS
            n_blocks = intermediate_size // BLOCK128
            row_gate_up = cute.make_tensor(
                gate_up + cutlass.Int64(row) * intermediate_size * 2,
                cute.make_layout(intermediate_size * 2),
            )
            row_output = cute.make_tensor(
                output + cutlass.Int64(row) * intermediate_size,
                cute.make_layout(intermediate_size),
            )
            row_scale = cute.make_tensor(
                output_scale + cutlass.Int64(row) * n_blocks,
                cute.make_layout(n_blocks),
            )
            values = cute.make_rmem_tensor(cute.make_layout(BLOCK128_ELEMS), cutlass.Float32)
            in_range = column < intermediate_size
            for item in cutlass.range_constexpr(BLOCK128_ELEMS):
                values[item] = cutlass.Float32(0.0)
            if in_range:
                for item in cutlass.range_constexpr(BLOCK128_ELEMS):
                    index = column + item
                    gate = cutlass.Float32(row_gate_up[index])
                    up = cutlass.Float32(row_gate_up[intermediate_size + index])
                    if swiglu_limit >= 0.0:
                        gate = cute.arch.fmin(gate, swiglu_limit)
                        up = cute.arch.fmin(cute.arch.fmax(up, -swiglu_limit), swiglu_limit)
                    activated = gate * cute.arch.rcp_approx(
                        cutlass.Float32(1.0)
                        + cute_math.exp2(-gate * 1.4426950408889634, fastmath=True)
                    )
                    values[item] = activated * up
            block_max = cutlass.Float32(0.0)
            for item in cutlass.range_constexpr(BLOCK128_ELEMS):
                block_max = cute.arch.fmax(block_max, cute_math.absf(values[item]))
            for step in cutlass.range_constexpr(4):
                block_max = cute.arch.fmax(
                    block_max, cute.arch.shuffle_sync_bfly(block_max, offset=1 << step)
                )
            if in_range:
                scale = block_max / cutlass.Float32(FP8_MAX)
                inv_scale = cutlass.Float32(1.0) / (scale + cutlass.Float32(1e-8))
                for item in cutlass.range_constexpr(BLOCK128_ELEMS):
                    quantized = cutlass.Float8E4M3FN(values[item] * inv_scale)
                    row_output[column + item] = quantized.bitcast(cutlass.Uint8)
                if cutlass.Int32(tidx) % BLOCK128_LANES == 0:
                    row_scale[column // BLOCK128] = scale

        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def block128_host(
        gate_up: cute.Pointer,
        output: cute.Pointer,
        output_scale: cute.Pointer,
        cu_m_tiles: cute.Pointer,
        row_counts: cute.Pointer,
        max_rows: cutlass.Int32,
        num_experts: cutlass.Int32,
        intermediate_size: cutlass.Int32,
        swiglu_limit: cutlass.Float32,
        stream: cuda.CUstream,
        threads: cutlass.Constexpr = BLOCK128_THREADS,
    ):
        vectors = intermediate_size // BLOCK128_ELEMS
        grid = [max_rows, (vectors + threads - 1) // threads, 1]
        block128_kernel(
            gate_up,
            output,
            output_scale,
            cu_m_tiles,
            row_counts,
            num_experts,
            intermediate_size,
            swiglu_limit,
            threads,
        ).launch(grid=grid, block=[threads, 1, 1], stream=stream, use_pdl=True)

    return block128_host


def _compiled_block128(tile_m: int, *args):
    key = ("activation_block128", BLOCK128_THREADS, tile_m)
    if key not in _cache:
        _, cute = import_cutlass()
        _cache[key] = launcher(cute.compile(_build_block128(tile_m), *args))
    return _cache[key]


def quantize_silu_mul_fp8_block128(
    gate_up: Tensor,
    output: Tensor,
    output_scale: Tensor,
    buffers,
    max_rows: int,
    num_expert_local: int,
    intermediate_size: int,
    swiglu_limit: float = -1.0,
) -> None:
    """SwiGLU then per-row 128-column FP8 quantization with float32 scales."""
    if intermediate_size % BLOCK128 != 0:
        raise ValueError(
            f"intermediate_size must be a multiple of {BLOCK128}; got {intermediate_size}"
        )

    cutlass, _ = import_cutlass()
    args = (
        make_ptr(cutlass.BFloat16, gate_up),
        make_ptr(cutlass.Uint8, output),
        make_ptr(cutlass.Float32, output_scale),
        make_ptr(cutlass.Int32, buffers.cu_m_tiles),
        make_ptr(cutlass.Int32, buffers.num_tokens_per_expert),
        scalar(cutlass.Int32, max_rows),
        scalar(cutlass.Int32, num_expert_local),
        scalar(cutlass.Int32, intermediate_size),
        scalar(cutlass.Float32, swiglu_limit),
        current_stream(),
    )
    _compiled_block128(buffers.tile_m, *args)(*args)
