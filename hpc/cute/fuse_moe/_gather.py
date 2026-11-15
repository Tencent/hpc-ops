# Copyright (C) 2026 Tencent.

"""Gather routed tokens into expert-contiguous rows.

One warp handles one route: it copies the E4M3 activation row and scatters the
token's UE8M0 scales into the interleaved SFA layout the block-scaled MMA
expects. The BF16 variant quantizes the row to MXFP8 on the way instead.

For a GEMM that reads A rows by token, :func:`gather_token_routes` places the
routes without copying rows: scales only, and a BF16 row quantized once rather
than once per route. A token's routes share a warp, one lane each.
"""

from typing import Any, Dict, Optional, Tuple

from torch import Tensor
import torch

from .._compile import current_stream, import_cutlass, launcher, make_ptr, scalar
from .._layouts import SCALE_K_ALIGNMENT, SCALE_VECTOR_SIZE, ceil_div, scale_offset_device
from .._quant import build_helpers as build_quant_helpers

_WARPS_PER_BLOCK = 4
_WARP_SIZE = 32
# gather_token_routes gives each of a token's routes one lane of its warp.
MAX_TOKEN_TOPK = _WARP_SIZE
_THREADS = _WARPS_PER_BLOCK * _WARP_SIZE
# One 16-byte vector holds four int32 lanes of E4M3 activation bytes.
_INTS_PER_VECTOR = 4
# BF16 input: one 16-byte vector per lane is eight elements, so four lanes
# cover one 32-element MX block.
_BF16_PER_LANE = 8
_LANES_PER_MX_BLOCK = SCALE_VECTOR_SIZE // _BF16_PER_LANE
# Vectors each lane has in flight per pass: a route is one warp, so at decode
# sizes the row's load latency is the gather's whole cost unless it overlaps.
_BF16_CHUNKS = 8
_BF16_PASS_COLUMNS = _BF16_CHUNKS * _WARP_SIZE * _BF16_PER_LANE


def _token_parts(assign: bool, bf16_to_mx: bool, token_topk: int) -> int:
    """Warps per token in :func:`gather_token_routes`.

    Quantizing a row once instead of once per route leaves a decode step with a
    top-k-th of the warps, each of them as long, and at one to four tokens that
    latency was 1 us of a 28 us layer. Rows already placed let as many warps as
    there were routes take a share of the passes each.
    """
    return token_topk if token_topk > 0 and bf16_to_mx and not assign else 1


_cache: Dict[Tuple[Any, ...], Any] = {}


def _build(
    assign_position: bool,
    scale_f32: bool = False,
    fp32_to_mx: bool = False,
    bf16_to_mx: bool = False,
    token_topk: int = 0,
):
    cutlass, cute = import_cutlass()
    import cuda.bindings.driver as cuda
    import cutlass.cute.math as cute_math

    _, quantization_scale, to_ue8m0_storage = build_quant_helpers()

    @cute.jit
    def quantize_bf16_row(
        input_words: cute.Pointer,
        source_base: cutlass.Int64,
        output_words: cute.Pointer,
        destination_base: cutlass.Int64,
        gathered_scale: cute.Pointer,
        rows: cute.Tensor,
        hidden_size: cutlass.Int32,
        k_atoms: cutlass.Int32,
        lane: cutlass.Int32,
        num_rows: cutlass.Constexpr,
        first_pass: cutlass.Int32,
        pass_stride: cutlass.Int32,
    ):
        """Quantize one BF16 row to MXFP8, writing each block's scale at every
        padded row in ``rows`` that is not -1. Covers passes ``first_pass``,
        ``first_pass + pass_stride``, ... of ``_BF16_PASS_COLUMNS`` columns.

        Same rounding as the up-projection epilogue's requantization: the scale
        is the block maximum over 448 rounded up to a power of two. A BF16
        element is the high half of its float32, so each int32 word unpacks
        into two exact float32 values.
        """
        raws = [
            cute.make_rmem_tensor(cute.make_layout(_INTS_PER_VECTOR), cutlass.Int32)
            for _ in range(_BF16_CHUNKS)
        ]
        packed = cute.make_rmem_tensor(cute.make_layout(_BF16_PER_LANE // 4), cutlass.Int32)
        values = cute.make_rmem_tensor(cute.make_layout(_BF16_PER_LANE), cutlass.Float32)
        # Warp-uniform bound: the block maximum is a shuffle, which every lane
        # has to join, so lanes past the row stay in the loop with zeros. Rows
        # are whole MX blocks, so a block is either all in range or all out.
        # All of a pass's loads are issued before any is used, so their
        # latencies overlap.
        base = first_pass * _BF16_PASS_COLUMNS
        while base < hidden_size:
            for chunk in cutlass.range_constexpr(_BF16_CHUNKS):
                column = base + (chunk * _WARP_SIZE + lane) * _BF16_PER_LANE
                for item in cutlass.range_constexpr(_INTS_PER_VECTOR):
                    raws[chunk][item] = cutlass.Int32(0)
                if column < hidden_size:
                    source = cute.make_tensor(
                        input_words + (source_base + column // 2),
                        cute.make_layout(_INTS_PER_VECTOR),
                    )
                    cute.autovec_copy(source, raws[chunk])
            for chunk in cutlass.range_constexpr(_BF16_CHUNKS):
                column = base + (chunk * _WARP_SIZE + lane) * _BF16_PER_LANE
                for item in cutlass.range_constexpr(_INTS_PER_VECTOR):
                    word = raws[chunk][item]
                    values[2 * item] = (word << 16).bitcast(cutlass.Float32)
                    values[2 * item + 1] = (word & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
                block_max = cutlass.Float32(0.0)
                for item in cutlass.range_constexpr(_BF16_PER_LANE):
                    block_max = cute.arch.fmax(block_max, cute_math.absf(values[item]))
                for step in cutlass.range_constexpr(2):
                    block_max = cute.arch.fmax(
                        block_max,
                        cute.arch.shuffle_sync_bfly(block_max, offset=1 << step),
                    )
                storage, inv_scale = quantization_scale(block_max)
                if column < hidden_size:
                    for half in cutlass.range_constexpr(_BF16_PER_LANE // 4):
                        word = cutlass.Int32(0)
                        for byte in cutlass.range_constexpr(4):
                            quantized = cutlass.Float8E4M3FN(values[4 * half + byte] * inv_scale)
                            bits = cutlass.Int32(quantized.bitcast(cutlass.Uint8)) & 0xFF
                            word = word | (bits << (8 * byte))
                        packed[half] = word
                    destination = cute.make_tensor(
                        output_words + (destination_base + column // 4),
                        cute.make_layout(_BF16_PER_LANE // 4),
                    )
                    cute.autovec_copy(packed, destination)
                    if lane % _LANES_PER_MX_BLOCK == 0:
                        for j in cutlass.range_constexpr(num_rows):
                            if rows[j] >= 0:
                                offset = scale_offset_device(rows[j], column, k_atoms)
                                slot = cute.make_tensor(
                                    gathered_scale + offset, cute.make_layout(1)
                                )
                                slot[0] = cutlass.Uint8(storage)
            base += pass_stride * _BF16_PASS_COLUMNS

    @cute.jit
    def scatter_mx_scales(
        input_scale: cute.Pointer,
        token: cutlass.Int32,
        gathered_scale: cute.Pointer,
        rows: cute.Tensor,
        hidden_size: cutlass.Int32,
        k_atoms: cutlass.Int32,
        lane: cutlass.Int32,
        num_rows: cutlass.Constexpr,
    ):
        """Write a token's UE8M0 scales at every padded row in ``rows`` that is
        not -1. The offset is a pure function of the row, so no per-expert base
        offset is needed."""
        num_k_blocks = hidden_size // SCALE_VECTOR_SIZE
        token_scale = cute.make_tensor(
            input_scale + cutlass.Int64(token) * num_k_blocks,
            cute.make_layout(num_k_blocks),
        )
        block_k = lane
        while block_k < num_k_blocks:
            value = token_scale[block_k]
            for j in cutlass.range_constexpr(num_rows):
                if rows[j] >= 0:
                    offset = scale_offset_device(rows[j], block_k * SCALE_VECTOR_SIZE, k_atoms)
                    slot = cute.make_tensor(gathered_scale + offset, cute.make_layout(1))
                    slot[0] = value
            block_k += _WARP_SIZE

    @cute.kernel
    def gather_kernel(
        input_words: cute.Pointer,
        input_scale: cute.Pointer,
        topk_ids_ptr: cute.Pointer,
        cu_tokens_ptr: cute.Pointer,
        routing_counters: cute.Pointer,
        topk_pos_ptr: cute.Pointer,
        row_to_route_ptr: cute.Pointer,
        gathered_words: cute.Pointer,
        gathered_scale: cute.Pointer,
        num_items: cutlass.Int32,
        num_topk: cutlass.Int32,
        hidden_size: cutlass.Int32,
        first_expert: cutlass.Int32,
        last_expert: cutlass.Int32,
        k_atoms: cutlass.Int32,
        warps: cutlass.Constexpr,
        assign: cutlass.Constexpr,
        scale_f32: cutlass.Constexpr,
        fp32_to_mx: cutlass.Constexpr,
        bf16_to_mx: cutlass.Constexpr,
        token_topk: cutlass.Constexpr,
    ):
        cute.arch.griddepcontrol_wait()

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cutlass.Int32(tidx) // _WARP_SIZE
        lane = cutlass.Int32(tidx) % _WARP_SIZE
        # A route, or with token_topk a token.
        item = cutlass.Int32(bidx) * warps + warp

        output_ptr = cute.arch.alloc_smem(cutlass.Int32, warps, alignment=16)
        output_rows = cute.make_tensor(output_ptr, cute.make_layout(warps))

        route_layout = cute.make_layout(num_items * token_topk if token_topk > 0 else num_items)
        topk_ids = cute.make_tensor(topk_ids_ptr, route_layout)
        topk_pos = cute.make_tensor(topk_pos_ptr, route_layout)
        cu_tokens = cute.make_tensor(
            cu_tokens_ptr, cute.make_layout(last_expert - first_expert + 1)
        )
        words_per_row = hidden_size // 4

        if cutlass.const_expr(token_topk > 0):
            # Lane j finds the padded row of the token's j-th route, -1 when the
            # expert is not local, and the shuffles hand every lane all of them.
            # Splitting a row's columns over warps needs its rows placed already:
            # each warp has to find them on its own.
            parts = _token_parts(assign, bf16_to_mx, token_topk)
            token = item // parts
            part = item % parts
            row = cutlass.Int32(-1)
            if token < num_items:
                if lane < token_topk:
                    route = token * token_topk + lane
                    global_expert = topk_ids[route]
                    if global_expert >= first_expert and global_expert < last_expert:
                        expert = global_expert - first_expert
                        if assign:
                            row = cu_tokens[expert] + cute.arch.atomic_add(
                                routing_counters + expert, cutlass.Int32(1), scope="gpu"
                            )
                            topk_pos[route] = row
                            row_to_route = cute.make_tensor(
                                row_to_route_ptr + row, cute.make_layout(1)
                            )
                            row_to_route[0] = route
                        else:
                            row = topk_pos[route]
            rows = cute.make_rmem_tensor(cute.make_layout(token_topk), cutlass.Int32)
            for j in cutlass.range_constexpr(token_topk):
                rows[j] = cute.arch.shuffle_sync(row, offset=j)
            if token < num_items:
                if cutlass.const_expr(bf16_to_mx):
                    quantize_bf16_row(
                        input_words,
                        cutlass.Int64(token) * (hidden_size // 2),
                        gathered_words,
                        cutlass.Int64(token) * words_per_row,
                        gathered_scale,
                        rows,
                        hidden_size,
                        k_atoms,
                        lane,
                        token_topk,
                        part,
                        parts,
                    )
                else:
                    scatter_mx_scales(
                        input_scale,
                        token,
                        gathered_scale,
                        rows,
                        hidden_size,
                        k_atoms,
                        lane,
                        token_topk,
                    )
        elif item < num_items:
            route = item
            global_expert = topk_ids[route]
            if global_expert >= first_expert and global_expert < last_expert:
                expert = global_expert - first_expert

                if lane == 0:
                    if assign:
                        local_row = cute.arch.atomic_add(
                            routing_counters + expert, cutlass.Int32(1), scope="gpu"
                        )
                        output_row = cu_tokens[expert] + local_row
                        output_rows[warp] = output_row
                        topk_pos[route] = output_row
                        # The inverse, for the epilogue that has to get from a
                        # row back to the token it belongs to.
                        row_to_route = cute.make_tensor(
                            row_to_route_ptr + output_row, cute.make_layout(1)
                        )
                        row_to_route[0] = route
                    else:
                        output_rows[warp] = topk_pos[route]
                cute.arch.sync_warp()

                token = route // num_topk
                output_row = output_rows[warp]
                rows = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Int32)
                rows[0] = output_row

                destination_base = cutlass.Int64(output_row) * words_per_row
                if cutlass.const_expr(not bf16_to_mx):
                    source_base = cutlass.Int64(token) * words_per_row
                    num_vectors = hidden_size // (_INTS_PER_VECTOR * 4)

                    frag = cute.make_rmem_tensor(cute.make_layout(_INTS_PER_VECTOR), cutlass.Int32)
                    vector = lane
                    while vector < num_vectors:
                        word = cutlass.Int64(vector) * _INTS_PER_VECTOR
                        source = cute.make_tensor(
                            input_words + (source_base + word),
                            cute.make_layout(_INTS_PER_VECTOR),
                        )
                        destination = cute.make_tensor(
                            gathered_words + (destination_base + word),
                            cute.make_layout(_INTS_PER_VECTOR),
                        )
                        cute.autovec_copy(source, frag)
                        cute.autovec_copy(frag, destination)
                        vector += _WARP_SIZE

                if cutlass.const_expr(bf16_to_mx):
                    quantize_bf16_row(
                        input_words,
                        cutlass.Int64(token) * (hidden_size // 2),
                        gathered_words,
                        destination_base,
                        gathered_scale,
                        rows,
                        hidden_size,
                        k_atoms,
                        lane,
                        1,
                        cutlass.Int32(0),
                        cutlass.Int32(1),
                    )
                elif cutlass.const_expr(fp32_to_mx):
                    # Qwen 128-block float32 → MX UE8M0: one byte per 32 K,
                    # repeating the 128-block scale across four MX vectors.
                    num_k_blocks_in = hidden_size // SCALE_K_ALIGNMENT
                    num_k_blocks_out = hidden_size // SCALE_VECTOR_SIZE
                    token_scale = cute.make_tensor(
                        input_scale + cutlass.Int64(token) * num_k_blocks_in,
                        cute.make_layout(num_k_blocks_in),
                    )
                    block_k = lane
                    while block_k < num_k_blocks_out:
                        offset = scale_offset_device(
                            output_row, block_k * SCALE_VECTOR_SIZE, k_atoms
                        )
                        slot = cute.make_tensor(gathered_scale + offset, cute.make_layout(1))
                        slot[0] = cutlass.Uint8(to_ue8m0_storage(token_scale[block_k // 4]))
                        block_k += _WARP_SIZE
                elif cutlass.const_expr(scale_f32):
                    num_k_blocks = hidden_size // SCALE_K_ALIGNMENT
                    token_scale = cute.make_tensor(
                        input_scale + cutlass.Int64(token) * num_k_blocks,
                        cute.make_layout(num_k_blocks),
                    )
                    dest_scale = cute.make_tensor(
                        gathered_scale + cutlass.Int64(output_row) * num_k_blocks,
                        cute.make_layout(num_k_blocks),
                    )
                    block_k = lane
                    while block_k < num_k_blocks:
                        dest_scale[block_k] = token_scale[block_k]
                        block_k += _WARP_SIZE
                else:
                    scatter_mx_scales(
                        input_scale, token, gathered_scale, rows, hidden_size, k_atoms, lane, 1
                    )

        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def gather_host(
        input_words: cute.Pointer,
        input_scale: cute.Pointer,
        topk_ids: cute.Pointer,
        cu_tokens: cute.Pointer,
        routing_counters: cute.Pointer,
        topk_pos: cute.Pointer,
        row_to_route: cute.Pointer,
        gathered_words: cute.Pointer,
        gathered_scale: cute.Pointer,
        num_items: cutlass.Int32,
        num_topk: cutlass.Int32,
        hidden_size: cutlass.Int32,
        first_expert: cutlass.Int32,
        last_expert: cutlass.Int32,
        k_atoms: cutlass.Int32,
        stream: cuda.CUstream,
        warps: cutlass.Constexpr = _WARPS_PER_BLOCK,
        assign: cutlass.Constexpr = assign_position,
        scale_f32: cutlass.Constexpr = scale_f32,
        fp32_to_mx: cutlass.Constexpr = fp32_to_mx,
        bf16_to_mx: cutlass.Constexpr = bf16_to_mx,
        token_topk: cutlass.Constexpr = token_topk,
    ):
        blocks = (num_items * _token_parts(assign, bf16_to_mx, token_topk) + warps - 1) // warps
        gather_kernel(
            input_words,
            input_scale,
            topk_ids,
            cu_tokens,
            routing_counters,
            topk_pos,
            row_to_route,
            gathered_words,
            gathered_scale,
            num_items,
            num_topk,
            hidden_size,
            first_expert,
            last_expert,
            k_atoms,
            warps,
            assign,
            scale_f32,
            fp32_to_mx,
            bf16_to_mx,
            token_topk,
        ).launch(
            grid=[blocks, 1, 1],
            block=[warps * _WARP_SIZE, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    return gather_host


def _compiled(
    assign_position: bool,
    scale_f32: bool,
    *args,
    fp32_to_mx: bool = False,
    bf16_to_mx: bool = False,
    token_topk: int = 0,
):
    key = (
        "gather",
        assign_position,
        scale_f32,
        fp32_to_mx,
        bf16_to_mx,
        token_topk,
        _WARPS_PER_BLOCK,
    )
    if key not in _cache:
        _, cute = import_cutlass()
        _cache[key] = launcher(
            cute.compile(
                _build(assign_position, scale_f32, fp32_to_mx, bf16_to_mx, token_topk), *args
            )
        )
    return _cache[key]


def gather_mxfp8_routes(
    x: Tensor,
    x_scale: Tensor,
    topk_ids: Tensor,
    buffers,
    gathered_input: Tensor,
    gathered_scale: Tensor,
    total_routes: int,
    num_topk: int,
    hidden_size: int,
    num_expert_local: int,
    rank_ep: int,
    assign_position: bool,
) -> None:
    """Scatter routed tokens into expert-contiguous buffers."""
    if hidden_size % 16 != 0:
        raise ValueError(f"hidden_size must be a multiple of 16; got {hidden_size}")

    cutlass, _ = import_cutlass()
    first_expert = rank_ep * num_expert_local

    args = (
        make_ptr(cutlass.Int32, x),
        make_ptr(cutlass.Uint8, x_scale),
        make_ptr(cutlass.Int32, topk_ids),
        make_ptr(cutlass.Int32, buffers.cu_tokens),
        make_ptr(cutlass.Int32, buffers.routing_counters),
        make_ptr(cutlass.Int32, buffers.topk_pos),
        make_ptr(cutlass.Int32, buffers.row_to_route),
        make_ptr(cutlass.Int32, gathered_input),
        make_ptr(cutlass.Uint8, gathered_scale),
        scalar(cutlass.Int32, total_routes),
        scalar(cutlass.Int32, num_topk),
        scalar(cutlass.Int32, hidden_size),
        scalar(cutlass.Int32, first_expert),
        scalar(cutlass.Int32, first_expert + num_expert_local),
        scalar(cutlass.Int32, ceil_div(hidden_size, SCALE_K_ALIGNMENT)),
        current_stream(),
    )
    _compiled(assign_position, False, *args)(*args)


def gather_bf16_to_mx_routes(
    x: Tensor,
    topk_ids: Tensor,
    buffers,
    gathered_input: Tensor,
    gathered_scale: Tensor,
    total_routes: int,
    num_topk: int,
    hidden_size: int,
    num_expert_local: int,
    rank_ep: int,
    assign_position: bool,
) -> None:
    """Scatter routed BF16 tokens, quantizing each to MXFP8 on the way.

    A token routed to ``top_k`` experts is quantized once per route; the
    quantization is a few instructions on a row the warp loads anyway, and it
    saves the caller a pass over the activations.
    """
    if hidden_size % SCALE_VECTOR_SIZE != 0:
        raise ValueError(
            f"hidden_size must be a multiple of {SCALE_VECTOR_SIZE}; got {hidden_size}"
        )
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x dtype must be bfloat16; got {x.dtype}")

    cutlass, _ = import_cutlass()
    first_expert = rank_ep * num_expert_local
    # The input scale slot is unused; any pointer of the right type fills it.
    args = (
        make_ptr(cutlass.Int32, x),
        make_ptr(cutlass.Uint8, gathered_scale),
        make_ptr(cutlass.Int32, topk_ids),
        make_ptr(cutlass.Int32, buffers.cu_tokens),
        make_ptr(cutlass.Int32, buffers.routing_counters),
        make_ptr(cutlass.Int32, buffers.topk_pos),
        make_ptr(cutlass.Int32, buffers.row_to_route),
        make_ptr(cutlass.Int32, gathered_input),
        make_ptr(cutlass.Uint8, gathered_scale),
        scalar(cutlass.Int32, total_routes),
        scalar(cutlass.Int32, num_topk),
        scalar(cutlass.Int32, hidden_size),
        scalar(cutlass.Int32, first_expert),
        scalar(cutlass.Int32, first_expert + num_expert_local),
        scalar(cutlass.Int32, ceil_div(hidden_size, SCALE_K_ALIGNMENT)),
        current_stream(),
    )
    _compiled(assign_position, False, *args, bf16_to_mx=True)(*args)


def gather_token_routes(
    x: Tensor,
    x_scale: Optional[Tensor],
    topk_ids: Tensor,
    buffers,
    quantized_input: Optional[Tensor],
    gathered_scale: Tensor,
    num_tokens: int,
    num_topk: int,
    hidden_size: int,
    num_expert_local: int,
    rank_ep: int,
    assign_position: bool,
) -> None:
    """Place routes for a GEMM that reads A rows by token, copying no rows.

    One warp per token, one lane per route. Each route to a local expert gets
    its padded row -- allocated here when ``assign_position``, as the route
    gather would, otherwise read from ``topk_pos`` -- and the token's UE8M0
    scales at that row of ``gathered_scale``. MXFP8 ``x`` is then the GEMM's A
    as it is. BF16 ``x`` (``x_scale=None``) is quantized once per token into
    ``quantized_input`` ``[num_tokens, hidden_size]``, with the same rounding
    as :func:`gather_bf16_to_mx_routes`.
    """
    if num_topk > MAX_TOKEN_TOPK:
        raise ValueError(f"at most {MAX_TOKEN_TOPK} routes per token; got {num_topk}")
    bf16 = x_scale is None
    if hidden_size % SCALE_VECTOR_SIZE != 0:
        raise ValueError(
            f"hidden_size must be a multiple of {SCALE_VECTOR_SIZE}; got {hidden_size}"
        )
    if bf16 and (x.dtype != torch.bfloat16 or quantized_input is None):
        raise ValueError("BF16 x needs quantized_input")

    cutlass, _ = import_cutlass()
    first_expert = rank_ep * num_expert_local
    # Unused slots take any pointer of the right type: the input scale when
    # quantizing, the output rows when not.
    args = (
        make_ptr(cutlass.Int32, x),
        make_ptr(cutlass.Uint8, gathered_scale if bf16 else x_scale),
        make_ptr(cutlass.Int32, topk_ids),
        make_ptr(cutlass.Int32, buffers.cu_tokens),
        make_ptr(cutlass.Int32, buffers.routing_counters),
        make_ptr(cutlass.Int32, buffers.topk_pos),
        make_ptr(cutlass.Int32, buffers.row_to_route),
        make_ptr(cutlass.Int32, quantized_input if bf16 else x),
        make_ptr(cutlass.Uint8, gathered_scale),
        scalar(cutlass.Int32, num_tokens),
        scalar(cutlass.Int32, num_topk),
        scalar(cutlass.Int32, hidden_size),
        scalar(cutlass.Int32, first_expert),
        scalar(cutlass.Int32, first_expert + num_expert_local),
        scalar(cutlass.Int32, ceil_div(hidden_size, SCALE_K_ALIGNMENT)),
        current_stream(),
    )
    _compiled(assign_position, False, *args, bf16_to_mx=bf16, token_topk=num_topk)(*args)


def gather_fp8_block128_routes(
    x: Tensor,
    x_scale: Tensor,
    topk_ids: Tensor,
    buffers,
    gathered_input: Tensor,
    gathered_scale: Tensor,
    total_routes: int,
    num_topk: int,
    hidden_size: int,
    num_expert_local: int,
    rank_ep: int,
    assign_position: bool,
) -> None:
    """Scatter routed FP8 tokens and their float32 128-block scales.

    ``x_scale`` is ``[num_tokens, hidden_size // 128]`` float32. The gathered
    scales are the same layout over padded expert rows, not the interleaved
    UE8M0 SFA atom.
    """
    if hidden_size % SCALE_K_ALIGNMENT != 0:
        raise ValueError(
            f"hidden_size must be a multiple of {SCALE_K_ALIGNMENT}; got {hidden_size}"
        )
    if x_scale.dtype != torch.float32:
        raise ValueError(f"x_scale dtype must be float32; got {x_scale.dtype}")

    cutlass, _ = import_cutlass()
    first_expert = rank_ep * num_expert_local
    args = (
        make_ptr(cutlass.Int32, x),
        make_ptr(cutlass.Float32, x_scale),
        make_ptr(cutlass.Int32, topk_ids),
        make_ptr(cutlass.Int32, buffers.cu_tokens),
        make_ptr(cutlass.Int32, buffers.routing_counters),
        make_ptr(cutlass.Int32, buffers.topk_pos),
        make_ptr(cutlass.Int32, buffers.row_to_route),
        make_ptr(cutlass.Int32, gathered_input),
        make_ptr(cutlass.Float32, gathered_scale),
        scalar(cutlass.Int32, total_routes),
        scalar(cutlass.Int32, num_topk),
        scalar(cutlass.Int32, hidden_size),
        scalar(cutlass.Int32, first_expert),
        scalar(cutlass.Int32, first_expert + num_expert_local),
        scalar(cutlass.Int32, ceil_div(hidden_size, SCALE_K_ALIGNMENT)),
        current_stream(),
    )
    _compiled(assign_position, True, *args)(*args)


def gather_fp8_to_mx_routes(
    x: Tensor,
    x_scale: Tensor,
    topk_ids: Tensor,
    buffers,
    gathered_input: Tensor,
    gathered_scale: Tensor,
    total_routes: int,
    num_topk: int,
    hidden_size: int,
    num_expert_local: int,
    rank_ep: int,
    assign_position: bool,
) -> None:
    """Scatter routed FP8 tokens and ceil their 128-block scales into MX SFA.

    ``x_scale`` is ``[num_tokens, hidden_size // 128]`` float32. Each 128-block
    scale is rounded up to UE8M0 and written four times into the interleaved
    MX atom, so the MMA applies the same scale across 128 K -- DeepGEMM's
    ``kGranKA=128`` broadcast, without a host-side scale tensor.
    """
    if hidden_size % SCALE_K_ALIGNMENT != 0:
        raise ValueError(
            f"hidden_size must be a multiple of {SCALE_K_ALIGNMENT}; got {hidden_size}"
        )
    if x_scale.dtype != torch.float32:
        raise ValueError(f"x_scale dtype must be float32; got {x_scale.dtype}")

    cutlass, _ = import_cutlass()
    first_expert = rank_ep * num_expert_local
    args = (
        make_ptr(cutlass.Int32, x),
        make_ptr(cutlass.Float32, x_scale),
        make_ptr(cutlass.Int32, topk_ids),
        make_ptr(cutlass.Int32, buffers.cu_tokens),
        make_ptr(cutlass.Int32, buffers.routing_counters),
        make_ptr(cutlass.Int32, buffers.topk_pos),
        make_ptr(cutlass.Int32, buffers.row_to_route),
        make_ptr(cutlass.Int32, gathered_input),
        make_ptr(cutlass.Uint8, gathered_scale),
        scalar(cutlass.Int32, total_routes),
        scalar(cutlass.Int32, num_topk),
        scalar(cutlass.Int32, hidden_size),
        scalar(cutlass.Int32, first_expert),
        scalar(cutlass.Int32, first_expert + num_expert_local),
        scalar(cutlass.Int32, ceil_div(hidden_size, SCALE_K_ALIGNMENT)),
        current_stream(),
    )
    _compiled(assign_position, False, *args, fp32_to_mx=True)(*args)
