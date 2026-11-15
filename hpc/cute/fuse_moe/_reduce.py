# Copyright (C) 2026 Tencent.

"""Top-k weighted reduction of expert outputs back onto token rows.

Each token row sums the BF16 outputs of the experts it was routed to, weighted
by the router scores, accumulating in FP32 and optionally folding in a
shared-expert result. Same role as ``reduce_kernel`` in
``src/fuse_moe/sm90/reduce.cu``.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from .._compile import current_stream, import_cutlass, launcher, make_ptr, scalar

_THREADS = 256
_ITEMS_PER_THREAD = 8  # 16 bytes of BF16

# A row's routes are staged in shared memory before the columns are walked, out
# of a fixed allocation. Not private to this kernel: the pipeline only reaches
# it for some batch sizes, so it has to enforce the same bound up front or the
# limit would depend on how large a batch happened to be.
MAX_TOPK = 128

_cache: Dict[Tuple[Any, ...], Any] = {}


def _build(has_shared: bool):
    cutlass, cute = import_cutlass()
    import cuda.bindings.driver as cuda

    @cute.kernel
    def reduce_kernel(
        y: cute.Pointer,
        x: cute.Pointer,
        topk_pos: cute.Pointer,
        topk_scale: cute.Pointer,
        shared_output: cute.Pointer,
        hidden_size: cutlass.Int32,
        num_topk: cutlass.Int32,
        num_block_col: cutlass.Int32,
        threads: cutlass.Constexpr,
        items: cutlass.Constexpr,
        with_shared: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        block = cutlass.Int32(bidx)
        row = block // num_block_col
        block_col = block % num_block_col
        col = (cutlass.Int32(tidx) + block_col * threads) * items

        pos_ptr = cute.arch.alloc_smem(cutlass.Int32, MAX_TOPK, alignment=16)
        scale_ptr = cute.arch.alloc_smem(cutlass.Float32, MAX_TOPK, alignment=16)
        pos_shm = cute.make_tensor(pos_ptr, cute.make_layout(MAX_TOPK))
        scale_shm = cute.make_tensor(scale_ptr, cute.make_layout(MAX_TOPK))

        cute.arch.griddepcontrol_wait()

        row_base = cutlass.Int64(row) * num_topk
        pos_src = cute.make_tensor(topk_pos + row_base, cute.make_layout(num_topk))
        scale_src = cute.make_tensor(topk_scale + row_base, cute.make_layout(num_topk))
        slot = cutlass.Int32(tidx)
        while slot < num_topk:
            pos_shm[slot] = pos_src[slot]
            scale_shm[slot] = scale_src[slot]
            slot += threads
        cute.arch.barrier()

        if col < hidden_size:
            acc = cute.make_rmem_tensor(cute.make_layout(items), cutlass.Float32)
            for i in cutlass.range_constexpr(items):
                acc[i] = cutlass.Float32(0.0)

            frag = cute.make_rmem_tensor(cute.make_layout(items), cutlass.BFloat16)
            for slot in cutlass.range(num_topk):
                position = pos_shm[slot]
                if position >= 0:
                    scale = scale_shm[slot]
                    source = cute.make_tensor(
                        x + (cutlass.Int64(position) * hidden_size + col),
                        cute.make_layout(items),
                    )
                    cute.autovec_copy(source, frag)
                    for i in cutlass.range_constexpr(items):
                        acc[i] = acc[i] + cutlass.Float32(frag[i]) * scale

            if with_shared:
                source = cute.make_tensor(
                    shared_output + (cutlass.Int64(row) * hidden_size + col),
                    cute.make_layout(items),
                )
                cute.autovec_copy(source, frag)
                for i in cutlass.range_constexpr(items):
                    acc[i] = acc[i] + cutlass.Float32(frag[i])

            for i in cutlass.range_constexpr(items):
                frag[i] = cutlass.BFloat16(acc[i])
            destination = cute.make_tensor(
                y + (cutlass.Int64(row) * hidden_size + col),
                cute.make_layout(items),
            )
            cute.autovec_copy(frag, destination)

        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def reduce_host(
        y: cute.Pointer,
        x: cute.Pointer,
        topk_pos: cute.Pointer,
        topk_scale: cute.Pointer,
        shared_output: cute.Pointer,
        num_seq: cutlass.Int32,
        hidden_size: cutlass.Int32,
        num_topk: cutlass.Int32,
        stream: cuda.CUstream,
        threads: cutlass.Constexpr = _THREADS,
        items: cutlass.Constexpr = _ITEMS_PER_THREAD,
        with_shared: cutlass.Constexpr = has_shared,
    ):
        num_block_col = (hidden_size // items + threads - 1) // threads
        reduce_kernel(
            y,
            x,
            topk_pos,
            topk_scale,
            shared_output,
            hidden_size,
            num_topk,
            num_block_col,
            threads,
            items,
            with_shared,
        ).launch(
            grid=[num_seq * num_block_col, 1, 1],
            block=[threads, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    return reduce_host


def _compiled(has_shared: bool, *args):
    key = ("reduce", has_shared, _THREADS, _ITEMS_PER_THREAD)
    if key not in _cache:
        _, cute = import_cutlass()
        _cache[key] = launcher(cute.compile(_build(has_shared), *args))
    return _cache[key]


def reduce_topk(
    output: Tensor,
    expert_output: Tensor,
    topk_pos: Tensor,
    topk_scale: Tensor,
    shared_output: Optional[Tensor],
    num_seq: int,
    hidden_size: int,
    num_topk: int,
) -> None:
    """Reduce ``expert_output`` rows into ``output`` in place."""
    if num_topk > MAX_TOPK:
        raise ValueError(f"at most {MAX_TOPK} experts per token are supported; got {num_topk}")
    if hidden_size % _ITEMS_PER_THREAD != 0:
        raise ValueError(
            f"hidden_size must be a multiple of {_ITEMS_PER_THREAD}; got {hidden_size}"
        )

    cutlass, _ = import_cutlass()
    has_shared = shared_output is not None
    # The kernel signature is uniform; when there is no shared expert the tensor
    # is never dereferenced, so any valid pointer works as a placeholder.
    shared = shared_output if has_shared else output

    args = (
        make_ptr(cutlass.BFloat16, output),
        make_ptr(cutlass.BFloat16, expert_output),
        make_ptr(cutlass.Int32, topk_pos),
        make_ptr(cutlass.Float32, topk_scale),
        make_ptr(cutlass.BFloat16, shared),
        scalar(cutlass.Int32, num_seq),
        scalar(cutlass.Int32, hidden_size),
        scalar(cutlass.Int32, num_topk),
        current_stream(),
    )
    _compiled(has_shared, *args)(*args)
