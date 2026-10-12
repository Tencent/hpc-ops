# Copyright (C) 2026 Tencent.

"""Route preparation for the CuTe DSL SM120 fused MoE.

Builds per-expert route counts and prefix sums for gather and grouped GEMM.
The two-kernel path counts routes in parallel and assigns output rows later
during gather; the single-CTA path does everything up front and is used for
small token counts, where the extra launches cost more than doing the counting
on one SM. Both end in the same CTA-wide scan over the per-expert counts,
executed serially within the CTA.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from .._compile import current_stream, import_cutlass, launcher, make_ptr, scalar
from .._layouts import SCALE_MN_ALIGNMENT
from ._tiles import gemm_tile_m_for

# ``1 <= E <= 512``. The small-route
# kernel keeps two counters per expert in static shared memory.
MAX_EXPERTS = 512

SMALL_ROUTE_TOKEN_LIMIT = 128

# Enough halvings to bisect MAX_EXPERTS entries.
GROUP_SEARCH_STEPS = 9

_THREADS = 256

WARP_SIZE = 32
# Both routing paths scan the per-expert counts with a single CTA, so one
# thread covers this many experts and the block covers MAX_EXPERTS.
SCAN_ITEMS = MAX_EXPERTS // _THREADS
SCAN_WARPS = _THREADS // WARP_SIZE
# Halvings needed for a warp-wide shuffle scan.
SCAN_SHUFFLE_STEPS = 5
# ``shfl.up`` reads the clamp operand as the segment's lower bound, where the
# other modes read it as the upper bound. The wrapper defaults to the latter,
# which leaves every lane out of range and turns the shuffle into a no-op.
SCAN_SHUFFLE_CLAMP = 0


def find_row_group(cu_m_tiles, num_groups, tile):
    """Expert owning ``tile``, for kernels that see padded rows but not experts.

    The trip count is fixed so the loop unrolls while tracing, which is what
    keeps this usable from a plain helper -- a data-dependent ``while`` would
    need its own ``@cute.jit`` wrapper. ``select_`` keeps it branchless, which
    matters because every row of the padded buffer runs it.
    """
    cutlass, _ = import_cutlass()

    low = cutlass.Int32(0)
    high = cutlass.Int32(num_groups - 1)
    for _ in range(GROUP_SEARCH_STEPS):
        mid = (low + high + 1) // 2
        take = cu_m_tiles[mid] <= tile
        low = cutlass.Int32(cutlass.select_(take, mid, low))
        high = cutlass.Int32(cutlass.select_(take, high, mid - 1))
    return low


@dataclass(frozen=True)
class RouteShape:
    """What a token count implies about the padded row buffers and tile shapes."""

    rows_per_expert: float
    tile_m: int
    m_tiles: int
    max_padded_rows: int


def route_shape(num_tokens: int, num_topk: int, num_experts: int) -> RouteShape:
    """Everything the host has to decide before the routing kernel has run.

    Separate from :class:`RouteBuffers` because the same numbers decide which
    kernel variants a token count needs, and ``_warmup`` wants them for every
    size a deployment can see without allocating a buffer for each.

    Routing is close enough to uniform that the expected number of live experts
    predicts the real tile count within a couple of percent, and these only pick
    tile shapes. Whether the real routing is skewed changes the answer by less
    than the choices' sensitivity to it.
    """
    total_routes = num_tokens * num_topk
    live = num_experts * (1.0 - (1.0 - 1.0 / num_experts) ** total_routes)
    # How many rows an expert gets, which is what the tile height and the GEMM's
    # A load box height both key off.
    rows_per_expert = total_routes / max(live, 1.0)
    # Fixed here rather than per kernel: it is the granularity the routing
    # kernel pads rows to, so everything downstream has to agree with it, and it
    # sets how much padding the buffers have to hold.
    tile_m = gemm_tile_m_for(rows_per_expert)
    # How many row tiles the GEMMs will really see, as opposed to the worst case
    # below.
    m_tiles = max(1, round(live + total_routes / tile_m))

    # Worst case for sum(round_up(count_e, tile_m)): every expert that got at
    # least one route wastes tile_m - 1 rows. entry.cc bounds the active expert
    # count by num_experts alone, which is far too loose for a 256-expert decode
    # step where only a handful of experts are live.
    max_padded_rows = total_routes + (tile_m - 1) * min(num_experts, total_routes)
    # The scale-factor layout is built out of whole 128-row atoms even when tiles
    # are a fraction of that tall, so the buffer height follows the atom.
    max_padded_rows += -max_padded_rows % SCALE_MN_ALIGNMENT
    return RouteShape(rows_per_expert, tile_m, m_tiles, max_padded_rows)


_cache: Dict[Tuple[Any, ...], Any] = {}


def _build(tile_m: int):
    cutlass, cute = import_cutlass()
    import cuda.bindings.driver as cuda

    @cute.jit
    def scan_expert_tiles(counts, num_experts, cu_m_tiles, cu_tokens, warp_totals):
        """Lay the experts out in padded rows, one CTA wide.

        Thread ``t`` owns experts ``[t * SCAN_ITEMS, (t + 1) * SCAN_ITEMS)``, so
        the running sum is serial within a thread, a shuffle scan across a warp,
        and a read of the preceding warps' totals out of shared memory -- one
        barrier for the whole thing. The slots past ``num_experts`` are carried
        as zeros rather than skipped: they keep the shuffles unpredicated and
        they make the block total fall out of the same scan.
        """
        tidx, _, _ = cute.arch.thread_idx()
        lane = cutlass.Int32(tidx) % WARP_SIZE
        warp = cutlass.Int32(tidx) // WARP_SIZE

        tiles = []
        thread_total = cutlass.Int32(0)
        for item in cutlass.range_constexpr(SCAN_ITEMS):
            slot = cutlass.Int32(tidx) * SCAN_ITEMS + item
            live = slot < num_experts
            # The padding slots still issue the load, so it has to stay in
            # bounds; they read expert 0 and drop the value.
            safe = cutlass.Int32(cutlass.select_(live, slot, cutlass.Int32(0)))
            count = cutlass.Int32(cutlass.select_(live, counts[safe], cutlass.Int32(0)))
            item_tiles = (count + tile_m - 1) // tile_m
            tiles.append(item_tiles)
            thread_total += item_tiles

        inclusive = thread_total
        for step in cutlass.range_constexpr(SCAN_SHUFFLE_STEPS):
            offset = 1 << step
            shifted = cute.arch.shuffle_sync_up(
                inclusive, offset=offset, mask_and_clamp=SCAN_SHUFFLE_CLAMP
            )
            inclusive = cutlass.Int32(
                cutlass.select_(lane >= offset, inclusive + shifted, inclusive)
            )

        if lane == WARP_SIZE - 1:
            warp_totals[warp] = inclusive
        cute.arch.barrier()

        warp_offset = cutlass.Int32(0)
        block_total = cutlass.Int32(0)
        for other in cutlass.range_constexpr(SCAN_WARPS):
            value = warp_totals[other]
            warp_offset += cutlass.Int32(cutlass.select_(warp > other, value, cutlass.Int32(0)))
            block_total += value

        prefix = warp_offset + inclusive - thread_total
        for item in cutlass.range_constexpr(SCAN_ITEMS):
            slot = cutlass.Int32(tidx) * SCAN_ITEMS + item
            if slot < num_experts:
                cu_m_tiles[slot] = prefix
                cu_tokens[slot] = prefix * tile_m
            prefix += tiles[item]

        if tidx == 0:
            cu_m_tiles[num_experts] = block_total
            cu_tokens[num_experts] = block_total * tile_m

    @cute.kernel
    def count_kernel(
        topk_ids_ptr: cute.Pointer,
        topk_pos_ptr: cute.Pointer,
        route_counts: cute.Pointer,
        total_routes: cutlass.Int32,
        first_expert: cutlass.Int32,
        last_expert: cutlass.Int32,
        threads: cutlass.Constexpr,
    ):
        cute.arch.griddepcontrol_wait()

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        route = cutlass.Int32(bidx) * threads + cutlass.Int32(tidx)

        route_layout = cute.make_layout(total_routes)
        topk_ids = cute.make_tensor(topk_ids_ptr, route_layout)
        topk_pos = cute.make_tensor(topk_pos_ptr, route_layout)

        if route < total_routes:
            expert = topk_ids[route]
            if expert >= first_expert and expert < last_expert:
                cute.arch.atomic_add(
                    route_counts + (expert - first_expert),
                    cutlass.Int32(1),
                    scope="gpu",
                )
            topk_pos[route] = cutlass.Int32(-1)

        cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def prefix_kernel(
        route_counts_ptr: cute.Pointer,
        counts_ptr: cute.Pointer,
        cu_tokens_ptr: cute.Pointer,
        cu_m_tiles_ptr: cute.Pointer,
        counters_ptr: cute.Pointer,
        num_experts: cutlass.Int32,
    ):
        cute.arch.griddepcontrol_wait()

        tidx, _, _ = cute.arch.thread_idx()

        expert_layout = cute.make_layout(num_experts)
        cu_layout = cute.make_layout(num_experts + 1)
        route_counts = cute.make_tensor(route_counts_ptr, expert_layout)
        num_tokens_per_expert = cute.make_tensor(counts_ptr, expert_layout)
        routing_counters = cute.make_tensor(counters_ptr, expert_layout)
        cu_tokens = cute.make_tensor(cu_tokens_ptr, cu_layout)
        cu_m_tiles = cute.make_tensor(cu_m_tiles_ptr, cu_layout)

        totals_ptr = cute.arch.alloc_smem(cutlass.Int32, SCAN_WARPS, alignment=16)
        warp_totals = cute.make_tensor(totals_ptr, cute.make_layout(SCAN_WARPS))

        for item in cutlass.range_constexpr(SCAN_ITEMS):
            slot = cutlass.Int32(tidx) * SCAN_ITEMS + item
            if slot < num_experts:
                num_tokens_per_expert[slot] = route_counts[slot]
                routing_counters[slot] = cutlass.Int32(0)

        scan_expert_tiles(route_counts, num_experts, cu_m_tiles, cu_tokens, warp_totals)

        # Hand the accumulator back zeroed so the next call's atomics need no
        # memset in front of them. A thread only clears the slots it just read,
        # so this cannot outrun another thread's load.
        for item in cutlass.range_constexpr(SCAN_ITEMS):
            slot = cutlass.Int32(tidx) * SCAN_ITEMS + item
            if slot < num_experts:
                route_counts[slot] = cutlass.Int32(0)

        cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def prepare_small_kernel(
        topk_ids_ptr: cute.Pointer,
        topk_pos_ptr: cute.Pointer,
        row_to_route_ptr: cute.Pointer,
        counts_ptr: cute.Pointer,
        cu_tokens_ptr: cute.Pointer,
        cu_m_tiles_ptr: cute.Pointer,
        total_routes: cutlass.Int32,
        max_rows: cutlass.Int32,
        num_experts: cutlass.Int32,
        first_expert: cutlass.Int32,
        last_expert: cutlass.Int32,
        threads: cutlass.Constexpr,
    ):
        cute.arch.griddepcontrol_wait()

        tidx, _, _ = cute.arch.thread_idx()

        route_layout = cute.make_layout(total_routes)
        cu_layout = cute.make_layout(num_experts + 1)
        topk_ids = cute.make_tensor(topk_ids_ptr, route_layout)
        topk_pos = cute.make_tensor(topk_pos_ptr, route_layout)
        # Indexed by padded row rather than by route: the experts' row blocks
        # are rounded up to whole tiles, so the rows outstrip the routes.
        row_to_route = cute.make_tensor(row_to_route_ptr, cute.make_layout(max_rows))
        num_tokens_per_expert = cute.make_tensor(counts_ptr, cute.make_layout(num_experts))
        cu_tokens = cute.make_tensor(cu_tokens_ptr, cu_layout)
        cu_m_tiles = cute.make_tensor(cu_m_tiles_ptr, cu_layout)

        counts_ptr = cute.arch.alloc_smem(cutlass.Int32, MAX_EXPERTS, alignment=16)
        rows_ptr = cute.arch.alloc_smem(cutlass.Int32, MAX_EXPERTS, alignment=16)
        totals_ptr = cute.arch.alloc_smem(cutlass.Int32, SCAN_WARPS, alignment=16)
        counts = cute.make_tensor(counts_ptr, cute.make_layout(MAX_EXPERTS))
        row_counters = cute.make_tensor(rows_ptr, cute.make_layout(MAX_EXPERTS))
        warp_totals = cute.make_tensor(totals_ptr, cute.make_layout(SCAN_WARPS))

        expert = cutlass.Int32(tidx)
        while expert < num_experts:
            counts[expert] = cutlass.Int32(0)
            row_counters[expert] = cutlass.Int32(0)
            expert += threads
        cute.arch.barrier()

        route = cutlass.Int32(tidx)
        while route < total_routes:
            global_expert = topk_ids[route]
            topk_pos[route] = cutlass.Int32(-1)
            if global_expert >= first_expert and global_expert < last_expert:
                cute.arch.atomic_add(
                    counts.iterator + (global_expert - first_expert),
                    cutlass.Int32(1),
                    scope="cta",
                )
            route += threads
        cute.arch.barrier()

        for item in cutlass.range_constexpr(SCAN_ITEMS):
            slot = cutlass.Int32(tidx) * SCAN_ITEMS + item
            if slot < num_experts:
                num_tokens_per_expert[slot] = counts[slot]

        scan_expert_tiles(counts, num_experts, cu_m_tiles, cu_tokens, warp_totals)
        # The row assignment below reads back every expert's base, not just the
        # ones this thread scanned.
        cute.arch.barrier()

        route = cutlass.Int32(tidx)
        while route < total_routes:
            global_expert = topk_ids[route]
            if global_expert >= first_expert and global_expert < last_expert:
                local_expert = global_expert - first_expert
                local_row = cute.arch.atomic_add(
                    row_counters.iterator + local_expert,
                    cutlass.Int32(1),
                    scope="cta",
                )
                row = cu_tokens[local_expert] + local_row
                topk_pos[route] = row
                row_to_route[row] = route
            route += threads

        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def count_host(
        topk_ids: cute.Pointer,
        topk_pos: cute.Pointer,
        route_counts: cute.Pointer,
        total_routes: cutlass.Int32,
        first_expert: cutlass.Int32,
        last_expert: cutlass.Int32,
        stream: cuda.CUstream,
        threads: cutlass.Constexpr = _THREADS,
    ):
        blocks = (total_routes + threads - 1) // threads
        count_kernel(
            topk_ids,
            topk_pos,
            route_counts,
            total_routes,
            first_expert,
            last_expert,
            threads,
        ).launch(grid=[blocks, 1, 1], block=[threads, 1, 1], stream=stream, use_pdl=True)

    @cute.jit
    def prefix_host(
        route_counts: cute.Pointer,
        num_tokens_per_expert: cute.Pointer,
        cu_tokens: cute.Pointer,
        cu_m_tiles: cute.Pointer,
        routing_counters: cute.Pointer,
        num_experts: cutlass.Int32,
        stream: cuda.CUstream,
        threads: cutlass.Constexpr = _THREADS,
    ):
        prefix_kernel(
            route_counts,
            num_tokens_per_expert,
            cu_tokens,
            cu_m_tiles,
            routing_counters,
            num_experts,
        ).launch(grid=[1, 1, 1], block=[threads, 1, 1], stream=stream, use_pdl=True)

    @cute.jit
    def prepare_small_host(
        topk_ids: cute.Pointer,
        topk_pos: cute.Pointer,
        row_to_route: cute.Pointer,
        num_tokens_per_expert: cute.Pointer,
        cu_tokens: cute.Pointer,
        cu_m_tiles: cute.Pointer,
        total_routes: cutlass.Int32,
        max_rows: cutlass.Int32,
        num_experts: cutlass.Int32,
        first_expert: cutlass.Int32,
        last_expert: cutlass.Int32,
        stream: cuda.CUstream,
        threads: cutlass.Constexpr = _THREADS,
    ):
        prepare_small_kernel(
            topk_ids,
            topk_pos,
            row_to_route,
            num_tokens_per_expert,
            cu_tokens,
            cu_m_tiles,
            total_routes,
            max_rows,
            num_experts,
            first_expert,
            last_expert,
            threads,
        ).launch(grid=[1, 1, 1], block=[threads, 1, 1], stream=stream, use_pdl=True)

    return count_host, prefix_host, prepare_small_host


def _compiled(name: str, index: int, tile_m: int, *args):
    key = (name, _THREADS, tile_m)
    if key not in _cache:
        _, cute = import_cutlass()
        _cache[key] = launcher(cute.compile(_build(tile_m)[index], *args))
    return _cache[key]


class RouteBuffers:
    """Integer buffers describing how routes map onto expert-contiguous rows."""

    def __init__(self, num_tokens: int, num_topk: int, num_experts: int, device):
        int_options = {"dtype": torch.int32, "device": device}

        self.topk_pos = torch.empty((num_tokens, num_topk), **int_options)
        self.num_tokens_per_expert = torch.empty((num_experts,), **int_options)
        self.cu_tokens = torch.empty((num_experts + 1,), **int_options)
        # Private to the multi-kernel path's atomics. The prefix kernel copies
        # route_counts into num_tokens_per_expert and clears both again, which
        # is what lets a subsequent launch skip a memset. prepare_small never
        # touches either buffer, so decode must not pay a zeros launch for them
        # -- under CUDA graph that fill is replayed every step (~1.6 us).
        if num_tokens <= SMALL_ROUTE_TOKEN_LIMIT:
            self.routing_counters = torch.empty((num_experts,), **int_options)
            self.route_counts = torch.empty((num_experts,), **int_options)
        else:
            # One fill for both: each zeros call is its own launch. The second
            # starts on a 16-byte boundary, which make_ptr assumes.
            stride = -(-num_experts // 4) * 4
            counters = torch.zeros((2 * stride,), **int_options)
            self.routing_counters = counters[:num_experts]
            self.route_counts = counters[stride : stride + num_experts]
        # Cumulative count of row tiles; doubles as the GEMM tile schedule and,
        # times tile_m, as the padded row offset of each expert.
        self.cu_m_tiles = torch.empty((num_experts + 1,), **int_options)

        shape = route_shape(num_tokens, num_topk, num_experts)
        self.expected_rows_per_expert = shape.rows_per_expert
        self.tile_m = shape.tile_m
        self.expected_m_tiles = shape.m_tiles
        self.max_padded_rows = shape.max_padded_rows

        # The inverse of topk_pos, for the fused reduce: an epilogue that has a
        # row in hand needs the token and routing weight that go with it. Only
        # the rows an expert really owns are read, so the padding is left alone.
        self.row_to_route = torch.empty((self.max_padded_rows,), **int_options)


def prepare_routes(
    topk_ids: Tensor,
    buffers: RouteBuffers,
    num_tokens: int,
    num_topk: int,
    num_expert_local: int,
    rank_ep: int,
) -> bool:
    """Fill ``buffers`` from ``topk_ids``.

    Returns:
        ``True`` when the multi-kernel path only counted rows and the gather
        still has to allocate them; the single-CTA path assigns ``topk_pos``
        itself and returns ``False``.
    """
    if num_expert_local > MAX_EXPERTS:
        raise ValueError(
            f"at most {MAX_EXPERTS} local experts are supported; got {num_expert_local}"
        )

    cutlass, _ = import_cutlass()
    stream = current_stream()

    total_routes = num_tokens * num_topk
    first_expert = rank_ep * num_expert_local
    last_expert = first_expert + num_expert_local

    topk_ids_flat = make_ptr(cutlass.Int32, topk_ids)
    topk_pos_flat = make_ptr(cutlass.Int32, buffers.topk_pos)
    counts = make_ptr(cutlass.Int32, buffers.num_tokens_per_expert)
    cu_tokens = make_ptr(cutlass.Int32, buffers.cu_tokens)
    cu_m_tiles = make_ptr(cutlass.Int32, buffers.cu_m_tiles)

    if num_tokens <= SMALL_ROUTE_TOKEN_LIMIT:
        args = (
            topk_ids_flat,
            topk_pos_flat,
            make_ptr(cutlass.Int32, buffers.row_to_route),
            counts,
            cu_tokens,
            cu_m_tiles,
            scalar(cutlass.Int32, total_routes),
            scalar(cutlass.Int32, buffers.max_padded_rows),
            scalar(cutlass.Int32, num_expert_local),
            scalar(cutlass.Int32, first_expert),
            scalar(cutlass.Int32, last_expert),
            stream,
        )
        _compiled("prepare_small", 2, buffers.tile_m, *args)(*args)
        return False

    route_counts = make_ptr(cutlass.Int32, buffers.route_counts)
    count_args = (
        topk_ids_flat,
        topk_pos_flat,
        route_counts,
        scalar(cutlass.Int32, total_routes),
        scalar(cutlass.Int32, first_expert),
        scalar(cutlass.Int32, last_expert),
        stream,
    )
    _compiled("count", 0, buffers.tile_m, *count_args)(*count_args)

    prefix_args = (
        route_counts,
        counts,
        cu_tokens,
        cu_m_tiles,
        make_ptr(cutlass.Int32, buffers.routing_counters),
        scalar(cutlass.Int32, num_expert_local),
        stream,
    )
    _compiled("prefix", 1, buffers.tile_m, *prefix_args)(*prefix_args)
    return True
