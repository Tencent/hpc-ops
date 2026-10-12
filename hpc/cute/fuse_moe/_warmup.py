# Copyright (C) 2026 Tencent.

"""Ahead-of-time compilation of the blockwise MoE's kernel variants.

The pipeline picks its tile height, its A load box height, its epilogue width
and whether the top-k sum is fused from the token count alone, so a process
that has only ever run a couple of batch sizes has only compiled the variants
those sizes reached. The rest are compiled the first time a batch reaches them,
on the critical path and once per rank: a server that profiled at its prefill
budget and captured CUDA graphs at its decode sizes still stalls for a second
or two mid-serve, the first time a batch lands between the two.

Nothing here decides anything. Every variant comes from running the same
host-side choices the pipeline runs -- :func:`_routing.route_shape`,
``_a_subtile_for``, ``_tile_n_for``, ``_fuse_reduce`` -- over the token counts a
deployment can see. ``tests/test_fuse_moe_dsl_warmup.py`` is what holds that
together: it warms a configuration and then asserts that real calls at sizes
across the range compile nothing further.

The kernels are compiled, not run. ``cute.compile`` only needs arguments of the
right type: every extent these kernels take is a runtime value, which is what
lets one compiled kernel serve every shape, so the buffers behind the pointers
are never read and one scratch allocation can stand in for all of them.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Tuple

import torch

from . import _gather, _grouped_gemm, _pipeline, _reduce, _routing
from .._compile import current_stream, import_cutlass, make_ptr
from .._layouts import SCALE_K_ALIGNMENT, ceil_div
from ._routing import SMALL_ROUTE_TOKEN_LIMIT, route_shape

# Enough to be a real allocation, small enough to be free. Nothing dereferences
# it; compiling only records the pointers.
_SCRATCH_BYTES = 1024


@dataclass(frozen=True)
class BlockwiseShape:
    """The parts of a layer's configuration that take part in the choices."""

    num_topk: int
    num_expert_local: int
    num_expert_total: int
    hidden_size: int
    intermediate_size: int


@dataclass(frozen=True)
class WarmupUnit:
    """One kernel variant to compile.

    ``key`` identifies the variant well enough to drop duplicates before
    compiling any of them; it is not the kernels' own cache key, which stays
    private to each module and does the real deduplication anyway.
    """

    key: Tuple[Any, ...]
    compile: Callable[[], None]


def blockwise_warmup_units(
    shape: BlockwiseShape,
    token_counts: Iterable[int],
    device: torch.device,
    has_shared_output: bool = False,
    bf16_input: bool = False,
) -> List[WarmupUnit]:
    """Every kernel variant ``fuse_moe_blockwise_w4a8_mxfp4_mxfp8`` can reach.

    Args:
        shape: The layer's routing and projection sizes.
        token_counts: Batch sizes to cover. Passing every size a scheduler can
            produce is cheap: the choices are arithmetic on a few integers and
            the variants collapse to a handful.
        device: Where the kernels will run. Its SM count and L2 size both take
            part in the choices.
        has_shared_output: Whether callers will hand the reduction a shared
            expert's output to add in.
        bf16_input: Whether callers pass BF16 activations for the gather to
            quantize, rather than MXFP8 ones.

    Returns:
        Deduplicated units, in the order the token counts first reached them.
    """
    cutlass, _ = import_cutlass()
    scratch = torch.empty(_SCRATCH_BYTES, dtype=torch.uint8, device=device)
    num_sms = _grouped_gemm._sm_count(device) * _grouped_gemm.CTAS_PER_SM
    gate_up_size = 2 * shape.intermediate_size

    units: Dict[Tuple[Any, ...], WarmupUnit] = {}

    def add(unit: WarmupUnit) -> None:
        units.setdefault(unit.key, unit)

    for num_tokens in token_counts:
        routes = route_shape(num_tokens, shape.num_topk, shape.num_expert_local)
        tile_m = routes.tile_m
        a_subtile = _grouped_gemm._a_subtile_for(routes.rows_per_expert, tile_m)
        # Reached through the module so that anything overriding the pipeline's
        # choice -- the tests do -- moves the warmup with it.
        fuse_reduce = _pipeline._fuse_reduce(
            _pipeline._local_routed_bytes(
                num_tokens * shape.num_topk,
                shape.hidden_size,
                shape.num_expert_local,
                shape.num_expert_total,
            ),
            device,
            num_tokens,
        )
        # The routing path splits on the token count alone and the gather
        # follows it: the multi-kernel path only counts rows, which leaves the
        # gather to allocate them.
        small = num_tokens <= SMALL_ROUTE_TOKEN_LIMIT
        if small:
            add(_routing_unit(cutlass, scratch, "prepare_small", 2, tile_m, pointers=6, ints=5))
        else:
            add(_routing_unit(cutlass, scratch, "count", 0, tile_m, pointers=3, ints=3))
            add(_routing_unit(cutlass, scratch, "prefix", 1, tile_m, pointers=5, ints=1))
        gather_a = _pipeline._gather_a(shape.num_topk)
        add(
            _gather_unit(
                cutlass,
                scratch,
                shape,
                assign_position=not small,
                bf16=bf16_input,
                token_topk=shape.num_topk if gather_a else 0,
            )
        )

        # The up projection always ends in the fused SwiGLU epilogue, whose slab
        # width is not a choice; the down projection picks its width from how
        # well the tiles fill the GPU, and its epilogue form from how much of L2
        # the routed rows would occupy.
        add(
            _gemm_unit(
                cutlass,
                scratch,
                num_sms=num_sms,
                n=gate_up_size,
                k=shape.hidden_size,
                tile_n=_grouped_gemm.FUSED_TILE_N,
                fused=True,
                scatter=False,
                a_subtile=a_subtile,
                tile_m=tile_m,
                a_gather=gather_a,
            )
        )
        gemm2_thin = _grouped_gemm._use_thin(False, tile_m, b_fp8=False)
        gemm2_num_sms = num_sms * _grouped_gemm.THIN_CTAS_PER_SM if gemm2_thin else num_sms
        add(
            _gemm_unit(
                cutlass,
                scratch,
                num_sms=gemm2_num_sms,
                n=shape.hidden_size,
                k=shape.intermediate_size,
                tile_n=_grouped_gemm._tile_n_for(routes.m_tiles, shape.hidden_size, gemm2_num_sms),
                fused=False,
                scatter=fuse_reduce,
                a_subtile=a_subtile,
                tile_m=tile_m,
                thin=gemm2_thin,
            )
        )
        if not fuse_reduce:
            add(_reduce_unit(cutlass, scratch, has_shared_output))

    return list(units.values())


def _routing_unit(
    cutlass, scratch, name: str, index: int, tile_m: int, pointers: int, ints: int
) -> WarmupUnit:
    """One of the three routing kernels. See :func:`_routing.prepare_routes`.

    All three take a run of ``int32`` buffers followed by a run of extents, so
    the counts are all a caller has to know to build an argument list of the
    right types.
    """

    def compile() -> None:
        args = (
            *(make_ptr(cutlass.Int32, scratch) for _ in range(pointers)),
            *(cutlass.Int32(0) for _ in range(ints)),
            current_stream(),
        )
        _routing._compiled(name, index, tile_m, *args)

    return WarmupUnit((name, tile_m), compile)


def _gather_unit(
    cutlass,
    scratch,
    shape: BlockwiseShape,
    assign_position: bool,
    bf16: bool = False,
    token_topk: int = 0,
) -> WarmupUnit:
    """See :func:`_gather.gather_mxfp8_routes`, :func:`_gather.gather_bf16_to_mx_routes`
    and, with ``token_topk``, :func:`_gather.gather_token_routes`."""

    def compile() -> None:
        args = (
            make_ptr(cutlass.Int32, scratch),
            make_ptr(cutlass.Uint8, scratch),
            *(make_ptr(cutlass.Int32, scratch) for _ in range(6)),
            make_ptr(cutlass.Uint8, scratch),
            cutlass.Int32(0),
            cutlass.Int32(shape.num_topk),
            cutlass.Int32(shape.hidden_size),
            cutlass.Int32(0),
            cutlass.Int32(shape.num_expert_local),
            cutlass.Int32(ceil_div(shape.hidden_size, SCALE_K_ALIGNMENT)),
            current_stream(),
        )
        _gather._compiled(assign_position, False, *args, bf16_to_mx=bf16, token_topk=token_topk)

    return WarmupUnit(("gather_bf16" if bf16 else "gather", assign_position, token_topk), compile)


def _gemm_unit(
    cutlass,
    scratch,
    num_sms: int,
    n: int,
    k: int,
    tile_n: int,
    fused: bool,
    scatter: bool,
    a_subtile: int,
    tile_m: int,
    b_fp8: bool = False,
    block128: bool = False,
    tile_k: int = _grouped_gemm.TILE_K,
    a_gather: bool = False,
    thin: bool = False,
) -> WarmupUnit:
    """See :func:`_grouped_gemm.grouped_block_scaled_gemm`."""

    def compile() -> None:
        args = (
            make_ptr(cutlass.Float8E4M3FN, scratch),
            make_ptr(cutlass.Float8E4M3FN if b_fp8 else cutlass.Float4E2M1FN, scratch),
            make_ptr(cutlass.Float8E8M0FNU, scratch),
            make_ptr(cutlass.Float8E8M0FNU, scratch),
            make_ptr(cutlass.Float8E4M3FN if fused else cutlass.BFloat16, scratch),
            make_ptr(
                cutlass.Float32 if fused and block128 else cutlass.Uint8,
                scratch,
            ),
            make_ptr(cutlass.Int32, scratch),
            make_ptr(cutlass.Int32, scratch),
            make_ptr(cutlass.Int32, scratch),
            make_ptr(cutlass.Float32, scratch),
            cutlass.Int32(tile_m),
            cutlass.Int32(n),
            cutlass.Int32(k),
            cutlass.Int32(1),
            cutlass.Int32(0),
            cutlass.Float32(-1.0),
            make_ptr(cutlass.Float32, scratch),
            make_ptr(cutlass.Float32, scratch),
            cutlass.Int32(1),
            cutlass.Int32(1),
            current_stream(),
        )
        _grouped_gemm._compiled(
            num_sms,
            tile_n,
            fused,
            scatter,
            a_subtile,
            tile_m,
            b_fp8,
            block128,
            tile_k,
            a_gather,
            *args,
            thin=thin,
        )

    return WarmupUnit(
        (
            "grouped_gemm",
            num_sms,
            tile_n,
            fused,
            scatter,
            a_subtile,
            tile_m,
            b_fp8,
            block128,
            tile_k,
            a_gather,
            thin,
        ),
        compile,
    )


def _reduce_unit(cutlass, scratch, has_shared: bool) -> WarmupUnit:
    """See :func:`_reduce.reduce_topk`."""

    def compile() -> None:
        args = (
            make_ptr(cutlass.BFloat16, scratch),
            make_ptr(cutlass.BFloat16, scratch),
            make_ptr(cutlass.Int32, scratch),
            make_ptr(cutlass.Float32, scratch),
            make_ptr(cutlass.BFloat16, scratch),
            *(cutlass.Int32(0) for _ in range(3)),
            current_stream(),
        )
        _reduce._compiled(has_shared, *args)

    return WarmupUnit(("reduce", has_shared), compile)


def _gather_fp8_unit(cutlass, scratch, shape: BlockwiseShape, assign_position: bool) -> WarmupUnit:
    """See :func:`_gather.gather_fp8_block128_routes`."""

    def compile() -> None:
        args = (
            make_ptr(cutlass.Int32, scratch),
            make_ptr(cutlass.Float32, scratch),
            *(make_ptr(cutlass.Int32, scratch) for _ in range(6)),
            make_ptr(cutlass.Float32, scratch),
            cutlass.Int32(0),
            cutlass.Int32(shape.num_topk),
            cutlass.Int32(shape.hidden_size),
            cutlass.Int32(0),
            cutlass.Int32(shape.num_expert_local),
            cutlass.Int32(ceil_div(shape.hidden_size, SCALE_K_ALIGNMENT)),
            current_stream(),
        )
        _gather._compiled(assign_position, True, *args)

    return WarmupUnit(("gather", assign_position, True), compile)


def _gather_fp8_to_mx_unit(
    cutlass, scratch, shape: BlockwiseShape, assign_position: bool
) -> WarmupUnit:
    """See :func:`_gather.gather_fp8_to_mx_routes`."""

    def compile() -> None:
        args = (
            make_ptr(cutlass.Int32, scratch),
            make_ptr(cutlass.Float32, scratch),
            *(make_ptr(cutlass.Int32, scratch) for _ in range(6)),
            make_ptr(cutlass.Uint8, scratch),
            cutlass.Int32(0),
            cutlass.Int32(shape.num_topk),
            cutlass.Int32(shape.hidden_size),
            cutlass.Int32(0),
            cutlass.Int32(shape.num_expert_local),
            cutlass.Int32(ceil_div(shape.hidden_size, SCALE_K_ALIGNMENT)),
            current_stream(),
        )
        _gather._compiled(assign_position, False, *args, fp32_to_mx=True)

    return WarmupUnit(("gather", assign_position, False, True), compile)


def fp8_blockwise_warmup_units(
    shape: BlockwiseShape,
    token_counts: Iterable[int],
    device: torch.device,
    has_shared_output: bool = False,
) -> List[WarmupUnit]:
    """Every kernel variant ``fuse_moe_blockwise_fp8`` can reach."""
    cutlass, _ = import_cutlass()
    scratch = torch.empty(_SCRATCH_BYTES, dtype=torch.uint8, device=device)
    num_sms = _grouped_gemm._sm_count(device) * _grouped_gemm.CTAS_PER_SM
    gate_up_size = 2 * shape.intermediate_size

    units: Dict[Tuple[Any, ...], WarmupUnit] = {}

    def add(unit: WarmupUnit) -> None:
        units.setdefault(unit.key, unit)

    for num_tokens in token_counts:
        routes = route_shape(num_tokens, shape.num_topk, shape.num_expert_local)
        tile_m = routes.tile_m
        a_subtile = _grouped_gemm._a_subtile_for(routes.rows_per_expert, tile_m)
        fuse_reduce = _pipeline._fuse_reduce(
            _pipeline._local_routed_bytes(
                num_tokens * shape.num_topk,
                shape.hidden_size,
                shape.num_expert_local,
                shape.num_expert_total,
            ),
            device,
            num_tokens,
        )
        small = num_tokens <= SMALL_ROUTE_TOKEN_LIMIT
        if small:
            add(_routing_unit(cutlass, scratch, "prepare_small", 2, tile_m, pointers=6, ints=5))
        else:
            add(_routing_unit(cutlass, scratch, "count", 0, tile_m, pointers=3, ints=3))
            add(_routing_unit(cutlass, scratch, "prefix", 1, tile_m, pointers=5, ints=1))
        add(_gather_fp8_to_mx_unit(cutlass, scratch, shape, assign_position=not small))
        add(
            _gemm_unit(
                cutlass,
                scratch,
                num_sms=num_sms,
                n=gate_up_size,
                k=shape.hidden_size,
                tile_n=_grouped_gemm._fused_tile_n_for(routes.rows_per_expert, True, False),
                fused=True,
                scatter=False,
                a_subtile=a_subtile,
                tile_m=tile_m,
                b_fp8=True,
                block128=False,
                tile_k=_grouped_gemm._tile_k_for(
                    routes.rows_per_expert, shape.hidden_size, True, False
                ),
            )
        )
        gemm2_thin = _grouped_gemm._use_thin(False, tile_m, b_fp8=True)
        gemm2_num_sms = num_sms * _grouped_gemm.THIN_CTAS_PER_SM if gemm2_thin else num_sms
        add(
            _gemm_unit(
                cutlass,
                scratch,
                num_sms=gemm2_num_sms,
                n=shape.hidden_size,
                k=shape.intermediate_size,
                tile_n=_grouped_gemm._tile_n_for(routes.m_tiles, shape.hidden_size, gemm2_num_sms),
                fused=False,
                scatter=fuse_reduce,
                a_subtile=a_subtile,
                tile_m=tile_m,
                b_fp8=True,
                block128=False,
                tile_k=_grouped_gemm._tile_k_for(
                    routes.rows_per_expert, shape.intermediate_size, True, False
                ),
                thin=gemm2_thin,
            )
        )
        if not fuse_reduce:
            add(_reduce_unit(cutlass, scratch, has_shared_output))

    return list(units.values())
