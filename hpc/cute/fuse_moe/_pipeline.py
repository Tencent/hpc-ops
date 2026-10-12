# Copyright (C) 2026 Tencent.

"""End-to-end CuTe DSL fused MoE for SM120 W4A8 MXFP4 x MXFP8.

End-to-end pipeline: route, gather, grouped GEMM, SwiGLU + requantize, grouped
GEMM, top-k reduce.

Two structural differences. The routed-row buffers are padded so each expert
starts on a tile boundary; see ``_grouped_gemm`` for why. Because the padded row
count is only known on the device, the buffers are sized for the worst case
``total_routes + (tile_m - 1) * num_experts`` rows, a tighter bound than the one
``entry.cc`` uses for its scale buffers. And the SwiGLU is not a pass of its own:
the up-projection GEMM applies it and requantizes in its epilogue, so the
bfloat16 intermediate never reaches DRAM.
The top-k reduce can go the same way, but only pays for a range of batch sizes;
see ``_fuse_reduce``.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from .._compile import require_sm120
from .._layouts import SCALE_MN_ALIGNMENT, scale_storage_size
from .._pack_scale import pack_mxfp_weight_scale_sm120
from .._quant import (
    UNIT_UE8M0,
    broadcast_ue8m0_128_to_mx,
    fp32_to_ue8m0,
    is_power_of_two,
    requant_fp8_weight_ue8m0,
)
from ._gather import (
    MAX_TOKEN_TOPK,
    gather_bf16_to_mx_routes,
    gather_fp8_block128_routes,
    gather_fp8_to_mx_routes,
    gather_mxfp8_routes,
    gather_token_routes,
)
from ._grouped_gemm import grouped_block_scaled_gemm
from ._reduce import MAX_TOPK, reduce_topk
from ._routing import RouteBuffers, prepare_routes


# How much of L2 the down projection's routed rows have to occupy before it
# pays to finish the top-k sum in its epilogue instead of storing those rows for
# reduce_topk to sum. What the fusion saves is a write and a read of one BF16
# row per route, and while those rows fit in cache that traffic never reaches
# DRAM and is nearly free, so all that is left is the cost of scattering with
# atomics rather than storing. End to end on the 4096-wide model with 6 routes
# per token, against the fraction of L2 the rows occupy: +2.5% at 0.25, +3.2%
# at 0.38, +2.6% at 0.50, -3.8% at 0.75, -7.5% at 1.0, -6.2% at 2.0, -7.4% at
# 4.1. Measuring the two GEMM-and-reduce sequences on their own says something
# different above 1.25 -- and is wrong, because it leaves out what a buffer of
# that size does to the rest of the pipeline's cache.
SCATTER_L2_MIN = 0.62

_l2_sizes: Dict[Any, int] = {}


def _local_routed_bytes(
    total_routes: int, hidden_size: int, num_expert_local: int, num_expert_total: int
) -> int:
    """BF16 bytes of routed rows this rank will produce, in expectation.

    Under expert parallelism only the routes that land on the local experts
    become rows here, and how many that is only settles on the device.
    """
    return total_routes * hidden_size * 2 * num_expert_local // num_expert_total


# Up to this many tokens the down projection always sums the top-k in its
# epilogue. The occupancy rule above is about DRAM traffic, which a handful of
# rows does not generate; what they do pay is a whole reduce_topk launch, about
# 3 us of a 50 us layer at one token. Past a few tokens the launch is noise and
# the atomics are not, so the rule takes over again.
DECODE_FUSE_REDUCE_TOKENS = 8


def _fuse_reduce(routed_bytes: int, device, num_tokens: int) -> bool:
    """Whether the down projection should sum the top-k in its own epilogue."""
    if num_tokens <= DECODE_FUSE_REDUCE_TOKENS:
        return True
    key = (device.type, device.index)
    l2 = _l2_sizes.get(key)
    if l2 is None:
        l2 = torch.cuda.get_device_properties(device).L2_cache_size
        _l2_sizes[key] = l2
    return routed_bytes >= SCATTER_L2_MIN * l2


def _gather_a(num_topk: int) -> bool:
    """Whether the up projection reads A rows by token instead of from a copy.

    Gathering skips the per-route copy of the activations (3 us at 64 tokens,
    17 us at 1024, 222 us at 8192 on DeepSeek-V4 TP4) and leaves the GEMM a
    token-major A a top-k-th the size, which stays in L2: with 32-row tiles the
    GEMM itself is 2-4% faster. With 64-row tiles the MMA is the bound and the
    gathers fall behind it: timed alone the GEMM is 1% slower at 2048 tokens and
    12-17% from 3072. Timed in the layer, which is what counts, it never costs
    more than it saves: 2-5% faster from 1 to 2048 tokens and 0-3% beyond, with
    BF16 or MXFP8 input alike.
    """
    return num_topk <= MAX_TOKEN_TOPK


def _packed_weight_scale(scale: Tensor, num_experts: int, rows: int, cols: int) -> Tensor:
    """Pack a logical ``[E, rows, cols / 32]`` scale, or pass a packed one through.

    Same 2-D-means-packed convention as ``pack_weight_scale_if_needed`` in
    ``entry.cc``; element counts cannot be used to tell the two apart because
    they coincide whenever ``rows`` and ``cols`` are already 128-aligned.
    """
    expected = scale_storage_size(rows, cols)
    if scale.dim() == 2:
        if tuple(scale.shape) != (num_experts, expected):
            raise ValueError(
                f"packed weight scale must be [{num_experts}, {expected}]; got {tuple(scale.shape)}"
            )
        return scale.reshape(-1)
    if tuple(scale.shape) != (num_experts, rows, cols // 32):
        raise ValueError(
            f"logical weight scale must be [{num_experts}, {rows}, {cols // 32}]; "
            f"got {tuple(scale.shape)}"
        )
    return pack_mxfp_weight_scale_sm120(scale, rows, cols).reshape(-1)


def fuse_moe_blockwise_w4a8_mxfp4_mxfp8(
    x: Tensor,
    x_scale: Optional[Tensor],
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    shared_output: Optional[Tensor] = None,
    rank_ep: int = 0,
    num_expert_total: int = 0,
    output: Optional[Tensor] = None,
    swiglu_limit: float = -1.0,
) -> Tensor:
    """Fused MoE with blockwise MXFP8 activations and MXFP4 x MXFP8 experts.

    ``x`` is either MXFP8 E4M3 with its UE8M0 ``x_scale``, or BF16 with
    ``x_scale=None``, which the gather quantizes to MXFP8: per token when the
    up projection reads A by token (see ``_gather_a``), per route otherwise.
    """
    require_sm120(x.device)

    bf16_input = x.dtype == torch.bfloat16
    if bf16_input:
        if x_scale is not None:
            raise ValueError("x_scale must be None when x is bfloat16")
    else:
        if x.dtype != torch.float8_e4m3fn:
            raise ValueError(f"x dtype must be float8_e4m3fn or bfloat16; got {x.dtype}")
        if x_scale is None or x_scale.dtype != torch.uint8:
            raise ValueError(
                "x_scale must be a uint8 tensor when x is float8_e4m3fn; got "
                f"{None if x_scale is None else x_scale.dtype}"
            )
    if topk_ids.dtype != torch.int32:
        raise ValueError(f"topk_ids dtype must be int32; got {topk_ids.dtype}")
    # Checked here and not left to reduce_topk, which is where the bound comes
    # from: whether this call reaches that kernel depends on the batch size, and
    # a limit that only some batch sizes enforce is worse than no limit at all.
    if topk_ids.shape[1] > MAX_TOPK:
        raise ValueError(
            f"at most {MAX_TOPK} experts per token are supported; got {topk_ids.shape[1]}"
        )

    num_tokens, hidden_size = x.shape
    num_expert_local = gate_up_weight.shape[0]
    gate_up_size = gate_up_weight.shape[1]
    intermediate_size = gate_up_size // 2
    num_topk = topk_ids.shape[1]
    total_routes = num_tokens * num_topk

    if num_expert_total == 0:
        num_expert_total = num_expert_local
    if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
        raise ValueError("hidden_size and intermediate_size must be multiples of 128")
    if gate_up_weight.shape[2] * 2 != hidden_size:
        raise ValueError("gate_up_weight packed K must equal hidden_size / 2")
    if tuple(down_weight.shape) != (num_expert_local, hidden_size, intermediate_size // 2):
        raise ValueError(
            f"down_weight must be [{num_expert_local}, {hidden_size}, {intermediate_size // 2}]"
        )

    device = x.device
    if output is None:
        output = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)

    gate_up_scale = _packed_weight_scale(
        gate_up_weight_scale, num_expert_local, gate_up_size, hidden_size
    )
    down_scale = _packed_weight_scale(
        down_weight_scale, num_expert_local, hidden_size, intermediate_size
    )

    routes = RouteBuffers(num_tokens, num_topk, num_expert_local, device)
    max_rows = routes.max_padded_rows

    # Rows past the routed ones are never initialized: they produce garbage in
    # both GEMMs, but every consumer is row-local and the epilogues only store
    # rows an expert really owns, so the garbage stays contained.
    gather_a = _gather_a(num_topk) and (
        bf16_input or (x.is_contiguous() and x.data_ptr() % 16 == 0)
    )
    if not gather_a:
        gate_up_input = torch.empty(
            (max_rows, hidden_size), dtype=torch.float8_e4m3fn, device=device
        )
    elif bf16_input:
        gate_up_input = torch.empty(
            (num_tokens, hidden_size), dtype=torch.float8_e4m3fn, device=device
        )
    else:
        gate_up_input = x
    gate_up_input_scale = torch.empty(
        scale_storage_size(max_rows, hidden_size), dtype=torch.uint8, device=device
    )
    down_input = torch.empty(
        (max_rows, intermediate_size), dtype=torch.float8_e4m3fn, device=device
    )
    down_input_scale = torch.empty(
        scale_storage_size(max_rows, intermediate_size), dtype=torch.uint8, device=device
    )

    fuse_reduce = _fuse_reduce(
        _local_routed_bytes(total_routes, hidden_size, num_expert_local, num_expert_total),
        device,
        num_tokens,
    )
    down_output = None
    if fuse_reduce:
        # The down projection accumulates into the output rather than parking a
        # weighted row per route for a second kernel to sum, so the output
        # starts as whatever it has to be added to.
        if shared_output is None:
            output.zero_()
        else:
            output.copy_(shared_output)
    else:
        down_output = torch.empty((max_rows, hidden_size), dtype=torch.bfloat16, device=device)

    assigned = prepare_routes(topk_ids, routes, num_tokens, num_topk, num_expert_local, rank_ep)
    if gather_a:
        gather_token_routes(
            x,
            x_scale,
            topk_ids,
            routes,
            gate_up_input if bf16_input else None,
            gate_up_input_scale,
            num_tokens,
            num_topk,
            hidden_size,
            num_expert_local,
            rank_ep,
            assign_position=assigned,
        )
    elif bf16_input:
        gather_bf16_to_mx_routes(
            x,
            topk_ids,
            routes,
            gate_up_input,
            gate_up_input_scale,
            total_routes,
            num_topk,
            hidden_size,
            num_expert_local,
            rank_ep,
            assign_position=assigned,
        )
    else:
        gather_mxfp8_routes(
            x,
            x_scale,
            topk_ids,
            routes,
            gate_up_input,
            gate_up_input_scale,
            total_routes,
            num_topk,
            hidden_size,
            num_expert_local,
            rank_ep,
            assign_position=assigned,
        )
    # The up projection ends in SwiGLU and requantization, both fused into its
    # epilogue: the intermediate tensor is four times the size of what comes out
    # of it, and writing it out only to read it straight back cost a tenth of
    # the layer at decode shapes.
    grouped_block_scaled_gemm(
        gate_up_input,
        gate_up_weight,
        gate_up_input_scale,
        gate_up_scale,
        down_input,
        routes.cu_m_tiles,
        routes.num_tokens_per_expert,
        num_expert_local,
        expected_m_tiles=routes.expected_m_tiles,
        expected_rows_per_expert=routes.expected_rows_per_expert,
        tile_m=routes.tile_m,
        c_scale=down_input_scale,
        swiglu_limit=swiglu_limit,
        a_row_to_route=routes.row_to_route if gather_a else None,
        num_topk=num_topk if gather_a else 0,
    )
    grouped_block_scaled_gemm(
        down_input,
        down_weight,
        down_input_scale,
        down_scale,
        output if fuse_reduce else down_output,
        routes.cu_m_tiles,
        routes.num_tokens_per_expert,
        num_expert_local,
        expected_m_tiles=routes.expected_m_tiles,
        expected_rows_per_expert=routes.expected_rows_per_expert,
        tile_m=routes.tile_m,
        row_to_route=routes.row_to_route if fuse_reduce else None,
        topk_scale=topk_scale if fuse_reduce else None,
        num_topk=num_topk if fuse_reduce else 0,
    )
    if not fuse_reduce:
        reduce_topk(
            output,
            down_output,
            routes.topk_pos,
            topk_scale,
            shared_output,
            num_tokens,
            hidden_size,
            num_topk,
        )
    return output


_unit_scale_cache: Dict[Tuple[Any, ...], Tensor] = {}


def _unit_weight_scale(device, num_experts: int, rows: int, cols: int) -> Tensor:
    """A 127-filled (UE8M0 1.0) scale-factor buffer for unit-SF MMA paths.

    Used by the exact float32 128-block FP8 path, which runs the MMA with unit
    scale factors and applies the real scales in software. Cached like
    ``get_unit_weight_scale`` in ``entry.cc``: the contents never
    change, so per-call memsets would be pure overhead.

    Activation-side callers pass the padded row count, which follows the batch
    size, so keying on it exactly would add a buffer per batch size and free
    none of them. Rows round up to a power of two instead: the contents are
    constant and the offsets a kernel computes do not depend on the buffer's
    length, so a handful of buffers serve every size and none is ever
    reallocated out from under a captured graph.
    """
    rows = max(SCALE_MN_ALIGNMENT, 1 << max(rows - 1, 1).bit_length())
    key = (device.type, device.index, num_experts, rows, cols)
    cached = _unit_scale_cache.get(key)
    if cached is None:
        cached = torch.full(
            (num_experts * scale_storage_size(rows, cols),),
            UNIT_UE8M0,
            dtype=torch.uint8,
            device=device,
        )
        _unit_scale_cache[key] = cached
    return cached


def _as_row_scales(scale: Tensor, num_tokens: int, k: int) -> Tensor:
    """``[num_tokens, K/128]`` activation scales, squeezing a trailing singleton."""
    blocks = k // 128
    if scale.dim() == 3 and scale.shape[1] == 1:
        scale = scale.reshape(scale.shape[0], scale.shape[2])
    if tuple(scale.shape) != (num_tokens, blocks):
        raise ValueError(
            f"activation scale must be [{num_tokens}, {blocks}]; got {tuple(scale.shape)}"
        )
    return scale


def prepare_fp8_mx_mma_weights(weight: Tensor, scale: Tensor) -> Tuple[Tensor, Tensor]:
    """Requantize 128-block FP8 weights and pack their scales for MX MMA.

    Idempotent for already-packed ``uint8`` scales. Requant is in-place on both
    ``weight`` and ``scale``, so the caller's float32 scales come back as the
    powers of two the stored E4M3 values are now quantized against. The packed
    SFB is cached on the weight tensor, which makes a second call on the same
    object free and cannot outlive the buffer it describes.

    Call this once per layer at load time. Doing it lazily inside a forward
    pass would either land on the critical path or, worse, be captured into a
    decode graph.
    """
    if scale.dtype == torch.uint8 and scale.dim() == 2:
        return weight, scale
    packed = getattr(weight, "_hpc_mx_sfb", None)
    if packed is not None:
        return weight, packed
    # Requant writes through both tensors. A non-contiguous scale would be
    # copied somewhere along the way and the caller would be left holding
    # float32 scales that no longer describe the weights next to them.
    if not scale.is_contiguous():
        raise ValueError("weight scale must be contiguous; requant writes through it")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous; requant writes through it")
    rows, cols = weight.shape[-2], weight.shape[-1]
    requant_fp8_weight_ue8m0(weight, scale)
    packed = pack_mxfp_weight_scale_sm120(
        broadcast_ue8m0_128_to_mx(fp32_to_ue8m0(scale), rows, cols), rows, cols
    )
    weight._hpc_mx_sfb = packed
    return weight, packed


# The MX path reads the activation's 128-block scales as UE8M0 and leaves the
# E4M3 data alone, so it is only correct for activations that were quantized
# against powers of two to begin with. Ceiling a scale the data was not divided
# by rescales that block by up to 2x, and because the factor differs per block
# it does not cancel across the reduction: on Qwen TEP4 shapes the output comes
# out at 83% relative error, still finite and still the right shape. What
# satisfies the requirement is several layers away -- vLLM's
# per_token_group_quant_fp8 takes use_ue8m0 from VLLM_USE_DEEP_GEMM_E8M0, which
# also needs DeepGEMM to be importable and the arch to be on its list -- and
# every way of losing it is silent. Hence a check here rather than a comment.
_ue8m0_checked = set()


def _require_ue8m0_activation_scale(x_scale: Tensor) -> None:
    """Check, once per device, that ``x_scale`` holds exact powers of two."""
    device = x_scale.device
    if device in _ue8m0_checked:
        return
    # Reading the scales syncs, and a capture cannot. vLLM profiles eagerly
    # through the same quantizer before it captures, so by then this has run.
    if torch.cuda.is_current_stream_capturing():
        return
    _ue8m0_checked.add(device)
    if is_power_of_two(x_scale):
        return
    raise ValueError(
        "the mx_mma FP8 MoE path needs activation scales that are exact powers "
        "of two, and this batch's are not. Quantize the activations with UE8M0 "
        "scales (under vLLM, set VLLM_USE_DEEP_GEMM_E8M0=1 and check that "
        "is_deep_gemm_e8m0_used() reports True), or pass mx_mma=False for the "
        "exact float32 128-block path."
    )


def _require_packed_sfb(scale: Tensor, name: str) -> Tensor:
    """Flatten an already-packed MX weight scale, or say what the caller must do.

    Preparing the weights from here would requantize them in place on the first
    call: a mutation the operator does not declare, and -- if that first call
    is a graph capture -- a few hundred megabytes of transform baked into every
    replay. The wrapper in ``_ops`` does it before the operator is entered.
    """
    if scale.dtype != torch.uint8 or scale.dim() != 2:
        raise ValueError(
            f"{name} weight scale must be packed UE8M0 for the mx_mma path; run "
            "hpc.prepare_fp8_mx_mma_weights over the weights at load time"
        )
    return scale.reshape(-1)


def fuse_moe_blockwise_fp8(
    x: Tensor,
    x_scale: Tensor,
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    shared_output: Optional[Tensor] = None,
    rank_ep: int = 0,
    num_expert_total: int = 0,
    output: Optional[Tensor] = None,
    swiglu_limit: float = -1.0,
    mx_mma: bool = True,
) -> Tensor:
    """Qwen-style 128x128 float32-block FP8 fused MoE on SM120.

    ``mx_mma=True`` (default, the serve path) is DeepGEMM's SM120 recipe:
    ceil the 128-block scales to UE8M0, broadcast them onto the MX 32-vector
    layout, and run the V4 fused 64-wide GEMM so scales land in the MMA
    instead of a software float32 multiply. ``mx_mma=False`` keeps the exact
    float32 128-block path for tests.
    """
    require_sm120(x.device)

    if x.dtype != torch.float8_e4m3fn:
        raise ValueError(f"x dtype must be float8_e4m3fn; got {x.dtype}")
    if x_scale.dtype != torch.float32:
        raise ValueError(f"x_scale dtype must be float32; got {x_scale.dtype}")
    if topk_ids.dtype != torch.int32:
        raise ValueError(f"topk_ids dtype must be int32; got {topk_ids.dtype}")
    if topk_ids.shape[1] > MAX_TOPK:
        raise ValueError(
            f"at most {MAX_TOPK} experts per token are supported; got {topk_ids.shape[1]}"
        )

    num_tokens, hidden_size = x.shape
    num_expert_local = gate_up_weight.shape[0]
    gate_up_size = gate_up_weight.shape[1]
    intermediate_size = gate_up_size // 2
    num_topk = topk_ids.shape[1]
    total_routes = num_tokens * num_topk
    x_scale = _as_row_scales(x_scale, num_tokens, hidden_size)

    if num_expert_total == 0:
        num_expert_total = num_expert_local
    if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
        raise ValueError("hidden_size and intermediate_size must be multiples of 128")
    if tuple(gate_up_weight.shape) != (num_expert_local, gate_up_size, hidden_size):
        raise ValueError(
            f"gate_up_weight must be [{num_expert_local}, {gate_up_size}, {hidden_size}]"
        )
    if tuple(down_weight.shape) != (num_expert_local, hidden_size, intermediate_size):
        raise ValueError(
            f"down_weight must be [{num_expert_local}, {hidden_size}, {intermediate_size}]"
        )
    if gate_up_weight.dtype != torch.float8_e4m3fn or down_weight.dtype != torch.float8_e4m3fn:
        raise ValueError("weights must be float8_e4m3fn")
    if mx_mma:
        _require_ue8m0_activation_scale(x_scale)

    device = x.device
    if output is None:
        output = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)

    routes = RouteBuffers(num_tokens, num_topk, num_expert_local, device)
    max_rows = routes.max_padded_rows
    fuse_reduce = _fuse_reduce(
        _local_routed_bytes(total_routes, hidden_size, num_expert_local, num_expert_total),
        device,
        num_tokens,
    )
    down_output = None
    if fuse_reduce:
        if shared_output is None:
            output.zero_()
        else:
            output.copy_(shared_output)
    else:
        down_output = torch.empty((max_rows, hidden_size), dtype=torch.bfloat16, device=device)

    assigned = prepare_routes(topk_ids, routes, num_tokens, num_topk, num_expert_local, rank_ep)
    if mx_mma:
        _fuse_moe_fp8_mx(
            x,
            x_scale,
            gate_up_weight,
            gate_up_weight_scale,
            down_weight,
            down_weight_scale,
            topk_ids,
            topk_scale,
            routes,
            assigned,
            output,
            down_output,
            shared_output,
            fuse_reduce,
            total_routes,
            num_tokens,
            num_topk,
            hidden_size,
            gate_up_size,
            intermediate_size,
            num_expert_local,
            rank_ep,
            swiglu_limit,
        )
    else:
        _fuse_moe_fp8_exact(
            x,
            x_scale,
            gate_up_weight,
            gate_up_weight_scale,
            down_weight,
            down_weight_scale,
            topk_ids,
            topk_scale,
            routes,
            assigned,
            output,
            down_output,
            shared_output,
            fuse_reduce,
            total_routes,
            num_tokens,
            num_topk,
            hidden_size,
            gate_up_size,
            intermediate_size,
            num_expert_local,
            rank_ep,
            swiglu_limit,
        )
    return output


def _fuse_moe_fp8_mx(
    x,
    x_scale,
    gate_up_weight,
    gate_up_weight_scale,
    down_weight,
    down_weight_scale,
    topk_ids,
    topk_scale,
    routes,
    assigned,
    output,
    down_output,
    shared_output,
    fuse_reduce,
    total_routes,
    num_tokens,
    num_topk,
    hidden_size,
    gate_up_size,
    intermediate_size,
    num_expert_local,
    rank_ep,
    swiglu_limit,
) -> None:
    """DeepGEMM-style UE8M0 MMA: V4 fused 64-wide GEMM, hardware block scales."""
    device = x.device
    max_rows = routes.max_padded_rows
    gate_up_sfb = _require_packed_sfb(gate_up_weight_scale, "gate_up")
    down_sfb = _require_packed_sfb(down_weight_scale, "down")

    gate_up_input = torch.empty((max_rows, hidden_size), dtype=torch.float8_e4m3fn, device=device)
    gate_up_input_scale = torch.empty(
        scale_storage_size(max_rows, hidden_size), dtype=torch.uint8, device=device
    )
    down_input = torch.empty(
        (max_rows, intermediate_size), dtype=torch.float8_e4m3fn, device=device
    )
    down_input_scale = torch.empty(
        scale_storage_size(max_rows, intermediate_size), dtype=torch.uint8, device=device
    )

    gather_fp8_to_mx_routes(
        x,
        x_scale,
        topk_ids,
        routes,
        gate_up_input,
        gate_up_input_scale,
        total_routes,
        num_topk,
        hidden_size,
        num_expert_local,
        rank_ep,
        assign_position=assigned,
    )
    grouped_block_scaled_gemm(
        gate_up_input,
        gate_up_weight,
        gate_up_input_scale,
        gate_up_sfb,
        down_input,
        routes.cu_m_tiles,
        routes.num_tokens_per_expert,
        num_expert_local,
        expected_m_tiles=routes.expected_m_tiles,
        expected_rows_per_expert=routes.expected_rows_per_expert,
        tile_m=routes.tile_m,
        c_scale=down_input_scale,
        swiglu_limit=swiglu_limit,
    )
    grouped_block_scaled_gemm(
        down_input,
        down_weight,
        down_input_scale,
        down_sfb,
        output if fuse_reduce else down_output,
        routes.cu_m_tiles,
        routes.num_tokens_per_expert,
        num_expert_local,
        expected_m_tiles=routes.expected_m_tiles,
        expected_rows_per_expert=routes.expected_rows_per_expert,
        tile_m=routes.tile_m,
        row_to_route=routes.row_to_route if fuse_reduce else None,
        topk_scale=topk_scale if fuse_reduce else None,
        num_topk=num_topk if fuse_reduce else 0,
    )
    if not fuse_reduce:
        reduce_topk(
            output,
            down_output,
            routes.topk_pos,
            topk_scale,
            shared_output,
            num_tokens,
            hidden_size,
            num_topk,
        )


def _fuse_moe_fp8_exact(
    x,
    x_scale,
    gate_up_weight,
    gate_up_weight_scale,
    down_weight,
    down_weight_scale,
    topk_ids,
    topk_scale,
    routes,
    assigned,
    output,
    down_output,
    shared_output,
    fuse_reduce,
    total_routes,
    num_tokens,
    num_topk,
    hidden_size,
    gate_up_size,
    intermediate_size,
    num_expert_local,
    rank_ep,
    swiglu_limit,
) -> None:
    """Exact float32 128-block path: unit-SF MMA plus a software scale multiply."""
    device = x.device
    max_rows = routes.max_padded_rows
    # Read as-is by the GEMM, and unlike the MX path nothing here writes back
    # through them, so a copy of a non-contiguous scale is harmless.
    gate_up_weight_scale = gate_up_weight_scale.contiguous()
    down_weight_scale = down_weight_scale.contiguous()

    gate_up_input = torch.empty((max_rows, hidden_size), dtype=torch.float8_e4m3fn, device=device)
    gate_up_input_scale = torch.empty(
        (max_rows, hidden_size // 128), dtype=torch.float32, device=device
    )
    down_input = torch.empty(
        (max_rows, intermediate_size), dtype=torch.float8_e4m3fn, device=device
    )
    down_input_scale = torch.empty(
        (max_rows, intermediate_size // 128), dtype=torch.float32, device=device
    )

    unit_sfa_gate = _unit_weight_scale(device, 1, max_rows, hidden_size)
    unit_sfb_gate = _unit_weight_scale(device, num_expert_local, gate_up_size, hidden_size)
    unit_sfa_down = _unit_weight_scale(device, 1, max_rows, intermediate_size)
    unit_sfb_down = _unit_weight_scale(device, num_expert_local, hidden_size, intermediate_size)

    gather_fp8_block128_routes(
        x,
        x_scale,
        topk_ids,
        routes,
        gate_up_input,
        gate_up_input_scale,
        total_routes,
        num_topk,
        hidden_size,
        num_expert_local,
        rank_ep,
        assign_position=assigned,
    )
    grouped_block_scaled_gemm(
        gate_up_input,
        gate_up_weight,
        unit_sfa_gate,
        unit_sfb_gate,
        down_input,
        routes.cu_m_tiles,
        routes.num_tokens_per_expert,
        num_expert_local,
        expected_m_tiles=routes.expected_m_tiles,
        expected_rows_per_expert=routes.expected_rows_per_expert,
        tile_m=routes.tile_m,
        c_scale=down_input_scale,
        swiglu_limit=swiglu_limit,
        a_scale128=gate_up_input_scale,
        b_scale128=gate_up_weight_scale,
    )
    grouped_block_scaled_gemm(
        down_input,
        down_weight,
        unit_sfa_down,
        unit_sfb_down,
        output if fuse_reduce else down_output,
        routes.cu_m_tiles,
        routes.num_tokens_per_expert,
        num_expert_local,
        expected_m_tiles=routes.expected_m_tiles,
        expected_rows_per_expert=routes.expected_rows_per_expert,
        tile_m=routes.tile_m,
        row_to_route=routes.row_to_route if fuse_reduce else None,
        topk_scale=topk_scale if fuse_reduce else None,
        num_topk=num_topk if fuse_reduce else 0,
        a_scale128=down_input_scale,
        b_scale128=down_weight_scale,
    )
    if not fuse_reduce:
        reduce_topk(
            output,
            down_output,
            routes.topk_pos,
            topk_scale,
            shared_output,
            num_tokens,
            hidden_size,
            num_topk,
        )
