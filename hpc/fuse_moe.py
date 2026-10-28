# Copyright (C) 2026 Tencent.

import torch
from torch import Tensor
from typing import Tuple, Optional


def count_and_gather(
    x: Tensor,
    topk_ids: Tensor,
    num_expert: int,
    rank_ep: int,
    intermediate_size: int,
    num_seq_per_group_avg: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Sorts and aggregates token based on expert assignments for MoE layers.

    This function organizes input token according to their assigned expert IDs
    after top-k expert selection, grouping token by expert for efficient
    parallel computation in mixture-of-experts architectures.

    Args:
        x: Input token features tensor
            Shape: [num_seq, hidden_size]
            Dtype: fp8
        topk_ids: Expert assignment indices for each token
            Shape: [num_seq, num_topk]
            Dtype: int32
        num_expert: Number of experts available on the current device
            Shape: scalar
            DType: int
        rank_ep: Current device rank in expert parallelism
            Shape: scalar
            DType: int

    Returns:
        Tuple containing eight tensors:
        - output: Sorted token features grouped by expert
            Shape: [num_seq * num_topk, hidden_size]
            Dtype: fp8
        - output_for_group_gemm: allocate output for group gemm
            Shape: [num_seq * num_topk, intermediate_size]
            Dtype: bfloat16
        - topk_pos: Position indices mapping each expert output back to original sequence
            Shape: [num_seq * num_topk]
            Dtype: int32
        - seqlens: Number of token assigned to each expert
            Shape: [num_expert]
            Dtype: int32
        - cu_seqlens: Cumulative token counts for expert indexing
            Shape: [num_expert + 1]
            Dtype: int32
        - tiles: Number of tiles assigned to each expert
            Shape: [num_expert]
            Dtype: int32
        - cu_tiles: Cumulative tiles counts for expert indexing
            Shape: [num_expert + 1]
            Dtype: int32
        - tmas: Describe the tma info of output
            Shape: [num_expert * 2 * 128]
            Dtype: int8

    Raises:
        RuntimeError: If input tensors have incompatible shapes or types,
            if expert assignments exceed available expert count,
            or if buffer sizes are insufficient for output tensors.

    Note:
        - This function is designed for expert-parallel distributed training
        - All input and output tensors must reside on the same device
        - The function modifies the output buffers in-place when provided
        - Expert assignments in topk_ids should be in range [0, num_expert-1]
    """
    return torch.ops.hpc.count_and_gather(
        x, topk_ids, num_expert, rank_ep, intermediate_size, num_seq_per_group_avg
    )


def reduce(
    x: Tensor,
    topk_pos: Tensor,
    topk_scale: Tensor,
    shared_output: Tensor = None,
) -> Tensor:
    """reduce token based on expert assignments for MoE layers.

    This kernel implements the reduction (scatter-add) operation in the final stage
    of Mixture of Experts (MoE) computation. It aggregates the weighted contributions
    from selected experts back to the original sequence positions.

    Args:
        x: Input token features tensor from expert processing
            Shape: [total_num_seq, hidden_size]
            Dtype: bfloat16
        topk_pos: Position indices mapping each expert output back to original sequence
            Shape: [num_seq, num_topk]
            Dtype: int32
        topk_scale: Scaling factors for expert outputs (typically gating scores)
            Shape: [num_seq, num_topk]
            Dtype: float32
        shared_output: output for shared experts, default is None
            Shape: [num_seq, hidden_size]
            Dtype: bfloat16

    Returns:
        Tensor: Reduced output tensor containing aggregated expert contributions
            Shape: [num_seq, hidden_size]
            Dtype: bfloat16

    Raises:
        RuntimeError: If tensor shapes are incompatible or CUDA kernel execution fails.
        ValueError: If input tensors are on different devices or have unsupported dtypes.

    Note:
        - This is a performance-critical kernel optimized for MoE workloads
        - The operation is equivalent to a weighted scatter-add operation
        - Input tensors must be contiguous and on the same device
        - For best performance, hidden_size should be a multiple of 32
        - topk_pos values must be in range [0, total_num_seq-1]
    """
    return torch.ops.hpc.reduce(x, topk_pos, topk_scale, shared_output)


_CP_ASYNC_N_TP_MAX = 512


def fuse_moe(
    x: Tensor,
    gate_up_weight: Tensor,
    down_weight: Tensor,
    gate_up_scale: Tensor,
    down_scale: Tensor,
    act_and_mul_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    use_bf16_mul: bool = True,
    shared_output: Tensor = None,
    output: Tensor = None,
) -> Tensor:
    """Run per-tensor FP8 FusedMoE."""
    return torch.ops.hpc.fuse_moe(
        x,
        gate_up_weight,
        down_weight,
        gate_up_scale,
        down_scale,
        act_and_mul_scale,
        topk_ids,
        topk_scale,
        shared_output,
        rank_ep,
        num_expert_total,
        use_bf16_mul,
        output,
    )


def fuse_moe_pertensor_fp8(
    x: Tensor,
    gate_up_weight: Tensor,
    down_weight: Tensor,
    gate_up_scale: Tensor,
    down_scale: Tensor,
    act_and_mul_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    use_bf16_mul: bool = True,
    shared_output: Tensor = None,
) -> Tensor:
    """Run per-tensor FP8 FusedMoE."""

    return torch.ops.hpc.fuse_moe_pertensor_fp8(
        x,
        gate_up_weight,
        down_weight,
        gate_up_scale,
        down_scale,
        act_and_mul_scale,
        topk_ids,
        topk_scale,
        shared_output,
        rank_ep,
        num_expert_total,
        use_bf16_mul,
        None,
    )


def fuse_moe_blockwise_fp8(
    x: Tensor,
    x_scale: Tensor,
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    shared_output: Tensor = None,
) -> Tensor:
    """Run blockwise FP8 FusedMoE."""
    return torch.ops.hpc.fuse_moe_blockwise_fp8(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        shared_output,
        rank_ep,
        num_expert_total,
        None,
    )


def fuse_moe_blockwise(
    x: Tensor,
    x_scale: Tensor,
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    shared_output: Tensor = None,
    output: Tensor = None,
) -> Tensor:
    """Run blockwise FP8 FusedMoE."""
    return torch.ops.hpc.fuse_moe_blockwise(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        shared_output,
        rank_ep,
        num_expert_total,
        output,
    )


_FP8_MAX = 448.0
# Smallest scale the kernels emit; keeps all-zero blocks well defined.
_MIN_SCALE = 2.0**-127
# Largest finite UE8M0 value is 2^127 (biased exponent 254).
_MAX_UE8M0 = 2.0**127


def _fp32_to_ue8m0(x: Tensor) -> Tensor:
    """Round positive float32 scales up to UE8M0, as ``cvt.rp.satfinite.ue8m0x2.f32`` does."""
    x = x.contiguous().float().clamp(min=_MIN_SCALE, max=_MAX_UE8M0)
    bits = x.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    tiny = (exponent == 0) & (mantissa <= 0x00400000)
    round_up = (mantissa > 0) & (exponent != 0xFE) & ~tiny
    return (exponent + round_up.to(torch.int32)).to(torch.uint8)


def _ue8m0_to_fp32(storage: Tensor) -> Tensor:
    """Decode UE8M0 bytes to float32."""
    bits = storage.to(torch.int32) << 23
    out = bits.view(torch.float32)
    return torch.where(storage == 0, torch.full_like(out, _MIN_SCALE), out)


def _requant_fp8_weight_ue8m0(weight: Tensor, scale: Tensor) -> None:
    """Requantize 128-block FP8 weights onto power-of-two scales in place.

    Same transform DeepGEMM applies on SM120: dequantize with the checkpoint
    float32 block scale, then quantize with ``2^ceil(log2(amax / 448))`` so the
    MX MMA's hardware scale is exact for the stored E4M3 values. Works one
    expert at a time, so no full-weight copy is held.
    """
    experts, rows, cols = weight.shape
    block = 128
    if rows % block != 0 or cols % block != 0:
        raise ValueError(f"weight [{rows}, {cols}] is not 128-block aligned")
    expected = (experts, rows // block, cols // block)
    if tuple(scale.shape) != expected:
        raise ValueError(f"scale must be {expected}; got {tuple(scale.shape)}")
    for expert in range(experts):
        dequant = weight[expert].float() * scale[expert].repeat_interleave(
            block, dim=0
        ).repeat_interleave(block, dim=1)
        tiled = dequant.view(rows // block, block, cols // block, block)
        amax = tiled.abs().amax(dim=(1, 3))
        sf = _ue8m0_to_fp32(_fp32_to_ue8m0(amax / _FP8_MAX))
        quantized = (tiled / sf[:, None, :, None]).clamp(-_FP8_MAX, _FP8_MAX)
        weight[expert].copy_(quantized.to(torch.float8_e4m3fn).reshape(rows, cols))
        scale[expert].copy_(sf)


def _pack_ue8m0_scale(scale: Tensor, rows: int, cols: int) -> Tensor:
    """Pack ``[experts, rows, cols // 32]`` UE8M0 bytes into the GEMMs' int32 layout."""
    num_experts = scale.shape[0]
    expected = (num_experts, rows, cols // 32)
    if scale.dtype != torch.uint8 or tuple(scale.shape) != expected:
        raise ValueError(f"scale must be uint8{expected}; got {scale.dtype}{tuple(scale.shape)}")
    scale_rows = (rows + 3) // 4 * 4
    scale_cols = (cols + 127) // 128
    padded = torch.zeros(
        (num_experts, scale_rows, scale_cols * 4),
        dtype=torch.uint8,
        device=scale.device,
    )
    padded[:, :rows, : cols // 32] = scale
    bytes4 = padded.reshape(num_experts, scale_rows, scale_cols, 4).to(torch.int32)
    words = bytes4[..., 0] | (bytes4[..., 1] << 8) | (bytes4[..., 2] << 16) | (bytes4[..., 3] << 24)
    return words.permute(0, 2, 1).contiguous().reshape(-1)


def prepare_mxfp4_moe_weights_sm120(scale: Tensor, rows: int, cols: int) -> Tensor:
    """Prepare MXFP4 MoE weight scales for ``fuse_moe_blockwise_mxfp4_sm120``.

    Packs ``[experts, rows, cols // 32]`` UE8M0 scales into the int32 layout
    the GEMMs read. Run it once when the weights are loaded.
    """
    return _pack_ue8m0_scale(scale, rows, cols)


def prepare_fp8_moe_weights_sm120(weight: Tensor, scale: Tensor) -> Tensor:
    """Prepare 128-block FP8 MoE weights for ``fuse_moe_blockwise_fp8_sm120``.

    Float32 ``[E, N/128, K/128]`` block scales are rounded up to powers of two and
    ``weight`` is requantized against them in place, as DeepGEMM does on SM120, so
    the MX MMA reads exact UE8M0 scales; ``scale`` is overwritten with those powers
    of two. Returns the packed int32 scale the GEMMs read.

    The weight is requantized in its own storage, so a misaligned weight stays
    misaligned; the fused MoE rejects a weight that does not start at a 16-byte
    aligned address.

    Run it once when the weights are loaded: requantizing cannot happen inside
    a CUDA graph capture.
    """
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 3:
        raise ValueError(
            f"weight must be float8_e4m3fn [E, N, K]; got {weight.dtype}{tuple(weight.shape)}"
        )
    if scale.dtype != torch.float32:
        raise ValueError(f"scale must be float32 [E, N/128, K/128]; got {scale.dtype}")
    if not weight.is_contiguous() or not scale.is_contiguous():
        raise ValueError("weight and scale must be contiguous; requant writes through both")
    _, rows, cols = weight.shape
    _requant_fp8_weight_ue8m0(weight, scale)
    # The MX MMA takes one scale per 32 elements, so each 128x128 block's byte
    # repeats 128 times along N and 4 times along K.
    ue8m0 = _fp32_to_ue8m0(scale).repeat_interleave(128, dim=1).repeat_interleave(4, dim=2)
    return _pack_ue8m0_scale(ue8m0, rows, cols)


def _fuse_moe_mx(
    x,
    x_scale,
    gate_up_weight,
    gate_up_weight_scale,
    down_weight,
    down_weight_scale,
    topk_ids,
    topk_scale,
    rank_ep,
    num_expert_total,
    shared_output,
    output,
    swiglu_limit,
):
    # The functional op allocates the result; the out op writes the caller's
    # output, which it declares mutable, and returns nothing.
    args = (
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        shared_output,
        rank_ep,
        num_expert_total,
        -1.0 if swiglu_limit is None else swiglu_limit,
    )
    if output is None:
        return torch.ops.hpc.fuse_moe_mx(*args)
    torch.ops.hpc.fuse_moe_mx_out(output, *args)
    return output


def fuse_moe_blockwise_mxfp4_sm120(
    x: Tensor,
    x_scale: Optional[Tensor],
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int = 0,
    num_expert_total: int = 0,
    shared_output: Optional[Tensor] = None,
    output: Optional[Tensor] = None,
    swiglu_limit: Optional[float] = None,
) -> Tensor:
    """Run the SM120 MXFP4 x MXFP8 fused MoE.

    The weights are packed MXFP4 E2M1 ``uint8`` ``[E, 2I, H/2]`` and ``[E, H, I/2]``,
    with scales from :func:`prepare_mxfp4_moe_weights_sm120`. The GEMMs read the
    weights and those packed scales with TMA, so all four must be contiguous and
    start at a 16-byte aligned address; a misaligned one is rejected. Tensors that
    PyTorch allocates, and the scales the prepare functions return, meet this; a
    view at an offset into a larger buffer may not. ``x`` is BF16 with
    ``x_scale=None``, or float8_e4m3fn with uint8 UE8M0 ``x_scale``
    ``[num_tokens, H/32]``.

    ``topk_ids`` are int32 global expert IDs, 1 to 128 per token, with float32
    ``topk_scale`` of the same shape. This rank holds experts
    ``[rank_ep * E, (rank_ep + 1) * E)``; routes to other experts are skipped,
    whatever ``num_expert_total`` is. ``num_expert_total`` only tunes the plan:
    with the global expert count it sizes the tiles for this rank's share of the
    routes, and ``0`` sizes them as if every route were local.

    ``shared_output``, when given, is added to the result. With ``output`` the
    result is written into that BF16 tensor of ``x``'s shape, which is also
    returned; without it a new tensor is returned. ``swiglu_limit`` clamps the
    SwiGLU inputs as DeepSeek does: gate to at most the limit, up to
    ``[-limit, limit]``; ``None`` disables it.
    """
    return _fuse_moe_mx(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_total,
        shared_output,
        output,
        swiglu_limit,
    )


def fuse_moe_blockwise_fp8_sm120(
    x: Tensor,
    x_scale: Optional[Tensor],
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int = 0,
    num_expert_total: int = 0,
    shared_output: Optional[Tensor] = None,
    output: Optional[Tensor] = None,
    swiglu_limit: Optional[float] = None,
) -> Tensor:
    """Run the SM120 fused MoE for 128x128-block FP8 weights.

    The weights are float8_e4m3fn ``[E, 2I, H]`` and ``[E, H, I]`` with scales from
    :func:`prepare_fp8_moe_weights_sm120`. ``x`` is BF16 with ``x_scale=None``, or
    float8_e4m3fn with float32 ``x_scale`` ``[num_tokens, H/128]``, or uint8 UE8M0
    ``x_scale`` ``[num_tokens, H/32]``. The MX MMA takes UE8M0 scales, so float32
    scales are expected to be powers of two (under vLLM,
    ``VLLM_USE_DEEP_GEMM_E8M0=1``). They are not checked: any other value is
    rounded up to the next power of two, which changes the result. The weights
    and packed scales have the same 16-byte alignment requirement as in
    :func:`fuse_moe_blockwise_mxfp4_sm120`, and the other arguments are as there.
    """
    return _fuse_moe_mx(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_total,
        shared_output,
        output,
        swiglu_limit,
    )


@torch.library.register_fake("hpc::count_and_gather")
def count_and_gather_fake(
    x, topk_ids, num_expert, rank_ep, intermediate_size, num_seq_per_group_avg
):
    return (
        torch.empty((topk_ids.shape[0] * topk_ids.shape[1], x.shape[1]), dtype=torch.float8_e4m3fn),
        torch.empty(
            (topk_ids.shape[0] * topk_ids.shape[1], intermediate_size), dtype=torch.bfloat16
        ),
        torch.empty((topk_ids.shape[0] * topk_ids.shape[1]), dtype=torch.int32),
        torch.empty((num_expert), dtype=torch.int32),
        torch.empty((num_expert + 1), dtype=torch.int32),
        torch.empty((num_expert), dtype=torch.int32),
        torch.empty((num_expert + 1), dtype=torch.int32),
        torch.empty((num_expert * 2 * 128), dtype=torch.int8),
    )


@torch.library.register_fake("hpc::reduce")
def reduce_fake(x, topk_pos, topk_scale, shared_output):
    return torch.empty((topk_pos.shape[0], x.shape[1]), dtype=torch.bfloat16)


@torch.library.register_fake("hpc::fuse_moe")
def fuse_moe_fake(
    x,
    gate_up_weight,
    down_weight,
    gate_up_scale,
    down_scale,
    act_and_mul_scale,
    topk_ids,
    topk_scale,
    shared_output,
    rank_ep,
    num_expert_total,
    use_bf16_mul,
    output,
):
    return (
        output
        if output is not None
        else torch.empty((x.shape[0], x.shape[1]), dtype=torch.bfloat16)
    )


@torch.library.register_fake("hpc::fuse_moe_pertensor_fp8")
def fuse_moe_pertensor_fp8_fake(
    x,
    gate_up_weight,
    down_weight,
    gate_up_scale,
    down_scale,
    act_and_mul_scale,
    topk_ids,
    topk_scale,
    shared_output,
    rank_ep,
    num_expert_total,
    use_bf16_mul,
    output,
):
    return (
        output
        if output is not None
        else torch.empty((x.shape[0], x.shape[1]), dtype=torch.bfloat16)
    )


@torch.library.register_fake("hpc::fuse_moe_blockwise")
def fuse_moe_blockwise_fake(
    x: Tensor,
    x_scale: Tensor,
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    shared_output,
    rank_ep: int,
    num_expert_total: int,
    output,
):
    return (
        output
        if output is not None
        else torch.empty((x.shape[0], x.shape[1]), dtype=torch.bfloat16)
    )


@torch.library.register_fake("hpc::fuse_moe_blockwise_fp8")
def fuse_moe_blockwise_fp8_fake(
    x: Tensor,
    x_scale: Tensor,
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    shared_output,
    rank_ep: int,
    num_expert_total: int,
    output,
):
    return (
        output
        if output is not None
        else torch.empty((x.shape[0], x.shape[1]), dtype=torch.bfloat16)
    )


@torch.library.register_fake("hpc::fuse_moe_mx")
def fuse_moe_mx_fake(
    x,
    x_scale,
    gate_up_weight,
    gate_up_weight_scale,
    down_weight,
    down_weight_scale,
    topk_ids,
    topk_scale,
    shared_output,
    rank_ep,
    num_expert_total,
    swiglu_limit,
):
    return x.new_empty(x.shape, dtype=torch.bfloat16)


@torch.library.register_fake("hpc::fuse_moe_mx_out")
def fuse_moe_mx_out_fake(
    output,
    x,
    x_scale,
    gate_up_weight,
    gate_up_weight_scale,
    down_weight,
    down_weight_scale,
    topk_ids,
    topk_scale,
    shared_output,
    rank_ep,
    num_expert_total,
    swiglu_limit,
):
    return None
