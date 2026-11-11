# Copyright (C) 2026 Tencent.

"""Offline preparation for the SM90 MXFP4-weight / FP8-activation GEMM.

The weight representation is E2M1 with one E8M0 scale for every 32 values.
The runtime kernel decodes a nibble to ``q4 * 2**(relative_exp - 6)`` and
multiplies the result by ``expert_scale``. Consequently the lossless strict
conversion below uses ``relative_exp = raw_exp - min_exp + 1`` and
``expert_scale = 2**(min_exp - 122)``. These formulas are an independent
mathematical implementation of the representation, with no Humming dependency.

These helpers validate tensor *contents* and can copy metadata from CUDA to
CPU. They are preparation utilities, not CUDA-graph-safe forward operators.
Prepare weights once after loading; forward builds routing metadata on the GPU.
"""

import torch


def prepare_mxfp4_w4a8(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    strict: bool = True,
):
    """Validate strict MXFP4 weights and normalize the group exponents.

    Args:
        weight_packed: Contiguous ``[E,N,K/2]`` uint8 tensor, low nibble first.
            A contiguous int32 ``[E,N,K/8]`` view of the same bytes is accepted.
        weight_scale: Contiguous ``[E,N,K/32]`` E8M0 exponent bytes (uint8), or
            a ``torch.float8_e8m0fnu`` view of those bytes. Both inputs must be
            on the same device. CPU preparation is supported.
        strict: Only strict conversion is implemented. Each expert's exponent
            range must be at most 11. No weights are silently requantized.

    Returns:
        ``(weight_uint8, relative_exp_uint8, expert_scale_float32)`` with shapes
        ``[E,N,K/2]``, ``[E,N,K/32]`` and ``[E]``. The weight is a byte view of
        the input; the scale tensors are newly allocated on the same device.

    Raises:
        ValueError: Invalid shape, noncontiguous storage, E8M0 NaN code 255,
            exponent range greater than 11, or an unrepresentable FP32 expert
            scale. N must be a multiple of 64 and K a multiple of 128.
        TypeError: An input has an unsupported dtype.
        NotImplementedError: ``strict=False``; lossy Humming-compatible
            requantization is deliberately not implicit in this interface.

    This function synchronizes when its inputs are on CUDA. Do not call it
    inside an inference forward or CUDA graph capture.
    """
    if not strict:
        raise NotImplementedError("Only strict MXFP4 preparation is implemented")
    if weight_packed.dtype not in (torch.uint8, torch.int32):
        raise TypeError("weight_packed must have uint8 or int32 dtype")
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if weight_scale.dtype != torch.uint8 and weight_scale.dtype != e8m0_dtype:
        raise TypeError("weight_scale must have uint8 or float8_e8m0fnu dtype")
    if weight_packed.ndim != 3 or weight_scale.ndim != 3:
        raise ValueError("weight_packed and weight_scale must be rank-3 tensors")
    if not weight_packed.is_contiguous() or not weight_scale.is_contiguous():
        raise ValueError("weight_packed and weight_scale must be contiguous")
    if weight_packed.device != weight_scale.device:
        raise ValueError("weight_packed and weight_scale must be on the same device")
    packed = weight_packed.view(torch.uint8)
    e, n, packed_k = packed.shape
    k = packed_k * 2
    if e < 1 or n < 1 or k < 1 or n % 64 or k % 128:
        raise ValueError("Require E>0, N a positive multiple of 64, K a positive multiple of 128")
    if weight_scale.shape != (e, n, k // 32):
        raise ValueError("weight_scale must have shape [E,N,K/32]")
    raw_exp = weight_scale.view(torch.uint8)
    if bool((raw_exp == 255).any()):
        raise ValueError("E8M0 exponent byte 255 encodes NaN and is not supported")
    exponents = raw_exp.to(torch.int16)
    flattened = exponents.reshape(e, -1)
    minimum = flattened.amin(dim=1)
    maximum = flattened.amax(dim=1)
    if bool(((maximum - minimum) > 11).any()):
        raise ValueError(
            "Per-expert E8M0 exponent range exceeds 11; strict preparation cannot "
            "represent these weights without additional FP4 rounding"
        )
    relative_exp = (exponents - minimum[:, None, None] + 1).to(torch.uint8)
    expert_scale = torch.exp2(minimum.to(torch.float32) - 122)
    if not bool((torch.isfinite(expert_scale) & (expert_scale > 0)).all()):
        raise ValueError("The per-expert scale must be positive and representable in FP32")
    return packed, relative_exp.contiguous(), expert_scale.contiguous()


def _pack_w4a8_operands(raw: torch.Tensor, relative: torch.Tensor):
    """Pack validated uint8 [E,N,K/2] and [E,N,K/32], outside inference.

    Returns int32 [E,K/64,8*N] and opaque uint8 [E,K/32,N]. Weights remain
    four bits each; no tensor-wide FP8/FP16 dequantization is performed.
    """
    e, n, half_k = raw.shape
    k = half_k * 2
    thread = torch.arange(128, device=raw.device)
    row0 = (thread // 4) % 8 + (thread // 32) * 16
    rows = torch.stack((row0, row0 + 8), dim=1)
    columns = (thread % 4)[:, None] * 2 + torch.tensor((0, 1, 8, 9), device=raw.device)[None, :]
    tiles = raw.reshape(e, n // 64, 64, k // 32, 16).permute(0, 1, 3, 2, 4)
    lane_bytes = tiles[..., rows[..., None], columns[:, None, :]].contiguous()
    # Magnitudes stay in their ordinary nibble positions. The signs for
    # nibble values 0..7 become [s0,s4,s1,s5,s2,s6,s3,s7], so register decode
    # can merge all four byte signs with one mask per output uint32.
    interleaved = lane_bytes & 0x77
    for i in range(4):
        lower = (lane_bytes[..., i // 2] >> ((i % 2) * 4)) & 0x08
        upper = (lane_bytes[..., 2 + i // 2] >> ((i % 2) * 4)) & 0x08
        interleaved[..., i] |= lower | (upper << 4)
    words = interleaved.contiguous().view(torch.int32).squeeze(-1)
    # [E,N64,K32,thread,channel-pair] -> [E,K64,N64,thread,K32-pair,channel-pair].
    packed = words.reshape(e, n // 64, k // 64, 2, 128, 2)
    packed = packed.permute(0, 2, 1, 4, 3, 5).contiguous().reshape(e, k // 64, n * 8)
    # Physical scales: [E,K128,N64,warp4,lane_group8,K32_4,rowpair2].
    # One aligned 8B vector holds all four K32 scales for an RS row pair.
    # Warp-major pairs give each warp eight consecutive vectors.
    scales = relative.reshape(e, n // 64, 4, 2, 8, k // 128, 4)
    scales = scales.permute(0, 5, 1, 2, 4, 6, 3).contiguous().reshape(e, k // 32, n)
    return packed, scales


def prepare_w4a8_weights(weight_packed: torch.Tensor, weight_scale: torch.Tensor):
    """Prepare strict MXFP4 weights for the native H20 W4A8 MoE, offline.

    Inputs are canonical uint8 [E,N,K/2] and E8M0 [E,N,K/32]. Returns
    packed int32 [E,K/64,8*N], relative uint8 [E,K/32,N], and FP32 [E]
    common scales. The layout is private to the native RS WGMMA kernel.
    CUDA inputs synchronize during validation; never call during graph capture.
    """
    raw, relative, common = prepare_mxfp4_w4a8(weight_packed, weight_scale)
    packed, relative = _pack_w4a8_operands(raw, relative)
    return packed, relative, common
