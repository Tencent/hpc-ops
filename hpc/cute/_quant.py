# Copyright (C) 2026 Tencent.

"""SwiGLU and MXFP8 quantization arithmetic, shared by every path that emits
the intermediate tensor: the standalone activation kernel and the fused
epilogue of the up-projection GEMM.

The standalone activation kernel and the fused up-projection epilogue must match
exactly, so the arithmetic lives here rather than being written twice.
"""

from typing import Tuple

import torch
from torch import Tensor

from ._compile import import_cutlass

FP8_MAX = 448.0
# Smallest scale any kernel here will emit; keeps all-zero blocks well defined.
MIN_SCALE = 2.0**-127
# UE8M0 encoding of 1.0, for the per-tensor paths whose scale is global.
UNIT_UE8M0 = 127
# Largest finite UE8M0 value is 2^127 (biased exponent 254).
MAX_UE8M0 = 2.0**127


def build_helpers():
    """Return the device-side helpers, bound to the traced ``cutlass`` module.

    They are built rather than defined at module scope because they close over
    the DSL modules, which are imported lazily.
    """
    cutlass, cute = import_cutlass()
    import cutlass.cute.math as cute_math

    def silu(x):
        """``hpc::silu``: approximate exp2/rcp, matching ``src/utils/utils.cuh``."""
        return x * cute.arch.rcp_approx(
            cutlass.Float32(1.0) + cute_math.exp2(-x * 1.4426950408889634, fastmath=True)
        )

    # These helpers run as plain Python during tracing, so their control flow has
    # to be expressed with select_/and_ rather than if statements.

    def to_ue8m0_storage(value):
        """Round ``value`` up to a power of two and return its UE8M0 byte.

        Mirrors ``cutlass::float_ue8m0_t::convert_from_float``, whose PTX form is
        ``cvt.rp.satfinite.ue8m0x2.f32`` -- round toward positive infinity. The
        input is always positive and finite here, so the NaN/Inf branch of the
        CUTLASS fallback is not reproduced.
        """
        bits = value.bitcast(cutlass.Int32)
        exponent = (bits >> 23) & 0xFF
        mantissa = bits & 0x7FFFFF

        round_up = cutlass.and_(
            mantissa > 0,
            cutlass.and_(
                # 0xFE is the largest exponent that can round up without saturating.
                exponent != 0xFE,
                # 2^-127 and below is already the smallest representable scale.
                cutlass.not_(cutlass.and_(exponent == 0, mantissa <= 0x00400000)),
            ),
        )
        return cutlass.Int32(cutlass.select_(round_up, exponent + 1, exponent))

    def ue8m0_to_float(storage):
        powered = (storage << 23).bitcast(cutlass.Float32)
        return cutlass.Float32(cutlass.select_(storage == 0, cutlass.Float32(MIN_SCALE), powered))

    def swiglu(gate, up, swiglu_limit):
        """SwiGLU on the bfloat16 values the unfused path would have written.

        The fused epilogue starts from an fp32 accumulator, so it has to round
        the way a bfloat16 store would; for a caller that already loaded
        bfloat16 the rounding is idempotent. A negative limit disables the
        clamp, selected rather than branched on because this runs as plain
        Python during tracing.
        """
        gate = cutlass.Float32(cutlass.BFloat16(gate))
        up = cutlass.Float32(cutlass.BFloat16(up))
        limited = swiglu_limit >= 0.0
        gate = cutlass.Float32(cutlass.select_(limited, cute.arch.fmin(gate, swiglu_limit), gate))
        up = cutlass.Float32(
            cutlass.select_(
                limited,
                cute.arch.fmin(cute.arch.fmax(up, -swiglu_limit), swiglu_limit),
                up,
            )
        )
        return silu(gate) * up

    def quantization_scale(max_abs):
        """UE8M0 byte for a block maximum, and the reciprocal to quantize with."""
        requested = cute.arch.fmax(max_abs / FP8_MAX, cutlass.Float32(MIN_SCALE))
        storage = to_ue8m0_storage(requested)
        return storage, cutlass.Float32(1.0) / ue8m0_to_float(storage)

    return swiglu, quantization_scale, to_ue8m0_storage


def fp32_to_ue8m0(x: Tensor) -> Tensor:
    """Ceil a positive float32 scale to UE8M0, matching ``to_ue8m0_storage``."""
    x = x.contiguous().float().clamp(min=MIN_SCALE, max=MAX_UE8M0)
    bits = x.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    tiny = (exponent == 0) & (mantissa <= 0x00400000)
    round_up = (mantissa > 0) & (exponent != 0xFE) & ~tiny
    return (exponent + round_up.to(torch.int32)).to(torch.uint8)


def is_power_of_two(scale: Tensor) -> bool:
    """Whether every entry of a positive float32 tensor is an exact power of two.

    A float32 is a power of two exactly when its mantissa bits are zero, so
    this is one pass over the scales and a single sync. Callers use it to check
    that activations were quantized the way the MX MMA reads them; ceiling a
    scale the data was not divided by rescales that block by up to 2x.
    """
    if scale.dtype != torch.float32:
        raise ValueError(f"scale must be float32; got {scale.dtype}")
    bits = scale.contiguous().view(torch.int32)
    return bool(((bits & 0x7FFFFF) == 0).all().item())


def ue8m0_to_fp32(storage: Tensor) -> Tensor:
    """Decode a UE8M0 byte tensor to float32."""
    bits = storage.to(torch.int32) << 23
    out = bits.view(torch.float32)
    return torch.where(storage == 0, torch.full_like(out, MIN_SCALE), out)


def requant_fp8_weight_ue8m0(weight: Tensor, scale: Tensor) -> Tuple[Tensor, Tensor]:
    """Re-quantize 128-block FP8 weights onto power-of-two scales in place.

    Same transform DeepGEMM applies on SM120 before packing UE8M0: dequantize
    with the checkpoint float32 block scale, then quantize with
    ``2^ceil(log2(amax / 448))`` so the MX MMA's hardware scale is exact for
    the stored E4M3 values. Writes back into ``weight`` / ``scale`` one expert
    at a time so the float32 workspace stays a single ``[N, K]`` tile and no
    full-weight clone is retained.
    """
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn; got {weight.dtype}")
    if scale.dtype != torch.float32:
        raise ValueError(f"scale must be float32; got {scale.dtype}")
    experts, rows, cols = weight.shape
    block = 128
    if rows % block != 0 or cols % block != 0:
        raise ValueError(f"weight [{rows}, {cols}] is not 128-block aligned")
    expected = (experts, rows // block, cols // block)
    if tuple(scale.shape) != expected:
        raise ValueError(f"scale must be {expected}; got {tuple(scale.shape)}")

    # Mutate in place. A cloned ``empty_like(weight)`` is ~400MB per gate_up
    # tensor; 48 Qwen layers would pin an extra ~28GB through CUDA-graph
    # capture and OOM the decode graphs.
    for expert in range(experts):
        dequant = weight[expert].float() * scale[expert].repeat_interleave(
            block, dim=0
        ).repeat_interleave(block, dim=1)
        tiled = dequant.view(rows // block, block, cols // block, block)
        amax = tiled.abs().amax(dim=(1, 3)).clamp(min=MIN_SCALE)
        ue8 = fp32_to_ue8m0((amax / FP8_MAX).clamp(min=MIN_SCALE))
        sf = ue8m0_to_fp32(ue8)
        quantized = (tiled / sf[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
        weight[expert].copy_(quantized.to(torch.float8_e4m3fn).reshape(rows, cols))
        scale[expert].copy_(sf)
        del dequant, tiled, quantized
    return weight, scale


def broadcast_ue8m0_128_to_mx(scale128: Tensor, rows: int, cols: int) -> Tensor:
    """Expand ``[E, rows/128, cols/128]`` UE8M0 to logical ``[E, rows, cols/32]``.

    SM120 MX MMA applies one scale per 32 elements. A 128-element Qwen block is
    four of those, so the same byte is repeated 128 times along N and 4 times
    along K -- the same broadcast DeepGEMM does inside ``kGranKA=128``.
    """
    if scale128.dtype != torch.uint8:
        raise ValueError(f"scale128 must be uint8; got {scale128.dtype}")
    expected = (scale128.shape[0], rows // 128, cols // 128)
    if tuple(scale128.shape) != expected:
        raise ValueError(f"scale128 must be {expected}; got {tuple(scale128.shape)}")
    return scale128.repeat_interleave(128, dim=1).repeat_interleave(4, dim=2)
