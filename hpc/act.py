from typing import Optional, Tuple

import torch
from torch import Tensor


def act_mul_and_quant(
    gate_up: Tensor, scale: Tensor, use_bf16_mul: bool = True, output: Tensor = None
) -> Tensor:
    """Applies activation, multiplication, and quantization to the gate_up projection.

    Specifically:
    1. Splits the `gate_up` tensor into gate (first half) and up (second half)
    2. Applies activation (typically SiLU) to the gate portion
    3. Computes element-wise multiplication: activated_gate × up
    4. Scales the result using the first element of `scale`
    5. Quantizes the output to fp8_e4m3 format

    Executes via a custom high-performance GPU kernel.

    Args:
        gate_up: Concatenated gate and up projections.
            Shape: [N, 2*C] (N = batch size, C = hidden dimension)
            Dtype: bfloat16
        scale: Quantization scale factor.
            Only the first tensor element is used.
            Dtype: float32

    Returns:
        Quantized output tensor. result = silu(gate_up[:, :d/2]) * gate_up[:, d/2:] * scale
            Shape: [N, C]
            Dtype: fp8_e4m3
    """
    return torch.ops.hpc.act_mul_and_quant(gate_up, scale, use_bf16_mul, output)


def masked_act_mul_and_quant(
    gate_up: Tensor, scale: Tensor, num_per_expert: Tensor, output: Optional[Tensor] = None
) -> Tensor:
    """Applies activation, multiplication, and quantization to the gate_up projection.

    Specifically:
    1. Splits the `gate_up` tensor into gate (first half) and up (second half)
    2. Applies activation (typically SiLU) to the gate portion
    3. Computes element-wise multiplication: activated_gate × up
    4. Scales the result using the first element of `scale`
    5. Quantizes the output to fp8_e4m3 format

    Executes via a custom high-performance GPU kernel.

    Args:
        gate_up: Concatenated gate and up projections.
            Shape: [N, 2*C] (N = num_expert * num_token_padded_per_expert, C = hidden dimension)
            Dtype: bfloat16
        scale: Quantization scale factor.
            Only the first tensor element is used.
            Dtype: float32
        num_per_expert: Real num tokens of per expert
            Shape: [num_expert, ]
            Dtype: int32

    Returns:
        Quantized output tensor.
            Shape: [N, C]
            Dtype: fp8_e4m3
    """
    return torch.ops.hpc.masked_act_mul_and_quant(gate_up, scale, num_per_expert, output)


def masked_act_mul_and_blockwise_quant(
    gate_up: Tensor,
    num_per_expert: Tensor,
    output: Optional[Tensor] = None,
    output_scale: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Applies activation, multiplication, and quantization to the gate_up projection.

    Specifically:
    1. Splits the `gate_up` tensor into gate (first half) and up (second half)
    2. Applies activation (typically SiLU) to the gate portion
    3. Computes element-wise multiplication: activated_gate × up
    3. Every 128 numbers are grouped for quantization, scale = max(abs(x)) / fp8_max
    5. Quantizes the output to fp8_e4m3 format, y = x / scale

    Executes via a custom high-performance GPU kernel.

    Args:
        gate_up: Concatenated gate and up projections.
            Shape: [N, 2*C] (N = num_expert * num_token_padded_per_expert, C = hidden dimension)
            Dtype: bfloat16
        num_per_expert: Real num tokens of per expert
            Shape: [num_expert, ]
            Dtype: int32

    Returns:
        Quantized output tensor. result = silu(gate_up[:, :d/2]) * gate_up[:, d/2:] * scale
            Shape: [N, C]
            Dtype: fp8_e4m3
        Scales output tensor. scale = max(abs(x)) / fp8_max
            Shape: [N, C / 128]
            Dtype: fp32
    """
    return torch.ops.hpc.masked_act_mul_and_blockwise_quant(
        gate_up, num_per_expert, output, output_scale
    )


@torch.library.register_fake("hpc::act_mul_and_quant")
def act_mul_and_quant_fake(input, scale, use_bf16_mul, output):
    return torch.empty(
        input.shape[0], input.shape[1] // 2, dtype=torch.float8_e4m3fn, device=input.device
    )


@torch.library.register_fake("hpc::masked_act_mul_and_quant")
def masked_act_mul_and_quant_fake(input, scale, num_per_expert, output=None):
    return torch.empty(
        input.shape[0], input.shape[1] // 2, dtype=torch.float8_e4m3fn, device=input.device
    )


@torch.library.register_fake("hpc::masked_act_mul_and_blockwise_quant")
def masked_act_mul_and_blockwise_quant_fake(input, num_per_expert, output=None, output_scale=None):
    return (
        torch.empty(
            input.shape[0], input.shape[1] // 2, dtype=torch.float8_e4m3fn, device=input.device
        ),
        torch.empty(
            input.shape[0], input.shape[1] // 2 // 128, dtype=torch.float32, device=input.device
        ),
    )
