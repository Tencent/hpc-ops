import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math


def reference_torch_layer_norm_with_scale_quant(
    x,
    weight,
    bias,
    before_norm_scale1,
    before_norm_scale2,
    scale,
    bias_scale,
    eps,
    quant_eps,
    group_size,
    is_elementwise_affine,
    use_before_norm_scale,
    use_after_norm_scale,
):
    # Step 1: Layer Normalization
    torch.set_printoptions(precision=6)
    if use_before_norm_scale:
        x = x + (before_norm_scale1.float() * before_norm_scale2.float())
    output_x = x
    mean = x.float().mean(dim=-1, keepdim=True)
    var = ((x.float() - mean) ** 2).mean(dim=-1, keepdim=True)
    var = torch.rsqrt(var + eps)
    x_normalized = (x.float() - mean) * var

    # Step 2: Affine transformation (optional)
    if is_elementwise_affine:
        x_normalized = x_normalized * weight.float() + bias.float()

    # Step 3: Scale and bias
    x_scaled = x_normalized
    if use_after_norm_scale:
        x_scaled = x_normalized * scale.float() + bias_scale.float()
    # Step 4: Per-token group quantization
    batch_size, hidden_states = x.shape
    num_groups = hidden_states // group_size

    # Reshape to [batch_size, num_groups, group_size]
    x_grouped = x_scaled.reshape(batch_size, num_groups, group_size)

    group_max = torch.abs(x_grouped).max(dim=-1, keepdim=True)[0]  # [batch_size, num_groups, 1]
    fp8_max = 448.0

    quant_scale = (group_max / fp8_max).squeeze(-1)  # [batch_size, num_groups]
    quant_scale = torch.clamp(quant_scale, min=quant_eps)

    x_grouped_scaled = x_grouped / quant_scale.unsqueeze(-1)
    x_grouped_clamped = torch.clamp(x_grouped_scaled, -fp8_max, fp8_max)

    x_quant = x_grouped_clamped.reshape(batch_size, hidden_states)

    output_fp8 = x_quant.to(torch.float8_e4m3fn)

    return output_fp8, quant_scale, output_x


@pytest.mark.parametrize("batch_size", [1, 2, 4, 5, 8, 16, 32, 64])
@pytest.mark.parametrize("hidden_states", [5120, 4096])
@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize("is_elementwise_affine", [True, False])
@pytest.mark.parametrize("use_before_norm_scale", [True, False])
@pytest.mark.parametrize("use_after_norm_scale", [True, False])
def test_fused_layer_norm_with_scale_quant(
    batch_size,
    hidden_states,
    group_size,
    is_elementwise_affine,
    use_before_norm_scale,
    use_after_norm_scale,
):
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    x = torch.rand(batch_size, hidden_states, dtype=torch.bfloat16, device="cuda")
    for i in range(batch_size):
        for j in range(0, hidden_states, group_size):
            x.flatten()[i * hidden_states + j] = fp8_max

    weight = torch.ones(hidden_states, dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros(hidden_states, dtype=torch.bfloat16, device="cuda")

    before_norm_scale1 = torch.rand(batch_size, hidden_states, dtype=torch.bfloat16, device="cuda")
    before_norm_scale2 = torch.rand(batch_size, hidden_states, dtype=torch.bfloat16, device="cuda")
    scale = torch.ones(batch_size, hidden_states, dtype=torch.bfloat16, device="cuda")
    bias_scale = torch.zeros(batch_size, hidden_states, dtype=torch.bfloat16, device="cuda")

    eps = 1e-5
    quant_eps = 1e-5

    gt_fp8, gt_quant_scale, gt_output_x = reference_torch_layer_norm_with_scale_quant(
        x,
        weight,
        bias,
        before_norm_scale1,
        before_norm_scale2,
        scale,
        bias_scale,
        eps,
        quant_eps,
        group_size,
        is_elementwise_affine,
        use_before_norm_scale,
        use_after_norm_scale,
    )

    output_fp8, quant_scale, output_x = hpc.normalization.fused_layer_norm_with_scale_quant(
        x,
        weight,
        bias,
        before_norm_scale1,
        before_norm_scale2,
        scale,
        bias_scale,
        eps,
        quant_eps,
        group_size,
        is_elementwise_affine,
        use_before_norm_scale,
        use_after_norm_scale,
    )

    assert output_fp8.dtype == torch.float8_e4m3fn, f"Expected FP8 output, got {output_fp8.dtype}"
    assert (
        quant_scale.dtype == torch.float32
    ), f"Expected float32 quant_scale, got {quant_scale.dtype}"

    assert output_fp8.shape == (
        batch_size,
        hidden_states,
    ), f"Output shape mismatch: {output_fp8.shape}"
    assert quant_scale.shape == (
        batch_size,
        hidden_states // group_size,
    ), f"Quant scale shape mismatch: {quant_scale.shape}"

    assert torch.allclose(
        quant_scale, gt_quant_scale, atol=0.5, rtol=0.05
    ), f"Quant scale mismatch. Max diff: {(quant_scale - gt_quant_scale).abs().max()}"

    gt_output_x = gt_output_x.to(torch.bfloat16)
    assert torch.allclose(
        output_x, gt_output_x, atol=0.5, rtol=0.05
    ), f"Output mismatch. Max diff: {(output_x - gt_output_x).abs().max()}"

    output_bf16 = output_fp8.to(torch.bfloat16)
    gt_bf16 = gt_fp8.to(torch.bfloat16)
    assert torch.allclose(
        output_bf16, gt_bf16, atol=0.5, rtol=0.05
    ), f"FP8 output mismatch. Max diff: {(output_bf16 - gt_bf16).abs().max()}"

    print(
        f"✓ Test passed: batch_size={batch_size}, hidden_states={hidden_states}, "
        f"group_size={group_size}, is_elementwise_affine={is_elementwise_affine}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
