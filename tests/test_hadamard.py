import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
from utils import allclose

try:
    from fast_hadamard_transform import hadamard_transform as fht_hadamard_transform

    HAS_FHT = True
except ImportError:
    HAS_FHT = False


def hadamard_transform_ref(x):
    n = x.shape[-1]
    x_float = x.float()
    batch_size = x_float.shape[0]

    # Determine decomposition
    if n == 64:
        base_size = 4
    else:
        raise ValueError(f"Unsupported n={n}")

    num_threads = n // base_size
    unit_iters = int(math.log2(num_threads))

    data = x_float.reshape(batch_size, num_threads, base_size).clone()

    # Unit Hadamard transforms (inter-thread butterfly)
    # Mirrors kernel: stride goes 1, 2, 4, 8 (from low bit to high bit)
    for step in range(unit_iters):
        stride = 1 << step
        new_data = data.clone()
        for t in range(num_threads):
            partner = t ^ stride
            if partner > t:
                a = data[:, t, :].clone()
                b = data[:, partner, :].clone()
                # t has bit stride == 0 (lower), partner has bit stride == 1 (upper)
                new_data[:, t, :] = a + b  # A' = A + B
                new_data[:, partner, :] = a - b  # B' = A - B
        data = new_data

    # Base Hadamard transform (intra-thread)
    if base_size == 4:
        H4 = torch.tensor(
            [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
            dtype=torch.float32,
            device=x.device,
        )
        # data shape: [batch_size, num_threads, 4]
        # y = data @ H4^T (each thread's 4 values multiplied by H4)
        data = torch.einsum("btn,mn->btm", data, H4)

    # Scale
    data = data * (1.0 / math.sqrt(n))

    return data.reshape(batch_size, n).to(x.dtype)


@pytest.mark.skipif(
    bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer, entry not actually used"
)
@pytest.mark.parametrize("batch_size", [1, 4, 17, 111, 1025])
@pytest.mark.parametrize("n", [64])
def test_hadamard_transform(batch_size, n):
    torch.manual_seed(42)
    x = torch.randn(batch_size, n, dtype=torch.bfloat16, device="cuda")

    gt = hadamard_transform_ref(x)
    output = hpc.hadamard_transform(x)

    assert output.dtype == torch.bfloat16, f"Expected bfloat16 output, got {output.dtype}"
    assert output.shape == (batch_size, n), f"Output shape mismatch: {output.shape}"

    assert allclose(
        gt, output, atol=0.1, rtol=0.02
    ), f"Hadamard transform mismatch. Max diff: {(gt.float() - output.float()).abs().max()}"

    print(f"✓ Test passed: batch_size={batch_size}, n={n}")


@pytest.mark.skipif(
    bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer, entry not actually used"
)
@pytest.mark.skipif(not HAS_FHT, reason="fast_hadamard_transform not installed")
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("n", [64])
def test_hadamard_transform_vs_fht(batch_size, n):
    """Test against the fast_hadamard_transform library (Dao-AILab)."""
    torch.manual_seed(42)
    x = torch.randn(batch_size, n, dtype=torch.bfloat16, device="cuda")

    inv_sqrt_n = 1.0 / math.sqrt(n)

    # Dao's library: hadamard_transform(x, scale) = H @ x * scale
    fht_output = fht_hadamard_transform(x.float(), scale=inv_sqrt_n).to(torch.bfloat16)
    our_output = hpc.hadamard_transform(x)

    assert allclose(fht_output, our_output, atol=0.1, rtol=0.02), (
        f"Mismatch with fast_hadamard_transform lib. "
        f"Max diff: {(fht_output.float() - our_output.float()).abs().max()}"
    )

    print(f"✓ FHT comparison test passed: batch_size={batch_size}, n={n}")


def act_mul_hadamard_blockwise_quant_ref(gate_up, upper_max=448.0):
    gate_up_f = gate_up.float()
    N, full_col = gate_up_f.shape
    C = full_col // 2
    gate = gate_up_f[:, :C]
    up = gate_up_f[:, C:]

    # silu * up
    x = torch.nn.functional.silu(gate) * up  # [N, C]

    # Hadamard per 64-wide block
    num_blocks = C // 64
    x_blocks = x.reshape(N, num_blocks, 64)  # [N, num_blocks, 64]
    x_had = torch.zeros_like(x_blocks)
    for b in range(num_blocks):
        x_had[:, b, :] = hadamard_transform_ref(x_blocks[:, b, :].to(torch.bfloat16)).float()

    # Blockwise quant: scale = max(|x|)/upper_max per block
    abs_max = x_had.abs().amax(dim=-1)  # [N, num_blocks]
    scale = abs_max / upper_max  # [N, num_blocks]
    inv_scale = 1.0 / (scale + 1e-8)

    x_quant = x_had * inv_scale.unsqueeze(-1)  # quantized values in fp8 range
    x_dequant = x_quant * scale.unsqueeze(-1)  # dequantized back to float

    # out_scale layout: [num_blocks, N]  (N-major, matching kernel output)
    out_scale = scale.T.contiguous()  # [num_blocks, N]

    return x_dequant.reshape(N, C), out_scale


@pytest.mark.skipif(
    bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer, entry not actually used"
)
@pytest.mark.parametrize("num_rows", [1, 4, 17, 111, 1025])
@pytest.mark.parametrize("num_col", [64, 128, 256, 512])
def test_act_mul_hadamard_blockwise_quant(num_rows, num_col):
    """Test fused act_mul_hadamard_blockwise_quant kernel correctness (default upper_max=448)."""
    torch.manual_seed(42)
    gate_up = torch.randn(num_rows, num_col * 2, dtype=torch.bfloat16, device="cuda") * 2.0

    ref_dequant, ref_scale = act_mul_hadamard_blockwise_quant_ref(gate_up)

    out_fp8, out_scale = hpc.act_mul_hadamard_blockwise_quant(gate_up)

    assert out_fp8.dtype == torch.float8_e4m3fn, f"Expected fp8_e4m3fn, got {out_fp8.dtype}"
    assert out_fp8.shape == (num_rows, num_col), f"out shape mismatch: {out_fp8.shape}"
    assert out_scale.dtype == torch.float32, f"Expected float32 scale, got {out_scale.dtype}"
    assert out_scale.shape == (num_col // 64, num_rows), f"scale shape mismatch: {out_scale.shape}"

    # out_scale: [num_col//64, num_rows] → broadcast over [num_rows, num_col]
    num_blocks = num_col // 64
    scale_per_elem = (
        out_scale.T.contiguous()
        .unsqueeze(-1)
        .expand(num_rows, num_blocks, 64)
        .reshape(num_rows, num_col)
    )
    out_dequant = out_fp8.float() * scale_per_elem

    assert allclose(
        ref_scale, out_scale, atol=1e-3, rtol=1e-2
    ), f"Scale mismatch. Max diff: {(ref_scale - out_scale).abs().max()}"

    assert allclose(
        ref_dequant, out_dequant, atol=0.5, rtol=0.05
    ), f"Dequant value mismatch. Max diff: {(ref_dequant - out_dequant).abs().max()}"

    print(f"✓ Test passed: num_rows={num_rows}, num_col={num_col}")


def act_mul_hadamard_per_tensor_quant_ref(gate_up, scale_inv):
    """Reference implementation: silu(gate)*up → Hadamard → per-tensor FP8 quant."""
    gate_up_f = gate_up.float()
    N, full_col = gate_up_f.shape
    C = full_col // 2
    gate = gate_up_f[:, :C]
    up = gate_up_f[:, C:]

    # silu * up
    x = torch.nn.functional.silu(gate) * up  # [N, C]

    # Hadamard per 64-wide block
    num_blocks = C // 64
    x_blocks = x.reshape(N, num_blocks, 64)
    x_had = torch.zeros_like(x_blocks)
    for b in range(num_blocks):
        x_had[:, b, :] = hadamard_transform_ref(x_blocks[:, b, :].to(torch.bfloat16)).float()

    # Per-tensor quant: multiply by scale_inv, cast to fp8
    x_quant_f = x_had * scale_inv
    # Simulate fp8 cast: clamp to fp8 range and round
    x_quant_f = x_quant_f.clamp(-448.0, 448.0)

    return x_had.reshape(N, C), x_quant_f.reshape(N, C)


@pytest.mark.parametrize("num_rows", [1, 113, 1025])
@pytest.mark.parametrize("num_col", [192])
def test_act_mul_hadamard_per_tensor_quant(num_rows, num_col):
    """Test fused act_mul_hadamard_per_tensor_quant kernel correctness."""
    torch.manual_seed(42)
    gate_up = torch.randn(num_rows, num_col * 2, dtype=torch.bfloat16, device="cuda") * 2.0

    # Compute a reasonable scale: amax of intermediate / 448
    # First compute the reference intermediate to get a realistic scale
    gate_up_f = gate_up.float()
    gate = gate_up_f[:, :num_col]
    up = gate_up_f[:, num_col:]
    x_intermediate = torch.nn.functional.silu(gate) * up
    amax = x_intermediate.abs().max().item()
    scale = amax / 448.0
    scale_inv = 1.0 / (scale + 1e-8)

    scale_inv_tensor = torch.tensor([scale_inv], dtype=torch.float32, device="cuda")

    ref_x_had, ref_quant_f = act_mul_hadamard_per_tensor_quant_ref(gate_up, scale_inv)

    out_fp8 = hpc.act_mul_hadamard_per_tensor_quant(gate_up, scale_inv_tensor)

    assert out_fp8.dtype == torch.float8_e4m3fn, f"Expected fp8_e4m3fn, got {out_fp8.dtype}"
    assert out_fp8.shape == (num_rows, num_col), f"out shape mismatch: {out_fp8.shape}"

    # Dequantize: out_fp8 / scale_inv = out_fp8 * scale
    out_dequant = out_fp8.float() * scale

    assert allclose(
        ref_x_had, out_dequant, atol=0.5, rtol=0.05
    ), f"Dequant value mismatch. Max diff: {(ref_x_had - out_dequant).abs().max()}"

    print(f"✓ Test passed: num_rows={num_rows}, num_col={num_col}")
