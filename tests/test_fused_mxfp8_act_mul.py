"""Unit test: fused gateup + act_mul kernel vs two-step reference.

Verifies: group_gemm_cp_async_mxfp8_act_mul == group_gemm_cp_async_mxfp8 + act_mul_and_mxfp8_quant

Input construction follows test_group_gemm_mxfp8.py style:
  - uniform seqlens per group (no routing / x_row_map)
  - config: num_group=32, seq_per_group variable, n=4096 (gate_up), k=6144
"""

import sys
import os
from pathlib import Path

sys.path.insert(
    0,
    os.path.realpath(
        sorted([p for p in Path(__file__).parent.glob("../build/lib.*/") if "linux" in str(p)])[0]
    ),
)

import hpc
import torch
import pytest
from utils import mxfp8_dispatch_kTileM
from utils import allclose

SF_VEC = 32


def interleave_n16(w):
    """Interleave first/second half every 16 in dim1.
    (E, 2*inter, K) → interleaved [gate16, up16, gate16, up16, ...]
    """
    E, N, K = w.shape
    half = N // 2
    gate = w[:, :half, :].reshape(E, half // 16, 16, K)
    up = w[:, half:, :].reshape(E, half // 16, 16, K)
    return torch.stack([gate, up], dim=2).reshape(E, N, K)


def _run_one(num_group: int, seq_per_group: int, n: int, k: int):
    torch.manual_seed(2026)
    device = torch.device("cuda:0")

    seqlens = torch.full((num_group,), seq_per_group, dtype=torch.int32, device=device)
    cu_seqlens = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(seqlens, 0).to(torch.int32)]
    )
    m_total = int(cu_seqlens[-1].item())

    # ==================== Data ====================
    x_f32 = torch.randint(-1, 2, (m_total, k), device=device, dtype=torch.float32)
    w_f32 = torch.randint(-1, 2, (num_group, n, k), device=device, dtype=torch.float32)

    x_fp8 = x_f32.to(torch.float8_e4m3fn).contiguous()
    # Non-interleaved weight (for reference path)
    w_fp8 = w_f32.to(torch.float8_e4m3fn).contiguous()
    # Interleaved weight (for fused path)
    w_interleaved_fp8 = interleave_n16(w_f32).to(torch.float8_e4m3fn).contiguous()

    # x_scale = torch.randint(124, 131, (m_total, k // SF_VEC), dtype=torch.uint8, device=device)
    # w_scale = torch.randint(124, 131, (num_group, n, k // SF_VEC), dtype=torch.uint8, device=device)
    x_scale = torch.full((m_total, k // SF_VEC), 127, dtype=torch.uint8, device=device)
    w_scale = torch.full((num_group, n, k // SF_VEC), 127, dtype=torch.uint8, device=device)

    w_scale_interleaved = interleave_n16(w_scale)

    kTileM = mxfp8_dispatch_kTileM(seq_per_group)

    # ==================== x_row_map (routing/gather) ====================
    # Both the reference (group_gemm_cp_async_mxfp8) and the fused op gather x
    # rows through x_row_map: out_row[i] = x[x_row_map[i]]. Feeding the SAME
    # x_row_map to both paths keeps their outputs comparable, so no permuted
    # copy of x / sfx is needed — just pass the map and x_num_rows = m_total.
    x_row_map = torch.randperm(m_total, device=device, dtype=torch.int32).contiguous()

    # Prepack scales
    _, sfw_packed = hpc.prepack_mxfp8_scale(
        x_scale, w_scale, cu_seqlens, num_seq_per_group_avg=seq_per_group
    )
    _, sfw_interleaved_packed = hpc.prepack_mxfp8_scale(
        None, w_scale_interleaved, None, num_seq_per_group_avg=seq_per_group
    )

    # ==================== Reference: two-step ====================
    # Step 1: group_gemm_cp_async_mxfp8 (produces bf16 gate_up_output)
    gate_up_output = torch.ops.hpc.group_gemm_cp_async_mxfp8(
        x_fp8,
        w_fp8,
        x_scale,
        sfw_packed,
        seqlens,
        cu_seqlens,
        seq_per_group,
        x_row_map,
        m_total,
        None,
        None,
    )  # (m_total, n) bf16
    # half = n // 2
    # gate = gate_up_output[:, :half].float()
    # up   = gate_up_output[:, half:].float()
    # ref_actmul = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)   # (m, n_half)
    ref, ref_scale = torch.ops.hpc.act_mul_and_mxfp8_quant(gate_up_output, m_total)
    print(ref_scale)

    # Step 2: act_mul_and_mxfp8_quant
    valid_rows = cu_seqlens[-1].item()
    # ==================== Fused kernel ====================
    fused_out, fused_fp8_out, fused_scale = torch.ops.hpc.group_gemm_cp_async_mxfp8_act_mul(
        x_fp8,
        w_interleaved_fp8,
        x_scale,
        sfw_interleaved_packed,
        seqlens,
        cu_seqlens,
        seq_per_group,
        x_row_map,
        m_total,
    )
    print(fused_scale)
    assert allclose(ref, fused_fp8_out, atol=0.016)
    assert allclose(fused_scale, ref_scale)

    # ==================== Compare ====================


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 10, reason="skip on non sm100!")
@pytest.mark.parametrize(
    "num_group, seq_per_group, n, k",
    [
        (32, 4, 4096, 6144),
        (32, 8, 4096, 6144),
        (32, 16, 4096, 6144),
        (32, 32, 4096, 6144),
        (32, 64, 4096, 6144),
        (32, 128, 4096, 6144),
    ],
)
def test_fused_act_mul(num_group, seq_per_group, n, k):
    _run_one(num_group, seq_per_group, n, k)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
