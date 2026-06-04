"""Compare group_gemm_cp_async_mxfp8 (A via cp.async) against group_gemm_mxfp8
(A via TMA). With x_row_map=None they should be numerically identical (same
inputs, just different load mechanism).

This is the smoke test before integrating the cp.async kernel into fuse_moe.
"""

import os
import sys
from pathlib import Path

sys.path.insert(
    0,
    os.path.realpath(
        sorted([p for p in Path(__file__).parent.glob("../build/lib.*/") if "linux" in str(p)])[0]
    ),
)

import torch
import pytest

import hpc
from utils import allclose

SF_VEC = 32


def _build_inputs(num_group, seq_per_group, n, k):
    torch.manual_seed(2026)
    device = torch.device("cuda:0")

    seqlens = torch.full((num_group,), seq_per_group, dtype=torch.int32, device=device)
    cu_seqlens = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(seqlens, 0).to(torch.int32)]
    )
    m_total = int(cu_seqlens[-1].item())

    x_f32 = torch.randint(-4, 5, (m_total, k), device=device, dtype=torch.float32)
    w_f32 = torch.randint(-4, 5, (num_group, n, k), device=device, dtype=torch.float32)
    x_fp8 = x_f32.to(torch.float8_e4m3fn).contiguous()
    w_fp8 = w_f32.to(torch.float8_e4m3fn).contiguous()

    sfx = torch.randint(124, 131, (m_total, k // SF_VEC), dtype=torch.uint8, device=device)
    sfw = torch.randint(124, 131, (num_group, n, k // SF_VEC), dtype=torch.uint8, device=device)

    sfx_packed, sfw_packed = hpc.prepack_mxfp8_scale(
        sfx, sfw, cu_seqlens, num_seq_per_group_avg=seq_per_group
    )

    return (x_fp8, w_fp8, sfx_packed, sfw_packed, seqlens, cu_seqlens)


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 10, reason="skip on non sm100!")
@pytest.mark.parametrize(
    "num_group, seq_per_group, n, k",
    [
        # (192, 8, 768, 4096),  # tp4 gate_up
        # (192, 32, 768, 4096),  # tp4 gate_up
        # (192, 8, 4096, 384),  # tp4 down
        # (192, 32, 4096, 384),  # tp4 down
        # tp8 cases (n=384/4096) — n%256==0 for n=4096, but down's avg<=32 stays 1SM
        (192, 8, 384, 4096),  # tp8 gate_up
        (192, 8, 4096, 192),  # tp8 down
        # Force-1SM kTileM=128 case (n=384, n%256!=0 → ref also 1SM)
        (192, 128, 384, 4096),  # 1SM-only kTileM=128
    ],
)
def test_cp_async_matches_tma(num_group, seq_per_group, n, k):
    """cp.async A path must produce identical output to TMA A path."""
    x_fp8, w_fp8, sfx_packed, sfw_packed, seqlens, cu_seqlens = _build_inputs(
        num_group, seq_per_group, n, k
    )

    # Reference: original TMA-A kernel.
    y_ref = hpc.group_gemm_mxfp8(
        x_fp8,
        w_fp8,
        sfx_packed,
        sfw_packed,
        seqlens,
        cu_seqlens,
        num_seq_per_group_avg=seq_per_group,
    )

    # cp.async kernel, no row_map (sequential read).
    y_cp = hpc.group_gemm_cp_async_mxfp8(
        x_fp8,
        w_fp8,
        sfx_packed,
        sfw_packed,
        seqlens,
        cu_seqlens,
        num_seq_per_group_avg=seq_per_group,
        x_row_map=None,
    )

    diff = (y_ref.to(torch.float32) - y_cp.to(torch.float32)).abs()
    print(
        f"[g={num_group} seq={seq_per_group} n={n} k={k}] "
        f"max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.6f}"
    )
    # Should be bit-identical or near-zero (same MMA, same inputs in SMEM).
    assert allclose(y_ref.to(torch.float32), y_cp.to(torch.float32), rtol=0.08, atol=1)


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 10, reason="skip on non sm100!")
@pytest.mark.parametrize(
    "num_group, seq_per_group, n, k",
    [
        (192, 8, 768, 4096),
        (192, 32, 4096, 384),
    ],
)
def test_cp_async_with_row_map(num_group, seq_per_group, n, k):
    """With x_row_map = identity permutation of a permuted source x', the
    kernel should produce the same result as feeding the un-permuted x."""
    x_fp8, w_fp8, sfx_packed, sfw_packed, seqlens, cu_seqlens = _build_inputs(
        num_group, seq_per_group, n, k
    )
    device = x_fp8.device
    m_total = x_fp8.shape[0]

    # Reference output (no permutation).
    y_ref = hpc.group_gemm_cp_async_mxfp8(
        x_fp8,
        w_fp8,
        sfx_packed,
        sfw_packed,
        seqlens,
        cu_seqlens,
        num_seq_per_group_avg=seq_per_group,
        x_row_map=None,
    )

    # Permute x rows; row_map = inverse permutation so post[i] = src[row_map[i]].
    perm = torch.randperm(m_total, device=device, dtype=torch.int64)
    x_permuted = x_fp8[perm].contiguous()  # x_permuted[i] = x[perm[i]]
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(m_total, device=device, dtype=torch.int64)
    # We want post[i] (i.e. the original i-th row) = x_permuted[inv_perm[i]],
    # so row_map[i] = inv_perm[i].
    row_map = inv_perm.to(torch.int32).contiguous()

    y_perm = hpc.group_gemm_cp_async_mxfp8(
        x_permuted,
        w_fp8,
        sfx_packed,
        sfw_packed,
        seqlens,
        cu_seqlens,
        num_seq_per_group_avg=seq_per_group,
        x_row_map=row_map,
    )

    diff = (y_ref.to(torch.float32) - y_perm.to(torch.float32)).abs()
    print(
        f"[ROW_MAP g={num_group} seq={seq_per_group} n={n} k={k}] "
        f"max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.6f}"
    )
    assert allclose(y_ref.to(torch.float32), y_perm.to(torch.float32), rtol=0.08, atol=1)


if __name__ == "__main__":
    test_cp_async_matches_tma(192, 8, 768, 4096)
    test_cp_async_with_row_map(192, 8, 768, 4096)
