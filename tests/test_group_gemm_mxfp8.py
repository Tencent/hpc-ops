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
from utils import allclose

SF_VEC = 32


def _mxfp8_kTileM(num_seq_per_group_avg: int, n: int) -> int:
    """Mirror C++ `mxfp8_dispatch_kTileM(avg, n)` in src/group_gemm/sm100/group_gemm.h."""
    use_2sm = (n % 256 == 0) and (num_seq_per_group_avg > 32)
    if use_2sm:
        for ktm in (64, 96, 128, 160, 192):
            if num_seq_per_group_avg <= ktm:
                return ktm
        return 256
    for ktm in (16, 32, 48, 64, 128):
        if num_seq_per_group_avg <= ktm:
            return ktm
    return 256


def dequant_mxfp8(x_fp8: torch.Tensor, sf_u8: torch.Tensor) -> torch.Tensor:
    *batch, k = x_fp8.shape
    assert k % SF_VEC == 0
    x = x_fp8.to(torch.float32)
    # UE8M0: value = 2^(bits - 127). bits=0 → subnormal/denorm; treat as 0.
    sf = torch.where(
        sf_u8 == 0,
        torch.zeros_like(sf_u8, dtype=torch.float32),
        torch.exp2((sf_u8.to(torch.float32) - 127.0)),
    )
    # broadcast sf over the last 32 elements
    x = x.view(*batch, k // SF_VEC, SF_VEC)
    sf = sf.view(*batch, k // SF_VEC, 1)
    return (x * sf).view(*batch, k)


def naive_group_gemm_mxfp8(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    sfx_u8: torch.Tensor,
    sfw_u8: torch.Tensor,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    m_total, k = x_fp8.shape
    num_group, n, _ = w_fp8.shape
    y = torch.zeros((m_total, n), dtype=torch.float32, device=x_fp8.device)
    for g in range(num_group):
        s = int(cu_seqlens[g].item())
        e = s + int(seqlens[g].item())
        if e == s:
            continue
        x_g = dequant_mxfp8(x_fp8[s:e], sfx_u8[s:e])  # fp32
        w_g = dequant_mxfp8(w_fp8[g], sfw_u8[g])  # fp32
        y[s:e] = torch.matmul(x_g, w_g.t())  # fp32 matmul
    return y.to(torch.bfloat16)


def _run_one(num_group: int, seq_per_group: int, n: int, k: int):
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

    kTileM = _mxfp8_kTileM(seq_per_group, n)

    sfx_packed, sfw_packed = hpc.prepack_mxfp8_scale(
        sfx, sfw, cu_seqlens, num_seq_per_group_avg=seq_per_group
    )

    for i in range(20):
        y = hpc.group_gemm_mxfp8(
            x_fp8,
            w_fp8,
            sfx_packed,
            sfw_packed,
            seqlens,
            cu_seqlens,
            num_seq_per_group_avg=seq_per_group,
        )

    gt = naive_group_gemm_mxfp8(x_fp8, w_fp8, sfx, sfw, seqlens, cu_seqlens)

    # Tolerance: mxfp8 has more numeric noise; use rtol=0.08 + atol=2.0 (8x int range)
    print(gt)
    print(y)
    abs_diff = (gt.to(torch.float32) - y.to(torch.float32)).abs()
    print(
        f"[g={num_group} seq={seq_per_group} kTileM={kTileM} n={n} k={k}] "
        f"max_err={abs_diff.max().item():.4f} mean_err={abs_diff.mean().item():.6f}"
    )
    assert allclose(gt.to(torch.float32), y.to(torch.float32), rtol=0.08, atol=2.0)


# hy3.0
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 10, reason="skip on non sm100!")
@pytest.mark.parametrize(
    "num_group, seq_per_group, n, k",
    [
        # tp8: gate_up
        (192, 32, 384, 4096),
        (192, 64, 384, 4096),
        # tp8: down
        (192, 8, 4096, 192),
        (192, 16, 4096, 192),
        (192, 128, 4096, 192),
        (192, 256, 4096, 192),
    ],
)
def test_group_gemm_mxfp8_(num_group, seq_per_group, n, k):
    _run_one(num_group, seq_per_group, n, k)
