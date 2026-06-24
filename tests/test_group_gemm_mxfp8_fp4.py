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
from utils import allclose, mxfp8_dispatch_kTileM

SF_VEC = 32

# e2m1 decode LUT (sign|exp2|mantissa1): index = 4bit nibble
#   [0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6]
_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def dequant_mxfp8(x_fp8: torch.Tensor, sf_u8: torch.Tensor) -> torch.Tensor:
    *batch, k = x_fp8.shape
    assert k % SF_VEC == 0
    x = x_fp8.to(torch.float32)
    sf = torch.where(
        sf_u8 == 0,
        torch.zeros_like(sf_u8, dtype=torch.float32),
        torch.exp2((sf_u8.to(torch.float32) - 127.0)),
    )
    x = x.view(*batch, k // SF_VEC, SF_VEC)
    sf = sf.view(*batch, k // SF_VEC, 1)
    return (x * sf).view(*batch, k)


def unpack_e2m1(w_packed: torch.Tensor, k: int) -> torch.Tensor:
    """[*, k/2] uint8 packed → [*, k] float32."""
    lut = _E2M1_LUT.to(w_packed.device)
    low = (w_packed & 0x0F).long()
    high = ((w_packed >> 4) & 0x0F).long()
    low_f = lut[low]  # [*, k/2]
    high_f = lut[high]  # [*, k/2]
    out = torch.stack([low_f, high_f], dim=-1)  # [*, k/2, 2]
    return out.reshape(*w_packed.shape[:-1], k)


def dequant_mxfp4(w_packed: torch.Tensor, sf_u8: torch.Tensor, k: int) -> torch.Tensor:
    *batch, _ = w_packed.shape
    w = unpack_e2m1(w_packed, k)  # [*, k] float32
    sf = torch.where(
        sf_u8 == 0,
        torch.zeros_like(sf_u8, dtype=torch.float32),
        torch.exp2((sf_u8.to(torch.float32) - 127.0)),
    )
    w = w.view(*batch, k // SF_VEC, SF_VEC)
    sf = sf.view(*batch, k // SF_VEC, 1)
    return (w * sf).view(*batch, k)


def naive_group_gemm_mxfp8_mxfp4(
    x_fp8: torch.Tensor,
    w_fp4_packed: torch.Tensor,
    sfx_u8: torch.Tensor,
    sfw_u8: torch.Tensor,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    k: int,
) -> torch.Tensor:
    m_total = x_fp8.shape[0]
    num_group, n, _ = w_fp4_packed.shape
    y = torch.zeros((m_total, n), dtype=torch.float32, device=x_fp8.device)
    for g in range(num_group):
        s = int(cu_seqlens[g].item())
        e = s + int(seqlens[g].item())
        if e == s:
            continue
        x_g = dequant_mxfp8(x_fp8[s:e], sfx_u8[s:e])  # [seq, k] fp32
        w_g = dequant_mxfp4(w_fp4_packed[g], sfw_u8[g], k)  # [n, k]  fp32
        y[s:e] = torch.matmul(x_g, w_g.t())
    return y.to(torch.bfloat16)


def _run_one(num_group: int, seq_per_group: int, n: int, k: int):
    torch.manual_seed(2026)
    device = torch.device("cuda:0")

    seqlens = torch.full((num_group,), seq_per_group, dtype=torch.int32, device=device)
    cu_seqlens = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(seqlens, 0).to(torch.int32)]
    )
    m_total = int(cu_seqlens[-1].item())

    # X: fp8
    x_f32 = torch.randint(-4, 5, (m_total, k), device=device, dtype=torch.float32)
    x_fp8 = x_f32.to(torch.float8_e4m3fn).contiguous()

    # W: fp4 packed [num_group, n, k/2]
    lo = torch.randint(0, 16, (num_group, n, k // 2), device=device, dtype=torch.uint8)
    hi = torch.randint(0, 16, (num_group, n, k // 2), device=device, dtype=torch.uint8)
    w_fp4_packed = ((hi << 4) | lo).contiguous()

    # scale: ue8m0
    sfx = torch.randint(124, 131, (m_total, k // SF_VEC), dtype=torch.uint8, device=device)
    sfw = torch.randint(124, 131, (num_group, n, k // SF_VEC), dtype=torch.uint8, device=device)

    kTileM = mxfp8_dispatch_kTileM(seq_per_group)

    sfx_packed, sfw_packed = hpc.prepack_mxfp8_scale(
        sfx, sfw, cu_seqlens, num_seq_per_group_avg=seq_per_group
    )

    y = hpc.group_gemm_mxfp8(
        x_fp8,
        w_fp4_packed,
        sfx_packed,
        sfw_packed,
        seqlens,
        cu_seqlens,
        num_seq_per_group_avg=seq_per_group,
    )

    gt = naive_group_gemm_mxfp8_mxfp4(x_fp8, w_fp4_packed, sfx, sfw, seqlens, cu_seqlens, k)
    print(gt)
    print(y)
    abs_diff = (gt.to(torch.float32) - y.to(torch.float32)).abs()
    print(
        f"[g={num_group} seq={seq_per_group} kTileM={kTileM} n={n} k={k}] "
        f"max_err={abs_diff.max().item():.4f} mean_err={abs_diff.mean().item():.6f}"
    )
    assert allclose(gt.to(torch.float32), y.to(torch.float32), rtol=0.08, atol=2.0)


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 10, reason="skip on non sm100!")
@pytest.mark.parametrize(
    "num_group, seq_per_group, n, k",
    [
        # tp8: gate_up
        (192, 4, 512, 6144),
        (192, 8, 512, 6144),
        (192, 16, 512, 6144),
        (192, 32, 512, 6144),
        (192, 64, 512, 6144),
        (192, 128, 512, 6144),
        (192, 256, 512, 6144),
        # tp8: down
        (192, 4, 6144, 256),
        (192, 8, 6144, 256),
        (192, 16, 6144, 256),
        (192, 32, 6144, 256),
        (192, 64, 6144, 256),
        (192, 128, 6144, 256),
        (192, 256, 6144, 256),
    ],
)
def test_group_gemm_mxfp8_mxfp4_(num_group, seq_per_group, n, k):
    _run_one(num_group, seq_per_group, n, k)
