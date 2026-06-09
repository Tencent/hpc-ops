import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import pytest
from utils import allclose

SF_VEC = 32

# e2m1 decode LUT (sign|exp2|mantissa1): index = 4bit nibble
#   [0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6]
_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
# Reverse: map fp32 values to the nearest e2m1 nibble (only used to pack integer weights to fp4)
_E2M1_VALUES = _E2M1_LUT.tolist()


def _mxfp8_kTileM(avg: int, n: int) -> int:
    use_2sm = (n % 256 == 0) and (avg > 32)
    if use_2sm:
        for ktm in (64, 96, 128, 160, 192):
            if avg <= ktm:
                return ktm
        return 256
    for ktm in (16, 32, 48, 64, 128):
        if avg <= ktm:
            return ktm
    return 256


def quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-32K block mxfp8 quant of fp32 tensor (mirror act_mul_and_mxfp8_quant)."""
    *batch, k = x.shape
    assert k % SF_VEC == 0
    blocks = x.view(*batch, k // SF_VEC, SF_VEC)
    absmax = blocks.abs().amax(dim=-1)

    bits = absmax.float().view(torch.int32)
    exp_biased = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    sf_bits = exp_biased - 8 + (mant > 0x600000).int()
    sf_bits = sf_bits.clamp(0, 255)
    sf_bits = torch.where(absmax == 0, torch.zeros_like(sf_bits), sf_bits)
    sf_bits = sf_bits.to(torch.uint8)

    sf_val = torch.where(
        sf_bits == 0,
        torch.ones_like(absmax),
        torch.exp2(sf_bits.float() - 127.0),
    ).unsqueeze(-1)
    quant = blocks / sf_val
    quant = torch.where(sf_bits.unsqueeze(-1) == 0, torch.zeros_like(quant), quant)

    fp8 = quant.to(torch.float8_e4m3fn).view(*batch, k)
    return fp8, sf_bits


def dequant_mxfp8(x_fp8: torch.Tensor, sf_u8: torch.Tensor) -> torch.Tensor:
    *batch, k = x_fp8.shape
    x = x_fp8.to(torch.float32).view(*batch, k // SF_VEC, SF_VEC)
    sf = torch.where(
        sf_u8 == 0,
        torch.zeros_like(sf_u8, dtype=torch.float32),
        torch.exp2(sf_u8.float() - 127.0),
    ).unsqueeze(-1)
    return (x * sf).view(*batch, k)


def unpack_e2m1(w_packed: torch.Tensor, k: int) -> torch.Tensor:
    """[*, k/2] uint8 packed → [*, k] float32 (low nibble first)."""
    lut = _E2M1_LUT.to(w_packed.device)
    low = (w_packed & 0x0F).long()
    high = ((w_packed >> 4) & 0x0F).long()
    out = torch.stack([lut[low], lut[high]], dim=-1)
    return out.reshape(*w_packed.shape[:-1], k)


def dequant_mxfp4(w_packed: torch.Tensor, sf_u8: torch.Tensor, k: int) -> torch.Tensor:
    *batch, _ = w_packed.shape
    w = unpack_e2m1(w_packed, k)
    sf = torch.where(
        sf_u8 == 0,
        torch.zeros_like(sf_u8, dtype=torch.float32),
        torch.exp2((sf_u8.to(torch.float32) - 127.0)),
    )
    w = w.view(*batch, k // SF_VEC, SF_VEC)
    sf = sf.view(*batch, k // SF_VEC, 1)
    return (w * sf).view(*batch, k)


def pack_e2m1(values: torch.Tensor) -> torch.Tensor:
    """[*, k] fp32 (values must be e2m1-representable) → [*, k/2] uint8 (low nibble first)."""
    lut = _E2M1_LUT.to(values.device)
    # nearest-nibble: find the closest LUT entry for each value
    flat = values.reshape(-1, 1)  # [N,1]
    idx = (flat - lut.view(1, -1)).abs().argmin(dim=-1).to(torch.uint8)
    idx = idx.reshape(values.shape)
    low = idx[..., 0::2]
    high = idx[..., 1::2]
    return ((high << 4) | low).contiguous()


def naive_fuse_moe(
    x_fp8,
    x_scale,
    gate_up_w_packed,
    gate_up_w_scale,
    down_w_packed,
    down_w_scale,
    topk_ids,
    topk_scale,
    rank_ep,
    num_expert_local,
    num_expert_total,
    hidden,
    intermediate,
):
    num_seq = x_fp8.shape[0]
    num_topk = topk_ids.shape[1]
    half = intermediate // 2
    start_e = rank_ep * num_expert_local
    end_e = (rank_ep + 1) * num_expert_local

    out = torch.zeros((num_seq, hidden), dtype=torch.float32, device=x_fp8.device)
    for i in range(num_seq):
        x_dq = dequant_mxfp8(x_fp8[i : i + 1], x_scale[i : i + 1]).squeeze(0)
        for k in range(num_topk):
            ge = int(topk_ids[i, k].item())
            if ge < start_e or ge >= end_e:
                continue
            le = ge - start_e
            scale = float(topk_scale[i, k].item())

            w_gu_dq = dequant_mxfp4(
                gate_up_w_packed[le], gate_up_w_scale[le], hidden
            )  # [interm,hidden]
            y_gu = x_dq @ w_gu_dq.t()
            gate, up = y_gu[:half], y_gu[half:]
            y_act = torch.nn.functional.silu(gate) * up

            yq, sf = quantize_mxfp8(y_act)
            y_act_dq = dequant_mxfp8(yq, sf)

            w_d_dq = dequant_mxfp4(down_w_packed[le], down_w_scale[le], half)  # [hidden, half]
            y_d = y_act_dq @ w_d_dq.t()

            out[i] += scale * y_d
    return out.to(torch.bfloat16)


def _prepack_weight_scale(sfw, avg):
    _, sfw_packed = hpc.prepack_mxfp8_scale(None, sfw, None, num_seq_per_group_avg=avg)
    return sfw_packed


def _run_one(
    num_seq, num_topk, num_expert_local, hidden, intermediate, rank_ep=0, num_expert_total=None
):
    if num_expert_total is None:
        num_expert_total = num_expert_local
    torch.manual_seed(2026)
    device = torch.device("cuda:0")
    half = intermediate // 2

    # X: fp8
    x_f32 = torch.randint(-2, 3, (num_seq, hidden), device=device, dtype=torch.float32)
    x_fp8 = x_f32.to(torch.float8_e4m3fn).contiguous()
    x_scale = torch.full((num_seq, hidden // SF_VEC), 127, dtype=torch.uint8, device=device)

    # Weights: small e2m1-representable integers, then pack to fp4 (uint8, K/2)
    gate_up_w_f32 = torch.randint(
        -2, 3, (num_expert_local, intermediate, hidden), device=device, dtype=torch.float32
    )
    down_w_f32 = torch.randint(
        -2, 3, (num_expert_local, hidden, half), device=device, dtype=torch.float32
    )
    gate_up_w_packed = pack_e2m1(gate_up_w_f32)  # [El, interm, hidden/2] uint8
    down_w_packed = pack_e2m1(down_w_f32)  # [El, hidden, half/2] uint8

    gate_up_w_scale = torch.full(
        (num_expert_local, intermediate, hidden // SF_VEC), 127, dtype=torch.uint8, device=device
    )
    down_w_scale = torch.full(
        (num_expert_local, hidden, half // SF_VEC), 127, dtype=torch.uint8, device=device
    )

    start_e = rank_ep * num_expert_local
    topk_ids = torch.randint(
        start_e, start_e + num_expert_local, (num_seq, num_topk), device=device, dtype=torch.int32
    )
    topk_scale = torch.rand((num_seq, num_topk), device=device, dtype=torch.float32) * 0.5 + 0.5

    avg = (num_seq * num_topk) // num_expert_total
    gate_up_w_scale_packed = _prepack_weight_scale(gate_up_w_scale, avg)
    down_w_scale_packed = _prepack_weight_scale(down_w_scale, avg)

    y = hpc.fuse_moe_mxfp8(
        x_fp8,
        x_scale,
        gate_up_w_packed,
        gate_up_w_scale_packed,
        down_w_packed,
        down_w_scale_packed,
        topk_ids,
        topk_scale,
        rank_ep=rank_ep,
        num_expert_total=num_expert_total,
    )

    gt = naive_fuse_moe(
        x_fp8,
        x_scale,
        gate_up_w_packed,
        gate_up_w_scale,
        down_w_packed,
        down_w_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_local,
        num_expert_total,
        hidden,
        intermediate,
    )

    abs_diff = (gt.to(torch.float32) - y.to(torch.float32)).abs()
    peak = gt.to(torch.float32).abs().max().item()
    max_err = abs_diff.max().item()
    rel_err = max_err / max(peak, 1e-3)
    print(
        f"[N={num_seq} topk={num_topk} El={num_expert_local} "
        f"H={hidden} I={intermediate} ep={rank_ep}/{num_expert_total} "
        f"avg={avg} kTileM={_mxfp8_kTileM(avg, intermediate)}] "
        f"peak={peak:.1f} max_err={max_err:.2f} rel_err(peak)={rel_err:.4f}"
    )
    assert allclose(
        gt.to(torch.float32), y.to(torch.float32), rtol=0.02, atol=max(peak * 0.02, 1.0)
    )


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 10, reason="skip on non sm100!")
@pytest.mark.parametrize(
    "num_seq,num_topk,num_expert,hidden,intermediate",
    [
        (16, 4, 8, 512, 1024),
        (32, 4, 8, 1024, 2048),
        (64, 8, 16, 1024, 2048),
    ],
)
def test_fuse_moe_mxfp8_fp4_basic(num_seq, num_topk, num_expert, hidden, intermediate):
    _run_one(num_seq, num_topk, num_expert, hidden, intermediate)


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 10, reason="skip on non sm100!")
@pytest.mark.parametrize("rank_ep,num_expert_total", [(0, 16), (1, 16), (2, 16)])
def test_fuse_moe_mxfp8_fp4_ep(rank_ep, num_expert_total):
    num_expert_local = num_expert_total // 4
    _run_one(
        num_seq=32,
        num_topk=4,
        num_expert_local=num_expert_local,
        hidden=512,
        intermediate=1024,
        rank_ep=rank_ep,
        num_expert_total=num_expert_total,
    )
