import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import pytest
from utils import allclose, mxfp8_dispatch_kTileM

SF_VEC = 32
FP8_MAX = 448.0


def quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-32K block mxfp8 quant of fp32 tensor.

    Mirrors the GPU formula in act_mul_and_mxfp8_quant:
      sf_bits = exp_biased(absmax) - 8 + (1 if mant > 0x600000 else 0), clamp [0,255]
    Returns (fp8_values [..., K] e4m3, sf_bits [..., K/32] uint8).
    """
    *batch, k = x.shape
    assert k % SF_VEC == 0
    blocks = x.view(*batch, k // SF_VEC, SF_VEC)
    absmax = blocks.abs().amax(dim=-1)  # [..., K/32]

    # bit-pattern math
    bits = absmax.float().view(torch.int32)
    exp_biased = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    sf_bits = exp_biased - 8 + (mant > 0x600000).int()
    sf_bits = sf_bits.clamp(0, 255)
    sf_bits = torch.where(absmax == 0, torch.zeros_like(sf_bits), sf_bits)
    sf_bits = sf_bits.to(torch.uint8)

    # SF value for division
    sf_val = torch.where(
        sf_bits == 0,
        torch.ones_like(absmax),  # avoids /0; we'll mask result to 0 below
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


def naive_fuse_moe_mxfp8(
    x_fp8,
    x_scale,
    gate_up_w,
    gate_up_w_scale,
    down_w,
    down_w_scale,
    topk_ids,
    topk_scale,
    rank_ep,
    num_expert_local,
    num_expert_total,
    shared_output=None,
):
    num_seq, hidden = x_fp8.shape
    num_topk = topk_ids.shape[1]
    intermediate = gate_up_w.shape[1]
    half = intermediate // 2
    start_e = rank_ep * num_expert_local
    end_e = (rank_ep + 1) * num_expert_local

    out = torch.zeros((num_seq, hidden), dtype=torch.float32, device=x_fp8.device)
    for i in range(num_seq):
        x_dq = dequant_mxfp8(x_fp8[i : i + 1], x_scale[i : i + 1]).squeeze(0)  # [hidden] fp32
        for k in range(num_topk):
            ge = int(topk_ids[i, k].item())
            if ge < start_e or ge >= end_e:
                continue
            le = ge - start_e
            scale = float(topk_scale[i, k].item())

            # gateup GEMM
            w_gu_dq = dequant_mxfp8(gate_up_w[le], gate_up_w_scale[le])  # [interm, hidden]
            y_gu = x_dq @ w_gu_dq.t()  # [interm]
            gate, up = y_gu[:half], y_gu[half:]
            y_act = torch.nn.functional.silu(gate) * up  # [half] fp32

            # mxfp8 quant + dequant (introduces quant noise consistent with kernel)
            yq, sf = quantize_mxfp8(y_act)
            y_act_dq = dequant_mxfp8(yq, sf)

            # down GEMM
            w_d_dq = dequant_mxfp8(down_w[le], down_w_scale[le])  # [hidden, half]
            y_d = y_act_dq @ w_d_dq.t()  # [hidden]

            out[i] += scale * y_d

    if shared_output is not None:
        out += shared_output.float()
    return out.to(torch.bfloat16)


def _prepack_weight_scale(sfw: torch.Tensor) -> torch.Tensor:
    """Offline weight-side SF prepack."""
    _, sfw_packed = hpc.prepack_mxfp8_scale(
        None,
        sfw,
        None,
    )
    return sfw_packed


def interleave_n16(w):
    """Interleave first/second half every 16 in dim1.
    (E, 2*inter, K) → interleaved [gate16, up16, gate16, up16, ...]
    """
    E, N, K = w.shape
    half = N // 2
    gate = w[:, :half, :].reshape(E, half // 16, 16, K)
    up = w[:, half:, :].reshape(E, half // 16, 16, K)
    return torch.stack([gate, up], dim=2).reshape(E, N, K)


def _run_one(
    num_seq,
    num_topk,
    num_expert_local,
    hidden,
    intermediate,
    rank_ep=0,
    num_expert_total=None,
    fuse_act=False,
):
    if num_expert_total is None:
        num_expert_total = num_expert_local
    torch.manual_seed(2026)
    device = torch.device("cuda:0")

    # Generate small int values to keep mxfp8 quantization noise bounded.
    x_f32 = torch.randint(-2, 3, (num_seq, hidden), device=device, dtype=torch.float32) / 100
    gate_up_w_f32 = torch.randint(
        -2, 3, (num_expert_local, intermediate, hidden), device=device, dtype=torch.float32
    )
    down_w_f32 = torch.randint(
        -2, 3, (num_expert_local, hidden, intermediate // 2), device=device, dtype=torch.float32
    )

    x_fp8 = x_f32.to(torch.float8_e4m3fn).contiguous()
    gate_up_w_fp8 = gate_up_w_f32.to(torch.float8_e4m3fn).contiguous()
    down_w_fp8 = down_w_f32.to(torch.float8_e4m3fn).contiguous()

    x_scale = torch.full((num_seq, hidden // SF_VEC), 127, dtype=torch.uint8, device=device)
    gate_up_w_scale = torch.full(
        (num_expert_local, intermediate, hidden // SF_VEC), 127, dtype=torch.uint8, device=device
    )
    down_w_scale = torch.full(
        (num_expert_local, hidden, (intermediate // 2) // SF_VEC),
        127,
        dtype=torch.uint8,
        device=device,
    )

    if fuse_act:
        gate_up_w_fp8_interleave = (
            interleave_n16(gate_up_w_f32).to(torch.float8_e4m3fn).contiguous()
        )
        gate_up_w_scale_interleave = interleave_n16(gate_up_w_scale)

    # routing: distribute uniformly over local experts (within EP range)
    start_e = rank_ep * num_expert_local
    topk_ids = torch.randint(
        start_e, start_e + num_expert_local, (num_seq, num_topk), device=device, dtype=torch.int32
    )
    topk_scale = torch.rand((num_seq, num_topk), device=device, dtype=torch.float32) * 0.5 + 0.5

    avg = (num_seq * num_topk) // num_expert_total
    gate_up_w_scale_packed = _prepack_weight_scale(gate_up_w_scale)
    if fuse_act:
        gate_up_w_scale_packed_interleave = _prepack_weight_scale(gate_up_w_scale_interleave)
    down_w_scale_packed = _prepack_weight_scale(down_w_scale)

    y = hpc.fuse_moe_mxfp8(
        x_fp8,
        x_scale,
        gate_up_w_fp8_interleave if fuse_act else gate_up_w_fp8,
        gate_up_w_scale_packed_interleave if fuse_act else gate_up_w_scale_packed,
        down_w_fp8,
        down_w_scale_packed,
        topk_ids,
        topk_scale,
        rank_ep=rank_ep,
        num_expert_total=num_expert_total,
    )

    gt = naive_fuse_moe_mxfp8(
        x_fp8,
        x_scale,
        gate_up_w_fp8,
        gate_up_w_scale,
        down_w_fp8,
        down_w_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_local,
        num_expert_total,
    )

    abs_diff = (gt.to(torch.float32) - y.to(torch.float32)).abs()
    peak = gt.to(torch.float32).abs().max().item()
    max_err = abs_diff.max().item()
    rel_err = max_err / max(peak, 1e-3)
    print(
        f"[N={num_seq} topk={num_topk} El={num_expert_local} "
        f"H={hidden} I={intermediate} ep={rank_ep}/{num_expert_total} "
        f"avg={avg} kTileM={mxfp8_dispatch_kTileM(avg)}] "
        f"peak={peak:.1f} max_err={max_err:.2f} mean_err={abs_diff.mean().item():.4f} "
        f"rel_err(peak)={rel_err:.4f}"
    )
    # mxfp8 + bf16 cumulative noise: use allclose with tolerances matching peak-relative bound.
    assert allclose(
        gt.to(torch.float32), y.to(torch.float32), rtol=0.02, atol=max(peak * 0.02, 1.0)
    )


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 10, reason="skip on non sm100!")
@pytest.mark.parametrize(
    "num_seq,num_topk,num_expert,hidden,intermediate",
    [
        (1, 8, 24, 6144, 2048),
        (32, 8, 24, 6144, 2048),
        (64, 8, 24, 6144, 2048),
        (128, 8, 24, 6144, 2048),
        (256, 8, 24, 6144, 2048),
        (512, 8, 24, 6144, 2048),
        (1024, 8, 24, 6144, 2048),
    ],
)
def test_fuse_moe_mxfp8_basic(num_seq, num_topk, num_expert, hidden, intermediate):
    _run_one(num_seq, num_topk, num_expert, hidden, intermediate)


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 10, reason="skip on non sm100!")
@pytest.mark.parametrize("rank_ep,num_expert_total", [(0, 16), (1, 16), (2, 16)])
def test_fuse_moe_mxfp8_ep(rank_ep, num_expert_total):
    """EP: weights are local (4 experts per rank) but topk_ids reference global ids."""
    num_expert_local = num_expert_total // 4
    _run_one(
        num_seq=32,
        num_topk=4,
        num_expert_local=num_expert_local,
        hidden=6144,
        intermediate=2048,
        rank_ep=rank_ep,
        num_expert_total=num_expert_total,
    )
