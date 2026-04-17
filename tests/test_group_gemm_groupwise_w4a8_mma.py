import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
from torch import Tensor
import math
import pytest
from utils import allclose


def break_int4_bytes_to_int8(packed):
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], packed.shape[1], -1)


def dequantize_int4_to_dtype(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
    weight_scale_2: torch.Tensor = None,
) -> torch.Tensor:
    # unpack: [E, N, K//2] -> [E, N, K]
    unpacked = break_int4_bytes_to_int8(packed_weight)
    scale_expanded = weight_scale.repeat_interleave(group_size, dim=-1)
    dequant = unpacked.float() * scale_expanded.float()
    if weight_scale_2 is not None:
        dequant = dequant / weight_scale_2.float()
    return dequant.to(dtype)


def naive_group_gemm_blockwise_w4a8(
    x: Tensor,
    weight: Tensor,
    seqlens: Tensor,
    cu_seqlens: Tensor,
    y_scale: Tensor,
    group_size: int,
):
    m = x.shape[0]
    num_group = weight.shape[0]
    n = weight.shape[1]
    k_half = weight.shape[2]

    x = x.to(torch.float16)
    unpack_weight = dequantize_int4_to_dtype(weight, y_scale, group_size, torch.float16)
    y = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    for i in range(num_group):
        start_idx = int(cu_seqlens[i].item())
        end_idx = int(start_idx + seqlens[i].item())  # cu_seqlens[i + 1].item()
        if seqlens[i].item() == 0:
            continue

        x_group = x[start_idx:end_idx]
        w_group = unpack_weight[i]

        y[start_idx:end_idx] = (torch.mm(x_group, w_group.t()).float()).to(torch.bfloat16)

    return y


@pytest.mark.parametrize(
    "actual_m",
    [1, 2, 3, 4, 8, 16],
)
@pytest.mark.parametrize(
    "m",
    [
        17,
    ],
)
@pytest.mark.parametrize(
    "num_group_n_k",
    [
        [192, 512, 4096],  # hy3.0 tp8 gate up, k=384 pad to k=512
        [192, 4096, 256],  # hy3.0 tp8 down, k=192 pad to k=256
        [96, 768, 4096],  # hy3.0 tp4ep2 gate up
        [96, 4096, 384],  # hy3.0 tp4ep2 down
    ],
)
@pytest.mark.parametrize(
    "group_size",
    [64, 128],
)
def test_group_gemm(actual_m, m, num_group_n_k, group_size):
    num_group, n, k = num_group_n_k

    assert k % 128 == 0

    torch.cuda.manual_seed(10086)

    total_seq_pad = m * num_group
    x = (torch.randn((total_seq_pad, k), dtype=torch.float, device="cuda") / 10).to(
        torch.float8_e4m3fn
    )
    w = torch.randint(-128, 127, (num_group, n, k // 2), dtype=torch.int8, device="cuda")
    wscale = torch.randn((num_group, n, k // group_size), dtype=torch.float, device="cuda").to(
        torch.bfloat16
    )

    # for debug

    # x = (torch.ones((total_seq_pad, k), dtype=torch.float, device="cuda")).to(
    #     torch.float8_e4m3fn
    # )
    # w = torch.ones((num_group, n, k // 2), dtype=torch.int8, device="cuda")
    # org_list = [
    #     0b00000000,
    #     0b00010001,
    #     0b00100010,
    #     0b00110011,
    #     0b01000100,
    #     0b01010101,
    #     0b01100110,
    #     0b01110111,
    # ]  # [0, 17, 34, 51, 68, 85, 102, 119]
    # reformat_list = [
    #     0b00010000,
    #     0b00110010,
    #     0b00010000,
    #     0b00110010,
    #     0b01010100,
    #     0b01110110,
    #     0b01010100,
    #     0b01110110,
    # ]  # [16, 50, 16, 50, 84, 118, 84, 118]

    # org_list = [0b10001000, 0b10011001, 0b10101010, 0b10111011, 0b11001100, 0b11011101, 0b11101110, 0b11111111] # [-120, -103,  -86,  -69,  -52,  -35,  -18,   -1]
    # reformat_list = [0b10011000, 0b10111010, 0b10011000, 0b10111010, 0b11011100, 0b11111110, 0b11011100, 0b11111110] # [-104,  -70, -104,  -70,  -36,   -2,  -36,   -2]

    # w = torch.tensor(
    #     [ org_list for row in range(n) for col in range(k // 16)],
    #     dtype=torch.uint8,
    #     device="cuda",
    # ).reshape(1, n, k // 2).repeat((num_group, 1, 1)).view(torch.int8)

    # wscale = torch.ones((num_group, n, k // 128), dtype=torch.float, device="cuda").to(
    #     torch.bfloat16
    # )

    # x = (
    #     torch.tensor(
    #         [
    #             [2**col + row * 2**col / 2] * 16
    #             for row in range(total_seq_pad)
    #             for col in range(k // 16)
    #         ],
    #         dtype=torch.float,
    #         device="cuda",
    #     )
    #     .reshape(m, k)
    #     .to(torch.float8_e4m3fn)
    # )
    # w = (
    #     torch.tensor(
    #         [[(row * k // 16 + col) % 128] * 8 for row in range(n) for col in range(k // 16)],
    #         dtype=torch.int8,
    #         device="cuda",
    #     )
    #     .reshape(1, n, k // 2)
    #     .repeat((num_group, 1, 1))
    # )
    # wscale = (
    #     torch.tensor([[row] * (k // 128) for row in range(n)], dtype=torch.float, device="cuda")
    #     .to(torch.bfloat16)
    #     .reshape(1, n, k // 128)
    #     .repeat((num_group, 1, 1))
    # )

    print(f"x: {x}")
    print(f"w: {w}")
    print(f"wscale: {wscale}")

    seqlens = torch.full((num_group,), actual_m, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.cumsum(
        torch.tensor([0] + [m] * num_group, dtype=torch.int32, device="cuda"), dim=0
    ).to(torch.int32)

    for _ in range(1):
        gt = naive_group_gemm_blockwise_w4a8(x, w, seqlens, cu_seqlens, wscale, group_size)
        w_refomat, wscale_reformat = hpc.group_gemm_groupwise_w4a8_mma_weight_reformat(
            w, wscale, group_size
        )
        print(f"reformat w: {w_refomat}")
        print(f"reformat wscale: {wscale_reformat}")
        my = hpc.group_gemm_groupwise_w4a8_mma(
            x, w_refomat, seqlens, cu_seqlens, wscale_reformat, group_size
        )

    for i in range(num_group):
        start = int(cu_seqlens[i].item() + seqlens[i].item())
        end = (i + 1) * m
        gt[start:end, :] = 0
        my[start:end, :] = 0

    print(f"gt: {gt}")
    print(f"my: {my}")

    assert allclose(gt, my, atol=0.6, rtol=1e-2)
