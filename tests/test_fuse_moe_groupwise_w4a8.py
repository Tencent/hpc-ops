import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))


sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))
import os

build_paths = list(Path(__file__).parent.glob("../build/lib.*/"))
if not build_paths:
    raise FileNotFoundError("No build/lib.* directory found in parent directory")
sys.path.insert(0, os.path.realpath(build_paths[0]))

import hpc
import torch
from torch import Tensor
import math
import pytest
from pathlib import Path
from utils import allclose


def naive_gather_expert_inputs(x, topk_ids, num_expert, rank_ep):
    num_tokens, num_topk = topk_ids.shape
    num_seq, hidden_size = x.shape

    unique_values, num_tokens_per_expert_partial = torch.unique(
        topk_ids.flatten(), return_counts=True, sorted=True
    )
    start_expert = rank_ep * num_expert
    end_expert = (rank_ep + 1) * num_expert
    mask = (unique_values >= start_expert) & (unique_values < end_expert)
    unique_values = unique_values[mask]
    num_tokens_per_expert_partial = num_tokens_per_expert_partial[mask]
    num_tokens_per_expert = torch.full([num_expert], 0, dtype=torch.int32, device="cuda")

    for i in range(unique_values.numel()):
        num_tokens_per_expert[unique_values[i] - start_expert] = num_tokens_per_expert_partial[i]

    cu_num_tokens_per_expert = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), num_tokens_per_expert]),
        dim=0,
    ).to(torch.int32)

    y = torch.zeros((num_tokens * num_topk, hidden_size), dtype=x.dtype, device=x.device)
    token_pos = torch.zeros((num_tokens, num_topk), dtype=torch.int32, device=x.device)
    token_pos.fill_(-1)

    print(f"num_tokens_per_expert: {num_tokens_per_expert}")
    print(f"cu_num_tokens_per_expert: {cu_num_tokens_per_expert}")

    # reset
    num_tokens_per_expert.fill_(0)

    for idx, iexpert in enumerate(topk_ids.flatten()):
        itoken = idx // num_topk
        icol = idx % num_topk
        if iexpert >= start_expert and iexpert < end_expert:
            pos = (
                cu_num_tokens_per_expert[iexpert - start_expert]
                + num_tokens_per_expert[iexpert - start_expert]
            )
            y[pos] = x[itoken]
            token_pos[itoken, icol] = pos.item()
            num_tokens_per_expert[iexpert - start_expert] += 1

    return (
        y,
        token_pos,
        num_tokens_per_expert,
        cu_num_tokens_per_expert,
        unique_values,
    )


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


def naive_act_mul_and_quant(gate_up, scale_inv):

    def silu(x):
        return x / (1 + (-x).exp())

        return x / (1 + (-x).exp())

        import torch.nn.functional as F

        return F.silu(x)

    gate, up = torch.chunk(gate_up.float(), 2, dim=1)
    out = (silu(gate) * up) * scale_inv
    outfp8 = out.to(torch.float8_e4m3fn)
    return outfp8


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

    num_blocks = C // 64
    assert C % 64 == 0, "C must be divisible by 64"
    x_blocks = x.reshape(N, num_blocks, 64)
    x_had = torch.zeros_like(x_blocks)
    for b in range(num_blocks):
        x_had[:, b, :] = hadamard_transform_ref(x_blocks[:, b, :].to(torch.bfloat16)).float()

    # Per-tensor quant: multiply by scale_inv, cast to fp8
    x_quant_f = x_had * scale_inv
    # Simulate fp8 cast: clamp to fp8 range and round
    x_quant_f = x_quant_f.clamp(-448.0, 448.0)

    return x_quant_f.to(torch.float8_e4m3fn).reshape(N, C)


def naive_reduce(x_bf16, topk_pos, topk_scale, shared_output=None):
    num_seq, num_topk = topk_pos.shape
    total_num_seq, hidden_size = x_bf16.shape

    y_bf16 = torch.zeros((num_seq, hidden_size), dtype=torch.bfloat16, device=x_bf16.device)
    for i in range(num_seq):
        acc = torch.zeros((1, hidden_size), dtype=torch.float, device="cuda")
        cur_topk_pos = topk_pos[i]
        cur_topk_scale = topk_scale[i]
        for j, pos in enumerate(cur_topk_pos):
            if pos >= 0:
                acc += x_bf16[pos].float() * cur_topk_scale[j].float()
        if shared_output is not None:
            acc += shared_output[i].float()
        y_bf16[i] = acc.to(torch.bfloat16)

    return y_bf16


def naive_fuse_moe_groupwise_w4a8(
    x,
    gate_up_weight,
    down_weight,
    gate_up_scale,
    down_scale,
    act_and_mul_scale,
    topk_ids,
    topk_scale,
    rank_ep,
    gateup_group_size,
    down_group_size,
    use_hadamard,
    shared_output=None,
):
    num_expert = gate_up_weight.size(0)
    # count_and_gather
    gate_up_input, topk_pos, seqlens, cu_seqlens, expert_ids = naive_gather_expert_inputs(
        x, topk_ids, num_expert, rank_ep
    )

    # gate_up_proj
    gate_up_output = naive_group_gemm_blockwise_w4a8(
        gate_up_input, gate_up_weight, seqlens, cu_seqlens, gate_up_scale, gateup_group_size
    )

    # act_and_mul
    if use_hadamard:
        down_input = act_mul_hadamard_per_tensor_quant_ref(gate_up_output, act_and_mul_scale)
    else:
        down_input = naive_act_mul_and_quant(gate_up_output, act_and_mul_scale)

    # down_proj
    down_output = naive_group_gemm_blockwise_w4a8(
        down_input, down_weight, seqlens, cu_seqlens, down_scale, down_group_size
    )

    # reduce
    y = naive_reduce(down_output, topk_pos, topk_scale, shared_output)

    return topk_pos, seqlens, cu_seqlens, gate_up_input, gate_up_output, down_input, down_output, y


def half_fuse_moe_groupwise_w4a8(
    x,
    gate_up_weight,
    down_weight,
    gate_up_scale,
    down_scale,
    act_and_mul_scale,
    topk_ids,
    topk_scale,
    rank_ep,
    gateup_group_size,
    down_group_size,
    use_hadamard,
    shared_output=None,
):
    num_expert = gate_up_weight.size(0)
    # count_and_gather
    gate_up_input, topk_pos, seqlens, cu_seqlens, expert_ids = naive_gather_expert_inputs(
        x, topk_ids, num_expert, rank_ep
    )

    # gate_up_proj
    gate_up_output = hpc.group_gemm_groupwise_w4a8_mma(
        gate_up_input, gate_up_weight, seqlens, cu_seqlens, gate_up_scale, gateup_group_size
    )

    # act_and_mul
    if use_hadamard:
        down_input = act_mul_hadamard_per_tensor_quant_ref(gate_up_output, act_and_mul_scale)
    else:
        down_input = naive_act_mul_and_quant(gate_up_output, act_and_mul_scale)

    # down_proj
    down_output = hpc.group_gemm_groupwise_w4a8_mma(
        down_input, down_weight, seqlens, cu_seqlens, down_scale, down_group_size
    )

    # reduce
    y = naive_reduce(down_output, topk_pos, topk_scale, shared_output)

    return topk_pos, seqlens, cu_seqlens, gate_up_input, gate_up_output, down_input, down_output, y


@pytest.mark.parametrize("num_seq", [128])
@pytest.mark.parametrize("num_topk", [8])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("intermediate_size", [192])
@pytest.mark.parametrize("num_expert", [192])
@pytest.mark.parametrize("gateup_group_size", [128])
@pytest.mark.parametrize("down_group_size", [64])
@pytest.mark.parametrize("rank_ep", [0, 1])
@pytest.mark.parametrize("size_ep", [1, 4, 8])
@pytest.mark.parametrize("use_hadamard", [False, True])
@pytest.mark.parametrize("has_shared_output", [False, True])
@pytest.mark.parametrize("use_output", [False, True])
def test_fuse_moe_groupwise_w4a8(
    num_seq,
    num_topk,
    hidden_size,
    intermediate_size,
    num_expert,
    gateup_group_size,
    down_group_size,
    rank_ep,
    size_ep,
    use_hadamard,
    has_shared_output,
    use_output,
):
    torch.manual_seed(0)
    dtype = torch.float8_e4m3fn

    topk_ids = torch.randint(0, num_expert, (num_seq, num_topk), dtype=torch.int32, device="cuda")
    topk_ids, _ = torch.sort(topk_ids, dim=1)
    print(topk_ids)

    x = (torch.randn((num_seq, hidden_size), dtype=torch.float, device="cuda")).to(dtype)
    gate_up_weight = torch.randint(
        -128,
        127,
        (num_expert // size_ep, intermediate_size * 2, hidden_size // 2),
        dtype=torch.int8,
        device="cuda",
    )
    down_weight = torch.randint(
        -128,
        127,
        (num_expert // size_ep, hidden_size, intermediate_size // 2),
        dtype=torch.int8,
        device="cuda",
    )

    affine_coeff = 0.005
    gate_up_scale = (
        torch.randn(
            (num_expert // size_ep, intermediate_size * 2, hidden_size // gateup_group_size),
            dtype=torch.float,
            device="cuda",
        ).to(torch.bfloat16)
        * affine_coeff
    )
    down_scale = (
        torch.randn(
            (num_expert // size_ep, hidden_size, intermediate_size // down_group_size),
            dtype=torch.float,
            device="cuda",
        ).to(torch.bfloat16)
        * affine_coeff
    )
    act_and_mul_scale = torch.randn((1), dtype=torch.float, device="cuda") * 0.02
    topk_scale = torch.randn((num_seq, num_topk), dtype=torch.float, device="cuda") / num_topk
    if has_shared_output:
        shared_output = torch.randn((num_seq, hidden_size), dtype=torch.bfloat16, device="cuda")
    else:
        shared_output = None

    gate_up_weight_refomat, gate_up_scale_reformat = (
        hpc.group_gemm_groupwise_w4a8_mma_weight_reformat(
            gate_up_weight, gate_up_scale, gateup_group_size
        )
    )
    down_weight_refomat, down_scale_reformat = hpc.group_gemm_groupwise_w4a8_mma_weight_reformat(
        down_weight, down_scale, down_group_size
    )

    if use_output:
        output = torch.empty_like(x, dtype=torch.bfloat16)
        hpc.fuse_moe_groupwise_w4a8(
            x,
            gate_up_weight_refomat,
            gate_up_scale_reformat,
            down_weight_refomat,
            down_scale_reformat,
            act_and_mul_scale,
            topk_ids,
            topk_scale,
            gateup_group_size,
            down_group_size,
            rank_ep,
            num_expert,
            use_hadamard,
            shared_output=shared_output,
            output=output,
        )
        my = output
    else:
        # (
        #     topk_pos_my,
        #     seqlens_my,
        #     cu_seqlens_my,
        #     gate_up_input_my,
        #     gate_up_output_my,
        #     down_input_my,
        #     down_output_my,
        # )
        my = hpc.fuse_moe_groupwise_w4a8(
            x,
            gate_up_weight_refomat,
            gate_up_scale_reformat,
            down_weight_refomat,
            down_scale_reformat,
            act_and_mul_scale,
            topk_ids,
            topk_scale,
            gateup_group_size,
            down_group_size,
            rank_ep,
            num_expert,
            use_hadamard,
            shared_output=shared_output,
        )

        # topk_pos_my, seqlens_my, cu_seqlens_my, gate_up_input_my, gate_up_output_my, down_input_my, down_output_my, my = half_fuse_moe_groupwise_w4a8(
        #     x,
        #     gate_up_weight_refomat,
        #     down_weight_refomat,
        #     gate_up_scale_reformat,
        #     down_scale_reformat,
        #     act_and_mul_scale,
        #     topk_ids,
        #     topk_scale,
        #     rank_ep,
        #     gateup_group_size,
        #     down_group_size,
        #     use_hadamard,
        #     shared_output=shared_output,
        # )
    (
        topk_pos_gt,
        seqlens_gt,
        cu_seqlens_gt,
        gate_up_input_gt,
        gate_up_output_gt,
        down_input_gt,
        down_output_gt,
        gt,
    ) = naive_fuse_moe_groupwise_w4a8(
        x,
        gate_up_weight,
        down_weight,
        gate_up_scale,
        down_scale,
        act_and_mul_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        gateup_group_size,
        down_group_size,
        use_hadamard,
        shared_output=shared_output,
    )

    # assert allclose(topk_pos_gt, topk_pos_my)
    # assert allclose(seqlens_gt, seqlens_my)
    # assert allclose(cu_seqlens_gt, cu_seqlens_my)
    # assert allclose(gate_up_input_gt, gate_up_input_my)
    # allclose(gate_up_output_gt, gate_up_output_my)
    # allclose(down_input_gt, down_input_my)
    # allclose(down_output_gt, down_output_my)

    print("gt")
    print(gt)

    print("my")
    print(my)

    assert allclose(gt, my, rtol=0.05, atol=0.01)
