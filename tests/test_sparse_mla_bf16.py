import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
import torch.nn.functional as F
from utils import allclose


def log2sumexp2(a: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(a * math.log(2), dim=dim) * math.log2(math.e)


def manual_softmax(x, sink, dim=-1):
    max_vals = torch.max(x, dim=dim).values
    x = torch.concat([x, sink.view(-1, 1)], dim=-1)

    x_stable = x - max_vals.view(-1, 1)
    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    softmax_output = exp_x / sum_exp

    return softmax_output[:, :-1]


def gt_index_attention_func(
    q,
    win_kvcache,
    win_block_ids,
    win_topk_ids,
    compress_kvcache,
    compress_block_ids,
    compress_topk_ids,
    cu_seqlens_q,
    sink_weight,
    sm_scale,
):

    num_batch, num_seq_q, num_head_q, num_dim_qk = q.shape
    num_win_blocks, block_size, num_head_kv, _ = win_kvcache.shape
    num_compress_blocks, block_size, num_head_kv, _ = compress_kvcache.shape

    output = torch.empty_like(q).reshape(-1, num_head_q, num_dim_qk)

    max_num_win_kv = num_win_blocks * block_size
    max_num_compress_kv = num_compress_blocks * block_size

    for itoken in range(num_batch * num_seq_q):
        ibatch = itoken // num_seq_q
        ires = itoken % num_seq_q

        Q = q[ibatch, ires, :, :]

        win_topk_block_ids = win_topk_ids[itoken] // block_size
        win_topk_block_res = win_topk_ids[itoken] % block_size
        win_topk_block_ids_true = win_block_ids[ibatch, win_topk_block_ids]
        win_topk_index_true = win_topk_block_ids_true * block_size + win_topk_block_res
        WK = win_kvcache.reshape(-1, num_head_kv, num_dim_qk)[win_topk_index_true, 0, :]

        compress_topk_block_ids = compress_topk_ids[itoken] // block_size
        compress_topk_block_res = compress_topk_ids[itoken] % block_size
        compress_topk_block_ids_true = compress_block_ids[ibatch, compress_topk_block_ids]
        compress_topk_index_true = (
            compress_topk_block_ids_true * block_size + compress_topk_block_res
        )
        CK = compress_kvcache.reshape(-1, num_head_kv, num_dim_qk)[compress_topk_index_true, 0, :]

        invalid_win_indices_mask = (win_topk_ids[itoken] < 0) | (
            win_topk_ids[itoken] >= max_num_win_kv
        )
        invalid_compress_indices_mask = (compress_topk_ids[itoken] < 0) | (
            compress_topk_ids[itoken] >= max_num_compress_kv
        )

        K = torch.cat([WK, CK], dim=0).float()
        invalid_indices_mask = torch.cat(
            [invalid_win_indices_mask, invalid_compress_indices_mask], dim=-1
        )
        scores = torch.matmul(Q.float(), K.T) * sm_scale  # / math.sqrt(num_dim_qk)
        scores[:, invalid_indices_mask] = float("-inf")

        """
        ntiles = scores.shape[1] // 64
        gMax = torch.empty(scores.shape[0], device='cuda') * 0 + float("-inf")
        gSum = torch.empty(scores.shape[0], device='cuda') * 0
        tile_out = torch.empty_like(Q) * 0
        klog2e = 1.442695040888963

        for itile in range(0, ntiles):
            tile_score = scores[:, itile * 64: (itile + 1) * 64]
            last_max = gMax

            gMax = torch.max(gMax, tile_score.max(dim=-1)[0] * sm_scale * klog2e)

            tile_score = torch.exp2(tile_score * sm_scale * klog2e - gMax[:, None])
            scale = torch.exp2(last_max - gMax)

            gSum = gSum * scale + tile_score.sum(dim=-1)

            tile_out *= scale[:, None]
            tile_out += torch.matmul(tile_score, K[itile * 64: (itile + 1) * 64, :])

        tile_out /= gSum[:, None]
        output[itoken] = tile_out
        """

        # attn_weights = F.softmax(scores, dim=-1).to(torch.bfloat16).float()
        attn_weights = manual_softmax(scores, sink_weight, dim=-1).to(torch.bfloat16).float()
        # attn_weights = scores.to(torch.bfloat16)
        # import pdb;pdb.set_trace()

        output[itoken] = torch.matmul(attn_weights, K)

        # print("compress_topk_block_ids:", compress_topk_block_ids, compress_topk_block_ids.dtype)
        # print("win_topk_block_ids:", win_topk_block_ids)
        # print("win_topk_block_res:", win_topk_block_res)
        # print("win_topk_block_ids_true:", win_topk_block_ids_true)
        # print("win_topk_index_true:", win_topk_index_true)
        # print("WK:", WK)
        # print("CK:", CK)
        # print("invalid_indices_mask:", invalid_indices_mask.shape, invalid_indices_mask.dtype)
        # print("invalid_win_indices_mask:", invalid_win_indices_mask)

    return output


@pytest.mark.parametrize("num_batch", [1])
@pytest.mark.parametrize("num_seq_q", [1000])
@pytest.mark.parametrize("max_num_seq_kv", [1024])
@pytest.mark.parametrize("num_seq_win_kv", [128])
@pytest.mark.parametrize("num_seq_compress_kv", [512])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("num_head_q", [64])
@pytest.mark.parametrize("num_head_kv", [1])
@pytest.mark.parametrize("num_dim_qk", [512])
@pytest.mark.parametrize("num_dim_v", [512])
@pytest.mark.parametrize("use_output", [False])
def test_index_attention_with_kvcache_prefill_bf16(
    num_batch,
    num_seq_q,
    max_num_seq_kv,
    num_seq_win_kv,
    num_seq_compress_kv,
    block_size,
    num_head_q,
    num_head_kv,
    num_dim_qk,
    num_dim_v,
    use_output,
):
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    Q = torch.randn(
        (num_batch, num_seq_q, num_head_q, num_dim_qk),
        dtype=torch.bfloat16,
        device="cuda",
    ) / math.sqrt(num_dim_qk)
    sink_weight = torch.randn((num_head_q), dtype=torch.float32, device="cuda") * 0

    seqlens_q = torch.full((num_batch,), num_seq_q, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens_q]), dim=0
    ).to(torch.int32)

    max_num_blocks = num_batch * (max_num_seq_kv + block_size - 1) // block_size * 2
    kvcache_blocks = torch.full(
        (num_batch,), max_num_seq_kv // block_size, dtype=torch.int32, device="cuda"
    )
    total_kvcache_blocks = sum(kvcache_blocks)
    max_kvcache_blocks = max(kvcache_blocks)

    kvcache = torch.randn(
        max_num_blocks, block_size, num_head_kv, num_dim_qk, dtype=torch.bfloat16, device="cuda"
    )
    packed_block_ids = torch.randperm(max_num_blocks)[:total_kvcache_blocks].to(torch.int32).cuda()
    print("max_num_blocks:", max_num_blocks)

    win_topk_ids = torch.randint(
        -1, 100, (num_batch * num_seq_q, num_seq_win_kv), device="cuda"
    ).to(torch.int32)
    compress_topk_ids = torch.randint(
        -1, num_seq_compress_kv, (num_batch * num_seq_q, num_seq_compress_kv), device="cuda"
    ).to(torch.int32)

    cu_blocks = 0
    block_ids = torch.empty(num_batch, max_kvcache_blocks, dtype=torch.int32, device="cuda")

    for i in range(num_batch):
        block_ids[i, : kvcache_blocks[i]] = packed_block_ids[
            cu_blocks : cu_blocks + kvcache_blocks[i]
        ]
        cu_blocks += kvcache_blocks[i]

    for i in range(1):
        gt = gt_index_attention_func(
            q=Q,
            win_kvcache=kvcache,
            win_block_ids=block_ids,
            win_topk_ids=win_topk_ids,
            compress_kvcache=kvcache,
            compress_block_ids=block_ids,
            compress_topk_ids=compress_topk_ids,
            cu_seqlens_q=cu_seqlens_q,
            sink_weight=sink_weight,
            sm_scale=0.5,
        )
        my = hpc.sparse_mla_with_kvcache_bf16(
            q=Q.reshape(-1, num_head_q, num_dim_qk),
            win_kvcache=kvcache,
            win_block_ids=block_ids,
            win_topk_ids=win_topk_ids,
            compress_kvcache=kvcache,
            compress_block_ids=block_ids,
            compress_topk_ids=compress_topk_ids,
            cu_seqlens_q=cu_seqlens_q,
            sink_weight=sink_weight,
            softmax_scale=0.5,
        )

    print("\ngt\n")
    print(gt[0, :, :])
    print("\nmy\n")
    print(my[0, :, :])

    assert allclose(gt, my, atol=0.05)
