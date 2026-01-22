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


def gt_mqa_indexer_logits_func(
    Q,
    KV,
    weight,
    kvcache,
    block_ids,
    nblocks,
    seqlenq,
    cu_seqlenq,
    num_seq_kv,
    ratio,
    topk,
    max_context_len,
):

    num_batch = seqlenq.shape[0]
    total_seq_q = Q.shape[0]
    num_head_q = Q.shape[1]
    head_dim = Q.shape[2]
    num_seq_kv_ratio = num_seq_kv // ratio
    output = torch.empty(total_seq_q, max_context_len, dtype=torch.float32, device="cuda")
    output.fill_(float("-inf"))

    for bi in range(num_batch):
        BQ = Q[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlenq[bi]].permute(1, 0, 2).float()
        blk_ids = block_ids[bi, : nblocks[bi]]
        BKV = (
            (
                kvcache[blk_ids, :, :]
                .reshape(-1, head_dim)
                .transpose(0, 1)[:, : num_seq_kv_ratio[bi]]
                .unsqueeze(0)
                .repeat_interleave(num_head_q, dim=0)
            )
            .float()
            .reshape(num_head_q, head_dim, -1)
        )

        BW = (
            weight[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlenq[bi]]
            .transpose(0, 1)[:, :, None]
            .float()
        )

        P = BQ @ BKV

        P = F.relu(P) * BW
        P = P.sum(0)
        mask = (
            torch.arange(num_seq_kv[bi] // ratio).repeat(num_seq_kv[bi], 1)
            >= torch.arange(1, num_seq_kv[bi] + 1).unsqueeze(1) // ratio
        )
        mask = torch.where(mask, float("-inf"), 0)
        mask = mask[-seqlenq[bi] :].cuda()
        P = P + mask

        # topk_idx = torch.topk(P, min(topk, num_seq_kv_ratio[bi]), dim=-1)[1]
        # output[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlenq[bi], : topk_idx.shape[1]] = topk_idx
        output[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlenq[bi], : P.shape[1]] = P

    return output


@pytest.mark.parametrize("num_batch", [1, 10, 20, 40, 60, 80])
@pytest.mark.parametrize("max_seq_q", [1, 2, 2 * 1024])
@pytest.mark.parametrize("max_seq_kv", [2 * 1024])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("num_head_q", [64])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("topk", [512])
@pytest.mark.parametrize("ratio", [4])
@pytest.mark.parametrize("skip_block_ids", [0, 2])
@pytest.mark.parametrize("max_context_len", [32 * 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_mqa_indexer_topk(
    num_batch,
    max_seq_q,
    max_seq_kv,
    block_size,
    num_head_q,
    topk,
    head_dim,
    ratio,
    skip_block_ids,
    max_context_len,
    dtype,
):
    if max_seq_q >= 2 * 1024 and num_batch >= 20:
        return

    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    max_num_blocks = int(num_batch * max_seq_kv // block_size * 1.2)
    num_seq_q = torch.randint(1, max_seq_q + 1, (num_batch,), dtype=torch.int32, device="cuda")
    num_seq_kvcache = torch.randint(
        1, max_seq_kv + 1, (num_batch,), dtype=torch.int32, device="cuda"
    )
    num_seq_kv = num_seq_q + num_seq_kvcache
    num_seq_kv_ratio = num_seq_kv // ratio

    total_seq_q = sum(num_seq_q)

    T = torch.bfloat16

    Q = (
        torch.randn((total_seq_q, num_head_q, head_dim), dtype=T, device="cuda")
        / math.sqrt(head_dim)
    ).to(dtype)
    KV = (torch.randn((total_seq_q, head_dim), dtype=T, device="cuda") / math.sqrt(head_dim)).to(
        dtype
    )

    W = torch.randn((total_seq_q, num_head_q), dtype=T, device="cuda")

    nblocks = (num_seq_kv_ratio + block_size - 1) // block_size
    total_blocks = sum(nblocks)
    kvcache = torch.randn(max_num_blocks, block_size, head_dim, dtype=T, device="cuda").to(dtype)
    packed_block_ids = torch.randperm(max_num_blocks)[:total_blocks].to(torch.int32).cuda()

    max_nblocks = max(nblocks)
    block_ids = torch.empty(num_batch, max_nblocks, dtype=torch.int32, device="cuda")

    cu_blocks = 0
    for i in range(num_batch):
        block_ids[i, : nblocks[i]] = packed_block_ids[cu_blocks : cu_blocks + nblocks[i]]
        cu_blocks += nblocks[i]

    cu_seqlenq = torch.cumsum(num_seq_q, dtype=torch.int32, dim=0)
    cu_seqlenq = torch.concat(
        [torch.tensor([0], dtype=torch.int32, device="cuda"), cu_seqlenq], dim=0
    )
    """
    gt = gt_mqa_indexer_topk_func(
        Q, KV, W, kvcache, block_ids, nblocks, num_seq_q, cu_seqlenq, num_seq_kv, ratio, topk
    )
    """
    block_ids = F.pad(block_ids, (skip_block_ids, 0), "constant", 0)
    gt = gt_mqa_indexer_logits_func(
        Q,
        KV,
        W,
        kvcache,
        block_ids[:, skip_block_ids:],
        nblocks,
        num_seq_q,
        cu_seqlenq,
        num_seq_kv,
        ratio,
        topk,
        max_context_len,
    )

    print(gt)

    my = torch.empty(Q.shape[0], max_context_len, dtype=torch.float32, device="cuda")

    my.fill_(float("-inf"))

    print(num_seq_kv)

    hpc.mqa_indexer_logits(
        Q,
        kvcache,
        W,
        block_ids[:, skip_block_ids:],
        cu_seqlenq,
        num_seq_kv,
        ratio,
        max_context_len,
        output=my,
    )

    print("\ngt\n")
    print(gt[:5, :10])
    print("\nmy\n")
    print(my[:5, :10])

    assert allclose(gt, my, atol=0.016)
