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


def naive_attn_with_paged_kvcache_func(Q, KV, kvcache, block_ids, nblocks, cu_seqlenq, num_seq_kv):

    num_batch = cu_seqlenq.shape[0] - 1
    num_head_q = Q.shape[1]
    head_dim = Q.shape[2]
    block_size = kvcache.shape[1]
    seqlenq = cu_seqlenq[1:] - cu_seqlenq[:-1]
    output = torch.empty_like(Q)
    for bi in range(num_batch):
        BQ = Q[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlenq[bi]].permute(1, 0, 2).float()
        blk_ids = block_ids[bi, : nblocks[bi]]
        seqlen = num_seq_kv[bi]

        BKV = (
            (
                kvcache[blk_ids, :, :]
                .reshape(-1, head_dim)
                .transpose(0, 1)[:, :seqlen]
                .unsqueeze(0)
                .repeat_interleave(num_head_q, dim=0)
            )
            .float()
            .reshape(num_head_q, head_dim, -1)
        )

        P = BQ @ BKV
        P = P / math.sqrt(head_dim)
        causal_mask = torch.ones(
            seqlenq[bi], seqlen - seqlenq[bi], device=Q.device, dtype=torch.bool
        )
        tail_causal_mask = torch.tril(
            torch.ones(seqlenq[bi], seqlenq[bi], device=Q.device, dtype=torch.bool)
        )
        causal_mask = torch.cat([causal_mask, tail_causal_mask], dim=-1).unsqueeze(0)

        P = P.masked_fill(~causal_mask, float("-inf"))
        attn_weights = F.softmax(P, dim=-1)
        # attn_weights = P
        Y = torch.matmul(attn_weights, BKV.transpose(-1, -2))
        output[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlenq[bi]] = Y.transpose(0, 1)

    return output.reshape(-1, num_head_q, head_dim)


@pytest.mark.parametrize("num_batch", [1])
@pytest.mark.parametrize("max_seq_q", [1024])
@pytest.mark.parametrize("max_seq_kv", [1024])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("num_head_q", [64])
@pytest.mark.parametrize("head_dim", [512])
def test_attention_decode_bf16(
    num_batch,
    max_seq_q,
    max_seq_kv,
    block_size,
    num_head_q,
    head_dim,
):
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    max_num_blocks = int(num_batch * (max_seq_kv + max_seq_q) // block_size * 1.2)

    num_seq_q = torch.randint(1, max_seq_q + 1, (num_batch,), dtype=torch.int32, device="cuda")
    num_seq_kvcache = torch.randint(
        1, max_seq_kv + 1, (num_batch,), dtype=torch.int32, device="cuda"
    )
    num_seq_kv = num_seq_q + num_seq_kvcache
    num_seq_kv_ratio = num_seq_kv

    total_seq_q = sum(num_seq_q)

    Q = torch.randn(
        (total_seq_q, num_head_q, head_dim), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(head_dim)
    KV = torch.randn((total_seq_q, head_dim), dtype=torch.bfloat16, device="cuda") / math.sqrt(
        head_dim
    )

    nblocks = (num_seq_kv_ratio + block_size - 1) // block_size
    total_blocks = sum(nblocks)
    kvcache = torch.randn(
        max_num_blocks, block_size, head_dim, dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(head_dim)
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

    gt = naive_attn_with_paged_kvcache_func(
        Q, KV, kvcache, block_ids, nblocks, cu_seqlenq, num_seq_kv
    )

    my = hpc.attention_mla_with_kvcache_bf16(
        Q,
        kvcache,
        block_ids,
        cu_seqlenq,
        num_seq_kv,
    )

    print("\ngt\n")
    print(gt[0, :, :32])
    print("\nmy\n")
    print(my[0, :, :32])

    assert allclose(gt, my, atol=0.016)
