import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import math

import torch
import torch.nn.functional as F

import hpc
from utils import allclose

# Set random seed for reproducibility
torch.manual_seed(41)
torch.cuda.manual_seed(41)


def ref_attn_with_paged_kvcache_func(
    q, k, v, kvcache, block_ids, nblocks, seqlenq, cu_seqlenq, num_seq_kvcache
):

    num_batch = seqlenq.shape[0]
    num_head_q = q.shape[1]
    num_head_kv = k.shape[1]
    head_dim = v.shape[2]
    block_size = kvcache.shape[2]
    head_per_group = num_head_q // num_head_kv
    q = q.reshape(num_batch, -1, num_head_q, head_dim)
    output = torch.empty_like(q)
    for bi in range(num_batch):
        q_batch = q[bi].transpose(0, 1).float()  # [num_heads, sq, head_dim]
        blk_ids = block_ids[bi, : nblocks[bi]]
        seqlen = seqlenq[bi] + num_seq_kvcache[bi]
        k_batch = (
            kvcache[blk_ids, 0, :, :, :]
            .reshape(-1, num_head_kv, head_dim)
            .transpose(0, 1)[:, :seqlen, :]
            .repeat_interleave(head_per_group, dim=0)
        ).float()
        v_batch = (
            kvcache[blk_ids, 1, :, :, :]
            .reshape(-1, num_head_kv, head_dim)
            .transpose(0, 1)[:, :seqlen, :]
            .repeat_interleave(head_per_group, dim=0)
        ).float()

        p = q_batch @ k_batch.transpose(-1, -2)
        p = p / math.sqrt(head_dim)
        causal_mask = torch.ones(
            seqlenq[bi], seqlen - seqlenq[bi], device=q.device, dtype=torch.bool
        )
        tail_causal_mask = torch.tril(
            torch.ones(seqlenq[bi], seqlenq[bi], device=q.device, dtype=torch.bool)
        )
        causal_mask = torch.cat([causal_mask, tail_causal_mask], dim=-1).unsqueeze(0)

        p = p.masked_fill(~causal_mask, float("-inf"))
        attn_weights = F.softmax(p, dim=-1)
        y = torch.matmul(attn_weights, v_batch)
        output[bi] = y.transpose(0, 1)

    return output.reshape(-1, num_head_q, head_dim)


@pytest.mark.parametrize("num_batch", [1, 16, 200])
@pytest.mark.parametrize("num_seq_q", [1])
@pytest.mark.parametrize("max_seq_kv", [1024, 4096])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("num_head_q", [4, 8])
@pytest.mark.parametrize("num_head_kv", [1])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("new_kv_included", [True, False])
@pytest.mark.parametrize("splitk", [True, False])
def test_attention_decode_bf16(
    num_batch,
    num_seq_q,
    max_seq_kv,
    block_size,
    num_head_q,
    num_head_kv,
    head_dim,
    new_kv_included,
    splitk,
):

    num_dim_qk = head_dim
    num_dim_v = head_dim
    max_num_blocks = int(num_batch * max_seq_kv // block_size * 1.2)

    q = torch.randn(
        (num_batch * num_seq_q, num_head_q, num_dim_qk), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(num_dim_qk)
    k = torch.randn(
        (num_batch * num_seq_q, num_head_kv, num_dim_qk), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(num_dim_qk)
    v = torch.randn(
        (num_batch * num_seq_q, num_head_kv, num_dim_v), dtype=torch.bfloat16, device="cuda"
    )

    num_seq_kvcache = (
        torch.randint(1, max_seq_kv, (num_batch,), dtype=torch.int32, device="cuda") * 0
        + max_seq_kv
    )
    nblocks = (num_seq_kvcache + num_seq_q + block_size - 1) // block_size
    total_blocks = sum(nblocks)
    kvcache = torch.randn(
        max_num_blocks, 2, block_size, num_head_kv, num_dim_qk, dtype=torch.bfloat16, device="cuda"
    )
    packed_block_ids = torch.randperm(max_num_blocks)[:total_blocks].to(torch.int32).cuda()

    max_num_block2 = max(nblocks)
    block_ids = torch.empty(num_batch, max_num_block2, dtype=torch.int32, device="cuda")
    seqlenq = torch.tensor([num_seq_q] * num_batch, dtype=torch.int32, device="cuda")
    cu_seqlenq = torch.cumsum(seqlenq, dtype=torch.int32, dim=0)

    cu_blocks = 0
    for i in range(num_batch):
        block_ids[i, : nblocks[i]] = packed_block_ids[cu_blocks : cu_blocks + nblocks[i]]
        cu_blocks += nblocks[i]
        for sqi in range(seqlenq[i]):
            si = sqi + num_seq_kvcache[i]
            blk_id = si // block_size
            slot_id = si % block_size
            kvcache[block_ids[i, blk_id], 0, slot_id] = k.reshape(
                num_batch, num_seq_q, num_head_kv, num_dim_qk
            )[i, sqi]
            kvcache[block_ids[i, blk_id], 1, slot_id] = v.reshape(
                num_batch, num_seq_q, num_head_kv, num_dim_qk
            )[i, sqi]

    gt = ref_attn_with_paged_kvcache_func(
        q, k, v, kvcache, block_ids, nblocks, seqlenq, cu_seqlenq, num_seq_kvcache
    )

    my = hpc.attention_decode_bf16(
        q,
        kvcache[:, 0, :, :, :],
        kvcache[:, 1, :, :, :],
        block_ids,
        num_seq_kvcache + 1 if new_kv_included else num_seq_kvcache,
        new_kv_included=new_kv_included,
        splitk=splitk,
    )

    assert allclose(gt, my, atol=0.016)
