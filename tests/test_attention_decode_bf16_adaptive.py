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


def report_cos_atol(gt, my, atol=0.1, cos_threshold=0.999, tag=""):
    gtf = gt.float().reshape(-1)
    myf = my.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(gtf, myf, dim=0).item()
    max_abs = (gtf - myf).abs().max().item()
    mean_abs = (gtf - myf).abs().mean().item()
    print(f"[cos] {tag} cos={cos:.6f} max_abs={max_abs:.5f} mean_abs={mean_abs:.6f}")
    assert cos > cos_threshold, f"{tag} cosine {cos:.6f} < {cos_threshold}"
    assert allclose(gt, my, atol=atol), f"{tag} max_abs={max_abs:.5f} > atol={atol}"


def flash_attn_with_kvcache_func(
    Q, K, V, kvcache, block_ids, nblocks, seqlenq, cu_seqlenq, num_seq_kvcache
):
    # from flash_attn.flash_attn_interface import flash_attn_with_kvcache
    from flash_attn_interface import flash_attn_with_kvcache

    num_batch = seqlenq.shape[0]
    num_head_q = Q.shape[1]
    head_dim = K.shape[2]
    Q = Q.reshape(num_batch, -1, num_head_q, head_dim)

    for _ in range(20):
        output = flash_attn_with_kvcache(
            q=Q,
            k_cache=kvcache[:, 0, :, :].contiguous(),
            v_cache=kvcache[:, 1, :, :].contiguous(),
            cache_seqlens=num_seq_kvcache + 1,
            # block_table=block_ids,
            page_table=block_ids,
            causal=True,
        )

    return output.reshape(-1, num_head_q, head_dim)


def naive_attn_with_paged_kvcache_func(
    Q, K, V, kvcache, block_ids, nblocks, seqlenq, cu_seqlenq, num_seq_kvcache
):

    num_batch = seqlenq.shape[0]
    num_head_q = Q.shape[1]
    num_head_kv = K.shape[1]
    head_dim = K.shape[2]
    block_size = kvcache.shape[2]
    head_per_group = num_head_q // num_head_kv
    Q = Q.reshape(num_batch, -1, num_head_q, head_dim)
    output = torch.empty_like(Q)
    for bi in range(num_batch):
        BQ = Q[bi].transpose(0, 1).float()  # [num_heads, sq, head_dim]
        blk_ids = block_ids[bi, : nblocks[bi]]
        seqlen = seqlenq[bi] + num_seq_kvcache[bi]
        BK = (
            kvcache[blk_ids, 0, :, :, :]
            .reshape(-1, num_head_kv, head_dim)
            .transpose(0, 1)[:, :seqlen, :]
            .repeat_interleave(head_per_group, dim=0)
        ).float()
        BV = (
            kvcache[blk_ids, 1, :, :, :]
            .reshape(-1, num_head_kv, head_dim)
            .transpose(0, 1)[:, :seqlen, :]
            .repeat_interleave(head_per_group, dim=0)
        ).float()
        P = BQ @ BK.transpose(-1, -2)
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
        Y = torch.matmul(attn_weights, BV)
        output[bi] = Y.transpose(0, 1)

    return output.reshape(-1, num_head_q, head_dim)


try:
    # fa2
    # from flash_attn.flash_attn_interface import flash_attn_with_kvcache
    # fa3
    from flash_attn_interface import flash_attn_with_kvcache

    # gt_attention_func = flash_attn_with_kvcache_func
    gt_attention_func = naive_attn_with_paged_kvcache_func
except Exception as e:
    print(f"execute naive_attn_func: {e}")
    gt_attention_func = naive_attn_with_paged_kvcache_func


def attention_decode_bf16_test_func(
    num_batch,
    num_seq_q,
    max_seq_kv,
    block_size,
    kv_head_q_head,
    head_dim,
    new_kv_included,
    use_output,
    kv_lens=None,
):
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    num_head_kv, num_head_q = kv_head_q_head

    num_dim_qk = head_dim
    num_dim_v = head_dim
    max_num_blocks = int(num_batch * max_seq_kv // block_size * 1.2)

    T = torch.bfloat16

    Q = torch.randn(
        (num_batch * num_seq_q, num_head_q, num_dim_qk), dtype=T, device="cuda"
    ) / math.sqrt(num_dim_qk)
    K = torch.randn(
        (num_batch * num_seq_q, num_head_kv, num_dim_qk), dtype=T, device="cuda"
    ) / math.sqrt(num_dim_qk)
    V = torch.randn((num_batch * num_seq_q, num_head_kv, num_dim_v), dtype=T, device="cuda")

    if kv_lens is not None:
        num_seq_kvcache = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    else:
        num_seq_kvcache = (
            torch.randint(1, max_seq_kv, (num_batch,), dtype=torch.int32, device="cuda") * 0
            + max_seq_kv
        )

    task_map = hpc.get_attention_decode_task_workspace_adaptive(
        num_batch, max_seq_kv + num_seq_q, num_head_kv, min_process_len=1024
    )
    hpc.assign_attention_decode_task_adaptive(
        num_seq_kvcache + num_seq_q if new_kv_included else num_seq_kvcache,
        task_map,
        num_head_kv,
        num_seq_q,
        new_kv_included,
        min_process_len=1024,
    )

    nblocks = (num_seq_kvcache + num_seq_q + block_size - 1) // block_size
    total_blocks = sum(nblocks)
    kvcache = torch.randn(
        max_num_blocks, 2, block_size, num_head_kv, num_dim_qk, dtype=T, device="cuda"
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
            kvcache[block_ids[i, blk_id], 0, slot_id] = K.reshape(
                num_batch, num_seq_q, num_head_kv, num_dim_qk
            )[i, sqi]
            kvcache[block_ids[i, blk_id], 1, slot_id] = V.reshape(
                num_batch, num_seq_q, num_head_kv, num_dim_qk
            )[i, sqi]

    gt = gt_attention_func(
        Q, K, V, kvcache, block_ids, nblocks, seqlenq, cu_seqlenq, num_seq_kvcache
    )

    if use_output:
        my = torch.empty_like(Q)
        hpc.attention_decode_bf16_adaptive(
            Q,
            kvcache[:, 0, :, :, :],
            kvcache[:, 1, :, :, :],
            block_ids,
            num_seq_kvcache + num_seq_q if new_kv_included else num_seq_kvcache,
            mtp=num_seq_q - 1,
            task_map=task_map,
            output=my,
        )
    else:
        my = hpc.attention_decode_bf16_adaptive(
            Q,
            kvcache[:, 0, :, :, :],
            kvcache[:, 1, :, :, :],
            block_ids,
            num_seq_kvcache + num_seq_q if new_kv_included else num_seq_kvcache,
            mtp=num_seq_q - 1,
            task_map=task_map,
        )

    report_cos_atol(
        gt,
        my,
        atol=0.016,
        tag=f"decode[{num_head_kv}x{num_head_q},b{num_batch},sq{num_seq_q},kv{max_seq_kv},"
        f"new{new_kv_included},out{use_output}]",
    )


@pytest.mark.skipif(True, reason="skip on ci!")
@pytest.mark.parametrize("num_batch", [1, 16])
@pytest.mark.parametrize("num_seq_q", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("max_seq_kv", [1024, 4096])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("kv_head_q_head", [(2, 8), (4, 32)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("new_kv_included", [True, False])
@pytest.mark.parametrize("use_output", [True, False])
def test_attn_bf16_offline(
    num_batch,
    num_seq_q,
    max_seq_kv,
    block_size,
    kv_head_q_head,
    head_dim,
    new_kv_included,
    use_output,
):
    attention_decode_bf16_test_func(
        num_batch,
        num_seq_q,
        max_seq_kv,
        block_size,
        kv_head_q_head,
        head_dim,
        new_kv_included,
        use_output,
    )


@pytest.mark.parametrize("num_batch", [1])
@pytest.mark.parametrize("num_seq_q", [3])
@pytest.mark.parametrize("max_seq_kv", [256])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("kv_head_q_head", [(1, 8)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("new_kv_included", [False])
@pytest.mark.parametrize("use_output", [False])
def test_attn_bf16(
    num_batch,
    num_seq_q,
    max_seq_kv,
    block_size,
    kv_head_q_head,
    head_dim,
    new_kv_included,
    use_output,
):
    attention_decode_bf16_test_func(
        num_batch,
        num_seq_q,
        max_seq_kv,
        block_size,
        kv_head_q_head,
        head_dim,
        new_kv_included,
        use_output,
    )
