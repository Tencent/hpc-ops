import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
import torch.nn.functional as F


def naive_attn_with_kvcache_func(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    page_table,
    causal=True,
):

    num_batch, num_seq_q, num_head_q, num_dim_qk = q.shape
    num_blocks, block_size, num_head_kv, _ = k_cache.shape
    _, _, _, num_dim_v = v_cache.shape

    num_group = num_head_q // num_head_kv
    output = torch.empty_like(q)

    kvcache_blocks = (cache_seqlens + block_size - 1) // block_size

    for i in range(num_batch):
        BQ = q[i].transpose(0, 1).float()  # [num_heads, sq, head_dim]
        blk_ids = page_table[i, : kvcache_blocks[i]]
        num_seq_kv = cache_seqlens[i]
        BK = (
            k_cache[blk_ids, :, :, :]
            .reshape(-1, num_head_kv, num_dim_qk)
            .transpose(0, 1)[:, :num_seq_kv, :]
            .repeat_interleave(num_group, dim=0)
        ).float()
        BV = (
            v_cache[blk_ids, :, :, :]
            .reshape(-1, num_head_kv, num_dim_v)
            .transpose(0, 1)[:, :num_seq_kv, :]
            .repeat_interleave(num_group, dim=0)
        ).float()

        scores = torch.matmul(BQ, BK.transpose(-2, -1)) / math.sqrt(num_dim_qk)
        if causal:
            causal_mask = (
                torch.tril(torch.ones(num_seq_kv, num_seq_kv, device=q.device, dtype=torch.bool))[
                    (num_seq_kv - num_seq_q) :, :
                ]
                .unsqueeze(0)
                .unsqueeze(0)
            )  # (1, 1, num_seq_q, num_seq_kv)
        else:
            causal_mask = causal_mask.view(1, 1, num_seq_q, num_seq_kv)

        scores = scores.masked_fill(~causal_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)

        output[i] = torch.matmul(attn_weights, BV).transpose(1, 2)

    return output


try:
    from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache

    gt_attention_func = flash_attn_with_kvcache
except Exception as e:
    print(f"execute naive_attn_func: {e}")
    gt_attention_func = naive_attn_with_kvcache_func


@pytest.mark.parametrize("num_batch", [1, 2, 4, 8])
@pytest.mark.parametrize("num_seq_q", [100, 500, 1000, 1500])
@pytest.mark.parametrize("num_seq_kv", [1500, 3000])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("num_head_q", [4])
@pytest.mark.parametrize("num_head_kv", [1])
@pytest.mark.parametrize("num_dim_qk", [128])
@pytest.mark.parametrize("num_dim_v", [128])
@pytest.mark.parametrize("use_output", [True])
def test_attention_with_kvcache_prefill_bf16(
    num_batch,
    num_seq_q,
    num_seq_kv,
    block_size,
    num_head_q,
    num_head_kv,
    num_dim_qk,
    num_dim_v,
    use_output,
):

    T = torch.bfloat16

    Q = torch.randn(
        (num_batch, num_seq_q, num_head_q, num_dim_qk),
        dtype=T,
        device="cuda",
    ) / math.sqrt(num_dim_qk)
    K = torch.randn(
        (num_batch * num_seq_q, num_head_kv, num_dim_qk), dtype=T, device="cuda"
    ) / math.sqrt(num_dim_qk)
    V = torch.randn((num_batch * num_seq_q, num_head_kv, num_dim_v), dtype=T, device="cuda")

    seqlens_q = torch.full((num_batch,), num_seq_q, dtype=torch.int32, device="cuda")
    seqlens_kvcache = torch.full((num_batch,), num_seq_kv, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens_q]), dim=0
    ).to(torch.int32)
    cu_seqlens_kvcache = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens_kvcache]), dim=0
    ).to(torch.int32)

    max_num_blocks = num_batch * (num_seq_kv + block_size - 1) // block_size * 2
    kvcache_blocks = (seqlens_kvcache + block_size - 1) // block_size
    total_kvcache_blocks = sum(kvcache_blocks)
    max_kvcache_blocks = max(kvcache_blocks)
    max_seqlens_q = max(seqlens_q)
    max_seqlens_kvcache = max(seqlens_kvcache)

    kvcache = torch.randn(
        max_num_blocks, 2, block_size, num_head_kv, num_dim_qk, dtype=T, device="cuda"
    )
    packed_block_ids = torch.randperm(max_num_blocks)[:total_kvcache_blocks].to(torch.int32).cuda()

    cu_blocks = 0
    block_ids = torch.empty(num_batch, max_kvcache_blocks, dtype=torch.int32, device="cuda")
    for i in range(num_batch):
        block_ids[i, : kvcache_blocks[i]] = packed_block_ids[
            cu_blocks : cu_blocks + kvcache_blocks[i]
        ]
        cu_blocks += kvcache_blocks[i]

    for i in range(1):
        gt = gt_attention_func(
            q=Q,
            k_cache=kvcache[:, 0, :, :],
            v_cache=kvcache[:, 1, :, :],
            cache_seqlens=seqlens_kvcache,
            page_table=block_ids,
            causal=True,
        )

        if use_output:
            my = torch.empty_like(Q.reshape(-1, num_head_q, num_dim_qk))
            hpc.attention_with_kvcache_prefill_bf16(
                Q.reshape(-1, num_head_q, num_dim_qk),
                kvcache[:, 0, :, :, :],
                kvcache[:, 1, :, :, :],
                cu_seqlens_q,
                block_ids,
                seqlens_kvcache,
                max_seqlens_q,
                output=my,
            )
        else:
            my = hpc.attention_with_kvcache_prefill_bf16(
                Q.reshape(-1, num_head_q, num_dim_qk),
                kvcache[:, 0, :, :, :],
                kvcache[:, 1, :, :, :],
                cu_seqlens_q,
                block_ids,
                seqlens_kvcache,
                max_seqlens_q,
            )
    gt = gt.reshape(-1, num_head_q, num_dim_v)

    print("\ngt\n")
    print(gt[0, :, 0])
    print("\nmy\n")
    print(my[0, :, 0], my.shape)

    abs_diff = torch.abs(gt - my)
    vals, idxs = torch.topk(abs_diff.flatten(), 10)
    idxs = [torch.unravel_index(idx, gt.shape) for idx in idxs]

    for i, idx in enumerate(idxs):
        cpu_idx = tuple(tensor.cpu().item() for tensor in idx)
        print(
            "{:+.4f} vs {:+.4f} with diff = {:.4f}, @ {}".format(gt[idx], my[idx], vals[i], cpu_idx)
        )

    assert torch.allclose(my, gt, atol=0.0156)
    assert gt.device == my.device
    assert gt.dtype == my.dtype
    assert gt.shape == my.shape
