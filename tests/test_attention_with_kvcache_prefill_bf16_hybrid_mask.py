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


def make_mm_prefix_range(seqlens_q, seqlens_kvcache, max_spans, device):
    """Build padded, disjoint, non-decreasing per-sequence inclusive image spans.

    Spans are expressed in sequence-local (== full-sequence) coordinates and laid
    inside each sequence's query region ``[start, num_seq_kv)`` so the hybrid path
    is actually exercised. Unused slots are padded with -1 (never match q_abs>=0).
    """
    num_batch = seqlens_q.shape[0]
    spans = torch.full((num_batch, max_spans, 2), -1, dtype=torch.int32, device=device)
    for b in range(num_batch):
        nq = int(seqlens_q[b])
        nkv = int(seqlens_kvcache[b])
        start = nkv - nq
        # Two disjoint spans inside the query region. e > q_abs for queries at the
        # span start, so they attend bidirectionally into the (future) span tail.
        anchors = [
            (start + nq // 8, start + nq // 4),
            (start + nq // 2, start + (5 * nq) // 8),
        ]
        for k in range(min(max_spans, len(anchors))):
            s, e = anchors[k]
            if 0 <= s <= e < nkv:
                spans[b, k, 0] = s
                spans[b, k, 1] = e
    return spans


def hybrid_mask(num_seq_q, num_seq_kv, mm_prefix_range, device):
    """Per-query keep mask: causal by default, bidirectional inside image spans.

    Each query at absolute position ``q_abs = (num_seq_kv - num_seq_q) + q_local``
    attends to keys ``[0, min(bound, num_seq_kv))`` where ``bound = e + 1`` if
    ``q_abs`` lies in an inclusive span ``[s, e]`` and ``bound = q_abs + 1``
    (causal) otherwise.
    """
    start = num_seq_kv - num_seq_q
    q_abs = start + torch.arange(num_seq_q, device=device, dtype=torch.int64)
    bound = q_abs + 1
    for j in range(mm_prefix_range.shape[0]):
        s = int(mm_prefix_range[j, 0])
        e = int(mm_prefix_range[j, 1])
        in_span = (q_abs >= s) & (q_abs <= e)
        bound = torch.where(in_span, torch.full_like(bound, e + 1), bound)
    bound = torch.clamp(bound, max=num_seq_kv)
    col = torch.arange(num_seq_kv, device=device).unsqueeze(0)
    return col < bound.unsqueeze(1)


def naive_attn_with_kvcache_hybrid_mask_func(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    page_table,
    mm_prefix_range,
):
    """Reference for the hybrid (causal + bidirectional-within-image-span) mask."""
    num_batch, num_seq_q, num_head_q, num_dim_qk = q.shape
    _, block_size, num_head_kv, _ = k_cache.shape
    _, _, _, num_dim_v = v_cache.shape

    num_group = num_head_q // num_head_kv
    output = torch.empty_like(q)

    kvcache_blocks = (cache_seqlens + block_size - 1) // block_size

    for i in range(num_batch):
        BQ = q[i].transpose(0, 1).float()  # [num_heads, sq, head_dim]
        blk_ids = page_table[i, : kvcache_blocks[i]]
        num_seq_kv = int(cache_seqlens[i])
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
        keep = hybrid_mask(num_seq_q, num_seq_kv, mm_prefix_range[i], q.device)
        scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)

        # matmul -> [num_head_q, num_seq_q, num_dim_v]; want [num_seq_q, num_head_q, num_dim_v]
        output[i] = torch.matmul(attn_weights, BV).transpose(0, 1)

    return output


@pytest.mark.parametrize("kv_layout", ["nhd"])
@pytest.mark.parametrize("num_head_q, num_head_kv", [(8, 1)])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize(
    "num_batch, num_seq_q, num_seq_kv",
    [(1, 256, 256)],
)
@pytest.mark.parametrize("use_output", [True])
def test_attention_with_kvcache_prefill_bf16_hybrid_mask(
    kv_layout,
    num_head_q,
    num_head_kv,
    block_size,
    num_batch,
    num_seq_q,
    num_seq_kv,
    use_output,
):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    T = torch.bfloat16
    device = "cuda"
    num_dim_qk = 128
    num_dim_v = 128
    max_spans = 4

    Q = torch.randn(
        (num_batch, num_seq_q, num_head_q, num_dim_qk), dtype=T, device=device
    ) / math.sqrt(num_dim_qk)

    seqlens_q = torch.full((num_batch,), num_seq_q, dtype=torch.int32, device=device)
    seqlens_kvcache = torch.full((num_batch,), num_seq_kv, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device=device), seqlens_q]), dim=0
    ).to(torch.int32)

    max_num_blocks = num_batch * (num_seq_kv + block_size - 1) // block_size * 2
    kvcache_blocks = (seqlens_kvcache + block_size - 1) // block_size
    total_kvcache_blocks = int(kvcache_blocks.sum().item())
    max_kvcache_blocks = int(kvcache_blocks.max().item())
    max_seqlens_q = int(seqlens_q.max().item())

    kvcache = torch.randn(
        max_num_blocks, 2, block_size, num_head_kv, num_dim_qk, dtype=T, device=device
    )
    if kv_layout == "hnd":
        kcache = kvcache[:, 0].transpose(1, 2).contiguous().transpose(1, 2)
        vcache = kvcache[:, 1].transpose(1, 2).contiguous().transpose(1, 2)
    else:
        kcache = kvcache[:, 0]
        vcache = kvcache[:, 1]
    packed_block_ids = torch.randperm(max_num_blocks, device=device)[:total_kvcache_blocks].to(
        torch.int32
    )

    block_ids = torch.empty(num_batch, max_kvcache_blocks, dtype=torch.int32, device=device)
    cu_blocks = 0
    for i in range(num_batch):
        block_ids[i, : kvcache_blocks[i]] = packed_block_ids[
            cu_blocks : cu_blocks + kvcache_blocks[i]
        ]
        cu_blocks += kvcache_blocks[i]

    mm_prefix_range = make_mm_prefix_range(seqlens_q, seqlens_kvcache, max_spans, device)

    gt = naive_attn_with_kvcache_hybrid_mask_func(
        Q,
        kcache,
        vcache,
        seqlens_kvcache,
        block_ids,
        mm_prefix_range,
    )

    if use_output:
        my = torch.empty_like(Q.reshape(-1, num_head_q, num_dim_v))
        hpc.attention_with_kvcache_prefill_bf16_hybrid_mask(
            Q.reshape(-1, num_head_q, num_dim_qk),
            kcache,
            vcache,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            max_seqlens_q,
            mm_prefix_range,
            output=my,
        )
    else:
        my = hpc.attention_with_kvcache_prefill_bf16_hybrid_mask(
            Q.reshape(-1, num_head_q, num_dim_qk),
            kcache,
            vcache,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            max_seqlens_q,
            mm_prefix_range,
        )

    gt = gt.reshape(-1, num_head_q, num_dim_v)
    report_cos_atol(
        gt,
        my,
        atol=0.016,
        tag=f"hybrid[{kv_layout},{num_head_q}x{num_head_kv},bs{block_size},"
        f"b{num_batch},sq{num_seq_q},kv{num_seq_kv},out{use_output}]",
    )


@pytest.mark.parametrize("kv_layout", ["nhd"])
@pytest.mark.parametrize("num_head_q, num_head_kv", [(8, 1)])
@pytest.mark.parametrize(
    "query_lens, kv_lens",
    [
        ([256], [256]),
    ],
)
def test_attention_with_kvcache_prefill_bf16_hybrid_mask_varlen(
    kv_layout,
    num_head_q,
    num_head_kv,
    query_lens,
    kv_lens,
):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    T = torch.bfloat16
    device = "cuda"
    block_size = 64
    num_dim_qk = 128
    num_dim_v = 128
    max_spans = 4

    num_batch = len(query_lens)
    total_seq_q = sum(query_lens)

    Q = torch.randn((total_seq_q, num_head_q, num_dim_qk), dtype=T, device=device) / math.sqrt(
        num_dim_qk
    )

    seqlens_q = torch.tensor(query_lens, dtype=torch.int32, device=device)
    seqlens_kvcache = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device=device), seqlens_q]), dim=0
    ).to(torch.int32)
    kvcache_blocks = (seqlens_kvcache + block_size - 1) // block_size
    total_kvcache_blocks = int(kvcache_blocks.sum().item())
    max_kvcache_blocks = int(kvcache_blocks.max().item())
    max_seqlens_q = int(seqlens_q.max().item())
    max_num_blocks = total_kvcache_blocks * 2

    kvcache = torch.randn(
        max_num_blocks, 2, block_size, num_head_kv, num_dim_qk, dtype=T, device=device
    )
    if kv_layout == "hnd":
        kcache = kvcache[:, 0].transpose(1, 2).contiguous().transpose(1, 2)
        vcache = kvcache[:, 1].transpose(1, 2).contiguous().transpose(1, 2)
    else:
        kcache = kvcache[:, 0]
        vcache = kvcache[:, 1]
    packed_block_ids = torch.randperm(max_num_blocks, device=device)[:total_kvcache_blocks].to(
        torch.int32
    )
    block_ids = torch.zeros(num_batch, max_kvcache_blocks, dtype=torch.int32, device=device)
    cu_blocks = 0
    for i in range(num_batch):
        block_ids[i, : kvcache_blocks[i]] = packed_block_ids[
            cu_blocks : cu_blocks + kvcache_blocks[i]
        ]
        cu_blocks += kvcache_blocks[i]

    mm_prefix_range = make_mm_prefix_range(seqlens_q, seqlens_kvcache, max_spans, device)

    gt_parts = []
    for i in range(num_batch):
        qs = int(cu_seqlens_q[i])
        qe = int(cu_seqlens_q[i + 1])
        gt_i = naive_attn_with_kvcache_hybrid_mask_func(
            Q[qs:qe].unsqueeze(0),
            kcache,
            vcache,
            seqlens_kvcache[i : i + 1],
            block_ids[i : i + 1],
            mm_prefix_range[i : i + 1],
        )
        gt_parts.append(gt_i.reshape(-1, num_head_q, num_dim_v))
    gt = torch.cat(gt_parts, dim=0)

    my = hpc.attention_with_kvcache_prefill_bf16_hybrid_mask(
        Q,
        kcache,
        vcache,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        max_seqlens_q,
        mm_prefix_range,
    )

    report_cos_atol(
        gt,
        my,
        atol=0.016,
        tag=f"varlen[{kv_layout},{num_head_q}x{num_head_kv},q{query_lens},kv{kv_lens}]",
    )


@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize(
    "num_batch, num_seq_q, num_seq_kv",
    [(1, 256, 256)],
)
@pytest.mark.parametrize("use_output", [True])
def test_attention_with_kvcache_prefill_bf16_hybrid_mask_causal(
    block_size,
    num_batch,
    num_seq_q,
    num_seq_kv,
    use_output,
):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    T = torch.bfloat16
    device = "cuda"
    num_head_q, num_head_kv = 8, 2
    num_dim_qk = 128
    num_dim_v = 128
    max_spans = 4

    Q = torch.randn(
        (num_batch, num_seq_q, num_head_q, num_dim_qk), dtype=T, device=device
    ) / math.sqrt(num_dim_qk)

    seqlens_q = torch.full((num_batch,), num_seq_q, dtype=torch.int32, device=device)
    seqlens_kvcache = torch.full((num_batch,), num_seq_kv, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device=device), seqlens_q]), dim=0
    ).to(torch.int32)

    max_num_blocks = num_batch * (num_seq_kv + block_size - 1) // block_size * 2
    kvcache_blocks = (seqlens_kvcache + block_size - 1) // block_size
    total_kvcache_blocks = int(kvcache_blocks.sum().item())
    max_kvcache_blocks = int(kvcache_blocks.max().item())
    max_seqlens_q = int(seqlens_q.max().item())

    kvcache = torch.randn(
        max_num_blocks, 2, block_size, num_head_kv, num_dim_qk, dtype=T, device=device
    )
    kcache = kvcache[:, 0]
    vcache = kvcache[:, 1]
    packed_block_ids = torch.randperm(max_num_blocks, device=device)[:total_kvcache_blocks].to(
        torch.int32
    )
    block_ids = torch.empty(num_batch, max_kvcache_blocks, dtype=torch.int32, device=device)
    cu_blocks = 0
    for i in range(num_batch):
        block_ids[i, : kvcache_blocks[i]] = packed_block_ids[
            cu_blocks : cu_blocks + kvcache_blocks[i]
        ]
        cu_blocks += kvcache_blocks[i]

    causal_spans = torch.full((num_batch, max_spans, 2), -1, dtype=torch.int32, device=device)
    gt = naive_attn_with_kvcache_hybrid_mask_func(
        Q, kcache, vcache, seqlens_kvcache, block_ids, causal_spans
    )

    if use_output:
        my = torch.empty_like(Q.reshape(-1, num_head_q, num_dim_v))
        hpc.attention_with_kvcache_prefill_bf16_hybrid_mask(
            Q.reshape(-1, num_head_q, num_dim_qk),
            kcache,
            vcache,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            max_seqlens_q,
            mm_prefix_range=None,
            output=my,
        )
    else:
        my = hpc.attention_with_kvcache_prefill_bf16_hybrid_mask(
            Q.reshape(-1, num_head_q, num_dim_qk),
            kcache,
            vcache,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            max_seqlens_q,
            mm_prefix_range=None,
        )

    gt = gt.reshape(-1, num_head_q, num_dim_v)
    report_cos_atol(
        gt,
        my,
        atol=0.016,
        tag=f"causal[bs{block_size},b{num_batch},sq{num_seq_q},kv{num_seq_kv},out{use_output}]",
    )
