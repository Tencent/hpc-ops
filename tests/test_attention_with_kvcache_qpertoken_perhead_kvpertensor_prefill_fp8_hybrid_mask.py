import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
from utils import allclose


def make_mm_prefix_range(seqlens_q, seqlens_kvcache, max_spans, device):
    num_batch = seqlens_q.shape[0]
    spans = torch.full((num_batch, max_spans, 2), -1, dtype=torch.int32, device=device)
    for b in range(num_batch):
        nq = int(seqlens_q[b])
        nkv = int(seqlens_kvcache[b])
        start = nkv - nq
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


def naive_attn_with_kvcache_qpertoken_kvpertensor_hybrid_mask(
    q,
    k_cache,
    v_cache,
    qscale,
    kscale,
    p_scale,
    vscale,
    cache_seqlens,
    page_table,
    mm_prefix_range,
):
    num_batch, num_seq_q, num_head_q, num_dim_qk = q.shape
    _, block_size, num_head_kv, _ = k_cache.shape
    _, _, _, num_dim_v = v_cache.shape
    num_group = num_head_q // num_head_kv
    output = torch.empty_like(q).to(torch.bfloat16)
    kvcache_blocks = (cache_seqlens + block_size - 1) // block_size

    for i in range(num_batch):
        BQ = q[i].transpose(0, 1).float()
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
        scale = qscale[i, :, :].unsqueeze(-1)[:, :num_seq_q, :]
        scores = torch.matmul(BQ, BK.transpose(-2, -1)) * scale * kscale[0] / math.sqrt(num_dim_qk)
        keep = hybrid_mask(num_seq_q, num_seq_kv, mm_prefix_range[i], q.device)
        scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
        attn_weights = torch.exp(scores - scores.max(dim=-1, keepdim=True)[0])
        gsum = attn_weights.sum(dim=-1, keepdim=True)
        attn_weights = attn_weights * p_scale[:, None, None]
        attn_weights = attn_weights.to(torch.float8_e4m3fn).float()
        out_head = torch.matmul(attn_weights, BV) / gsum
        out_head = out_head * vscale[0] / p_scale[:, None, None]
        output[i] = out_head.transpose(0, 1)

    return output


@pytest.mark.parametrize("kv_layout", ["nhd"])
@pytest.mark.parametrize("use_output", [True])
def test_attention_with_kvcache_qpertoken_perhead_kvpertensor_prefill_fp8_hybrid_mask(
    kv_layout,
    use_output,
):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    T = torch.bfloat16
    T1 = torch.float8_e4m3fn
    device = "cuda"
    num_batch = 2
    num_seq_q = 512
    num_seq_kv = 1536
    block_size = 64
    num_head_q = 4
    num_head_kv = 1
    num_dim_qk = 128
    num_dim_v = 128
    max_spans = 4
    num_seq_q_pad = (num_seq_q + 127) // 128 * 128

    Q = (
        torch.randn((num_batch, num_seq_q, num_head_q, num_dim_qk), dtype=T, device=device)
        / math.sqrt(num_dim_qk)
    ).to(T1)
    qscale = (
        torch.abs(
            torch.randn((num_batch, num_head_q, num_seq_q_pad), dtype=torch.float32, device=device)
        )
        / 10
    )
    kscale = torch.randn((1), dtype=torch.float32, device=device).abs() * 10
    vscale = torch.randn((1), dtype=torch.float32, device=device)
    p_scale = torch.linspace(128.0, 256.0, num_head_q, dtype=torch.float32, device=device)
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
    ).to(T1)
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
    gt = naive_attn_with_kvcache_qpertoken_kvpertensor_hybrid_mask(
        Q,
        kcache,
        vcache,
        qscale,
        kscale,
        p_scale,
        vscale,
        seqlens_kvcache,
        block_ids,
        mm_prefix_range,
    )

    if use_output:
        my = torch.empty_like(Q.reshape(-1, num_head_q, num_dim_v), dtype=torch.bfloat16)
        hpc.attention_with_kvcache_prefill_fp8_hybrid_mask(
            Q.reshape(-1, num_head_q, num_dim_qk),
            kcache,
            vcache,
            qscale,
            kscale,
            vscale,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            max_seqlens_q,
            mm_prefix_range,
            p_scale=p_scale,
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
            output=my,
        )
    else:
        my = hpc.attention_with_kvcache_prefill_fp8_hybrid_mask(
            Q.reshape(-1, num_head_q, num_dim_qk),
            kcache,
            vcache,
            qscale,
            kscale,
            vscale,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            max_seqlens_q,
            mm_prefix_range,
            p_scale=p_scale,
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
        )

    gt = gt.reshape(-1, num_head_q, num_dim_v)
    assert allclose(gt, my, atol=0.1)


@pytest.mark.parametrize("kv_layout", ["nhd"])
@pytest.mark.parametrize("use_output", [True])
def test_attention_with_kvcache_qpertoken_perhead_kvpertensor_prefill_fp8_hybrid_mask_causal(
    kv_layout,
    use_output,
):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    T = torch.bfloat16
    T1 = torch.float8_e4m3fn
    device = "cuda"
    num_batch = 2
    num_seq_q = 512
    num_seq_kv = 1536
    block_size = 64
    num_head_q = 4
    num_head_kv = 1
    num_dim_qk = 128
    num_dim_v = 128
    max_spans = 4
    num_seq_q_pad = (num_seq_q + 127) // 128 * 128

    Q = (
        torch.randn((num_batch, num_seq_q, num_head_q, num_dim_qk), dtype=T, device=device)
        / math.sqrt(num_dim_qk)
    ).to(T1)
    qscale = (
        torch.abs(
            torch.randn((num_batch, num_head_q, num_seq_q_pad), dtype=torch.float32, device=device)
        )
        / 10
    )
    kscale = torch.randn((1), dtype=torch.float32, device=device).abs() * 10
    vscale = torch.randn((1), dtype=torch.float32, device=device)
    p_scale = torch.linspace(128.0, 256.0, num_head_q, dtype=torch.float32, device=device)
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
    ).to(T1)
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

    causal_spans = torch.full((num_batch, max_spans, 2), -1, dtype=torch.int32, device=device)
    gt = naive_attn_with_kvcache_qpertoken_kvpertensor_hybrid_mask(
        Q,
        kcache,
        vcache,
        qscale,
        kscale,
        p_scale,
        vscale,
        seqlens_kvcache,
        block_ids,
        causal_spans,
    )

    if use_output:
        my = torch.empty_like(Q.reshape(-1, num_head_q, num_dim_v), dtype=torch.bfloat16)
        hpc.attention_with_kvcache_prefill_fp8_hybrid_mask(
            Q.reshape(-1, num_head_q, num_dim_qk),
            kcache,
            vcache,
            qscale,
            kscale,
            vscale,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            max_seqlens_q,
            None,
            p_scale=p_scale,
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
            output=my,
        )
    else:
        my = hpc.attention_with_kvcache_prefill_fp8_hybrid_mask(
            Q.reshape(-1, num_head_q, num_dim_qk),
            kcache,
            vcache,
            qscale,
            kscale,
            vscale,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            max_seqlens_q,
            None,
            p_scale=p_scale,
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
        )

    gt = gt.reshape(-1, num_head_q, num_dim_v)
    assert allclose(gt, my, atol=0.1)
