import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
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


def make_p_scale(p_scale_mode, num_head_q, device):
    if p_scale_mode == "256":
        return torch.full((num_head_q,), 256.0, dtype=torch.float32, device=device)
    if p_scale_mode == "linspace":
        return torch.linspace(128.0, 256.0, num_head_q, dtype=torch.float32, device=device)
    if p_scale_mode == "all_2":
        return torch.full((num_head_q,), 2.0, dtype=torch.float32, device=device)
    if p_scale_mode == "per_head_random":
        return 0.7 + 0.8 * torch.rand(num_head_q, dtype=torch.float32, device=device)
    raise ValueError(p_scale_mode)


def naive_attn_with_kvcache_qkpertoken_vperhead_hybrid_mask(
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
        BKS = (
            kscale[blk_ids, :, :, :]
            .permute(0, 1, 3, 2)
            .reshape(-1, num_head_kv)
            .transpose(0, 1)[:, :num_seq_kv]
            .repeat_interleave(num_group, dim=0)
        ).float()
        scale = qscale[i, :, :].unsqueeze(-1)[:, :num_seq_q, :]
        scores = torch.matmul(BQ, BK.transpose(-2, -1)) / math.sqrt(num_dim_qk)
        scores = scores * scale * BKS.unsqueeze(1)
        keep = hybrid_mask(num_seq_q, num_seq_kv, mm_prefix_range[i], q.device)
        scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
        attn_weights = torch.exp(scores - scores.max(dim=-1, keepdim=True)[0])
        gsum = attn_weights.sum(dim=-1, keepdim=True)
        p_scale_eff = p_scale[:, None, None]
        attn_weights = attn_weights * p_scale_eff
        attn_weights = attn_weights.to(torch.float8_e4m3fn).float()
        out_head = torch.matmul(attn_weights, BV) / gsum
        v_scale_eff = vscale[:, None, None].repeat_interleave(num_group, dim=0)
        out_head = out_head * v_scale_eff / p_scale_eff
        output[i] = out_head.transpose(0, 1)

    return output


@pytest.mark.parametrize("kv_layout", ["nhd"])
@pytest.mark.parametrize("num_head_q, num_head_kv", [(8, 1)])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize(
    "num_batch, num_seq_q, num_seq_kv",
    [(1, 256, 256)],
)
@pytest.mark.parametrize("p_scale_mode", ["256"])
@pytest.mark.parametrize("use_output", [True])
def test_attention_with_kvcache_qkpertoken_perhead_vperhead_prefill_fp8_hybrid_mask(
    kv_layout,
    num_head_q,
    num_head_kv,
    block_size,
    num_batch,
    num_seq_q,
    num_seq_kv,
    p_scale_mode,
    use_output,
):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    T = torch.bfloat16
    T1 = torch.float8_e4m3fn
    device = "cuda"
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
    vscale = torch.randn((num_head_kv), dtype=torch.float32, device=device)
    p_scale = make_p_scale(p_scale_mode, num_head_q, device)
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

    kScaleBlockSize = block_size // 32
    num_dim_scale = num_dim_qk // 4
    kscale = torch.abs(
        torch.randn(
            (max_num_blocks, kScaleBlockSize, num_head_kv, num_dim_scale),
            dtype=torch.float32,
            device=device,
        )
    ).view(torch.float8_e4m3fn)
    mm_prefix_range = make_mm_prefix_range(seqlens_q, seqlens_kvcache, max_spans, device)
    gt = naive_attn_with_kvcache_qkpertoken_vperhead_hybrid_mask(
        Q,
        kcache,
        vcache,
        qscale,
        kscale.view(torch.float32),
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
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
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
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
        )

    gt = gt.reshape(-1, num_head_q, num_dim_v)
    report_cos_atol(
        gt,
        my,
        atol=0.1,
        tag=f"hybrid[{kv_layout},{num_head_q}x{num_head_kv},bs{block_size},"
        f"b{num_batch},sq{num_seq_q},kv{num_seq_kv},{p_scale_mode},out{use_output}]",
    )


@pytest.mark.parametrize("kv_layout", ["nhd"])
@pytest.mark.parametrize("num_head_q, num_head_kv", [(8, 1)])
@pytest.mark.parametrize(
    "query_lens, kv_lens",
    [
        ([256], [256]),
    ],
)
def test_attention_with_kvcache_qkpertoken_perhead_vperhead_prefill_fp8_hybrid_mask_varlen(
    kv_layout,
    num_head_q,
    num_head_kv,
    query_lens,
    kv_lens,
):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    T = torch.bfloat16
    T1 = torch.float8_e4m3fn
    device = "cuda"
    block_size = 64
    num_dim_qk = 128
    num_dim_v = 128
    max_spans = 4

    num_batch = len(query_lens)
    total_seq_q = sum(query_lens)
    max_seq_q = max(query_lens)
    num_seq_q_pad = (max_seq_q + 127) // 128 * 128

    Q = (
        torch.randn((total_seq_q, num_head_q, num_dim_qk), dtype=T, device=device)
        / math.sqrt(num_dim_qk)
    ).to(T1)
    qscale = (
        torch.abs(
            torch.randn((num_batch, num_head_q, num_seq_q_pad), dtype=torch.float32, device=device)
        )
        / 10
    )
    vscale = torch.randn((num_head_kv), dtype=torch.float32, device=device)
    p_scale = make_p_scale("256", num_head_q, device)
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
    block_ids = torch.zeros(num_batch, max_kvcache_blocks, dtype=torch.int32, device=device)
    cu_blocks = 0
    for i in range(num_batch):
        block_ids[i, : kvcache_blocks[i]] = packed_block_ids[
            cu_blocks : cu_blocks + kvcache_blocks[i]
        ]
        cu_blocks += kvcache_blocks[i]

    kScaleBlockSize = block_size // 32
    num_dim_scale = num_dim_qk // 4
    kscale = torch.abs(
        torch.randn(
            (max_num_blocks, kScaleBlockSize, num_head_kv, num_dim_scale),
            dtype=torch.float32,
            device=device,
        )
    ).view(torch.float8_e4m3fn)
    mm_prefix_range = make_mm_prefix_range(seqlens_q, seqlens_kvcache, max_spans, device)

    gt_parts = []
    for i in range(num_batch):
        qs = int(cu_seqlens_q[i])
        qe = int(cu_seqlens_q[i + 1])
        gt_i = naive_attn_with_kvcache_qkpertoken_vperhead_hybrid_mask(
            Q[qs:qe].unsqueeze(0),
            kcache,
            vcache,
            qscale[i : i + 1],
            kscale.view(torch.float32),
            p_scale,
            vscale,
            seqlens_kvcache[i : i + 1],
            block_ids[i : i + 1],
            mm_prefix_range[i : i + 1],
        )
        gt_parts.append(gt_i.reshape(-1, num_head_q, num_dim_v))
    gt = torch.cat(gt_parts, dim=0)

    my = hpc.attention_with_kvcache_prefill_fp8_hybrid_mask(
        Q,
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
        quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
    )

    report_cos_atol(
        gt,
        my,
        atol=0.1,
        tag=f"varlen[{kv_layout},{num_head_q}x{num_head_kv},q{query_lens},kv{kv_lens}]",
    )


@pytest.mark.parametrize("kv_layout", ["nhd"])
@pytest.mark.parametrize("p_scale_mode", ["256"])
@pytest.mark.parametrize("use_output", [True])
def test_attention_with_kvcache_qkpertoken_perhead_vperhead_prefill_fp8_hybrid_mask_causal(
    kv_layout,
    p_scale_mode,
    use_output,
):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    T = torch.bfloat16
    T1 = torch.float8_e4m3fn
    device = "cuda"
    num_head_q, num_head_kv = 8, 2
    block_size = 64
    num_batch, num_seq_q, num_seq_kv = 2, 512, 1536
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
    vscale = torch.randn((num_head_kv), dtype=torch.float32, device=device)
    p_scale = make_p_scale(p_scale_mode, num_head_q, device)
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

    kScaleBlockSize = block_size // 32
    num_dim_scale = num_dim_qk // 4
    kscale = torch.abs(
        torch.randn(
            (max_num_blocks, kScaleBlockSize, num_head_kv, num_dim_scale),
            dtype=torch.float32,
            device=device,
        )
    ).view(torch.float8_e4m3fn)

    causal_spans = torch.full((num_batch, max_spans, 2), -1, dtype=torch.int32, device=device)
    gt = naive_attn_with_kvcache_qkpertoken_vperhead_hybrid_mask(
        Q,
        kcache,
        vcache,
        qscale,
        kscale.view(torch.float32),
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
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
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
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
        )

    gt = gt.reshape(-1, num_head_q, num_dim_v)
    report_cos_atol(
        gt,
        my,
        atol=0.1,
        tag=f"causal[{kv_layout},{num_head_q}x{num_head_kv},bs{block_size},"
        f"b{num_batch},sq{num_seq_q},kv{num_seq_kv},{p_scale_mode},out{use_output}]",
    )
