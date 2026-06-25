"""Pure-pytorch reference utilities for the cutedsl attention test suite.

Provides a random input builder, an fp32 ground-truth attention reference,
and paged KV quantization helpers. These have no GPU / cute dependency and
are self-contained so the regression suite always runs.
"""

from __future__ import annotations

import math

import torch


__all__ = [
    "allclose",
    "build_inputs",
    "naive_attn_with_kvcache_func",
    "quant_paged_cache_perhead",
    "quant_paged_cache_pertoken",
]


LOG2E = math.log2(math.e)
LOG2_E4M3_MAX = math.log2(448.0)


def allclose(a, b, atol=0.1, rtol=0.0):
    if a.dtype != b.dtype:
        a = a.float()
        b = b.float()
    diff = (a - b).abs()
    threshold = atol + rtol * b.abs()
    num_mismatch = int((diff > threshold).sum().item())
    max_diff = float(diff.max().item()) if diff.numel() else 0.0
    if num_mismatch:
        total = a.numel()
        pct = num_mismatch / total * 100
        print(
            f"allclose FAILED: {num_mismatch}/{total} ({pct:.1f}%) elements mismatch, "
            f"max abs diff = {max_diff:.6f}, atol = {atol}"
        )
        return False
    print(f"allclose PASSED: max abs diff = {max_diff:.6f}, atol = {atol}")
    return True


def _fp8_bmm2_softmax_payload(scores):
    row_max = scores.max(dim=-1, keepdim=True).values
    p_scaled = torch.exp2((scores - row_max) * LOG2E + LOG2_E4M3_MAX)
    p_fp8 = p_scaled.to(torch.float8_e4m3fn).float()
    p_sum = p_scaled.sum(dim=-1, keepdim=True)
    return p_fp8, p_sum


def quant_paged_cache_pertoken(cache, block_size):
    num_blocks = cache.shape[0]
    head_dim = cache.shape[-1]
    num_head_kv = cache.shape[-2]
    scale = cache[:, :block_size, :, :].float().abs().max(-1)[0] / 448
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))

    cache_fp8 = torch.empty_like(cache, dtype=torch.float8_e4m3fn)
    cache_fp8[:, :block_size, :, :] = (cache[:, :block_size, :, :] / scale[:, :, :, None]).to(
        torch.float8_e4m3fn
    )
    scale = (
        scale.permute(0, 2, 1)
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(num_blocks, num_head_kv, -1, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    cache_fp8[:, block_size:, :, :] = scale
    return cache_fp8, cache_fp8[:, block_size:, :, :]


def quant_paged_cache_perhead(cache, block_size):
    num_head_kv = cache.shape[-2]
    scale = (
        cache[:, :block_size, :, :]
        .float()
        .abs()
        .permute(2, 0, 1, 3)
        .reshape(num_head_kv, -1)
        .max(-1)[0]
        / 448
    )
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    cache_fp8 = (cache.float() / scale[None, None, :, None]).to(torch.float8_e4m3fn)
    return cache_fp8, scale


def naive_attn_with_kvcache_func(
    q, k_cache, v_cache, qscale, kscale, vscale, cache_seqlens, page_table
):
    num_batch, num_seq_q, num_head_q, num_dim_qk = q.shape
    _, block_size, num_head_kv, _ = k_cache.shape
    _, _, _, num_dim_v = v_cache.shape
    num_group = num_head_q // num_head_kv
    output = torch.empty_like(q).to(torch.bfloat16)

    for i in range(num_batch):
        bq = q[i].transpose(0, 1).float()
        num_seq_kv = int(cache_seqlens[i].item())
        num_blocks = (num_seq_kv + block_size - 1) // block_size
        blk_ids = page_table[i, :num_blocks]
        bk = (
            k_cache[blk_ids, :, :, :]
            .reshape(-1, num_head_kv, num_dim_qk)
            .transpose(0, 1)[:, :num_seq_kv, :]
            .repeat_interleave(num_group, dim=0)
        ).float()
        bv = (
            v_cache[blk_ids, :, :, :]
            .reshape(-1, num_head_kv, num_dim_v)
            .transpose(0, 1)[:, :num_seq_kv, :]
            .repeat_interleave(num_group, dim=0)
        ).float()
        bks = (
            kscale[blk_ids, :, :, :]
            .permute(0, 1, 3, 2)
            .reshape(-1, num_head_kv)
            .transpose(0, 1)
            .contiguous()
            .view(torch.float32)[:, :num_seq_kv]
            .repeat_interleave(num_group, dim=0)
        ).float()

        scale = qscale[i, :, :].unsqueeze(-1)[:, :num_seq_q, :]
        scores = bq @ bk.transpose(-1, -2)
        scores = scores / math.sqrt(num_dim_qk) * scale * bks.unsqueeze(1)
        causal_mask = (
            torch.tril(torch.ones(num_seq_kv, num_seq_kv, device=q.device, dtype=torch.bool))[
                (num_seq_kv - num_seq_q) :, :
            ]
            .unsqueeze(0)
            .unsqueeze(0)
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        p_fp8, p_sum = _fp8_bmm2_softmax_payload(scores)
        pv = torch.matmul(p_fp8, bv) / p_sum
        output[i] = (pv * vscale[:, None, None].repeat_interleave(num_group, dim=0)).transpose(1, 2)

    return output


def build_inputs(num_batch, num_seq_q, num_seq_kv, num_head_q, num_head_kv, head_dim, block_size):
    torch.cuda.manual_seed(10086)
    dtype = torch.bfloat16
    fp8 = torch.float8_e4m3fn
    num_seq_q_pad = (num_seq_q + 127) // 128 * 128
    scale_rows = block_size * 4 // head_dim

    q = (
        torch.randn((num_batch, num_seq_q, num_head_q, head_dim), dtype=dtype, device="cuda")
        / math.sqrt(head_dim)
    ).to(fp8)
    k = torch.randn(
        (num_batch, num_seq_kv, num_head_kv, head_dim), dtype=dtype, device="cuda"
    ) / math.sqrt(head_dim)
    v = torch.randn((num_batch, num_seq_kv, num_head_kv, head_dim), dtype=dtype, device="cuda")
    qscale = (
        torch.abs(
            torch.randn((num_batch, num_head_q, num_seq_q_pad), dtype=torch.float32, device="cuda")
        )
        / 10
    )

    seqlens_q = torch.full((num_batch,), num_seq_q, dtype=torch.int32, device="cuda")
    seqlens_kvcache = torch.full((num_batch,), num_seq_kv, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens_q]), dim=0
    ).to(torch.int32)

    kvcache_blocks = (seqlens_kvcache + block_size - 1) // block_size
    total_kvcache_blocks = int(kvcache_blocks.sum().item())
    max_kvcache_blocks = int(kvcache_blocks.max().item())
    max_num_blocks = num_batch * (num_seq_kv + block_size - 1) // block_size * 2

    kvcache = torch.zeros(
        max_num_blocks,
        2,
        block_size + scale_rows,
        num_head_kv,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    packed_block_ids = torch.randperm(max_num_blocks, device="cuda")[:total_kvcache_blocks].to(
        torch.int32
    )
    block_ids = torch.empty(num_batch, max_kvcache_blocks, dtype=torch.int32, device="cuda")

    cu_blocks = 0
    for batch_idx in range(num_batch):
        num_blocks = int(kvcache_blocks[batch_idx].item())
        block_ids[batch_idx, :num_blocks] = packed_block_ids[cu_blocks : cu_blocks + num_blocks]
        cu_blocks += num_blocks
        for blk_id in range(num_blocks):
            start = blk_id * block_size
            end = min(start + block_size, num_seq_kv)
            page = int(block_ids[batch_idx, blk_id].item())
            kvcache[page, 0, : end - start] = k[batch_idx, start:end]
            kvcache[page, 1, : end - start] = v[batch_idx, start:end]

    kvcache_fp8 = torch.empty_like(kvcache, dtype=fp8)
    kcache, _ = quant_paged_cache_pertoken(kvcache[:, 0, :, :, :], block_size)
    vcache, vscale = quant_paged_cache_perhead(kvcache[:, 1, :, :, :], block_size)
    kvcache_fp8[:, 0, :, :, :] = kcache
    kvcache_fp8[:, 1, :, :, :] = vcache

    return q, kvcache_fp8, qscale, vscale, cu_seqlens_q, block_ids, seqlens_kvcache
