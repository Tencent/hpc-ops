"""CI correctness tests for block-sparse FP8 prefill attention kernels.

Covers two operators:
  1. attention_with_kvcache_blocksparse_prefill_fp8  (paged KV cache, dim=128)
  2. attention_blocksparse_prefill_fp8               (varlen MLA, dim_qk=192, dim_v=128)
"""

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

BSA_BLOCK = 128


# ====================================================================
# Shared helpers
# ====================================================================


def generate_block_sparse_mask(batch, heads, nrow, ncol, skip_ratio, causal=True, device="cuda"):
    """Block-level sparse mask.  True = attend."""
    mask = torch.rand(batch, heads, nrow, ncol, device=device) >= skip_ratio

    row_idx = torch.arange(nrow, device=device).view(nrow, 1)
    col_idx = torch.arange(ncol, device=device).view(1, ncol)

    if causal:
        causal_boundary = row_idx + (ncol - nrow)
        valid_causal = col_idx <= causal_boundary
        mask = mask & valid_causal
        diag_col = torch.clamp(causal_boundary, max=ncol - 1)
        mask = mask | (col_idx == diag_col)

    return mask


# ====================================================================
# Kernel 1: paged KV cache blocksparse prefill fp8 (dim=128)
# ====================================================================


def naive_attn_with_kvcache_sparse(
    q,
    k_cache,
    v_cache,
    qkscale,
    vscale,
    cache_seqlens,
    page_table,
    block_mask=None,
    causal=True,
):
    """FP8 paged-KV-cache naive reference with optional block-sparse mask."""
    num_batch, num_seq_q, num_head_q, num_dim_qk = q.shape
    num_blocks, block_size, num_head_kv, _ = k_cache.shape
    _, _, _, num_dim_v = v_cache.shape
    num_group = num_head_q // num_head_kv
    output = torch.empty_like(q).to(torch.bfloat16)
    kvcache_blocks = (cache_seqlens + block_size - 1) // block_size

    for i in range(num_batch):
        BQ = q[i].transpose(0, 1).float()
        blk_ids = page_table[i, : kvcache_blocks[i]]
        num_seq_kv = cache_seqlens[i]
        BK = (
            k_cache[blk_ids]
            .reshape(-1, num_head_kv, num_dim_qk)
            .transpose(0, 1)[:, :num_seq_kv, :]
            .repeat_interleave(num_group, dim=0)
        ).float()
        BV = (
            v_cache[blk_ids]
            .reshape(-1, num_head_kv, num_dim_v)
            .transpose(0, 1)[:, :num_seq_kv, :]
            .repeat_interleave(num_group, dim=0)
        ).float()

        scale = qkscale[i, :, :].unsqueeze(-1)[:, :num_seq_q, :]
        scores = torch.matmul(BQ, BK.transpose(-2, -1)) * scale / math.sqrt(num_dim_qk)

        if block_mask is not None:
            bm = block_mask[i]
            elem_mask = bm.repeat_interleave(BSA_BLOCK, dim=-2)[:, :num_seq_q, :]
            elem_mask = elem_mask.repeat_interleave(BSA_BLOCK, dim=-1)[:, :, :num_seq_kv]
            scores = scores.masked_fill(~elem_mask, float("-inf"))

        if causal:
            cm = (
                torch.tril(torch.ones(num_seq_kv, num_seq_kv, device=q.device, dtype=torch.bool))[
                    (num_seq_kv - num_seq_q) :, :
                ]
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = scores.masked_fill(~cm, float("-inf"))

        attn_w = F.softmax(scores, dim=-1)
        output[i] = torch.matmul(attn_w, BV).transpose(1, 2) * vscale[0]
    return output


@pytest.mark.parametrize("num_batch", [2])
@pytest.mark.parametrize("num_seq", [1024, 2048])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("skip_ratio", [0.0, 0.5])
def test_kvcache_blocksparse_prefill_fp8(
    num_batch,
    num_seq,
    num_head_q,
    num_head_kv,
    head_dim,
    skip_ratio,
):
    T_bf16 = torch.bfloat16
    T_fp8 = torch.float8_e4m3fn
    device = "cuda"
    block_size = 64

    num_seq_q_pad = (num_seq + 127) // 128 * 128

    Q = (
        torch.randn(num_batch, num_seq, num_head_q, head_dim, dtype=T_bf16, device=device)
        / math.sqrt(head_dim)
    ).to(T_fp8)
    qscale = (
        torch.abs(
            torch.randn(num_batch, num_head_q, num_seq_q_pad, dtype=torch.float32, device=device)
        )
        / 10
    )
    kscale = torch.rand(1, dtype=torch.float32, device=device) + 0.5
    vscale = torch.randn(1, dtype=torch.float32, device=device)

    seqlens_q = torch.full((num_batch,), num_seq, dtype=torch.int32, device=device)
    seqlens_kvcache = torch.full((num_batch,), num_seq, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.zeros(num_batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)

    max_num_blocks = num_batch * ((num_seq + block_size - 1) // block_size) * 2
    kvcache_blocks = (seqlens_kvcache + block_size - 1) // block_size
    total_kb = kvcache_blocks.sum().item()
    max_kb = kvcache_blocks.max().item()

    kvcache = torch.randn(
        max_num_blocks, 2, block_size, num_head_kv, head_dim, dtype=T_bf16, device=device
    ).to(T_fp8)
    packed_ids = torch.randperm(max_num_blocks, device=device)[:total_kb].to(torch.int32)
    block_ids = torch.empty(num_batch, max_kb, dtype=torch.int32, device=device)
    cu = 0
    for i in range(num_batch):
        nb = kvcache_blocks[i].item()
        block_ids[i, :nb] = packed_ids[cu : cu + nb]
        cu += nb

    # Build block mask
    block_mask = None
    block_mask_u8 = None
    if skip_ratio > 0:
        ntiles = (num_seq + BSA_BLOCK - 1) // BSA_BLOCK
        block_mask = generate_block_sparse_mask(
            num_batch, num_head_q, ntiles, ntiles, skip_ratio, causal=True, device=device
        )
        block_mask_u8 = block_mask.to(torch.uint8).contiguous()

    # Reference
    gt = naive_attn_with_kvcache_sparse(
        Q,
        kvcache[:, 0],
        kvcache[:, 1],
        qscale * kscale,
        vscale,
        seqlens_kvcache,
        block_ids,
        block_mask=block_mask,
        causal=True,
    ).reshape(-1, num_head_q, head_dim)

    # HPC kernel
    q_flat = Q.reshape(-1, num_head_q, head_dim)
    my = hpc.attention_with_kvcache_blocksparse_prefill_fp8(
        q_flat,
        kvcache[:, 0],
        kvcache[:, 1],
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        num_seq,
        block_mask_u8,
    )

    assert allclose(gt, my, atol=0.1)


# ====================================================================
# Kernel 2: varlen MLA blocksparse prefill fp8 (dim_qk=192, dim_v=128)
# ====================================================================


def naive_attention_varlen_fp8(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    softmax_scale,
    q_descale=1.0,
    k_descale=1.0,
    v_descale=1.0,
    causal=True,
    block_mask=None,
):
    """Varlen FP8 attention reference (float32 internal, bf16 output)."""
    num_batch = cu_seqlens_q.shape[0] - 1
    Hq = q.shape[1]
    Hkv = k.shape[1]
    dim_v = v.shape[2]
    num_groups = Hq // Hkv
    output = torch.empty(q.shape[0], Hq, dim_v, dtype=torch.bfloat16, device=q.device)

    for i in range(num_batch):
        sq_s = cu_seqlens_q[i].item()
        sq_e = cu_seqlens_q[i + 1].item()
        skv_s = cu_seqlens_kv[i].item()
        skv_e = cu_seqlens_kv[i + 1].item()
        Sq = sq_e - sq_s
        Skv = skv_e - skv_s

        BQ = q[sq_s:sq_e].transpose(0, 1).float() * q_descale
        BK = k[skv_s:skv_e].transpose(0, 1).float() * k_descale
        BV = v[skv_s:skv_e].transpose(0, 1).float()
        if num_groups > 1:
            BK = BK.repeat_interleave(num_groups, dim=0)
            BV = BV.repeat_interleave(num_groups, dim=0)

        scores = torch.matmul(BQ, BK.transpose(-2, -1)) * softmax_scale

        if block_mask is not None:
            bm = block_mask[i]
            elem_mask = bm.repeat_interleave(BSA_BLOCK, dim=-2)[:, :Sq, :]
            elem_mask = elem_mask.repeat_interleave(BSA_BLOCK, dim=-1)[:, :, :Skv]
            scores.masked_fill_(~elem_mask, float("-inf"))

        if causal:
            cm = torch.tril(torch.ones(Skv, Skv, device=q.device, dtype=torch.bool))[
                (Skv - Sq) :, :
            ]
            scores.masked_fill_(~cm.unsqueeze(0), float("-inf"))

        attn_w = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_w, BV) * v_descale
        output[sq_s:sq_e] = out.transpose(0, 1).to(torch.bfloat16)

    return output


@pytest.mark.parametrize("num_batch", [2])
@pytest.mark.parametrize("num_seq", [1024, 2048])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1)])
@pytest.mark.parametrize("dim_qk,dim_v", [(192, 128)])
@pytest.mark.parametrize("skip_ratio", [0.0, 0.5])
def test_blocksparse_prefill_fp8(
    num_batch,
    num_seq,
    num_head_q,
    num_head_kv,
    dim_qk,
    dim_v,
    skip_ratio,
):
    T_fp8 = torch.float8_e4m3fn
    device = "cuda"
    total_tokens = num_batch * num_seq

    Q = (
        torch.randn(total_tokens, num_head_q, dim_qk, dtype=torch.bfloat16, device=device)
        / math.sqrt(dim_qk)
    ).to(T_fp8)
    K = (
        torch.randn(total_tokens, num_head_kv, dim_qk, dtype=torch.bfloat16, device=device)
        / math.sqrt(dim_qk)
    ).to(T_fp8)
    V = torch.randn(total_tokens, num_head_kv, dim_v, dtype=torch.bfloat16, device=device).to(T_fp8)

    seqlens = torch.full((num_batch,), num_seq, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.zeros(num_batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens, dim=0)
    cu_seqlens_kv = cu_seqlens_q.clone()

    softmax_scale = dim_qk ** (-0.5)
    q_scale = torch.rand(1, dtype=torch.float32, device=device) + 0.5
    k_scale = torch.rand(1, dtype=torch.float32, device=device) + 0.5
    v_scale = torch.rand(1, dtype=torch.float32, device=device) + 0.5

    # Build block mask
    block_mask = None
    block_mask_u8 = None
    if skip_ratio > 0:
        ntiles = (num_seq + BSA_BLOCK - 1) // BSA_BLOCK
        block_mask = generate_block_sparse_mask(
            num_batch, num_head_q, ntiles, ntiles, skip_ratio, causal=True, device=device
        )
        block_mask_u8 = block_mask.to(torch.uint8).contiguous()

    # Reference
    gt = naive_attention_varlen_fp8(
        Q,
        K,
        V,
        cu_seqlens_q,
        cu_seqlens_kv,
        softmax_scale,
        q_descale=q_scale.item(),
        k_descale=k_scale.item(),
        v_descale=v_scale.item(),
        causal=True,
        block_mask=block_mask,
    )

    # HPC kernel
    my = hpc.attention_blocksparse_prefill_fp8(
        Q,
        K,
        V,
        cu_seqlens_q,
        cu_seqlens_kv,
        num_seq,
        num_seq,
        q_scale,
        k_scale,
        v_scale,
        softmax_scale=softmax_scale,
        block_mask=block_mask_u8,
    )

    assert allclose(gt, my, atol=0.1)
