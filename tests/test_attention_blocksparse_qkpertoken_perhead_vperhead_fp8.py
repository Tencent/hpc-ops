"""CI correctness test for the qkpertoken_perhead_vperhead BSA fp8 prefill kernel.

Covers `attention_with_kvcache_blocksparse_prefill_fp8` with
`QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD` (paged KV cache, dim=128):
  - Q per-token-per-head scale
  - K per-token-group(32) per-head per-dim-group(4) scale loaded via TMA (4D tensor)
  - V per-head scale
"""

import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
from utils import allclose

BSA_BLOCK = 128
QUANT_TYPE_NEW = hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD


def generate_block_sparse_mask(batch, heads, nrow, ncol, skip_ratio, causal=True, device="cuda"):
    """Block-level sparse mask.  True = attend.  Diagonal always kept under causal."""
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


def naive_attn_with_kvcache_qkpv_sparse_fixed_pscale(
    q,
    k_cache,
    v_cache,
    qscale,
    kscale,
    vscale,
    cache_seqlens,
    page_table,
    block_mask=None,
    causal=True,
):
    """Sparse qkpertoken/vperhead reference for the fixed P-scale=256 FP8 path."""
    num_batch, num_seq_q, num_head_q, num_dim_qk = q.shape
    _, block_size, num_head_kv, _ = k_cache.shape
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

        attn_weights = torch.exp(scores - scores.max(dim=-1, keepdim=True)[0])
        gsum = attn_weights.sum(dim=-1, keepdim=True)
        attn_weights = attn_weights * 256.0
        attn_weights = attn_weights.to(torch.float8_e4m3fn).float()

        out_head = torch.matmul(attn_weights, BV) / gsum
        vscale_eff = vscale[:, None, None].repeat_interleave(num_group, dim=0)
        vscale_eff = vscale_eff / 256.0
        output[i] = (out_head * vscale_eff).transpose(1, 2)

    return output


@pytest.mark.parametrize("num_batch", [2])
@pytest.mark.parametrize("num_seq", [1024, 2048])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("skip_ratio", [0.0, 0.5])
@pytest.mark.parametrize("kv_layout", ["nhd", "hnd"])
def test_kvcache_blocksparse_qkpertoken_perhead_vperhead_prefill_fp8(
    kv_layout,
    num_batch,
    num_seq,
    num_head_q,
    num_head_kv,
    head_dim,
    skip_ratio,
):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)

    T_bf16 = torch.bfloat16
    T_fp8 = torch.float8_e4m3fn
    device = "cuda"
    block_size = 64

    # K-scale geometry (matches dense kernel): 32 tokens per scale-group, 4 dims per dim-group.
    kScaleBlockSize = block_size // 32
    num_dim_scale = head_dim // 4

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
    vscale = torch.abs(torch.randn(num_head_kv, dtype=torch.float32, device=device))

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
    if kv_layout == "hnd":
        kcache = kvcache[:, 0].transpose(1, 2).contiguous().transpose(1, 2)
        vcache = kvcache[:, 1].transpose(1, 2).contiguous().transpose(1, 2)
    else:
        kcache = kvcache[:, 0]
        vcache = kvcache[:, 1]
    packed_ids = torch.randperm(max_num_blocks, device=device)[:total_kb].to(torch.int32)
    block_ids = torch.empty(num_batch, max_kb, dtype=torch.int32, device=device)
    cu = 0
    for i in range(num_batch):
        nb = kvcache_blocks[i].item()
        block_ids[i, :nb] = packed_ids[cu : cu + nb]
        cu += nb

    # K-scale: generated as fp32, viewed as fp8 to exercise the fp8-dtype dispatch path.
    kscale = torch.abs(
        torch.randn(
            (max_num_blocks, kScaleBlockSize, num_head_kv, num_dim_scale),
            dtype=torch.float32,
            device=device,
        )
    ).view(torch.float8_e4m3fn)
    kscale_f32 = kscale.view(torch.float32)

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
    gt = naive_attn_with_kvcache_qkpv_sparse_fixed_pscale(
        Q,
        kcache,
        vcache,
        qscale,
        kscale_f32,
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
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        num_seq,
        quant_type=QUANT_TYPE_NEW,
        block_mask=block_mask_u8,
    )

    assert allclose(gt, my, atol=0.1)


@pytest.mark.parametrize("skip_ratio", [0.0, 0.5])
def test_kvcache_blocksparse_qkpertoken_perhead_vperhead_prefill_fp8_fixed_pscale(skip_ratio):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)

    T_bf16 = torch.bfloat16
    T_fp8 = torch.float8_e4m3fn
    device = "cuda"
    block_size = 64
    num_batch = 2
    num_seq = 1024
    num_head_q = 4
    num_head_kv = 1
    head_dim = 128
    kScaleBlockSize = block_size // 32
    num_dim_scale = head_dim // 4
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
    vscale = torch.abs(torch.randn(num_head_kv, dtype=torch.float32, device=device))

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
    kcache = kvcache[:, 0]
    vcache = kvcache[:, 1]
    packed_ids = torch.randperm(max_num_blocks, device=device)[:total_kb].to(torch.int32)
    block_ids = torch.empty(num_batch, max_kb, dtype=torch.int32, device=device)
    cu = 0
    for i in range(num_batch):
        nb = kvcache_blocks[i].item()
        block_ids[i, :nb] = packed_ids[cu : cu + nb]
        cu += nb

    kscale = torch.abs(
        torch.randn(
            (max_num_blocks, kScaleBlockSize, num_head_kv, num_dim_scale),
            dtype=torch.float32,
            device=device,
        )
    ).view(torch.float8_e4m3fn)
    kscale_f32 = kscale.view(torch.float32)

    block_mask = None
    block_mask_u8 = None
    if skip_ratio > 0:
        ntiles = (num_seq + BSA_BLOCK - 1) // BSA_BLOCK
        block_mask = generate_block_sparse_mask(
            num_batch, num_head_q, ntiles, ntiles, skip_ratio, causal=True, device=device
        )
        block_mask_u8 = block_mask.to(torch.uint8).contiguous()

    gt = naive_attn_with_kvcache_qkpv_sparse_fixed_pscale(
        Q,
        kcache,
        vcache,
        qscale,
        kscale_f32,
        vscale,
        seqlens_kvcache,
        block_ids,
        block_mask=block_mask,
        causal=True,
    ).reshape(-1, num_head_q, head_dim)

    my = hpc.attention_with_kvcache_blocksparse_prefill_fp8(
        Q.reshape(-1, num_head_q, head_dim),
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        num_seq,
        quant_type=QUANT_TYPE_NEW,
        block_mask=block_mask_u8,
    )

    assert allclose(gt, my, atol=0.05)
