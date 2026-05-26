"""CI correctness tests for dim192 varlen block-sparse FP8 prefill attention."""

import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
from utils import allclose

BSA_BLOCK = 128


def generate_block_sparse_mask(batch, heads, nrow, ncol, skip_ratio, causal=True, device="cuda"):
    """Block-level sparse mask. True = attend."""
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
    """Varlen FP8 attention reference with FP32 internal math and BF16 output."""
    num_batch = cu_seqlens_q.shape[0] - 1
    num_head_q = q.shape[1]
    num_head_kv = k.shape[1]
    dim_v = v.shape[2]
    num_groups = num_head_q // num_head_kv
    output = torch.empty(q.shape[0], num_head_q, dim_v, dtype=torch.bfloat16, device=q.device)

    for i in range(num_batch):
        sq_s = cu_seqlens_q[i].item()
        sq_e = cu_seqlens_q[i + 1].item()
        skv_s = cu_seqlens_kv[i].item()
        skv_e = cu_seqlens_kv[i + 1].item()
        seq_q = sq_e - sq_s
        seq_kv = skv_e - skv_s

        bq = q[sq_s:sq_e].transpose(0, 1).float() * q_descale
        bk = k[skv_s:skv_e].transpose(0, 1).float() * k_descale
        bv = v[skv_s:skv_e].transpose(0, 1).float()
        if num_groups > 1:
            bk = bk.repeat_interleave(num_groups, dim=0)
            bv = bv.repeat_interleave(num_groups, dim=0)

        scores = torch.matmul(bq, bk.transpose(-2, -1)) * softmax_scale

        if block_mask is not None:
            bm = block_mask[i]
            elem_mask = bm.repeat_interleave(BSA_BLOCK, dim=-2)[:, :seq_q, :]
            elem_mask = elem_mask.repeat_interleave(BSA_BLOCK, dim=-1)[:, :, :seq_kv]
            scores.masked_fill_(~elem_mask, float("-inf"))

        if causal:
            cm = torch.tril(torch.ones(seq_kv, seq_kv, device=q.device, dtype=torch.bool))[
                (seq_kv - seq_q) :, :
            ]
            scores.masked_fill_(~cm.unsqueeze(0), float("-inf"))

        # Mirror the kernel's online-softmax + fixed-P-scale fp8 quant path.
        attn_w = torch.exp(scores - scores.max(dim=-1, keepdim=True)[0])
        gsum = attn_w.sum(dim=-1, keepdim=True)
        attn_w = (attn_w * 256.0).to(torch.float8_e4m3fn).float()
        out = torch.matmul(attn_w, bv) / gsum
        out = out * (v_descale / 256.0)
        output[sq_s:sq_e] = out.transpose(0, 1).to(torch.bfloat16)

    return output


@pytest.mark.parametrize("num_batch", [2])
@pytest.mark.parametrize("num_seq", [1024, 2048])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1)])
@pytest.mark.parametrize("dim_qk,dim_v", [(192, 128)])
@pytest.mark.parametrize("skip_ratio", [0.0, 0.5])
def test_blocksparse_prefill_fp8_dim192(
    num_batch,
    num_seq,
    num_head_q,
    num_head_kv,
    dim_qk,
    dim_v,
    skip_ratio,
):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)

    dtype_fp8 = torch.float8_e4m3fn
    device = "cuda"
    total_tokens = num_batch * num_seq

    q = (
        torch.randn(total_tokens, num_head_q, dim_qk, dtype=torch.bfloat16, device=device)
        / math.sqrt(dim_qk)
    ).to(dtype_fp8)
    k = (
        torch.randn(total_tokens, num_head_kv, dim_qk, dtype=torch.bfloat16, device=device)
        / math.sqrt(dim_qk)
    ).to(dtype_fp8)
    v = torch.randn(total_tokens, num_head_kv, dim_v, dtype=torch.bfloat16, device=device).to(
        dtype_fp8
    )

    seqlens = torch.full((num_batch,), num_seq, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.zeros(num_batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens, dim=0)
    cu_seqlens_kv = cu_seqlens_q.clone()

    softmax_scale = dim_qk**-0.5
    q_scale = torch.rand(1, dtype=torch.float32, device=device) + 0.5
    k_scale = torch.rand(1, dtype=torch.float32, device=device) + 0.5
    v_scale = torch.rand(1, dtype=torch.float32, device=device) + 0.5

    block_mask = None
    block_mask_u8 = None
    if skip_ratio > 0:
        ntiles = (num_seq + BSA_BLOCK - 1) // BSA_BLOCK
        block_mask = generate_block_sparse_mask(
            num_batch, num_head_q, ntiles, ntiles, skip_ratio, causal=True, device=device
        )
        block_mask_u8 = block_mask.to(torch.uint8).contiguous()

    gt = naive_attention_varlen_fp8(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        softmax_scale,
        q_descale=q_scale.item(),
        k_descale=k_scale.item(),
        v_descale=v_scale.item(),
        causal=True,
        block_mask=block_mask,
    )

    out = hpc.attention_blocksparse_prefill_fp8_dim192(
        q,
        k,
        v,
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

    assert allclose(gt, out, atol=0.1)
