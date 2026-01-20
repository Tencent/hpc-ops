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


def naive_attn_func(q, k, v, causal=True):
    num_batch, num_seq_q, num_head_q, num_dim_qk = q.shape
    _, num_seq_kv, num_head_kv, _ = k.shape
    _, _, _, num_dim_v = v.shape

    assert k.shape[:3] == v.shape[:3]
    assert num_head_q % num_head_kv == 0

    num_groups = num_head_q // num_head_kv

    q = q.transpose(1, 2)
    k = k.transpose(1, 2).repeat_interleave(num_groups, dim=1)
    v = v.transpose(1, 2).repeat_interleave(num_groups, dim=1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(num_dim_qk)
    if causal:
        causal_mask = (
            torch.tril(torch.ones(num_seq_q, num_seq_kv, device=q.device, dtype=torch.bool))
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, num_seq_q, num_seq_kv)
    else:
        causal_mask = causal_mask.view(1, 1, num_seq_q, num_seq_kv)

    scores = scores.masked_fill(~causal_mask, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, v)

    return output.transpose(1, 2)


def naive_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
):
    """
    q: (total_q, num_heads, head_dim)
    k, v: (total_k, num_heads, head_dim)
    cu_seqlens_q/k: (batch_size + 1,)
    """
    total_q, num_heads, head_dim = q.shape
    total_k = k.shape[0]
    batch_size = cu_seqlens_q.shape[0] - 1

    outputs = []
    attn_probs = [] if return_attn_probs else None

    for i in range(batch_size):
        q_start, q_end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        k_start, k_end = cu_seqlens_k[i], cu_seqlens_k[i + 1]

        qi = q[q_start:q_end]  # (seqlen_q, num_heads, head_dim)
        ki = k[k_start:k_end]  # (seqlen_k, num_heads, head_dim)
        vi = v[k_start:k_end]  # (seqlen_k, num_heads, head_dim)

        scores = torch.einsum("qhd,khd->hqk", qi, ki)  # (num_heads, seqlen_q, seqlen_k)

        if softmax_scale is None:
            softmax_scale = 1.0 / (head_dim**0.5)
        scores = scores * softmax_scale

        if causal:
            causal_mask = (
                torch.triu(torch.ones(qi.shape[0], ki.shape[0]), diagonal=1)
                .bool()
                .to(device=qi.device)
            )  # (seqlen_q, seqlen_k)
            scores.masked_fill_(causal_mask, float("-inf"))

        attn_prob = F.softmax(scores, dim=-1)  # (num_heads, seqlen_q, seqlen_k)

        out = torch.einsum("hqk,khd->qhd", attn_prob, vi)  # (seqlen_q, num_heads, head_dim)
        outputs.append(out)

        if return_attn_probs:
            attn_probs.append(attn_prob)

    output = torch.cat(outputs, dim=0)  # (total_q, num_heads, head_dim)

    if return_attn_probs:
        return output, attn_probs
    return output


try:
    from flash_attn_interface import flash_attn_func, flash_attn_varlen_func

    gt_attention_func = flash_attn_varlen_func
except Exception as e:
    print(f"execute naive_attn_func: {e}")
    gt_attention_func = naive_attn_varlen_func


@pytest.mark.parametrize("num_batch", [1, 4])
@pytest.mark.parametrize("num_seq_q", [2007, 3907])
@pytest.mark.parametrize("num_seq_kv", [2007, 3907])
@pytest.mark.parametrize("num_head_q", [4])
@pytest.mark.parametrize("num_head_kv", [1])
@pytest.mark.parametrize("num_dim_qk", [128])
@pytest.mark.parametrize("num_dim_v", [128])
@pytest.mark.parametrize("use_output", [True, False])
def test_attention_prefill_bf16(
    num_batch, num_seq_q, num_seq_kv, num_head_q, num_head_kv, num_dim_qk, num_dim_v, use_output
):

    total_seq_q = num_batch * num_seq_q
    q = torch.randn(
        (total_seq_q, num_head_q, num_dim_qk), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(num_dim_qk)
    k = torch.randn(
        (total_seq_q, num_head_kv, num_dim_qk), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(num_dim_qk)
    v = torch.randn((total_seq_q, num_head_kv, num_dim_v), dtype=torch.bfloat16, device="cuda")

    seqlens_q = torch.full((num_batch,), num_seq_q, dtype=torch.int32, device="cuda")

    cu_seqlens_q = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens_q]), dim=0
    ).to(torch.int32)

    max_seqlens_q = max(seqlens_q)

    gt = gt_attention_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_q,
        max_seqlen_q=max_seqlens_q,
        max_seqlen_k=max_seqlens_q,
        causal=True,
    )

    if use_output:
        my = torch.empty_like(q)
        hpc.attention_prefill_bf16(q, k, v, seqlens_q, cu_seqlens_q, max_seqlens_q, output=my)
    else:
        my = hpc.attention_prefill_bf16(q, k, v, seqlens_q, cu_seqlens_q, max_seqlens_q)

    gt = gt.reshape(-1, num_head_q, num_dim_v)

    assert allclose(gt, my, atol=0.016)
