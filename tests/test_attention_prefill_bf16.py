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


def naive_attn_func(Q, K, V, causal=True):
    num_batch, num_seq_q, num_head_q, num_dim_qk = Q.shape
    _, num_seq_kv, num_head_kv, _ = K.shape
    _, _, _, num_dim_v = V.shape

    assert K.shape[:3] == V.shape[:3]
    assert num_head_q % num_head_kv == 0

    num_groups = num_head_q // num_head_kv

    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2).repeat_interleave(num_groups, dim=1)
    V = V.transpose(1, 2).repeat_interleave(num_groups, dim=1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(num_dim_qk)
    if causal:
        causal_mask = (
            torch.tril(torch.ones(num_seq_q, num_seq_kv, device=Q.device, dtype=torch.bool))
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, num_seq_q, num_seq_kv)
    else:
        causal_mask = causal_mask.view(1, 1, num_seq_q, num_seq_kv)

    scores = scores.masked_fill(~causal_mask, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, V)

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

    torch.cuda.manual_seed(2)

    T = torch.bfloat16

    total_seq_q = num_batch * num_seq_q
    Q = torch.randn((total_seq_q, num_head_q, num_dim_qk), dtype=T, device="cuda") / math.sqrt(
        num_dim_qk
    )
    K = torch.randn((total_seq_q, num_head_kv, num_dim_qk), dtype=T, device="cuda") / math.sqrt(
        num_dim_qk
    )
    V = torch.randn((total_seq_q, num_head_kv, num_dim_v), dtype=T, device="cuda")

    seqlens_q = torch.full((num_batch,), num_seq_q, dtype=torch.int32, device="cuda")
    # seqlens_q = (
    #   num_seq_q + torch.randint(-2, 2, (num_batch,), device="cuda", dtype=torch.int32) * 1000
    # )
    cu_seqlens_q = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens_q]), dim=0
    ).to(torch.int32)

    max_seqlens_q = max(seqlens_q)

    for i in range(1):
        gt = gt_attention_func(
            Q,
            K,
            V,
            cu_seqlens_q,
            cu_seqlens_q,
            max_seqlen_q=max_seqlens_q,
            max_seqlen_k=max_seqlens_q,
            causal=True,
        )
        if use_output:
            my = torch.empty_like(Q)
            hpc.attention_prefill_bf16(Q, K, V, seqlens_q, cu_seqlens_q, max_seqlens_q, output=my)
        else:
            my = hpc.attention_prefill_bf16(Q, K, V, seqlens_q, cu_seqlens_q, max_seqlens_q)

    gt = gt.reshape(-1, num_head_q, num_dim_v)
    print("\ngt\n")
    print(gt[:5, :, 9:10])
    print("\nmy\n")
    print(my[:5, :, 9:10])

    assert allclose(gt, my, atol=0.016)


@pytest.mark.parametrize("num_batch", [1])
@pytest.mark.parametrize("num_seq_q", [8000, 16000])
@pytest.mark.parametrize("num_seq_kv", [8000, 16000])
@pytest.mark.parametrize("num_head_q", [16])
@pytest.mark.parametrize("num_head_kv", [16])
@pytest.mark.parametrize("num_dim_qk", [192])
@pytest.mark.parametrize("num_dim_v", [128])
@pytest.mark.parametrize("use_output", [False])
def test_mla_prefill_bf16(
    num_batch, num_seq_q, num_seq_kv, num_head_q, num_head_kv, num_dim_qk, num_dim_v, use_output
):

    torch.cuda.manual_seed(2)

    T = torch.bfloat16

    total_seq_q = num_batch * num_seq_q
    Q = torch.randn((total_seq_q, num_head_q, num_dim_qk), dtype=T, device="cuda")
    K = torch.randn((total_seq_q, num_head_kv, num_dim_qk), dtype=T, device="cuda")
    # V = K.clone()[:, :, :num_dim_v]
    V = K.clone()

    seqlens_q = torch.full((num_batch,), num_seq_q, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens_q]), dim=0
    ).to(torch.int32)

    max_seqlens_q = max(seqlens_q)

    my = hpc.mla_prefill_bf16(Q, K, seqlens_q, cu_seqlens_q, num_dim_qk, num_dim_v, max_seqlens_q)

    gt = gt_attention_func(
        Q,
        K,
        V,
        cu_seqlens_q,
        cu_seqlens_q,
        max_seqlen_q=max_seqlens_q,
        max_seqlen_k=max_seqlens_q,
        causal=True,
    )

    gt = gt.reshape(-1, num_head_q, num_dim_qk)[:, :, :num_dim_v]
    # print("\ngt\n")
    # print(gt)
    # print("\nmy\n")
    # print(my)

    assert allclose(gt, my, atol=0.016)
