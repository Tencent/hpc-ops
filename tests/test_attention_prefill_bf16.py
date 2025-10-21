import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
import torch.nn.functional as F


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


try:
    from flash_attn_interface import flash_attn_func

    gt_attention_func = flash_attn_func
except Exception as e:
    print(f"execute naive_attn_func: {e}")
    gt_attention_func = naive_attn_func


@pytest.mark.parametrize("num_batch", [4])
@pytest.mark.parametrize("num_seq_q", [3904])
@pytest.mark.parametrize("num_seq_kv", [3904])
@pytest.mark.parametrize("num_head_q", [4])
@pytest.mark.parametrize("num_head_kv", [1])
@pytest.mark.parametrize("num_dim_qk", [128])
@pytest.mark.parametrize("num_dim_v", [128])
def test_attention_prefill_bf16(
    num_batch, num_seq_q, num_seq_kv, num_head_q, num_head_kv, num_dim_qk, num_dim_v
):

    # torch.cuda.manual_seed(10086)

    T = torch.bfloat16

    Q = torch.randn(
        (num_batch, num_seq_q, num_head_q, num_dim_qk), dtype=T, device="cuda"
    ) / math.sqrt(num_dim_qk)
    K = torch.randn(
        (num_batch, num_seq_kv, num_head_kv, num_dim_qk), dtype=T, device="cuda"
    ) / math.sqrt(num_dim_qk)
    V = torch.randn((num_batch, num_seq_kv, num_head_kv, num_dim_v), dtype=T, device="cuda")

    for i in range(10):
        gt = gt_attention_func(Q, K, V, causal=True)
        my = hpc.attention_prefill_bf16(Q, K, V)

    print("\ngt\n")
    print(gt[0, :, :])
    print("\nmy\n")
    print(my[0, :, :])

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
