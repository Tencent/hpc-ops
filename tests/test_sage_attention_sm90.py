import sys
import os
import math
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import pytest
from utils import allclose

from hpc.sage_attention import sageattn_qk_int8_pv_fp8


def ref_attention(q, k, v, tensor_layout="NHD", is_causal=False):
    """bf16 scaled dot-product attention reference (with optional causal mask)."""
    head_dim = q.size(-1)
    sm_scale = 1.0 / math.sqrt(head_dim)
    if tensor_layout == "NHD":
        q_r, k_r, v_r = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    else:
        q_r, k_r, v_r = q, k, v
    gqa_ratio = q_r.size(1) // k_r.size(1)
    q_r = q_r.float()
    k_r = k_r.float().repeat_interleave(gqa_ratio, dim=1)
    v_r = v_r.float().repeat_interleave(gqa_ratio, dim=1)
    scores = q_r @ k_r.transpose(-2, -1) * sm_scale
    if is_causal:
        qo_len = q_r.size(-2)
        kv_len = k_r.size(-2)
        mask = torch.ones(qo_len, kv_len, dtype=torch.bool, device=q.device).tril(
            diagonal=kv_len - qo_len
        )
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = attn @ v_r
    if tensor_layout == "NHD":
        out = out.transpose(1, 2)
    return out.bfloat16()


@pytest.mark.parametrize("seqlen", [1, 3, 16, 64, 128, 256, 512, 1024, 2048, 4099])
@pytest.mark.parametrize("hq,hkv", [(1, 1), (8, 2), (16, 4), (32, 8)])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_sage_attention_sm90(batch, seqlen, hq, hkv, layout):
    torch.manual_seed(42)
    if layout == "NHD":
        q = torch.randn(batch, seqlen, hq, 128, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, seqlen, hkv, 128, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, seqlen, hkv, 128, dtype=torch.bfloat16, device="cuda")
    else:
        q = torch.randn(batch, hq, seqlen, 128, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, hkv, seqlen, 128, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, hkv, seqlen, 128, dtype=torch.bfloat16, device="cuda")

    out = sageattn_qk_int8_pv_fp8(q, k, v, tensor_layout=layout)
    ref = ref_attention(q, k, v, tensor_layout=layout)

    cos = torch.nn.functional.cosine_similarity(
        out.float().reshape(-1, 128), ref.float().reshape(-1, 128), dim=-1
    )
    assert cos.mean().item() > 0.99, f"cos_mean={cos.mean():.6f}"
    assert allclose(out, ref, atol=0.2, rtol=0.5)
