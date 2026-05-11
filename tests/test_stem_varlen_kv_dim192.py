"""CI smoke tests for dim192 Stem varlen FP8 scalar-scale path."""

import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch

STEM_BLOCK_SIZE = 128
STEM_STRIDE = 16
LAMBDA_MAG = 0.3
ALPHA = 1.0
INITIAL_BLOCKS = 4
WINDOW_SIZE = 4
K_BLOCK_NUM_RATE = 0.1
K_BLOCK_NUM_BIAS = 30


def _setup_varlen_fp8_data(
    num_batch,
    seq_len,
    num_head_q,
    num_head_kv,
    dim_qk=192,
    dim_v=128,
):
    """Create dim192 ragged FP8 data with scalar Q/K/V scales."""
    dtype_fp8 = torch.float8_e4m3fn
    device = "cuda"
    total_tokens = num_batch * seq_len

    q_fp8 = (
        torch.randn(total_tokens, num_head_q, dim_qk, dtype=torch.bfloat16, device=device)
        / math.sqrt(dim_qk)
    ).to(dtype_fp8)
    k_fp8 = (
        torch.randn(total_tokens, num_head_kv, dim_qk, dtype=torch.bfloat16, device=device)
        / math.sqrt(dim_qk)
    ).to(dtype_fp8)
    v_fp8 = torch.randn(total_tokens, num_head_kv, dim_v, dtype=torch.bfloat16, device=device).to(
        dtype_fp8
    )

    seqlens = torch.full((num_batch,), seq_len, dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros(num_batch + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)

    return dict(
        q_fp8=q_fp8,
        k_fp8=k_fp8,
        v_fp8=v_fp8,
        qscale=torch.ones(1, dtype=torch.float32, device=device),
        kscale=torch.ones(1, dtype=torch.float32, device=device),
        vscale=torch.ones(1, dtype=torch.float32, device=device),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens.clone(),
        q_seq_lens=seqlens,
        kv_seq_lens=seqlens.clone(),
    )


@pytest.mark.parametrize("num_batch", [1])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1)])
def test_stem_varlen_prep_dim192(num_batch, seq_len, num_head_q, num_head_kv):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    d = _setup_varlen_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)

    kflat, vbias = hpc.stem_oam_prep_varlen_kv_dim192(
        d["k_fp8"],
        d["v_fp8"],
        d["kscale"],
        d["vscale"],
        d["kv_seq_lens"],
        d["cu_seqlens_kv"],
        lambda_mag=LAMBDA_MAG,
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
    )
    qflat = hpc.stem_oam_prep_varlen_q_dim192(
        d["q_fp8"],
        d["qscale"],
        d["q_seq_lens"],
        d["cu_seqlens_q"],
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
    )

    max_blocks = (seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
    assert kflat.shape == (num_batch, num_head_kv, max_blocks, STEM_STRIDE * 192)
    assert qflat.shape == (num_batch, num_head_q, max_blocks, STEM_STRIDE * 192)
    assert vbias.shape == (num_batch, num_head_kv, max_blocks)
    assert kflat.dtype == torch.bfloat16
    assert qflat.dtype == torch.bfloat16
    assert vbias.dtype == torch.float32


@pytest.mark.parametrize("num_batch", [1])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1)])
def test_stem_oam_gemm_dim192(num_batch, seq_len, num_head_q, num_head_kv):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    d = _setup_varlen_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)
    q_seq_lens = d["q_seq_lens"]

    kflat, vbias = hpc.stem_oam_prep_varlen_kv_dim192(
        d["k_fp8"],
        d["v_fp8"],
        d["kscale"],
        d["vscale"],
        d["kv_seq_lens"],
        d["cu_seqlens_kv"],
        lambda_mag=LAMBDA_MAG,
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
    )
    qflat = hpc.stem_oam_prep_varlen_q_dim192(
        d["q_fp8"],
        d["qscale"],
        q_seq_lens,
        d["cu_seqlens_q"],
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
    )
    block_logits = hpc.stem_oam_gemm_dim192(
        qflat,
        kflat,
        vbias,
        q_seq_lens,
        d["kv_seq_lens"],
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
        causal=True,
    )

    max_blocks = (seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
    assert block_logits.shape == (num_batch, num_head_q, max_blocks, max_blocks)
    assert block_logits.dtype == torch.bfloat16
    finite_ratio = torch.isfinite(block_logits).float().mean().item()
    assert finite_ratio > 0.1, f"Too few finite logits: {finite_ratio:.2%}"


@pytest.mark.parametrize("num_batch", [1])
@pytest.mark.parametrize("seq_len", [2048])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1)])
def test_stem_varlen_kv_dim192_e2e(num_batch, seq_len, num_head_q, num_head_kv):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    d = _setup_varlen_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)

    mask = hpc.stem_varlen_kv_dim192(
        d["q_fp8"],
        d["k_fp8"],
        d["v_fp8"],
        d["qscale"],
        d["kscale"],
        d["vscale"],
        d["cu_seqlens_q"],
        d["cu_seqlens_kv"],
        lambda_mag=LAMBDA_MAG,
        alpha=ALPHA,
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
        causal=True,
        initial_blocks=INITIAL_BLOCKS,
        window_size=WINDOW_SIZE,
        k_block_num_rate=K_BLOCK_NUM_RATE,
        k_block_num_bias=K_BLOCK_NUM_BIAS,
    )

    max_blocks = (seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
    assert mask.shape == (num_batch, num_head_q, max_blocks, max_blocks)
    assert mask.dtype == torch.uint8
    density = mask.float().mean().item()
    assert 0.0 < density < 1.0, f"Mask density {density:.4f} looks degenerate"
    diag = mask[:, :, range(max_blocks), range(max_blocks)]
    assert (diag == 1).all(), "Diagonal blocks must be selected"
