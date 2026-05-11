"""CI correctness tests for dim128 Stem with Q per-token and KV per-tensor scales.

Lightweight smoke tests covering:
  1. stem_oam_prep_paged_kv  — KV prep (paged, dim=128): shape & dtype check
  2. stem_oam_prep_varlen_q  — Q prep (dim=128): shape & dtype check
  3. stem_oam_gemm           — OAM block-logits GEMM: shape & finite check
  4. stem_tpd                — TPD mask generation: shape, dtype & mask sanity
  5. stem_paged_kv           — End-to-end pipeline (dim=128): mask shape & sanity
"""

import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math


# ====================================================================
# Shared helpers
# ====================================================================

STEM_BLOCK_SIZE = 128
STEM_STRIDE = 16
LAMBDA_MAG = 0.3
ALPHA = 1.0
INITIAL_BLOCKS = 4
WINDOW_SIZE = 4
K_BLOCK_NUM_RATE = 0.1
K_BLOCK_NUM_BIAS = 30


def _setup_paged_fp8_data(
    num_batch, seq_len, num_head_q, num_head_kv, dim_qk=128, dim_v=128, kv_block_size=64
):
    """Create random paged FP8 data for dim_qk=128 Stem pipeline."""
    T_fp8 = torch.float8_e4m3fn
    device = "cuda"
    total_tokens = num_batch * seq_len

    q_fp8 = (
        torch.randn(total_tokens, num_head_q, dim_qk, dtype=torch.bfloat16, device=device)
        / math.sqrt(dim_qk)
    ).to(T_fp8)

    num_blocks_per_req = (seq_len + kv_block_size - 1) // kv_block_size
    total_blocks = num_batch * num_blocks_per_req

    kcache_fp8 = (
        torch.randn(
            total_blocks, kv_block_size, num_head_kv, dim_qk, dtype=torch.bfloat16, device=device
        )
        / math.sqrt(dim_qk)
    ).to(T_fp8)
    vcache_fp8 = torch.randn(
        total_blocks,
        kv_block_size,
        num_head_kv,
        dim_v,
        dtype=torch.bfloat16,
        device=device,
    ).to(T_fp8)

    kv_indices = torch.arange(total_blocks, device=device, dtype=torch.int32).reshape(
        num_batch, num_blocks_per_req
    )

    max_seqlen_pad128 = ((seq_len + 127) // 128) * 128
    qscale = torch.ones(
        num_batch, num_head_q, max_seqlen_pad128, dtype=torch.float32, device=device
    )
    kscale = torch.ones(1, dtype=torch.float32, device=device)
    vscale = torch.tensor(1.0, dtype=torch.float32, device=device)

    seqlens = torch.full((num_batch,), seq_len, dtype=torch.int32, device=device)
    cu_q_seqlens = torch.zeros(num_batch + 1, dtype=torch.int32, device=device)
    cu_q_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    kv_seqlens = seqlens.clone()

    return dict(
        q_fp8=q_fp8,
        kcache_fp8=kcache_fp8,
        vcache_fp8=vcache_fp8,
        qscale=qscale,
        kscale=kscale,
        vscale=vscale,
        kv_indices=kv_indices,
        cu_q_seqlens=cu_q_seqlens,
        kv_seqlens=kv_seqlens,
    )


# ====================================================================
# Test 1: stem_oam_prep_paged_kv
# ====================================================================


@pytest.mark.parametrize("num_batch", [1])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1)])
def test_stem_oam_prep_paged_kv(num_batch, seq_len, num_head_q, num_head_kv):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)
    vscale_t = d["vscale"].unsqueeze(0)

    kflat, vbias = hpc.stem_oam_prep_paged_kv(
        d["kcache_fp8"],
        d["vcache_fp8"],
        d["kscale"],
        vscale_t,
        d["kv_indices"],
        d["kv_seqlens"],
        lambda_mag=LAMBDA_MAG,
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
        quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
    )

    max_kv_padded = ((seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE) * STEM_BLOCK_SIZE
    max_kb = max_kv_padded // STEM_BLOCK_SIZE

    assert kflat.dim() == 4
    assert kflat.shape[0] == num_batch
    assert kflat.shape[1] == num_head_kv
    assert kflat.shape[2] == max_kb
    assert kflat.dtype == torch.bfloat16

    assert vbias.dim() == 3
    assert vbias.shape[0] == num_batch
    assert vbias.shape[1] == num_head_kv
    assert vbias.shape[2] == max_kb
    assert vbias.dtype == torch.float32


# ====================================================================
# Test 2: stem_oam_prep_varlen_q
# ====================================================================


@pytest.mark.parametrize("num_batch", [1])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1)])
def test_stem_oam_prep_varlen_q(num_batch, seq_len, num_head_q, num_head_kv):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)
    q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)

    qflat = hpc.stem_oam_prep_varlen_q(
        d["q_fp8"],
        d["qscale"],
        q_seqlens,
        d["cu_q_seqlens"],
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
    )

    max_q_padded = ((seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE) * STEM_BLOCK_SIZE
    max_qb = max_q_padded // STEM_BLOCK_SIZE

    assert qflat.dim() == 4
    assert qflat.shape[0] == num_batch
    assert qflat.shape[1] == num_head_q
    assert qflat.shape[2] == max_qb
    assert qflat.dtype == torch.bfloat16


# ====================================================================
# Test 3: stem_oam_gemm
# ====================================================================


@pytest.mark.parametrize("num_batch", [1])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1)])
def test_stem_oam_gemm(num_batch, seq_len, num_head_q, num_head_kv):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)
    q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)
    vscale_t = d["vscale"].unsqueeze(0)

    kflat, vbias = hpc.stem_oam_prep_paged_kv(
        d["kcache_fp8"],
        d["vcache_fp8"],
        d["kscale"],
        vscale_t,
        d["kv_indices"],
        d["kv_seqlens"],
        lambda_mag=LAMBDA_MAG,
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
        quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
    )
    qflat = hpc.stem_oam_prep_varlen_q(
        d["q_fp8"],
        d["qscale"],
        q_seqlens,
        d["cu_q_seqlens"],
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
    )
    block_logits = hpc.stem_oam_gemm(
        qflat,
        kflat,
        vbias,
        q_seqlens,
        d["kv_seqlens"],
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
        causal=True,
    )

    assert block_logits.dim() == 4
    assert block_logits.shape[0] == num_batch
    assert block_logits.shape[1] == num_head_q
    # Logits should be mostly finite (causal masked positions are -inf)
    finite_ratio = torch.isfinite(block_logits).float().mean().item()
    assert finite_ratio > 0.1, f"Too few finite logits: {finite_ratio:.2%}"


# ====================================================================
# Test 4: stem_tpd
# ====================================================================


@pytest.mark.parametrize("num_batch", [1])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("num_head_q", [4])
def test_stem_tpd(num_batch, seq_len, num_head_q):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    device = "cuda"
    stem_block_size = STEM_BLOCK_SIZE
    max_qb = (seq_len + stem_block_size - 1) // stem_block_size
    max_kb = max_qb

    # Simulate block_logits (bf16, causal-masked)
    block_logits = torch.randn(
        num_batch, num_head_q, max_qb, max_kb, dtype=torch.bfloat16, device=device
    )
    # Apply causal mask
    qr = torch.arange(max_qb, device=device).unsqueeze(1)
    kr = torch.arange(max_kb, device=device).unsqueeze(0)
    causal_mask = kr > qr
    block_logits.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    q_seq_lens = torch.full((num_batch,), seq_len, dtype=torch.int32, device=device)
    kv_seq_lens = q_seq_lens.clone()

    mask = hpc.stem_tpd(
        block_logits,
        q_seq_lens,
        kv_seq_lens,
        block_size=stem_block_size,
        alpha=ALPHA,
        initial_blocks=INITIAL_BLOCKS,
        window_size=WINDOW_SIZE,
        k_block_num_rate=K_BLOCK_NUM_RATE,
        k_block_num_bias=K_BLOCK_NUM_BIAS,
    )

    assert mask.shape == (num_batch, num_head_q, max_qb, max_kb)
    assert mask.dtype == torch.uint8
    # Diagonal should always be selected
    diag = mask[:, :, range(min(max_qb, max_kb)), range(min(max_qb, max_kb))]
    assert (diag == 1).all(), "Diagonal blocks must be selected"
    # initial_blocks should be selected for all Q blocks
    if max_kb >= INITIAL_BLOCKS:
        init = mask[:, :, :, :INITIAL_BLOCKS]
        assert (init == 1).all(), "Initial blocks must be selected"


# ====================================================================
# Test 5: stem_paged_kv (end-to-end pipeline)
# ====================================================================


@pytest.mark.parametrize("num_batch", [1])
@pytest.mark.parametrize("seq_len", [2048])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1)])
def test_stem_paged_kv_e2e(num_batch, seq_len, num_head_q, num_head_kv):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)
    vscale_t = d["vscale"].unsqueeze(0)

    mask = hpc.stem_paged_kv(
        d["q_fp8"],
        d["kcache_fp8"],
        d["vcache_fp8"],
        d["qscale"],
        d["kscale"],
        vscale_t,
        d["kv_indices"],
        d["cu_q_seqlens"],
        d["kv_seqlens"],
        lambda_mag=LAMBDA_MAG,
        alpha=ALPHA,
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
        causal=True,
        initial_blocks=INITIAL_BLOCKS,
        window_size=WINDOW_SIZE,
        k_block_num_rate=K_BLOCK_NUM_RATE,
        k_block_num_bias=K_BLOCK_NUM_BIAS,
        quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
    )

    max_qb = (seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
    assert mask.dim() == 4
    assert mask.shape[0] == num_batch
    assert mask.shape[1] == num_head_q
    assert mask.dtype == torch.uint8
    # Mask should be non-trivial (not all zeros, not all ones for seq_len >= 2048)
    density = mask.float().mean().item()
    assert 0.0 < density < 1.0, f"Mask density {density:.4f} looks degenerate"
    # Diagonal should always be selected
    min_dim = min(mask.shape[2], mask.shape[3])
    diag = mask[:, :, range(min_dim), range(min_dim)]
    assert (diag == 1).all(), "Diagonal blocks must be selected"
