"""CI correctness tests for dim128 Stem with Q/K per-token and V per-head scales.

Exercises `QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD`:
  - kscale: 4D [num_blocks, kScaleBlockSize, num_kv_heads, num_dim_scale]
            * fp32 dtype OR fp8 view of the same fp32 storage (both supported)
            * semantically per-KV-token (one fp32 scale per token in page)
  - vscale: per-head fp32 vector [num_kv_heads]

Tests:
  1. test_qkpertoken_perhead_vperhead_prep_paged_kv  -- kflat / vbias correctness
  2. test_qkpertoken_perhead_vperhead_fp8_view       -- fp8 view matches fp32
  3. test_qkpertoken_perhead_vperhead_e2e            -- end-to-end stem_paged_kv
"""

import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch

# ====================================================================
# Shared constants
# ====================================================================

STEM_BLOCK_SIZE = 128
STEM_STRIDE = 16
LAMBDA_MAG = 0.3
ALPHA = 1.0
INITIAL_BLOCKS = 4
WINDOW_SIZE = 4
K_BLOCK_NUM_RATE_MEDIUM = 0.2
K_BLOCK_NUM_BIAS_MEDIUM = 30
K_BLOCK_NUM_RATE_LARGE = 0.1
K_BLOCK_NUM_BIAS_LARGE = 30


def _setup_paged_fp8_data(
    num_batch,
    seq_len,
    num_head_q,
    num_head_kv,
    dim_qk=128,
    dim_v=128,
    kv_block_size=64,
):
    """Create random paged FP8 data with new-path K-scale (4D fp32) + V-scale (per-head)."""
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
            total_blocks,
            kv_block_size,
            num_head_kv,
            dim_qk,
            dtype=torch.bfloat16,
            device=device,
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

    # 4D per-token K-scale: kscale_fp32[block, row_in_page // 32, head_kv, row_in_page % 32].
    num_dim_scale = dim_v // 4
    assert (
        kv_block_size % num_dim_scale == 0
    ), f"kv_block_size={kv_block_size} must be a multiple of num_dim_scale={num_dim_scale}"
    scale_block_size = kv_block_size // num_dim_scale
    kscale_fp32 = (
        torch.abs(
            torch.randn(
                total_blocks,
                scale_block_size,
                num_head_kv,
                num_dim_scale,
                dtype=torch.float32,
                device=device,
            )
        )
        / 10
    )

    vscale = torch.abs(torch.ones(num_head_kv, dtype=torch.float32, device=device))

    seqlens = torch.full((num_batch,), seq_len, dtype=torch.int32, device=device)
    cu_q_seqlens = torch.zeros(num_batch + 1, dtype=torch.int32, device=device)
    cu_q_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    kv_seqlens = seqlens.clone()

    return dict(
        q_fp8=q_fp8,
        kcache_fp8=kcache_fp8,
        vcache_fp8=vcache_fp8,
        qscale=qscale,
        kscale_fp32=kscale_fp32,
        vscale=vscale,
        kv_indices=kv_indices,
        cu_q_seqlens=cu_q_seqlens,
        kv_seqlens=kv_seqlens,
        kv_block_size=kv_block_size,
        scale_block_size=scale_block_size,
        num_dim_scale=num_dim_scale,
    )


# ====================================================================
# PyTorch reference for the new quant_type=0 KV prep
# ====================================================================


def _gather_per_token_kscale(kscale_fp32, kv_indices, kv_seqlens, kv_block_size):
    """Materialize a per-token K-scale tensor [num_batch, num_head_kv, max_kv_pad].

    For each request and each KV token position t < kv_seqlens[b]:
      page_idx       = t // kv_block_size
      row_in_page    = t %  kv_block_size
      phys_block     = kv_indices[b, page_idx]
      kscale_per_tok = kscale_fp32[phys_block, row_in_page // 32, head_kv, row_in_page % 32]
    """
    num_batch = kv_seqlens.shape[0]
    num_head_kv = kscale_fp32.shape[2]
    max_kv_padded = (
        (int(kv_seqlens.max().item()) + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
    ) * STEM_BLOCK_SIZE
    device = kscale_fp32.device
    dtype = kscale_fp32.dtype
    out = torch.zeros(num_batch, num_head_kv, max_kv_padded, dtype=dtype, device=device)
    for b in range(num_batch):
        seqlen_b = int(kv_seqlens[b].item())
        for t in range(seqlen_b):
            page_idx = t // kv_block_size
            row_in_page = t % kv_block_size
            phys_block = int(kv_indices[b, page_idx].item())
            row_chunk = row_in_page // 32
            row_offset = row_in_page % 32
            for h in range(num_head_kv):
                out[b, h, t] = kscale_fp32[phys_block, row_chunk, h, row_offset]
    return out


def _kv_prep_reference(
    kcache_fp8,
    vcache_fp8,
    kscale_fp32,
    vscale,
    kv_indices,
    kv_seqlens,
    kv_block_size,
    lambda_mag=LAMBDA_MAG,
    stem_block_size=STEM_BLOCK_SIZE,
    stem_stride=STEM_STRIDE,
):
    """Pure PyTorch reference matching the new qkpertoken_perhead_vperhead kernel.

    K processing: per-KV-token scale folded into the FP32 group sum.
    V processing: per-head vscale folded into element-wise scaling before L2 norm.
    Vbias path is shared with legacy (log normalize is invariant to global per-head scale).
    """
    num_batch = kv_seqlens.shape[0]
    num_head_kv = kcache_fp8.shape[2]
    dim_qk = kcache_fp8.shape[3]
    dim_v = vcache_fp8.shape[3]

    sample_per_block = stem_block_size // stem_stride  # = 8

    max_kv = int(kv_seqlens.max().item())
    max_kv_padded = ((max_kv + stem_block_size - 1) // stem_block_size) * stem_block_size
    max_num_stem_blocks = max_kv_padded // stem_block_size
    max_k_down_len = max_kv_padded // stem_stride

    device = kcache_fp8.device

    # Materialize per-token K-scale (B, Hkv, max_kv_pad).
    kscale_per_token = _gather_per_token_kscale(kscale_fp32, kv_indices, kv_seqlens, kv_block_size)

    # Materialize gathered K/V per (batch, head_kv, token, dim).
    K_dense = torch.zeros(
        num_batch, num_head_kv, max_kv_padded, dim_qk, dtype=torch.float32, device=device
    )
    V_dense = torch.zeros(
        num_batch, num_head_kv, max_kv_padded, dim_v, dtype=torch.float32, device=device
    )
    for b in range(num_batch):
        seqlen_b = int(kv_seqlens[b].item())
        for t in range(seqlen_b):
            page_idx = t // kv_block_size
            row_in_page = t % kv_block_size
            phys_block = int(kv_indices[b, page_idx].item())
            K_dense[b, :, t, :] = kcache_fp8[phys_block, row_in_page, :, :].float()
            V_dense[b, :, t, :] = vcache_fp8[phys_block, row_in_page, :, :].float()

    # Apply per-token K-scale to K (broadcast on dim).
    K_scaled = K_dense * kscale_per_token.unsqueeze(-1)

    # K group-sum -> kflat (BF16 with reversed group order).
    kflat = torch.zeros(
        num_batch,
        num_head_kv,
        max_num_stem_blocks,
        stem_stride * dim_qk,
        dtype=torch.bfloat16,
        device=device,
    )
    for b in range(num_batch):
        seqlen_b = int(kv_seqlens[b].item())
        nblock = (seqlen_b + stem_block_size - 1) // stem_block_size
        for sb in range(nblock):
            row_base = sb * stem_block_size
            for g in range(stem_stride):
                acc = torch.zeros(num_head_kv, dim_qk, dtype=torch.float32, device=device)
                for r in range(sample_per_block):
                    row = row_base + g + r * stem_stride
                    if row < seqlen_b:
                        acc += K_scaled[b, :, row, :]
                seg = stem_stride - 1 - g  # reversed group order
                kflat[b, :, sb, seg * dim_qk : (seg + 1) * dim_qk] = acc.to(torch.bfloat16)

    # V L2 norm with per-head vscale -> v_norm_down (FP32).
    v_norm_down = torch.zeros(
        num_batch, num_head_kv, max_k_down_len, dtype=torch.float32, device=device
    )
    for b in range(num_batch):
        seqlen_b = int(kv_seqlens[b].item())
        k_padded = ((seqlen_b + stem_block_size - 1) // stem_block_size) * stem_block_size
        nblock = k_padded // stem_block_size
        for sb in range(nblock):
            row_base = sb * stem_block_size
            for g in range(sample_per_block):
                # Warp g handles rows [g*stride .. g*stride+stride-1] within the block.
                max_norm = torch.zeros(num_head_kv, dtype=torch.float32, device=device)
                for s in range(stem_stride):
                    row = row_base + g * stem_stride + s
                    if row < seqlen_b:
                        v_scaled = V_dense[b, :, row, :] * vscale.unsqueeze(-1)
                        norm = torch.sqrt((v_scaled * v_scaled).sum(dim=-1))
                        max_norm = torch.maximum(max_norm, norm)
                down_idx = sb * sample_per_block + g
                if down_idx < max_k_down_len:
                    v_norm_down[b, :, down_idx] = max_norm

    # vbias_reduce: log normalize -> ReLU -> block average (per (batch, head_kv)).
    vbias = torch.zeros(
        num_batch, num_head_kv, max_num_stem_blocks, dtype=torch.float32, device=device
    )
    for b in range(num_batch):
        seqlen_b = int(kv_seqlens[b].item())
        k_padded = ((seqlen_b + stem_block_size - 1) // stem_block_size) * stem_block_size
        k_down = k_padded // stem_stride
        nblock = k_padded // stem_block_size
        if k_down == 0:
            continue
        for h in range(num_head_kv):
            slice_h = v_norm_down[b, h, :k_down]
            log_vals = torch.log(slice_h + 1e-6)
            mean = log_vals.mean()
            std = log_vals.std(unbiased=True) if k_down > 1 else torch.tensor(0.0, device=device)
            inv_std = 1.0 / (std + 1e-6)
            for sb in range(nblock):
                acc = torch.tensor(0.0, dtype=torch.float32, device=device)
                for r in range(sample_per_block):
                    idx = sb * sample_per_block + r
                    if idx < k_down:
                        log_val = torch.log(v_norm_down[b, h, idx] + 1e-6)
                        normalized = (log_val - mean) * inv_std
                        acc = acc + lambda_mag * torch.clamp(normalized, min=0.0)
                vbias[b, h, sb] = acc / sample_per_block

    return kflat, vbias


# ====================================================================
# Test 1: stem_oam_prep_paged_kv with quant_type=0 (QK per-token, V per-head)
# ====================================================================


@pytest.mark.parametrize("num_batch", [1])
@pytest.mark.parametrize("seq_len", [256])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1)])
@pytest.mark.parametrize("kv_block_size", [32, 64])
def test_qkpertoken_perhead_vperhead_prep_paged_kv(
    num_batch, seq_len, num_head_q, num_head_kv, kv_block_size
):
    torch.manual_seed(20260423)
    torch.cuda.manual_seed(20260423)
    d = _setup_paged_fp8_data(
        num_batch, seq_len, num_head_q, num_head_kv, kv_block_size=kv_block_size
    )

    # Reference (FP32).
    ref_kflat, ref_vbias = _kv_prep_reference(
        d["kcache_fp8"],
        d["vcache_fp8"],
        d["kscale_fp32"],
        d["vscale"],
        d["kv_indices"],
        d["kv_seqlens"],
        d["kv_block_size"],
        lambda_mag=LAMBDA_MAG,
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
    )

    # CUDA kernel via new quant_type=0 path (kscale fp32 dtype).
    cuda_kflat, cuda_vbias = hpc.stem_oam_prep_paged_kv(
        d["kcache_fp8"],
        d["vcache_fp8"],
        d["kscale_fp32"],
        d["vscale"],
        d["kv_indices"],
        d["kv_seqlens"],
        lambda_mag=LAMBDA_MAG,
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
        quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
    )

    # Shape sanity.
    assert cuda_kflat.shape == ref_kflat.shape
    assert cuda_vbias.shape == ref_vbias.shape
    assert cuda_kflat.dtype == torch.bfloat16
    assert cuda_vbias.dtype == torch.float32

    # Compare valid blocks per request.
    for b in range(num_batch):
        seqlen_b = int(d["kv_seqlens"][b].item())
        valid_kb = (seqlen_b + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
        ref_k = ref_kflat[b, :, :valid_kb].float()
        cuda_k = cuda_kflat[b, :, :valid_kb].float()
        assert torch.allclose(ref_k, cuda_k, atol=5e-2, rtol=5e-2), (
            f"Kflat mismatch (batch={b}): "
            f"max_abs={(ref_k - cuda_k).abs().max():.4e} "
            f"mean_abs={(ref_k - cuda_k).abs().mean():.4e}"
        )
        ref_vb = ref_vbias[b, :, :valid_kb]
        cuda_vb = cuda_vbias[b, :, :valid_kb]
        assert torch.allclose(ref_vb, cuda_vb, atol=5e-3, rtol=5e-3), (
            f"Vbias mismatch (batch={b}): " f"max_abs={(ref_vb - cuda_vb).abs().max():.4e}"
        )


# ====================================================================
# Test 2: kscale passed as fp8 view (mirrors attention-side calling pattern)
# ====================================================================


@pytest.mark.parametrize("num_batch", [1])
@pytest.mark.parametrize("seq_len", [512])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1)])
@pytest.mark.parametrize("kv_block_size", [64])
def test_qkpertoken_perhead_vperhead_fp8_view(
    num_batch, seq_len, num_head_q, num_head_kv, kv_block_size
):
    """fp8 view of an fp32-backed kscale tensor must produce identical output.

    The entry layer folds fp8 strides back to fp32 element units so the kernel
    sees the same indexing in both cases.
    """
    torch.manual_seed(20260424)
    torch.cuda.manual_seed(20260424)
    d = _setup_paged_fp8_data(
        num_batch, seq_len, num_head_q, num_head_kv, kv_block_size=kv_block_size
    )
    kscale_fp32 = d["kscale_fp32"]
    # fp8 view: reinterpret the same memory as fp8; last dim grows by 4x in the byte view.
    kscale_fp8_view = kscale_fp32.view(torch.float8_e4m3fn)
    assert kscale_fp8_view.shape[3] == kscale_fp32.shape[3] * 4

    out_fp32 = hpc.stem_oam_prep_paged_kv(
        d["kcache_fp8"],
        d["vcache_fp8"],
        kscale_fp32,
        d["vscale"],
        d["kv_indices"],
        d["kv_seqlens"],
        lambda_mag=LAMBDA_MAG,
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
        quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
    )
    out_fp8 = hpc.stem_oam_prep_paged_kv(
        d["kcache_fp8"],
        d["vcache_fp8"],
        kscale_fp8_view,
        d["vscale"],
        d["kv_indices"],
        d["kv_seqlens"],
        lambda_mag=LAMBDA_MAG,
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
        quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
    )
    # Bit-for-bit equal (both paths use identical kernel + identical underlying data).
    assert torch.equal(out_fp32[0], out_fp8[0]), "Kflat differs between fp32 and fp8-view inputs"
    assert torch.equal(out_fp32[1], out_fp8[1]), "Vbias differs between fp32 and fp8-view inputs"


# ====================================================================
# Test 3: end-to-end stem_paged_kv with quant_type=0
# ====================================================================


@pytest.mark.parametrize("num_batch", [1])
@pytest.mark.parametrize("seq_len", [2048])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1)])
def test_qkpertoken_perhead_vperhead_e2e(num_batch, seq_len, num_head_q, num_head_kv):
    torch.manual_seed(20260425)
    torch.cuda.manual_seed(20260425)
    d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)

    mask = hpc.stem_paged_kv(
        d["q_fp8"],
        d["kcache_fp8"],
        d["vcache_fp8"],
        d["qscale"],
        d["kscale_fp32"],
        d["vscale"],
        d["kv_indices"],
        d["cu_q_seqlens"],
        d["kv_seqlens"],
        d["kv_seqlens"],  # num_prompt_tokens (normal prefill: equals kv_seqlens)
        lambda_mag=LAMBDA_MAG,
        alpha=ALPHA,
        stem_block_size=STEM_BLOCK_SIZE,
        stem_stride=STEM_STRIDE,
        causal=True,
        initial_blocks=INITIAL_BLOCKS,
        window_size=WINDOW_SIZE,
        k_block_num_rate_medium=K_BLOCK_NUM_RATE_MEDIUM,
        k_block_num_bias_medium=K_BLOCK_NUM_BIAS_MEDIUM,
        k_block_num_rate_large=K_BLOCK_NUM_RATE_LARGE,
        k_block_num_bias_large=K_BLOCK_NUM_BIAS_LARGE,
        quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
    )

    max_qb = (seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
    assert mask.dim() == 4
    assert mask.shape[0] == num_batch
    assert mask.shape[1] == num_head_q
    assert mask.dtype == torch.uint8
    density = mask.float().mean().item()
    assert 0.0 < density < 1.0, f"Mask density {density:.4f} looks degenerate"
    # Diagonal must always be selected (TPD post-fix retention).
    min_dim = min(mask.shape[2], mask.shape[3])
    diag = mask[:, :, range(min_dim), range(min_dim)]
    assert (diag == 1).all(), "Diagonal blocks must be selected"
