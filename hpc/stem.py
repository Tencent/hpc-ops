"""Stem sparse attention operators.

This module provides CUDA-accelerated operators for the Stem sparse attention
pipeline, including QKV preprocessing (prep), scoring (oam), and
top-k/mask generation (tpd).
"""

from typing import Tuple

import torch
from torch import Tensor

from .attention import QuantType


def stem_oam_prep_paged_kv(
    kcache: Tensor,
    vcache: Tensor,
    kscale: Tensor,
    vscale: Tensor,
    kv_indices: Tensor,
    kv_seq_lens: Tensor,
    lambda_mag: float = 0.3,
    stem_block_size: int = 128,
    stem_stride: int = 16,
    quant_type: QuantType = QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
) -> Tuple[Tensor, Tensor]:
    """Precompute K_flat and V_bias from paged FP8 KV cache.

    First stage of the Stem sparse scoring pipeline:
      - K_flat (BF16): group-summed K vectors (scaled by kscale) for OAM-GEMM.
      - V_bias (FP32): per-block importance bias from V norm statistics.

    Args:
        kcache: Paged K cache.
            Shape: [num_blocks, kv_block_size, num_kv_heads, dim_qk=128]
            Dtype: float8_e4m3fn
        vcache: Paged V cache.
            Shape: [num_blocks, kv_block_size, num_kv_heads, dim_v=128]
            Dtype: float8_e4m3fn
        kscale: K FP8 dequantization scale.
            Shape depends on `quant_type`:
              - QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR: [1], fp32.
              - QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD:
                  [num_blocks, scale_block_size, num_kv_heads, num_dim_scale],
                  fp32 dtype OR fp8 view of the same fp32 storage.
        vscale: V FP8 dequantization scale.
            QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR: [1], fp32.
            QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD: per-head vector [num_kv_heads], fp32.
        kv_indices: Paged block table mapping logical to physical blocks.
            Shape: [num_batch, max_blocks_per_req], Dtype: int32
        kv_seq_lens: KV sequence length per request.
            Shape: [num_batch], Dtype: int32
        lambda_mag: Scaling coefficient for V_bias (default 0.3).
        stem_block_size: Stem sparse scoring block size (default 128).
        stem_stride: Downsampling stride (default 16).
        quant_type: Scale dispatch for the dim128 paged path.

    Returns:
        kflat: Group-summed K vectors with reversed group order for anti-diag scoring.
            Shape: [num_batch, num_kv_heads, max_Kb, stem_stride * dim_qk]
            Dtype: bfloat16
        vbias: Per-block V importance bias scalar.
            Shape: [num_batch, num_kv_heads, max_Kb]
            Dtype: float32
    """
    return torch.ops.hpc.stem_oam_prep_paged_kv(
        kcache,
        vcache,
        kscale,
        vscale,
        kv_indices,
        kv_seq_lens,
        lambda_mag,
        stem_block_size,
        stem_stride,
        quant_type.value,
    )


def stem_oam_prep_varlen_kv_dim192(
    k_fp8: Tensor,
    v_fp8: Tensor,
    kscale: Tensor,
    vscale: Tensor,
    kv_seq_lens: Tensor,
    cu_seqlens_kv: Tensor,
    lambda_mag: float = 0.3,
    stem_block_size: int = 128,
    stem_stride: int = 16,
) -> Tuple[Tensor, Tensor]:
    """Precompute dim192 K_flat and V_bias from ragged varlen FP8 KV.

    K processing applies kscale (per-tensor K dequant scale) to the group-sum
    before BF16 write-back. vscale is used for V L2 norm computation.

    Args:
        k_fp8: Varlen K tensor.
            Shape: [total_kv, num_kv_heads, dim_qk=192], Dtype: float8_e4m3fn
        v_fp8: Varlen V tensor.
            Shape: [total_kv, num_kv_heads, dim_v=128], Dtype: float8_e4m3fn
        kscale: Per-tensor K FP8 dequantization scale.
            Shape: [1], Dtype: float32
        vscale: Per-tensor V FP8 dequantization scale.
            Shape: [1], Dtype: float32
        kv_seq_lens: KV sequence length per request.
            Shape: [num_batch], Dtype: int32
        cu_seqlens_kv: Cumulative KV sequence lengths.
            Shape: [num_batch + 1], Dtype: int32
        lambda_mag: V_bias scaling coefficient (default 0.3).
        stem_block_size: Stem sparse scoring block size (default 128).
        stem_stride: Downsampling stride (default 16).

    Returns:
        kflat: Group-summed K vectors (scaled by kscale) with reversed group order.
            Shape: [num_batch, num_kv_heads, max_Kb, stem_stride * dim_qk]
            Dtype: bfloat16
        vbias: Per-block V importance bias scalar.
            Shape: [num_batch, num_kv_heads, max_Kb]
            Dtype: float32
    """
    return torch.ops.hpc.stem_oam_prep_varlen_kv_dim192(
        k_fp8,
        v_fp8,
        kscale,
        vscale,
        kv_seq_lens,
        cu_seqlens_kv,
        lambda_mag,
        stem_block_size,
        stem_stride,
    )


def stem_oam_prep_varlen_q(
    q_fp8: Tensor,
    qscale: Tensor,
    q_seq_lens: Tensor,
    cu_seqlens_q: Tensor,
    stem_block_size: int = 128,
    stem_stride: int = 16,
) -> Tensor:
    """Precompute dim128 Q_flat from packed FP8 Q tensor.

    Computes weighted group-sum of Q tokens using per-token qscale (Q FP8
    dequantization scale). kscale is handled separately in KV prep.

    Args:
        q_fp8: Per-token query vectors (packed across requests).
            Shape: [total_tokens, num_q_heads, dim_qk=128]
            Dtype: float8_e4m3fn
        qscale: Q FP8 dequantization scale.
            Shape: [num_batch, num_q_heads, max_seq_q_pad]
            Dtype: float32
        q_seq_lens: Q sequence length per request.
            Shape: [num_batch], Dtype: int32
        cu_seqlens_q: Cumulative Q sequence lengths.
            Shape: [num_batch + 1], Dtype: int32
        stem_block_size: Stem sparse scoring block size (default 128).
        stem_stride: Downsampling stride (default 16).

    Returns:
        qflat: Weighted group-summed Q vectors in natural group order.
            Shape: [num_batch, num_q_heads, max_Qb, stem_stride * dim_qk]
            Dtype: bfloat16
    """
    return torch.ops.hpc.stem_oam_prep_varlen_q(
        q_fp8,
        qscale,
        q_seq_lens,
        cu_seqlens_q,
        stem_block_size,
        stem_stride,
    )


def stem_oam_prep_varlen_q_dim192(
    q_fp8: Tensor,
    qscale: Tensor,
    q_seq_lens: Tensor,
    cu_seqlens_q: Tensor,
    stem_block_size: int = 128,
    stem_stride: int = 16,
) -> Tensor:
    """Precompute dim192 Q_flat from packed FP8 Q tensor with scalar Q scale.

    Args:
        q_fp8: Per-token query vectors (packed across requests).
            Shape: [total_tokens, num_q_heads, dim_qk=192], Dtype: float8_e4m3fn
        qscale: Per-tensor Q FP8 dequantization scale.
            Shape: [1], Dtype: float32
        q_seq_lens: Q sequence length per request.
            Shape: [num_batch], Dtype: int32
        cu_seqlens_q: Cumulative Q sequence lengths.
            Shape: [num_batch + 1], Dtype: int32
        stem_block_size: Stem sparse scoring block size (default 128).
        stem_stride: Downsampling stride (default 16).

    Returns:
        qflat: Weighted group-summed Q vectors in natural group order.
            Shape: [num_batch, num_q_heads, max_Qb, stem_stride * dim_qk]
            Dtype: bfloat16
    """
    return torch.ops.hpc.stem_oam_prep_varlen_q_dim192(
        q_fp8,
        qscale,
        q_seq_lens,
        cu_seqlens_q,
        stem_block_size,
        stem_stride,
    )


def stem_oam_gemm(
    qflat: Tensor,
    kflat: Tensor,
    vbias: Tensor,
    q_seq_lens: Tensor,
    kv_seq_lens: Tensor,
    stem_block_size: int = 128,
    stem_stride: int = 16,
    causal: bool = True,
) -> Tensor:
    """Compute dim128 block_logits via OAM GEMM with fused causal mask epilogue.

    Performs block_logits = FrobScale * (Qflat @ Kflat^T) + V_bias, with
    optional causal masking. Invalid positions are set to -inf.

    Args:
        qflat: Precomputed Q flat vectors.
            Shape: [num_batch, num_q_heads, max_Qb, stem_stride * dim_qk=128]
            Dtype: bfloat16
        kflat: Precomputed K flat vectors.
            Shape: [num_batch, num_kv_heads, max_Kb, stem_stride * dim_qk=128]
            Dtype: bfloat16
        vbias: Per-block V importance bias.
            Shape: [num_batch, num_kv_heads, max_Kb]
            Dtype: float32
        q_seq_lens: Q sequence length per request.
            Shape: [num_batch], Dtype: int32
        kv_seq_lens: KV sequence length per request.
            Shape: [num_batch], Dtype: int32
        stem_block_size: Stem sparse scoring block size (default 128).
        stem_stride: Downsampling stride (default 16).
        causal: Whether to apply causal masking (default True).

    Returns:
        block_logits: OAM GEMM scores + V_bias, -inf for masked positions.
            Shape: [num_batch, num_q_heads, max_Qb, max_Kb]
            Dtype: bfloat16
    """
    return torch.ops.hpc.stem_oam_gemm(
        qflat,
        kflat,
        vbias,
        q_seq_lens,
        kv_seq_lens,
        stem_block_size,
        stem_stride,
        causal,
    )


def stem_oam_gemm_dim192(
    qflat: Tensor,
    kflat: Tensor,
    vbias: Tensor,
    q_seq_lens: Tensor,
    kv_seq_lens: Tensor,
    stem_block_size: int = 128,
    stem_stride: int = 16,
    causal: bool = True,
) -> Tensor:
    """Compute dim192 block_logits via OAM GEMM with fused causal mask epilogue."""
    return torch.ops.hpc.stem_oam_gemm_dim192(
        qflat,
        kflat,
        vbias,
        q_seq_lens,
        kv_seq_lens,
        stem_block_size,
        stem_stride,
        causal,
    )


def stem_tpd(
    block_logits: Tensor,
    q_seq_lens: Tensor,
    kv_seq_lens: Tensor,
    num_prompt_tokens: Tensor,
    block_size: int = 128,
    alpha: float = 1.0,
    initial_blocks: int = 4,
    window_size: int = 4,
    k_block_num_rate_medium: float = 0.2,
    k_block_num_bias_medium: int = 30,
    k_block_num_rate_large: float = 0.1,
    k_block_num_bias_large: int = 30,
) -> Tensor:
    """Generate sparse block mask via top-k policy denoising.

    Fuses per-row budget (3-regime k_schedule + linear decay, both keyed on
    the full prompt KV length so the result is chunked-prefill invariant),
    radix top-k threshold, and fixed retention (initial / window / diagonal).

    Args:
        block_logits: Block-level OAM scores.
            Shape: [num_batch, num_q_heads, max_Qb, max_Kb]
            Dtype: bfloat16 (invalid positions set to -inf)
        q_seq_lens: Q sequence length per request (current chunk).
            Shape: [num_batch], Dtype: int32
        kv_seq_lens: KV sequence length per request (cumulative through current chunk).
            Shape: [num_batch], Dtype: int32
        num_prompt_tokens: Full prompt KV-token count per request. For
            chunked prefill pass the same value for every chunk of one
            prompt; for normal prefill pass ``kv_seq_lens``.
            Shape: [num_batch], Dtype: int32
        block_size: Stem sparse scoring block size (default 128).
        alpha: Per-row budget decay factor (default 1.0 disables decay).
        initial_blocks: Leading KV blocks always retained (default 4).
        window_size: Recent diagonal-adjacent blocks always retained (default 4).
        k_block_num_rate_medium: k_schedule multiplier when
            56 <= prompt_kv_blocks < 160 (default 0.2).
        k_block_num_bias_medium: k_schedule bias in the medium regime (default 30).
        k_block_num_rate_large: k_schedule multiplier when
            prompt_kv_blocks >= 160 (default 0.1).
        k_block_num_bias_large: k_schedule bias in the large regime (default 30).

    Returns:
        mask: Per-block selection byte-mask.
            Shape: [num_batch, num_q_heads, max_Qb, max_Kb]
            Dtype: uint8  (1 = selected, 0 = skipped)
    """
    return torch.ops.hpc.stem_tpd(
        block_logits,
        q_seq_lens,
        kv_seq_lens,
        num_prompt_tokens,
        block_size,
        alpha,
        initial_blocks,
        window_size,
        k_block_num_rate_medium,
        k_block_num_bias_medium,
        k_block_num_rate_large,
        k_block_num_bias_large,
    )


def stem_paged_kv(
    q_fp8: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    qscale: Tensor,
    kscale: Tensor,
    vscale: Tensor,
    kv_indices: Tensor,
    cu_seqlens_q: Tensor,
    kv_seq_lens: Tensor,
    num_prompt_tokens: Tensor,
    lambda_mag: float = 0.3,
    alpha: float = 1.0,
    stem_block_size: int = 128,
    stem_stride: int = 16,
    causal: bool = True,
    initial_blocks: int = 4,
    window_size: int = 4,
    k_block_num_rate_medium: float = 0.2,
    k_block_num_bias_medium: int = 30,
    k_block_num_rate_large: float = 0.1,
    k_block_num_bias_large: int = 30,
    quant_type: QuantType = QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
) -> Tensor:
    """End-to-end Stem sparse mask generation for paged FP8 KV cache (dim_qk=128).

    Fuses the full OAM + TPD pipeline into a single call:
      1. prep_paged_kv  -> kflat, vbias
      2. prep_varlen_q  -> qflat
      3. oam_gemm       -> block_logits
      4. tpd            -> u8 sparse mask

    Args:
        q_fp8: Packed query vectors.
            Shape: [total_tokens, num_q_heads, 128], Dtype: float8_e4m3fn
        kcache: Paged K cache.
            Shape: [num_blocks, kv_block_size, num_kv_heads, dim_qk=128], Dtype: float8_e4m3fn
        vcache: Paged V cache.
            Shape: [num_blocks, kv_block_size, num_kv_heads, dim_v=128], Dtype: float8_e4m3fn
        qscale: Per-token per-head Q dequantization scale.
            Shape: [num_batch, num_q_heads, max_seq_q_pad], Dtype: float32
        kscale: K dequantization scale (shape depends on `quant_type`, see
            `stem_oam_prep_paged_kv` for details).
        vscale: V dequantization scale (shape depends on `quant_type`).
        kv_indices: Block table (logical -> physical).
            Shape: [num_batch, max_blocks_per_req], Dtype: int32
        cu_seqlens_q: Cumulative Q sequence lengths.
            Shape: [num_batch + 1], Dtype: int32
        kv_seq_lens: KV sequence length per request (cumulative through current chunk).
            Shape: [num_batch], Dtype: int32
        num_prompt_tokens: Full prompt KV-token count per request. For
            chunked prefill pass the same value for every chunk of one
            prompt; for normal prefill pass ``kv_seq_lens``.
            Shape: [num_batch], Dtype: int32
        lambda_mag: V-bias scaling coefficient (default 0.3).
        alpha: TPD budget decay factor (default 1.0; see ``stem_tpd``).
        stem_block_size: Sparse scoring block size (default 128).
        stem_stride: Downsampling stride (default 16).
        causal: Apply causal masking (default True).
        initial_blocks: Leading KV blocks always retained (default 4).
        window_size: Recent diagonal-adjacent blocks always retained (default 4).
        k_block_num_rate_medium: k_schedule multiplier when
            56 <= prompt_kv_blocks < 160 (default 0.2).
        k_block_num_bias_medium: k_schedule bias in the medium regime (default 30).
        k_block_num_rate_large: k_schedule multiplier when
            prompt_kv_blocks >= 160 (default 0.1).
        k_block_num_bias_large: k_schedule bias in the large regime (default 30).
        quant_type: Scale dispatch for dim128. Defaults to
            QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR; pass
            QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD for per-token K-scale
            and per-head V-scale.

    Returns:
        mask: Per-block selection byte-mask.
            Shape: [num_batch, num_q_heads, max_Qb, max_Kb], Dtype: uint8
    """
    q_seq_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)

    kflat, vbias = stem_oam_prep_paged_kv(
        kcache,
        vcache,
        kscale,
        vscale,
        kv_indices,
        kv_seq_lens,
        lambda_mag,
        stem_block_size,
        stem_stride,
        quant_type,
    )
    qflat = stem_oam_prep_varlen_q(
        q_fp8,
        qscale,
        q_seq_lens,
        cu_seqlens_q,
        stem_block_size,
        stem_stride,
    )
    block_logits = stem_oam_gemm(
        qflat,
        kflat,
        vbias,
        q_seq_lens,
        kv_seq_lens,
        stem_block_size,
        stem_stride,
        causal,
    )
    mask = stem_tpd(
        block_logits,
        q_seq_lens,
        kv_seq_lens,
        num_prompt_tokens,
        stem_block_size,
        alpha,
        initial_blocks,
        window_size,
        k_block_num_rate_medium,
        k_block_num_bias_medium,
        k_block_num_rate_large,
        k_block_num_bias_large,
    )
    return mask


def stem_varlen_kv_dim192(
    q_fp8: Tensor,
    k_fp8: Tensor,
    v_fp8: Tensor,
    qscale: Tensor,
    kscale: Tensor,
    vscale: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_kv: Tensor,
    num_prompt_tokens: Tensor,
    lambda_mag: float = 0.3,
    alpha: float = 1.0,
    stem_block_size: int = 128,
    stem_stride: int = 16,
    causal: bool = True,
    initial_blocks: int = 4,
    window_size: int = 4,
    k_block_num_rate_medium: float = 0.2,
    k_block_num_bias_medium: int = 30,
    k_block_num_rate_large: float = 0.1,
    k_block_num_bias_large: int = 30,
) -> Tensor:
    """End-to-end Stem sparse mask generation for ragged dim192 FP8 KV.

    Fuses the full OAM + TPD pipeline into a single call:
      1. prep_varlen_kv  -> kflat, vbias
      2. prep_varlen_q   -> qflat
      3. oam_gemm        -> block_logits
      4. tpd             -> u8 sparse mask

    Args:
        q_fp8: Packed query vectors.
            Shape: [total_tokens, num_q_heads, dim_qk=192], Dtype: float8_e4m3fn
        k_fp8: Varlen K tensor.
            Shape: [total_kv, num_kv_heads, dim_qk=192], Dtype: float8_e4m3fn
        v_fp8: Varlen V tensor.
            Shape: [total_kv, num_kv_heads, dim_v=128], Dtype: float8_e4m3fn
        qscale: Per-tensor Q dequantization scale.
            Shape: [1], Dtype: float32
        kscale: Per-tensor K dequantization scale.
            Shape: [1], Dtype: float32
        vscale: Per-tensor V quantization scale.
            Shape: [1], Dtype: float32
        cu_seqlens_q: Cumulative Q sequence lengths.
            Shape: [num_batch + 1], Dtype: int32
        cu_seqlens_kv: Cumulative KV sequence lengths (current chunk).
            Shape: [num_batch + 1], Dtype: int32
        num_prompt_tokens: Full prompt KV-token count per request. For
            chunked prefill pass the same value for every chunk of one
            prompt; for normal prefill pass the chunk KV lengths.
            Shape: [num_batch], Dtype: int32
        lambda_mag: V-bias scaling coefficient (default 0.3).
        alpha: TPD budget decay factor (default 1.0; see ``stem_tpd``).
        stem_block_size: Sparse scoring block size (default 128).
        stem_stride: Downsampling stride (default 16).
        causal: Apply causal masking (default True).
        initial_blocks: Leading KV blocks always retained (default 4).
        window_size: Recent diagonal-adjacent blocks always retained (default 4).
        k_block_num_rate_medium: k_schedule multiplier when
            56 <= prompt_kv_blocks < 160 (default 0.2).
        k_block_num_bias_medium: k_schedule bias in the medium regime (default 30).
        k_block_num_rate_large: k_schedule multiplier when
            prompt_kv_blocks >= 160 (default 0.1).
        k_block_num_bias_large: k_schedule bias in the large regime (default 30).

    Returns:
        mask: Per-block selection byte-mask.
            Shape: [num_batch, num_q_heads, max_Qb, max_Kb], Dtype: uint8
    """
    q_seq_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)
    kv_seq_lens = (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).to(torch.int32)

    kflat, vbias = stem_oam_prep_varlen_kv_dim192(
        k_fp8,
        v_fp8,
        kscale,
        vscale,
        kv_seq_lens,
        cu_seqlens_kv,
        lambda_mag,
        stem_block_size,
        stem_stride,
    )
    qflat = stem_oam_prep_varlen_q_dim192(
        q_fp8,
        qscale,
        q_seq_lens,
        cu_seqlens_q,
        stem_block_size,
        stem_stride,
    )
    block_logits = stem_oam_gemm_dim192(
        qflat,
        kflat,
        vbias,
        q_seq_lens,
        kv_seq_lens,
        stem_block_size,
        stem_stride,
        causal,
    )
    mask = stem_tpd(
        block_logits,
        q_seq_lens,
        kv_seq_lens,
        num_prompt_tokens,
        stem_block_size,
        alpha,
        initial_blocks,
        window_size,
        k_block_num_rate_medium,
        k_block_num_bias_medium,
        k_block_num_rate_large,
        k_block_num_bias_large,
    )
    return mask


@torch.library.register_fake("hpc::stem_oam_prep_paged_kv")
def _stem_oam_prep_paged_kv_fake(
    kcache,
    vcache,
    kscale,
    vscale,
    kv_indices,
    kv_seq_lens,
    lambda_mag,
    stem_block_size,
    stem_stride,
    quant_type,
):
    num_batch = kv_seq_lens.size(0)
    num_head_kv = kcache.size(2)
    dim_qk = kcache.size(3)
    # Conservative upper bound: assume max sequence fills all pages
    max_blocks_per_req = kv_indices.size(1)
    kv_block_size = kcache.size(1)
    max_kv_len = max_blocks_per_req * kv_block_size
    max_kv_padded = ((max_kv_len + stem_block_size - 1) // stem_block_size) * stem_block_size
    max_Kb = max_kv_padded // stem_block_size
    kflat_dim = stem_stride * dim_qk

    kflat = torch.empty(
        (num_batch, num_head_kv, max_Kb, kflat_dim),
        dtype=torch.bfloat16,
        device=kcache.device,
    )
    vbias = torch.empty(
        (num_batch, num_head_kv, max_Kb),
        dtype=torch.float32,
        device=kcache.device,
    )
    return kflat, vbias


@torch.library.register_fake("hpc::stem_oam_prep_varlen_kv_dim192")
def _stem_oam_prep_varlen_kv_dim192_fake(
    k_fp8,
    v_fp8,
    kscale,
    vscale,
    kv_seq_lens,
    cu_seqlens_kv,
    lambda_mag,
    stem_block_size,
    stem_stride,
):
    num_batch = kv_seq_lens.size(0)
    num_head_kv = k_fp8.size(1)
    dim_qk = k_fp8.size(2)
    max_kv_len = k_fp8.size(0)
    max_kv_padded = ((max_kv_len + stem_block_size - 1) // stem_block_size) * stem_block_size
    max_Kb = max_kv_padded // stem_block_size
    kflat_dim = stem_stride * dim_qk
    kflat = torch.empty(
        (num_batch, num_head_kv, max_Kb, kflat_dim),
        dtype=torch.bfloat16,
        device=k_fp8.device,
    )
    vbias = torch.empty(
        (num_batch, num_head_kv, max_Kb),
        dtype=torch.float32,
        device=k_fp8.device,
    )
    return kflat, vbias


@torch.library.register_fake("hpc::stem_oam_prep_varlen_q")
def _stem_oam_prep_varlen_q_fake(
    q_fp8,
    qscale,
    q_seq_lens,
    cu_seqlens_q,
    stem_block_size,
    stem_stride,
):
    num_batch = q_seq_lens.size(0)
    num_head_q = q_fp8.size(1)
    dim_qk = q_fp8.size(2)
    qflat_dim = stem_stride * dim_qk
    max_seq_q_pad = qscale.size(2)
    max_q_padded = ((max_seq_q_pad + stem_block_size - 1) // stem_block_size) * stem_block_size
    max_Qb = max_q_padded // stem_block_size
    return torch.empty(
        (num_batch, num_head_q, max_Qb, qflat_dim),
        dtype=torch.bfloat16,
        device=q_fp8.device,
    )


@torch.library.register_fake("hpc::stem_oam_prep_varlen_q_dim192")
def _stem_oam_prep_varlen_q_dim192_fake(
    q_fp8,
    qscale,
    q_seq_lens,
    cu_seqlens_q,
    stem_block_size,
    stem_stride,
):
    num_batch = q_seq_lens.size(0)
    num_head_q = q_fp8.size(1)
    dim_qk = q_fp8.size(2)
    qflat_dim = stem_stride * dim_qk
    max_seq_q_pad = q_fp8.size(0)
    max_q_padded = ((max_seq_q_pad + stem_block_size - 1) // stem_block_size) * stem_block_size
    max_Qb = max_q_padded // stem_block_size
    return torch.empty(
        (num_batch, num_head_q, max_Qb, qflat_dim),
        dtype=torch.bfloat16,
        device=q_fp8.device,
    )


@torch.library.register_fake("hpc::stem_oam_gemm")
def _stem_oam_gemm_fake(
    qflat,
    kflat,
    vbias,
    q_seq_lens,
    kv_seq_lens,
    stem_block_size,
    stem_stride,
    causal,
):
    num_batch = qflat.size(0)
    num_head_q = qflat.size(1)
    max_Qb = qflat.size(2)
    max_Kb = kflat.size(2)
    return torch.empty(
        (num_batch, num_head_q, max_Qb, max_Kb),
        dtype=torch.bfloat16,
        device=qflat.device,
    )


@torch.library.register_fake("hpc::stem_oam_gemm_dim192")
def _stem_oam_gemm_dim192_fake(
    qflat,
    kflat,
    vbias,
    q_seq_lens,
    kv_seq_lens,
    stem_block_size,
    stem_stride,
    causal,
):
    num_batch = qflat.size(0)
    num_head_q = qflat.size(1)
    max_Qb = qflat.size(2)
    max_Kb = kflat.size(2)
    return torch.empty(
        (num_batch, num_head_q, max_Qb, max_Kb),
        dtype=torch.bfloat16,
        device=qflat.device,
    )


@torch.library.register_fake("hpc::stem_tpd")
def _stem_tpd_fake(
    block_logits,
    q_seq_lens,
    kv_seq_lens,
    num_prompt_tokens,
    block_size,
    alpha,
    initial_blocks,
    window_size,
    k_block_num_rate_medium,
    k_block_num_bias_medium,
    k_block_num_rate_large,
    k_block_num_bias_large,
):
    return torch.empty_like(block_logits, dtype=torch.uint8)
