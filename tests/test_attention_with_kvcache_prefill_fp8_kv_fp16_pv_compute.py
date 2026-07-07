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


# Golden reference for `attention_with_kvcache_prefill_fp8_kv_fp16_pv_compute`.
# Hybrid: Q/K/V all FP8 (e4m3). Q@Kᵀ is computed from dequantized fp32 Q/K
# (qscale·kscale applied to scores); P@V keeps V in fp32, V scale applied once
# in the bf16 epilogue. quant_type: 20 = QFP8_KPERTOKEN_PERHEAD_VPERHEAD_FP16PV,
#                                   21 = QFP8_KPERTENSOR_VPERTENSOR_FP16PV.


def _dequant_K_fp32(kcache_fp8, kscale, quant_type, num_head_kv, num_dim_qk):
    """Return K_fp32 = fp32(kcache_fp8) * kscale_per_token_per_head."""
    K_fp32 = kcache_fp8.to(torch.float32)
    if quant_type == 21:
        return K_fp32 * kscale.item()
    # mode 20: K per-token+per-head scale; fp8-byte-aligned packed layout.
    num_blocks, block_size, _, _ = kcache_fp8.shape
    assert kscale.dtype == torch.float8_e4m3fn
    kscale_logical = (
        kscale.permute(0, 2, 1, 3)
        .contiguous()
        .view(torch.float32)
        .reshape(num_blocks, num_head_kv, block_size)
        .permute(0, 2, 1)
        .contiguous()
    )
    expanded = kscale_logical.unsqueeze(-1).expand(num_blocks, block_size, num_head_kv, num_dim_qk)
    return K_fp32 * expanded


def _apply_V_scale_post(out_fp32, vscale, quant_type, num_head_kv, num_head_q):
    """Apply V scale to the per-head output [Hq, sq, Dv] (fp32 accumulator)."""
    if quant_type == 21:
        return out_fp32 * vscale.item()
    # mode 20: vscale [num_head_kv]; broadcast to Hq via head group.
    num_group = num_head_q // num_head_kv
    vs = vscale.repeat_interleave(num_group, dim=0).reshape(num_head_q, 1, 1).to(torch.float32)
    return out_fp32 * vs


def _dequant_Q_fp32(q_fp8, qscale, num_head_q):
    """Return Q_fp32 = fp32(q_fp8) * qscale_per_token_per_head.

    q_fp8: [num_batch, num_seq_q, num_head_q, num_dim_qk]
    qscale: [num_batch, num_head_q, max_seqlens_q_pad] fp32
    """
    num_batch, num_seq_q, num_head_q_, num_dim_qk = q_fp8.shape
    Q_fp32 = q_fp8.to(torch.float32)
    qs = qscale[:, :, :num_seq_q].permute(0, 2, 1).unsqueeze(-1)  # [B, sq, Hq, 1]
    return Q_fp32 * qs.to(torch.float32)


def naive_fp8_kv_fp16_pv_compute_attn(
    q,
    kcache_fp8,
    vcache_fp8,
    qscale,
    kscale,
    vscale,
    seqlens_kvcache,
    page_table,
    quant_type,
    causal=True,
):
    num_batch, num_seq_q, num_head_q, num_dim_qk = q.shape
    _, block_size, num_head_kv, _ = kcache_fp8.shape
    _, _, _, num_dim_v = vcache_fp8.shape
    num_group = num_head_q // num_head_kv
    kvcache_blocks = (seqlens_kvcache + block_size - 1) // block_size
    output = torch.empty(
        num_batch, num_seq_q, num_head_q, num_dim_v, dtype=torch.bfloat16, device=q.device
    )

    K_fp32 = _dequant_K_fp32(kcache_fp8, kscale, quant_type, num_head_kv, num_dim_qk)
    Q_fp32 = _dequant_Q_fp32(q, qscale, num_head_q)

    for i in range(num_batch):
        BQ = Q_fp32[i].transpose(0, 1)  # [Hq, sq, D]
        blk_ids = page_table[i, : kvcache_blocks[i]]
        num_seq_kv = int(seqlens_kvcache[i])

        BK = (
            K_fp32[blk_ids, :, :, :]
            .reshape(-1, num_head_kv, num_dim_qk)
            .transpose(0, 1)[:, :num_seq_kv, :]
            .repeat_interleave(num_group, dim=0)
        )  # [Hq, N_KV, D]

        V_fp8 = vcache_fp8[blk_ids, :, :, :].reshape(-1, num_head_kv, num_dim_v)
        BV = (
            V_fp8.to(torch.float32)
            .transpose(0, 1)[:, :num_seq_kv, :]
            .repeat_interleave(num_group, dim=0)
        )  # [Hq, N_KV, Dv]

        scores = torch.matmul(BQ, BK.transpose(-2, -1)) / math.sqrt(num_dim_qk)
        if causal:
            causal_mask = torch.tril(
                torch.ones(num_seq_kv, num_seq_kv, device=q.device, dtype=torch.bool)
            )[(num_seq_kv - num_seq_q) :, :].unsqueeze(0)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        attn_w = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_w, BV)  # [Hq, sq, Dv] fp32
        out = _apply_V_scale_post(out, vscale, quant_type, num_head_kv, num_head_q)
        output[i] = out.transpose(0, 1).to(torch.bfloat16)

    return output


def _build_kv_caches(
    num_batch, num_seq_q, num_seq_kv, block_size, num_head_kv, num_dim_qk, num_dim_v, quant_type
):
    """Allocate KV cache tensors + scales matching the schema for `quant_type`."""
    T = torch.bfloat16
    T8 = torch.float8_e4m3fn

    seqlens_q = torch.full((num_batch,), num_seq_q, dtype=torch.int32, device="cuda")
    seqlens_kvcache = torch.full((num_batch,), num_seq_kv, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens_q]), dim=0
    ).to(torch.int32)

    max_num_blocks = num_batch * ((num_seq_kv + block_size - 1) // block_size) * 2
    kv_bf16 = torch.randn(
        max_num_blocks, 2, block_size, num_head_kv, num_dim_qk, dtype=T, device="cuda"
    )

    if quant_type == 21:
        kscale = (kv_bf16[:, 0].abs().max().to(torch.float32) / 448.0).reshape(1).contiguous()
        vscale = (kv_bf16[:, 1].abs().max().to(torch.float32) / 448.0).reshape(1).contiguous()
        kcache = (kv_bf16[:, 0].to(torch.float32) / kscale).to(T8)
        vcache = (kv_bf16[:, 1].to(torch.float32) / vscale).to(T8)
    else:
        # mode 20: K per-(token,head), V per-head. kcache+kscale share one fp8
        # allocation [nb, bs + bs*4/dim, Hkv, dim] (identical byte strides).
        scale_extra_rows = block_size * 4 // num_dim_qk  # bs*4/dim
        cache_fp8 = torch.empty(
            max_num_blocks,
            block_size + scale_extra_rows,
            num_head_kv,
            num_dim_qk,
            dtype=T8,
            device="cuda",
        )
        K_fp32 = kv_bf16[:, 0].to(torch.float32)
        kscale_fp32 = (K_fp32.abs().amax(dim=-1) / 448.0).clamp(min=1e-30).contiguous()
        cache_fp8[:, :block_size, :, :] = (K_fp32 / kscale_fp32.unsqueeze(-1)).to(T8)
        kscale_packed = (
            kscale_fp32.permute(0, 2, 1)
            .contiguous()
            .view(torch.float8_e4m3fn)
            .reshape(max_num_blocks, num_head_kv, scale_extra_rows, num_dim_qk)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        cache_fp8[:, block_size:, :, :] = kscale_packed
        kcache = cache_fp8[:, :block_size, :, :]
        kscale = cache_fp8[:, block_size:, :, :]
        V_fp32 = kv_bf16[:, 1].to(torch.float32)
        vscale = (V_fp32.abs().amax(dim=(0, 1, 3)) / 448.0).clamp(min=1e-30).contiguous()
        vcache = (V_fp32 / vscale.reshape(1, 1, num_head_kv, 1)).to(T8)

    kvcache_blocks = (seqlens_kvcache + block_size - 1) // block_size
    max_kvcache_blocks = int(max(kvcache_blocks))
    total_kvcache_blocks = int(sum(kvcache_blocks))
    packed_block_ids = torch.randperm(max_num_blocks)[:total_kvcache_blocks].to(torch.int32).cuda()
    block_ids = torch.empty(num_batch, max_kvcache_blocks, dtype=torch.int32, device="cuda")
    cu_blocks = 0
    for i in range(num_batch):
        block_ids[i, : kvcache_blocks[i]] = packed_block_ids[
            cu_blocks : cu_blocks + kvcache_blocks[i]
        ]
        cu_blocks += int(kvcache_blocks[i])

    return kcache, vcache, kscale, vscale, block_ids, seqlens_kvcache, cu_seqlens_q


def _build_Q_fp8_and_qscale(num_batch, num_seq_q, num_head_q, num_dim_qk):
    """Build fp8 Q + per-token+per-head qscale [B, Hq, max_seq_q_pad] fp32."""
    T = torch.bfloat16
    T8 = torch.float8_e4m3fn
    Q_bf16 = torch.randn(
        (num_batch, num_seq_q, num_head_q, num_dim_qk), dtype=T, device="cuda"
    ) / math.sqrt(num_dim_qk)

    max_seqlens_q_pad = ((num_seq_q + 127) // 128) * 128
    Q_fp32 = Q_bf16.to(torch.float32)
    qs_amax = Q_fp32.abs().amax(dim=-1)  # [B, sq, Hq]
    qs = (qs_amax / 448.0).clamp(min=1e-30).permute(0, 2, 1).contiguous()  # [B, Hq, sq]
    if max_seqlens_q_pad > num_seq_q:
        pad = torch.zeros(
            (num_batch, num_head_q, max_seqlens_q_pad - num_seq_q),
            dtype=torch.float32,
            device="cuda",
        )
        qs = torch.cat([qs, pad], dim=2).contiguous()

    qs_per_elem = qs[:, :, :num_seq_q].permute(0, 2, 1).unsqueeze(-1)  # [B, sq, Hq, 1]
    Q_fp8 = (Q_fp32 / qs_per_elem).to(T8)
    return Q_fp8, qs


@pytest.mark.parametrize(
    "quant_type",
    [
        hpc.QuantType.QFP8_KPERTENSOR_VPERTENSOR_FP16PV,
        hpc.QuantType.QFP8_KPERTOKEN_PERHEAD_VPERHEAD_FP16PV,
    ],
    ids=lambda qt: qt.name if hasattr(qt, "name") else str(qt),
)
@pytest.mark.parametrize("num_batch", [1, 4])
@pytest.mark.parametrize("num_seq_q", [128, 1500])
@pytest.mark.parametrize("num_seq_kv", [1500, 3000])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("num_head_q", [8])
@pytest.mark.parametrize("num_head_kv", [1])
@pytest.mark.parametrize("num_dim_qk", [128])
@pytest.mark.parametrize("num_dim_v", [128])
def test_attention_with_kvcache_prefill_fp8_kv_fp16_pv_compute_smoke(
    quant_type,
    num_batch,
    num_seq_q,
    num_seq_kv,
    block_size,
    num_head_q,
    num_head_kv,
    num_dim_qk,
    num_dim_v,
):
    qt = quant_type.value if hasattr(quant_type, "value") else int(quant_type)

    Q, qscale = _build_Q_fp8_and_qscale(num_batch, num_seq_q, num_head_q, num_dim_qk)
    (kcache, vcache, kscale, vscale, block_ids, seqlens_kvcache, cu_seqlens_q) = _build_kv_caches(
        num_batch, num_seq_q, num_seq_kv, block_size, num_head_kv, num_dim_qk, num_dim_v, qt
    )

    total_seq = num_batch * num_seq_q
    max_seqlens_q = num_seq_q

    gt = naive_fp8_kv_fp16_pv_compute_attn(
        Q,
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        seqlens_kvcache,
        block_ids,
        quant_type=qt,
        causal=True,
    ).reshape(total_seq, num_head_q, num_dim_v)

    out = hpc.attention_with_kvcache_prefill_fp8_kv_fp16_pv_compute(
        Q.reshape(total_seq, num_head_q, num_dim_qk),
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        max_seqlens_q,
        quant_type=quant_type,
    )
    torch.cuda.synchronize()

    assert out.shape == (total_seq, num_head_q, num_dim_v)
    assert out.dtype == torch.bfloat16
    assert out.device == Q.device

    assert allclose(gt, out, atol=0.02)
