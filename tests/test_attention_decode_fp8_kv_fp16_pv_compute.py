# Copyright 2025 hpc-ops authors

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


# Golden reference for hybrid FP8-QK / FP16-PV decode (dense, no causal mask for
# num_seq_q == 1). Q/K/V all fp8; Q@Kᵀ from dequantized fp32 Q/K (qscale·kscale
# applied to scores), P@V keeps V fp32 with V scale applied once in the epilogue.
# quant_type: 20 = QFP8_KPERTOKEN_PERHEAD_VPERHEAD_FP16PV,
#             21 = QFP8_KPERTENSOR_VPERTENSOR_FP16PV.


def _dequant_K_fp32_decode(kcache_fp8, kscale, quant_type):
    K_fp32 = kcache_fp8.to(torch.float32)
    if quant_type == 21:
        return K_fp32 * kscale.item()
    num_blocks, block_size, num_head_kv, num_dim_qk = kcache_fp8.shape
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


def _apply_V_scale_post_decode(out_fp32, vscale, quant_type, num_head_q, num_head_kv):
    if quant_type == 21:
        return out_fp32 * vscale.item()
    num_group = num_head_q // num_head_kv
    vs = vscale.repeat_interleave(num_group, dim=0).reshape(num_head_q, 1, 1).to(torch.float32)
    return out_fp32 * vs


def _dequant_Q_fp32_decode(q, qscale, num_seq_q):
    """q: [num_batch, num_seq_q, num_head_q, num_dim_qk] fp8.
    qscale: [num_batch * num_seq_q, num_head_q] fp32.
    """
    num_batch, sq, num_head_q, num_dim_qk = q.shape
    Q_fp32 = q.to(torch.float32)
    qs = qscale.reshape(num_batch, sq, num_head_q).unsqueeze(-1)  # [B, sq, Hq, 1]
    return Q_fp32 * qs.to(torch.float32)


def naive_fp8_kv_fp16_pv_compute_decode_attn(
    q, kcache_fp8, vcache_fp8, qscale, kscale, vscale, seqlens_kvcache, page_table, quant_type
):
    num_batch, num_seq_q, num_head_q, num_dim_qk = q.shape
    _, block_size, num_head_kv, _ = kcache_fp8.shape
    _, _, _, num_dim_v = vcache_fp8.shape
    num_group = num_head_q // num_head_kv
    kvcache_blocks = (seqlens_kvcache + block_size - 1) // block_size
    output = torch.empty(
        num_batch, num_seq_q, num_head_q, num_dim_v, dtype=torch.bfloat16, device=q.device
    )

    K_fp32 = _dequant_K_fp32_decode(kcache_fp8, kscale, quant_type)
    Q_fp32 = _dequant_Q_fp32_decode(q, qscale, num_seq_q)

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
        )

        scores = torch.matmul(BQ, BK.transpose(-2, -1)) / math.sqrt(num_dim_qk)
        attn_w = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_w, BV)
        out = _apply_V_scale_post_decode(out, vscale, quant_type, num_head_q, num_head_kv)
        output[i] = out.transpose(0, 1).to(torch.bfloat16)

    return output


def _build_kv_caches(
    num_batch, num_seq_kv, block_size, num_head_kv, num_dim_qk, num_dim_v, quant_type, num_seq_q=1
):
    T = torch.bfloat16
    T8 = torch.float8_e4m3fn

    seqlens_kvcache = torch.full((num_batch,), num_seq_kv, dtype=torch.int32, device="cuda")

    blocks_per_seq = (num_seq_kv + num_seq_q + block_size - 1) // block_size
    max_num_blocks = num_batch * blocks_per_seq * 2
    kv_bf16 = torch.randn(
        max_num_blocks, 2, block_size, num_head_kv, num_dim_qk, dtype=T, device="cuda"
    )

    if quant_type == 21:
        kscale = (kv_bf16[:, 0].abs().max().to(torch.float32) / 448.0).reshape(1).contiguous()
        vscale = (kv_bf16[:, 1].abs().max().to(torch.float32) / 448.0).reshape(1).contiguous()
        kcache = (kv_bf16[:, 0].to(torch.float32) / kscale).to(T8)
        vcache = (kv_bf16[:, 1].to(torch.float32) / vscale).to(T8)
    else:
        scale_extra_rows = block_size * 4 // num_dim_qk
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

    kvcache_blocks = (seqlens_kvcache + num_seq_q + block_size - 1) // block_size
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

    return kcache, vcache, kscale, vscale, block_ids, seqlens_kvcache


def _build_Q_fp8_and_qscale(num_batch, num_seq_q, num_head_q, num_dim_qk):
    """fp8 Q + per-(token,head) qscale [num_batch * num_seq_q, num_head_q] fp32."""
    T = torch.bfloat16
    T8 = torch.float8_e4m3fn
    Q_bf16 = torch.randn(
        (num_batch, num_seq_q, num_head_q, num_dim_qk), dtype=T, device="cuda"
    ) / math.sqrt(num_dim_qk)
    Q_fp32 = Q_bf16.to(torch.float32)
    qs = (Q_fp32.abs().amax(dim=-1) / 448.0).clamp(min=1e-30)  # [B, sq, Hq]
    Q_fp8 = (Q_fp32 / qs.unsqueeze(-1)).to(T8)
    qscale = qs.reshape(num_batch * num_seq_q, num_head_q).contiguous()
    return Q_fp8, qscale


@pytest.mark.parametrize("quant_type", [20, 21])
@pytest.mark.parametrize("num_batch", [16])
@pytest.mark.parametrize("num_seq_q", [1])
@pytest.mark.parametrize("num_seq_kv", [4096])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("num_head_q", [8])
@pytest.mark.parametrize("num_head_kv", [1])
@pytest.mark.parametrize("num_dim_qk", [128])
@pytest.mark.parametrize("num_dim_v", [128])
@pytest.mark.parametrize("use_splitk", [False, True])
@pytest.mark.parametrize("use_dynamic_sched", [False, True])
def test_attention_decode_fp8_kv_fp16_pv_compute_smoke(
    quant_type,
    num_batch,
    num_seq_q,
    num_seq_kv,
    block_size,
    num_head_q,
    num_head_kv,
    num_dim_qk,
    num_dim_v,
    use_splitk,
    use_dynamic_sched,
):
    if use_dynamic_sched and not use_splitk:
        pytest.skip("dynamic schedule requires splitk")

    Q, qscale = _build_Q_fp8_and_qscale(num_batch, num_seq_q, num_head_q, num_dim_qk)
    (kcache, vcache, kscale, vscale, block_ids, seqlens_kvcache) = _build_kv_caches(
        num_batch,
        num_seq_kv,
        block_size,
        num_head_kv,
        num_dim_qk,
        num_dim_v,
        quant_type,
        num_seq_q=num_seq_q,
    )

    task_map = None
    if use_dynamic_sched:
        # The dynamic scheduler's assign step takes num_seq_q (not mtp) as its
        # 4th arg, and num_seq_kvcache *including* the new KV tokens with
        # new_kv_included=True (mirrors the pure-fp8 dynamic decode test).
        task_map = hpc.get_attention_decode_task_workspace(
            num_batch, num_seq_kv + num_seq_q, num_head_kv, min_process_len=512
        )
        hpc.assign_attention_decode_task(
            seqlens_kvcache + num_seq_q,
            task_map,
            num_head_kv,
            num_seq_q,
            True,
            min_process_len=512,
        )

    gt = naive_fp8_kv_fp16_pv_compute_decode_attn(
        Q, kcache, vcache, qscale, kscale, vscale, seqlens_kvcache, block_ids, quant_type=quant_type
    ).reshape(num_batch * num_seq_q, num_head_q, num_dim_v)

    out = hpc.attention_decode_fp8_kv_fp16_pv_compute(
        Q.reshape(num_batch * num_seq_q, num_head_q, num_dim_qk),
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        block_ids,
        seqlens_kvcache,
        mtp=num_seq_q - 1,
        new_kv_included=False,
        use_splitk=use_splitk,
        quant_type=quant_type,
        task_map=task_map,
    )
    torch.cuda.synchronize()

    assert out.shape == (num_batch * num_seq_q, num_head_q, num_dim_v)
    assert out.dtype == torch.bfloat16
    assert out.device == Q.device

    assert allclose(gt, out, atol=0.05)


# Regression for the dynamic-splitk task_map workspace under-allocation: an
# illegal memory access fired for num_seq_q in {2,3} at long kv whose tile count
# is not a multiple of the CTA count. Root cause was a shared sizing bug in
# `get_attention_decode_task_workspace` (it sized the task region for
# kMaxCtaPerSm=4 CTAs, but the assigner uses kCtaPerSmMap[num_seq_q-1]*num_sm =
# 3*num_sm for num_seq_q in {2,3}; (ceil(T/n)+1)*n is non-monotonic in n).
# Affected every dynamic decode family (pure-fp8, a16c8, fp16-PV). These shapes
# reliably crashed before the fix; afterwards they must run and match.
@pytest.mark.parametrize("quant_type", [20, 21])
@pytest.mark.parametrize("num_seq_q", [2, 3])
@pytest.mark.parametrize("num_seq_kv", [16384, 32770])
@pytest.mark.parametrize("num_batch", [32])
@pytest.mark.parametrize("num_head_q,num_head_kv", [(64, 8)])
def test_attention_decode_fp8_kv_fp16_pv_compute_dynamic_longkv_oob(
    quant_type, num_seq_q, num_seq_kv, num_batch, num_head_q, num_head_kv
):
    block_size, num_dim_qk, num_dim_v = 64, 128, 128

    Q, qscale = _build_Q_fp8_and_qscale(num_batch, num_seq_q, num_head_q, num_dim_qk)
    (kcache, vcache, kscale, vscale, block_ids, seqlens_kvcache) = _build_kv_caches(
        num_batch,
        num_seq_kv,
        block_size,
        num_head_kv,
        num_dim_qk,
        num_dim_v,
        quant_type,
        num_seq_q=num_seq_q,
    )

    task_map = hpc.get_attention_decode_task_workspace(
        num_batch, num_seq_kv + num_seq_q, num_head_kv, min_process_len=512
    )
    hpc.assign_attention_decode_task(
        seqlens_kvcache + num_seq_q, task_map, num_head_kv, num_seq_q, True, min_process_len=512
    )

    gt = naive_fp8_kv_fp16_pv_compute_decode_attn(
        Q, kcache, vcache, qscale, kscale, vscale, seqlens_kvcache, block_ids, quant_type=quant_type
    ).reshape(num_batch * num_seq_q, num_head_q, num_dim_v)

    out = hpc.attention_decode_fp8_kv_fp16_pv_compute(
        Q.reshape(num_batch * num_seq_q, num_head_q, num_dim_qk),
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        block_ids,
        seqlens_kvcache,
        mtp=num_seq_q - 1,
        new_kv_included=False,
        use_splitk=True,
        quant_type=quant_type,
        task_map=task_map,
    )
    torch.cuda.synchronize()

    assert out.shape == (num_batch * num_seq_q, num_head_q, num_dim_v)
    assert allclose(gt, out, atol=0.05)
