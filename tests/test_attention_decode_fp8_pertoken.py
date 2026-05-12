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


def quant_paged_cache_pertoken(cache, block_size):
    num_blocks = cache.shape[0]
    head_dim = cache.shape[-1]
    num_head_kv = cache.shape[-2]
    scale = cache[:, :block_size, :, :].float().abs().max(-1)[0] / 448

    cache_fp8 = torch.empty_like(cache, dtype=torch.float8_e4m3fn)
    cache_fp8[:, :block_size, :, :] = (cache[:, :block_size, :, :] / scale[:, :, :, None]).to(
        torch.float8_e4m3fn
    )

    scale = (
        scale.permute(0, 2, 1)
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(num_blocks, num_head_kv, -1, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    cache_fp8[:, block_size:, :, :] = scale

    return cache_fp8, cache_fp8[:, block_size:, :, :]


def quant_paged_cache_perhead(cache, block_size):
    num_head_kv = cache.shape[-2]
    scale = (
        cache[:, :block_size, :, :]
        .float()
        .abs()
        .permute(2, 0, 1, 3)
        .reshape(num_head_kv, -1)
        .max(-1)[0]
        / 448
    )
    cache_fp8 = (cache.float() / scale[None, None, :, None]).to(torch.float8_e4m3fn)

    return cache_fp8, scale * 0.1


def naive_attn_with_paged_kvcache_func(
    Q,
    K,
    V,
    kvcache,
    block_ids,
    nblocks,
    seqlenq,
    cu_seqlenq,
    num_seq_kvcache,
    QS,
    KS,
    VS,
):

    num_batch = seqlenq.shape[0]
    num_head_q = Q.shape[1]
    num_head_kv = K.shape[1]
    head_dim = K.shape[2]
    block_size = kvcache.shape[2]
    head_per_group = num_head_q // num_head_kv
    Q = Q.reshape(num_batch, -1, num_head_q, head_dim)
    output = torch.empty_like(Q, dtype=torch.bfloat16)
    for bi in range(num_batch):
        BQ = Q[bi].transpose(0, 1).float()  # [num_heads, sq, head_dim]
        blk_ids = block_ids[bi, : nblocks[bi]]
        seqlen = seqlenq[bi] + num_seq_kvcache[bi]
        BK = (
            kvcache[blk_ids, 0, :, :, :]
            .reshape(-1, num_head_kv, head_dim)
            .transpose(0, 1)[:, :seqlen, :]
            .repeat_interleave(head_per_group, dim=0)
        ).float()
        BV = (
            kvcache[blk_ids, 1, :, :, :]
            .reshape(-1, num_head_kv, head_dim)
            .transpose(0, 1)[:, :seqlen, :]
            .repeat_interleave(head_per_group, dim=0)
        ).float()

        BKS = (
            KS[blk_ids, :, :, :]
            .view(torch.float32)
            .permute(0, 1, 3, 2)
            .reshape(-1, num_head_kv)
            .transpose(0, 1)[:, :seqlen]
            .repeat_interleave(head_per_group, dim=0)
        ).float()

        P = BQ @ BK.transpose(-1, -2)

        P = P / math.sqrt(head_dim) * BKS.unsqueeze(1) * QS[bi][:, None, None]

        causal_mask = torch.ones(
            seqlenq[bi], seqlen - seqlenq[bi], device=Q.device, dtype=torch.bool
        )
        tail_causal_mask = torch.tril(
            torch.ones(seqlenq[bi], seqlenq[bi], device=Q.device, dtype=torch.bool)
        )
        causal_mask = torch.cat([causal_mask, tail_causal_mask], dim=-1).unsqueeze(0)

        P = P.masked_fill(~causal_mask, float("-inf"))

        # attn_weights = F.softmax(P, dim=-1)
        attn_weights = torch.exp(P - P.max(dim=-1)[0][:, :, None])
        gSum = attn_weights.sum(dim=-1)[:, :, None]

        # attn_weights = attn_weights / gSum

        attn_weights = attn_weights.to(torch.float8_e4m3fn).float()

        Y = torch.matmul(attn_weights, BV)

        Y = Y / gSum

        Y = Y * VS[:, None, None].repeat_interleave(head_per_group, dim=0)

        output[bi] = Y.transpose(0, 1)

    return output.reshape(-1, num_head_q, head_dim)


def online_attn_with_paged_kvcache_func(
    Q,
    K,
    V,
    kvcache,
    block_ids,
    nblocks,
    seqlenq,
    cu_seqlenq,
    num_seq_kvcache,
    QS,
    KS,
    VS,
    splitk=1,
):

    num_batch = seqlenq.shape[0]
    num_head_q = Q.shape[1]
    num_head_kv = K.shape[1]
    head_dim = K.shape[2]
    block_size = kvcache.shape[2]
    head_per_group = num_head_q // num_head_kv
    Q = Q.reshape(num_batch, -1, num_head_q, head_dim)
    num_seq_q = Q.shape[1]
    output = torch.empty_like(Q, dtype=torch.bfloat16)
    one_over_dk_log2e = 1.4426950408889634 / math.sqrt(head_dim)
    for bi in range(num_batch):
        BQ = Q[bi].transpose(0, 1).float()  # [num_heads, sq, head_dim]
        blk_ids = block_ids[bi, : nblocks[bi]]
        seqlen = seqlenq[bi] + num_seq_kvcache[bi]
        num_tile_full = num_seq_kvcache[bi] // 64
        num_tile_kv = (seqlen + 63) // 64
        num_tile_causal = num_tile_kv - num_tile_full

        blk_ids = torch.cat([blk_ids[num_tile_full:], blk_ids[:num_tile_full]], dim=-1)

        gMax = torch.zeros(num_head_q, num_seq_q, dtype=torch.float32, device=Q.device) - 1e7
        gSum = torch.zeros(num_head_q, num_seq_q, dtype=torch.float32, device=Q.device)
        Y = torch.zeros_like(BQ)

        for idx, blk_id in enumerate(blk_ids):
            BK = (
                kvcache[blk_id, 0, :, :, :]
                .reshape(-1, num_head_kv, head_dim)
                .transpose(0, 1)
                .repeat_interleave(head_per_group, dim=0)
            ).float()

            BV = (
                kvcache[blk_id, 1, :, :, :]
                .reshape(-1, num_head_kv, head_dim)
                .transpose(0, 1)
                .repeat_interleave(head_per_group, dim=0)
            ).float()

            BKS = (
                KS[blk_id, :, :, :]
                .view(torch.float32)
                .permute(0, 2, 1)
                .reshape(-1, num_head_kv)
                .transpose(0, 1)
                .repeat_interleave(head_per_group, dim=0)
            ).float()

            P = BQ @ BK.transpose(-1, -2)

            if idx < num_tile_causal:
                causal_mask = torch.ones(
                    seqlenq[bi],
                    max(0, num_seq_kvcache[bi] - (idx + num_tile_full) * block_size),
                    device=Q.device,
                    dtype=torch.bool,
                )

                tail_causal_mask = torch.tril(
                    torch.ones(
                        seqlenq[bi],
                        min(
                            seqlenq[bi],
                            (idx + num_tile_full + 1) * block_size - num_seq_kvcache[bi],
                            seqlen - (idx + num_tile_full) * block_size,
                        ),
                        device=Q.device,
                        dtype=torch.bool,
                    )
                )

                empty_mask = torch.zeros(
                    (seqlenq[bi], max(0, (idx + num_tile_full + 1) * block_size - seqlen)),
                    device=Q.device,
                    dtype=torch.bool,
                )

                causal_mask = torch.cat(
                    [causal_mask, tail_causal_mask, empty_mask], dim=-1
                ).unsqueeze(0)
                P = P.masked_fill(~causal_mask, float("-inf"))

            P = P * BKS.unsqueeze(1) * QS[bi][:, None, None] * one_over_dk_log2e

            last_max = gMax

            gMax = torch.max(gMax, P.max(-1)[0])

            P = torch.exp2(P - gMax[:, :, None])

            scale = torch.exp2(last_max - gMax)

            gSum = scale * gSum + P.sum(-1)

            Y = scale[:, :, None] * Y

            P = P.to(torch.float8_e4m3fn).float()

            Y = torch.matmul(P, BV) + Y

        Y = Y / gSum[:, :, None]

        Y = Y * VS[:, None, None].repeat_interleave(head_per_group, dim=0)

        output[bi] = Y.transpose(0, 1)

    return output.reshape(-1, num_head_q, head_dim)


try:
    # fa2
    # from flash_attn.flash_attn_interface import flash_attn_with_kvcache
    # fa3
    from flash_attn_interface import flash_attn_with_kvcache

    # gt_attention_func = flash_attn_with_kvcache_func
    gt_attention_func = naive_attn_with_paged_kvcache_func
except Exception as e:
    print(f"execute naive_attn_func: {e}")
    gt_attention_func = naive_attn_with_paged_kvcache_func


def attention_decode_fp8_test_func(
    num_batch,
    num_seq_q,
    max_seq_kv,
    block_size,
    kv_head_q_head,
    head_dim,
    new_kv_included,
    use_output,
    splitk,
    use_dynamic_sched,
    kvcache_shape,
):
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    num_head_kv, num_head_q = kv_head_q_head

    num_dim_qk = head_dim
    num_dim_v = head_dim
    max_num_blocks = int(num_batch * max_seq_kv // block_size * 1.2)

    T = torch.float8_e4m3fn

    Q = torch.randn(
        (num_batch * num_seq_q, num_head_q, num_dim_qk), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(num_dim_qk)
    QS = Q.float().abs().max(-1)[0] / 10
    Q = (Q / QS[:, :, None]).to(T)

    K = torch.randn(
        (num_batch * num_seq_q, num_head_kv, num_dim_qk), dtype=torch.bfloat16, device="cuda"
    )
    V = torch.randn(
        (num_batch * num_seq_q, num_head_kv, num_dim_v), dtype=torch.bfloat16, device="cuda"
    )

    VS = torch.randn((num_head_kv), dtype=torch.float32, device="cuda")

    num_seq_kvcache = torch.randint(1, max_seq_kv, (num_batch,), dtype=torch.int32, device="cuda")

    task_map = None
    if use_dynamic_sched:
        task_map_for_cpu = hpc.get_attention_decode_task_workspace(
            num_batch, max_seq_kv + num_seq_q, num_head_kv, min_process_len=1024
        )
        task_map_for_cuda = hpc.get_attention_decode_task_workspace(
            num_batch, max_seq_kv + num_seq_q, num_head_kv, min_process_len=1024
        )

        hpc.assign_attention_decode_task(
            num_seq_kvcache.cpu() + num_seq_q,
            task_map_for_cpu,
            num_head_kv,
            num_seq_q,
            new_kv_included,
            min_process_len=1024,
        )

        hpc.assign_attention_decode_task(
            num_seq_kvcache + num_seq_q,
            task_map_for_cuda,
            num_head_kv,
            num_seq_q,
            new_kv_included,
            min_process_len=1024,
        )

        num_sm_count = (
            torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
            // num_head_kv
        )

        sched_need_byte_size = (task_map_for_cpu.view(torch.int32)[0] * num_sm_count + 1) * 32 + (
            num_batch * 4 + 31
        ) // 32 * 32
        assert torch.allclose(
            task_map_for_cpu[:sched_need_byte_size], task_map_for_cuda[:sched_need_byte_size]
        )

        task_map = task_map_for_cuda
        # hpc.print_attention_decode_task(task_map)

    nblocks = (num_seq_kvcache + num_seq_q + block_size - 1) // block_size
    total_blocks = sum(nblocks)
    kvcache_scale_rows = block_size * 4 // num_dim_qk

    kvcache = torch.randn(
        max_num_blocks,
        2,
        block_size + kvcache_scale_rows,
        num_head_kv,
        num_dim_qk,
        dtype=torch.bfloat16,
        device="cuda",
    )

    if kvcache_shape == "HND":
        kvcache = kvcache.permute(0, 1, 3, 2, 4).contiguous().permute(0, 1, 3, 2, 4)

    packed_block_ids = torch.randperm(max_num_blocks)[:total_blocks].to(torch.int32).cuda()

    max_num_block2 = max(nblocks)
    block_ids = torch.empty(num_batch, max_num_block2, dtype=torch.int32, device="cuda")
    seqlenq = torch.tensor([num_seq_q] * num_batch, dtype=torch.int32, device="cuda")
    cu_seqlenq = torch.cumsum(seqlenq, dtype=torch.int32, dim=0)

    cu_blocks = 0
    for i in range(num_batch):
        block_ids[i, : nblocks[i]] = packed_block_ids[cu_blocks : cu_blocks + nblocks[i]]
        cu_blocks += nblocks[i]
        for sqi in range(seqlenq[i]):
            si = sqi + num_seq_kvcache[i]
            blk_id = si // block_size
            slot_id = si % block_size
            kvcache[block_ids[i, blk_id], 0, slot_id] = K.reshape(
                num_batch, num_seq_q, num_head_kv, num_dim_qk
            )[i, sqi]
            kvcache[block_ids[i, blk_id], 1, slot_id] = V.reshape(
                num_batch, num_seq_q, num_head_kv, num_dim_qk
            )[i, sqi]

    kvcache_fp8 = torch.empty_like(kvcache, dtype=T)

    kcache, KS = quant_paged_cache_pertoken(kvcache[:, 0, :, :, :], block_size)
    vcache, VS = quant_paged_cache_perhead(kvcache[:, 1, :, :, :], block_size)

    kvcache_fp8[:, 0, :, :, :] = kcache
    kvcache_fp8[:, 1, :, :, :] = vcache

    KS = kvcache_fp8[:, 0, block_size:, :, :]

    gt = gt_attention_func(
        Q,
        K,
        V,
        kvcache_fp8[:, :, :block_size, :, :],
        block_ids,
        nblocks,
        seqlenq,
        cu_seqlenq,
        num_seq_kvcache,
        QS,
        KS,
        VS,
    )

    if use_output:
        my = torch.empty_like(Q, dtype=torch.bfloat16)
        hpc.attention_decode_fp8(
            Q,
            kvcache_fp8[:, 0, :block_size, :, :],
            kvcache_fp8[:, 1, :block_size, :, :],
            block_ids,
            num_seq_kvcache + num_seq_q if new_kv_included else num_seq_kvcache,
            QS,
            KS,
            VS,
            mtp=num_seq_q - 1,
            new_kv_included=new_kv_included,
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
            splitk=splitk,
            task_map=task_map,
            output=my,
        )
    else:
        for i in range(1):
            my = hpc.attention_decode_fp8(
                Q,
                kvcache_fp8[:, 0, :block_size, :, :],
                kvcache_fp8[:, 1, :block_size, :, :],
                block_ids,
                num_seq_kvcache + num_seq_q if new_kv_included else num_seq_kvcache,
                QS,
                KS,
                VS,
                mtp=num_seq_q - 1,
                new_kv_included=new_kv_included,
                quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
                splitk=splitk,
                task_map=task_map,
            )

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

    assert allclose(my, gt, atol=0.05)
    assert gt.device == my.device
    assert gt.dtype == my.dtype
    assert gt.shape == my.shape


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 9, reason="skip on non sm90!")
@pytest.mark.parametrize("num_batch", [1, 16, 200])
@pytest.mark.parametrize("num_seq_q", [1, 2, 3])
@pytest.mark.parametrize("max_seq_kv", [1024, 4096])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("kv_head_q_head", [(2, 8), (4, 32)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("new_kv_included", [True])
@pytest.mark.parametrize("use_output", [False])
@pytest.mark.parametrize("splitk", [True])
@pytest.mark.parametrize("use_dynamic_sched", [False])
@pytest.mark.parametrize("kvcache_shape", ["NHD", "HND"])
def test_attn_fp8_sm90(
    num_batch,
    num_seq_q,
    max_seq_kv,
    block_size,
    kv_head_q_head,
    head_dim,
    new_kv_included,
    use_output,
    splitk,
    use_dynamic_sched,
    kvcache_shape,
):
    attention_decode_fp8_test_func(
        num_batch,
        num_seq_q,
        max_seq_kv,
        block_size,
        kv_head_q_head,
        head_dim,
        new_kv_included,
        use_output,
        splitk,
        use_dynamic_sched,
        kvcache_shape,
    )


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 10, reason="skip on non sm100!")
@pytest.mark.parametrize("num_batch", [1, 16, 200])
@pytest.mark.parametrize("num_seq_q", [1, 2, 3])
@pytest.mark.parametrize("max_seq_kv", [1024, 4096])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("kv_head_q_head", [(1, 8), (4, 32)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("new_kv_included", [True])
@pytest.mark.parametrize("use_output", [False])
@pytest.mark.parametrize("splitk", [True])
@pytest.mark.parametrize("use_dynamic_sched", [True, False])
@pytest.mark.parametrize("kvcache_shape", ["NHD", "HND"])
def test_attn_fp8_sm100(
    num_batch,
    num_seq_q,
    max_seq_kv,
    block_size,
    kv_head_q_head,
    head_dim,
    new_kv_included,
    use_output,
    splitk,
    use_dynamic_sched,
    kvcache_shape,
):
    attention_decode_fp8_test_func(
        num_batch,
        num_seq_q,
        max_seq_kv,
        block_size,
        kv_head_q_head,
        head_dim,
        new_kv_included,
        use_output,
        splitk,
        use_dynamic_sched,
        kvcache_shape,
    )


# -----------------------------------------------------------------------------
# P_scale (kHasPScale=true) coverage
# -----------------------------------------------------------------------------
#
# attention_decode_fp8 now accepts an optional pair (p_scale, p_scale_inv) of
# shape [num_head_q] (float32, on the same device as q). When both are passed,
# the kernel goes through the kHasPScale=true template instance which:
#   * scales softmax(P) by p_scale[h] before fp8-quantizing P
#   * folds p_scale_inv[h] into the trailing vscale multiplication
# Mathematically the output is unchanged when p_scale * p_scale_inv == 1; the
# golden mirrors naive_attn_with_paged_kvcache_func (which already does the
# attn_weights -> fp8 round-trip) so the comparison stays apples-to-apples.


def naive_attn_with_paged_kvcache_pscale_func(
    Q,
    K,
    V,
    kvcache,
    block_ids,
    nblocks,
    seqlenq,
    cu_seqlenq,
    num_seq_kvcache,
    QS,
    KS,
    VS,
    p_scale=None,
    p_scale_inv=None,
):
    """P_scale-aware variant of naive_attn_with_paged_kvcache_func (pertoken K
    + per-head V scale).
    """
    num_batch = seqlenq.shape[0]
    num_head_q = Q.shape[1]
    num_head_kv = K.shape[1]
    head_dim = K.shape[2]
    block_size = kvcache.shape[2]
    head_per_group = num_head_q // num_head_kv

    has_ps = p_scale is not None
    if has_ps:
        assert p_scale_inv is not None and p_scale.shape == (num_head_q,)

    Q = Q.reshape(num_batch, -1, num_head_q, head_dim)
    output = torch.empty_like(Q, dtype=torch.bfloat16)
    for bi in range(num_batch):
        BQ = Q[bi].transpose(0, 1).float()  # [num_heads, sq, head_dim]
        blk_ids = block_ids[bi, : nblocks[bi]]
        seqlen = seqlenq[bi] + num_seq_kvcache[bi]
        BK = (
            kvcache[blk_ids, 0, :, :, :]
            .reshape(-1, num_head_kv, head_dim)
            .transpose(0, 1)[:, :seqlen, :]
            .repeat_interleave(head_per_group, dim=0)
        ).float()
        BV = (
            kvcache[blk_ids, 1, :, :, :]
            .reshape(-1, num_head_kv, head_dim)
            .transpose(0, 1)[:, :seqlen, :]
            .repeat_interleave(head_per_group, dim=0)
        ).float()

        BKS = (
            KS[blk_ids, :, :, :]
            .view(torch.float32)
            .permute(0, 1, 3, 2)
            .reshape(-1, num_head_kv)
            .transpose(0, 1)[:, :seqlen]
            .repeat_interleave(head_per_group, dim=0)
        ).float()

        P = BQ @ BK.transpose(-1, -2)
        P = P / math.sqrt(head_dim) * BKS.unsqueeze(1) * QS[bi][:, None, None]

        causal_mask = torch.ones(
            seqlenq[bi], seqlen - seqlenq[bi], device=Q.device, dtype=torch.bool
        )
        tail_causal_mask = torch.tril(
            torch.ones(seqlenq[bi], seqlenq[bi], device=Q.device, dtype=torch.bool)
        )
        causal_mask = torch.cat([causal_mask, tail_causal_mask], dim=-1).unsqueeze(0)
        P = P.masked_fill(~causal_mask, float("-inf"))

        attn_weights = torch.exp(P - P.max(dim=-1)[0][:, :, None])
        gSum = attn_weights.sum(dim=-1)[:, :, None]

        # Mirror the kernel: P_scale is applied per-q-head before fp8 quant.
        if has_ps:
            attn_weights = attn_weights * p_scale[:, None, None]

        attn_weights = attn_weights.to(torch.float8_e4m3fn).float()

        Y = torch.matmul(attn_weights, BV)
        Y = Y / gSum

        # Per-head V scale, optionally compensated by p_scale_inv.
        v_scale_eff = VS[:, None, None].repeat_interleave(head_per_group, dim=0)
        if has_ps:
            v_scale_eff = v_scale_eff * p_scale_inv[:, None, None]
        Y = Y * v_scale_eff

        output[bi] = Y.transpose(0, 1)

    return output.reshape(-1, num_head_q, head_dim)


def _make_pscale(mode, num_head_q, device):
    """Helper for the four canonical p_scale modes."""
    if mode == "none":
        return None, None
    if mode == "all_ones":
        p = torch.ones(num_head_q, dtype=torch.float32, device=device)
        return p, p.clone()
    if mode == "all_2":
        p = torch.full((num_head_q,), 2.0, dtype=torch.float32, device=device)
        pi = torch.full((num_head_q,), 0.5, dtype=torch.float32, device=device)
        return p, pi
    if mode == "per_head_random":
        g = torch.Generator(device=device).manual_seed(20240514)
        p = 0.7 + 0.8 * torch.rand(num_head_q, generator=g, device=device, dtype=torch.float32)
        return p, 1.0 / p
    raise ValueError(mode)


def attention_decode_fp8_pertoken_pscale_test_func(
    num_batch,
    num_seq_q,
    max_seq_kv,
    block_size,
    kv_head_q_head,
    head_dim,
    p_scale_mode,
):
    """Same data construction as attention_decode_fp8_test_func, but routed
    through hpc.attention_decode_fp8(p_scale=..., p_scale_inv=...) and the
    matching P_scale-aware golden."""
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    num_head_kv, num_head_q = kv_head_q_head
    num_dim_qk = head_dim
    num_dim_v = head_dim
    max_num_blocks = int(num_batch * max_seq_kv // block_size * 1.2)

    T = torch.float8_e4m3fn

    Q = torch.randn(
        (num_batch * num_seq_q, num_head_q, num_dim_qk), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(num_dim_qk)
    QS = Q.float().abs().max(-1)[0] / 10
    Q = (Q / QS[:, :, None]).to(T)

    K = torch.randn(
        (num_batch * num_seq_q, num_head_kv, num_dim_qk), dtype=torch.bfloat16, device="cuda"
    )
    V = torch.randn(
        (num_batch * num_seq_q, num_head_kv, num_dim_v), dtype=torch.bfloat16, device="cuda"
    )

    VS = torch.randn((num_head_kv), dtype=torch.float32, device="cuda")

    num_seq_kvcache = torch.randint(1, max_seq_kv, (num_batch,), dtype=torch.int32, device="cuda")

    nblocks = (num_seq_kvcache + num_seq_q + block_size - 1) // block_size
    total_blocks = sum(nblocks)
    kvcache_scale_rows = block_size * 4 // num_dim_qk

    kvcache = torch.randn(
        max_num_blocks,
        2,
        block_size + kvcache_scale_rows,
        num_head_kv,
        num_dim_qk,
        dtype=torch.bfloat16,
        device="cuda",
    )

    packed_block_ids = torch.randperm(max_num_blocks)[:total_blocks].to(torch.int32).cuda()
    max_num_block2 = max(nblocks)
    block_ids = torch.empty(num_batch, max_num_block2, dtype=torch.int32, device="cuda")
    seqlenq = torch.tensor([num_seq_q] * num_batch, dtype=torch.int32, device="cuda")
    cu_seqlenq = torch.cumsum(seqlenq, dtype=torch.int32, dim=0)

    cu_blocks = 0
    for i in range(num_batch):
        block_ids[i, : nblocks[i]] = packed_block_ids[cu_blocks : cu_blocks + nblocks[i]]
        cu_blocks += nblocks[i]
        for sqi in range(seqlenq[i]):
            si = sqi + num_seq_kvcache[i]
            blk_id = si // block_size
            slot_id = si % block_size
            kvcache[block_ids[i, blk_id], 0, slot_id] = K.reshape(
                num_batch, num_seq_q, num_head_kv, num_dim_qk
            )[i, sqi]
            kvcache[block_ids[i, blk_id], 1, slot_id] = V.reshape(
                num_batch, num_seq_q, num_head_kv, num_dim_qk
            )[i, sqi]

    kvcache_fp8 = torch.empty_like(kvcache, dtype=T)
    kcache, KS = quant_paged_cache_pertoken(kvcache[:, 0, :, :, :], block_size)
    vcache, VS = quant_paged_cache_perhead(kvcache[:, 1, :, :, :], block_size)
    kvcache_fp8[:, 0, :, :, :] = kcache
    kvcache_fp8[:, 1, :, :, :] = vcache
    KS = kvcache_fp8[:, 0, block_size:, :, :]

    p_scale, p_scale_inv = _make_pscale(p_scale_mode, num_head_q, device="cuda")

    gt = naive_attn_with_paged_kvcache_pscale_func(
        Q,
        K,
        V,
        kvcache_fp8[:, :, :block_size, :, :],
        block_ids,
        nblocks,
        seqlenq,
        cu_seqlenq,
        num_seq_kvcache,
        QS,
        KS,
        VS,
        p_scale=p_scale,
        p_scale_inv=p_scale_inv,
    )

    my = hpc.attention_decode_fp8(
        Q,
        kvcache_fp8[:, 0, :block_size, :, :],
        kvcache_fp8[:, 1, :block_size, :, :],
        block_ids,
        num_seq_kvcache + num_seq_q,
        QS,
        KS,
        VS,
        mtp=num_seq_q - 1,
        new_kv_included=True,
        quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
        splitk=True,
        task_map=None,
        p_scale=p_scale,
        p_scale_inv=p_scale_inv,
    )

    # atol = 0.05: golden mirrors the kernel's full round-trip; max abs diff
    # over a 30-seed sweep stays under ~0.009; 0.05 leaves ample headroom.
    assert allclose(my, gt, atol=0.05), (
        f"[p_scale={p_scale_mode}] decode_fp8_pertoken diverges from golden " f"beyond atol=0.05"
    )
    assert gt.shape == my.shape and gt.dtype == my.dtype and gt.device == my.device


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 9, reason="skip on non sm90!")
@pytest.mark.parametrize("num_batch", [1, 16])
@pytest.mark.parametrize("num_seq_q", [1, 2])
@pytest.mark.parametrize("max_seq_kv", [1024])
@pytest.mark.parametrize("kv_head_q_head", [(2, 8), (4, 32)])
@pytest.mark.parametrize("p_scale_mode", ["none", "all_ones", "all_2", "per_head_random"])
def test_attn_fp8_pscale_sm90(
    num_batch,
    num_seq_q,
    max_seq_kv,
    kv_head_q_head,
    p_scale_mode,
):
    attention_decode_fp8_pertoken_pscale_test_func(
        num_batch=num_batch,
        num_seq_q=num_seq_q,
        max_seq_kv=max_seq_kv,
        block_size=64,
        kv_head_q_head=kv_head_q_head,
        head_dim=128,
        p_scale_mode=p_scale_mode,
    )
