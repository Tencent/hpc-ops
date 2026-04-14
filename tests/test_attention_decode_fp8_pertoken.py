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

    return cache_fp8, scale


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


@pytest.mark.parametrize("num_batch", [1, 16, 200])
@pytest.mark.parametrize("num_seq_q", [1, 2, 3])
@pytest.mark.parametrize("max_seq_kv", [1024, 4096])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("kv_head_q_head", [(2, 8), (4, 32)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("new_kv_included", [True])
@pytest.mark.parametrize("use_output", [False])
@pytest.mark.parametrize("splitk", [True])
def test_attention_decode_fp8(
    num_batch,
    num_seq_q,
    max_seq_kv,
    block_size,
    kv_head_q_head,
    head_dim,
    new_kv_included,
    use_output,
    splitk,
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
            new_kv_included=new_kv_included,
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
            splitk=splitk,
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

    assert allclose(my, gt, atol=0.1)
    assert gt.device == my.device
    assert gt.dtype == my.dtype
    assert gt.shape == my.shape
