import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
from utils import allclose


def report_cos_atol(gt, my, atol=0.1, cos_threshold=0.999, tag=""):
    gtf = gt.float().reshape(-1)
    myf = my.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(gtf, myf, dim=0).item()
    max_abs = (gtf - myf).abs().max().item()
    mean_abs = (gtf - myf).abs().mean().item()
    print(f"[cos] {tag} cos={cos:.6f} max_abs={max_abs:.5f} mean_abs={mean_abs:.6f}")
    assert cos > cos_threshold, f"{tag} cosine {cos:.6f} < {cos_threshold}"
    assert allclose(gt, my, atol=atol), f"{tag} max_abs={max_abs:.5f} > atol={atol}"


def make_p_scale(p_scale_mode, num_head_q, device):
    if p_scale_mode == "256":
        return torch.full((num_head_q,), 256.0, dtype=torch.float32, device=device)
    if p_scale_mode == "linspace":
        return torch.linspace(128.0, 256.0, num_head_q, dtype=torch.float32, device=device)
    if p_scale_mode == "all_2":
        return torch.full((num_head_q,), 2.0, dtype=torch.float32, device=device)
    if p_scale_mode == "per_head_random":
        return 0.7 + 0.8 * torch.rand(num_head_q, dtype=torch.float32, device=device)
    raise ValueError(p_scale_mode)


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


def ref_attn_with_paged_kvcache_func(
    q,
    k,
    v,
    kvcache,
    block_ids,
    nblocks,
    seqlenq,
    cu_seqlenq,
    num_seq_kvcache,
    q_scale,
    k_scale,
    p_scale,
    v_scale,
):

    num_batch = seqlenq.shape[0]
    num_head_q = q.shape[1]
    num_head_kv = k.shape[1]
    head_dim = v.shape[2]
    block_size = kvcache.shape[2]
    head_per_group = num_head_q // num_head_kv
    q = q.reshape(num_batch, -1, num_head_q, head_dim)
    output = torch.empty_like(q, dtype=torch.bfloat16)
    for bi in range(num_batch):
        q_batch = q[bi].transpose(0, 1).float()  # [num_heads, sq, head_dim]
        blk_ids = block_ids[bi, : nblocks[bi]]
        seqlen = seqlenq[bi] + num_seq_kvcache[bi]
        k_batch = (
            kvcache[blk_ids, 0, :, :, :]
            .reshape(-1, num_head_kv, head_dim)
            .transpose(0, 1)[:, :seqlen, :]
            .repeat_interleave(head_per_group, dim=0)
        ).float()
        v_batch = (
            kvcache[blk_ids, 1, :, :, :]
            .reshape(-1, num_head_kv, head_dim)
            .transpose(0, 1)[:, :seqlen, :]
            .repeat_interleave(head_per_group, dim=0)
        ).float()

        k_scale_batch = (
            k_scale[blk_ids, :, :, :]
            .view(torch.float32)
            .permute(0, 1, 3, 2)
            .reshape(-1, num_head_kv)
            .transpose(0, 1)[:, :seqlen]
            .repeat_interleave(head_per_group, dim=0)
        ).float()

        p = q_batch @ k_batch.transpose(-1, -2)

        p = p / math.sqrt(head_dim) * k_scale_batch.unsqueeze(1) * q_scale[bi][:, None, None]

        causal_mask = torch.ones(
            seqlenq[bi], seqlen - seqlenq[bi], device=q.device, dtype=torch.bool
        )
        tail_causal_mask = torch.tril(
            torch.ones(seqlenq[bi], seqlenq[bi], device=q.device, dtype=torch.bool)
        )
        causal_mask = torch.cat([causal_mask, tail_causal_mask], dim=-1).unsqueeze(0)

        p = p.masked_fill(~causal_mask, float("-inf"))

        attn_weights = torch.exp(p - p.max(dim=-1)[0][:, :, None])
        gSum = attn_weights.sum(dim=-1)[:, :, None]

        p_scale_eff = p_scale[:, None, None]
        attn_weights = attn_weights * p_scale_eff
        attn_weights = attn_weights.to(torch.float8_e4m3fn).float()

        y = torch.matmul(attn_weights, v_batch)

        y = y / gSum

        y = y * v_scale[:, None, None].repeat_interleave(head_per_group, dim=0)
        y = y / p_scale_eff

        output[bi] = y.transpose(0, 1)

    return output.reshape(-1, num_head_q, head_dim)


def online_attn_with_paged_kvcache_func(
    q,
    k,
    v,
    kvcache,
    block_ids,
    nblocks,
    seqlenq,
    cu_seqlenq,
    num_seq_kvcache,
    q_scale,
    k_scale,
    v_scale,
    splitk=1,
):

    num_batch = seqlenq.shape[0]
    num_head_q = q.shape[1]
    num_head_kv = k.shape[1]
    head_dim = k.shape[2]
    block_size = kvcache.shape[2]
    head_per_group = num_head_q // num_head_kv
    q = q.reshape(num_batch, -1, num_head_q, head_dim)
    num_seq_q = q.shape[1]
    output = torch.empty_like(q, dtype=torch.bfloat16)
    one_over_dk_log2e = 1.4426950408889634 / math.sqrt(head_dim)
    for bi in range(num_batch):
        q_batch = q[bi].transpose(0, 1).float()  # [num_heads, sq, head_dim]
        blk_ids = block_ids[bi, : nblocks[bi]]
        seqlen = seqlenq[bi] + num_seq_kvcache[bi]
        num_tile_full = num_seq_kvcache[bi] // 64
        num_tile_kv = (seqlen + 63) // 64
        num_tile_causal = num_tile_kv - num_tile_full

        blk_ids = torch.cat([blk_ids[num_tile_full:], blk_ids[:num_tile_full]], dim=-1)

        gMax = torch.zeros(num_head_q, num_seq_q, dtype=torch.float32, device=Q.device) - 1e7
        gSum = torch.zeros(num_head_q, num_seq_q, dtype=torch.float32, device=Q.device)
        Y = torch.zeros_like(q_batch)

        for idx, blk_id in enumerate(blk_ids):
            k_batch = (
                kvcache[blk_id, 0, :, :, :]
                .reshape(-1, num_head_kv, head_dim)
                .transpose(0, 1)
                .repeat_interleave(head_per_group, dim=0)
            ).float()

            v_batch = (
                kvcache[blk_id, 1, :, :, :]
                .reshape(-1, num_head_kv, head_dim)
                .transpose(0, 1)
                .repeat_interleave(head_per_group, dim=0)
            ).float()

            k_scale_batch = (
                k_scale[blk_id, :, :, :]
                .view(torch.float32)
                .permute(0, 2, 1)
                .reshape(-1, num_head_kv)
                .transpose(0, 1)
                .repeat_interleave(head_per_group, dim=0)
            ).float()

            p = q_batch @ k_batch.transpose(-1, -2)

            if idx < num_tile_causal:
                causal_mask = torch.ones(
                    seqlenq[bi],
                    max(0, num_seq_kvcache[bi] - (idx + num_tile_full) * block_size),
                    device=q.device,
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
                        device=q.device,
                        dtype=torch.bool,
                    )
                )

                empty_mask = torch.zeros(
                    (seqlenq[bi], max(0, (idx + num_tile_full + 1) * block_size - seqlen)),
                    device=q.device,
                    dtype=torch.bool,
                )

                causal_mask = torch.cat(
                    [causal_mask, tail_causal_mask, empty_mask], dim=-1
                ).unsqueeze(0)
                p = p.masked_fill(~causal_mask, float("-inf"))

            p = p * k_scale_batch.unsqueeze(1) * q_scale[bi][:, None, None] * one_over_dk_log2e
            last_max = gMax
            gMax = torch.max(gMax, p.max(-1)[0])
            p = torch.exp2(p - gMax[:, :, None])
            scale = torch.exp2(last_max - gMax)
            gSum = scale * gSum + p.sum(-1)
            y = scale[:, :, None] * y
            p = p.to(torch.float8_e4m3fn).float()
            y = torch.matmul(p, v_batch) + y

        y = y / gSum[:, :, None]
        y = y * v_scale[:, None, None].repeat_interleave(head_per_group, dim=0)
        output[bi] = y.transpose(0, 1)

    return output.reshape(-1, num_head_q, head_dim)


def attention_decode_fp8_test_func(
    num_batch,
    num_seq_q,
    max_seq_kv,
    block_size,
    kv_head_q_head,
    head_dim,
    new_kv_included,
    use_output,
    kvcache_shape,
    p_scale_mode,
    kv_lens=None,
):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)

    num_head_kv, num_head_q = kv_head_q_head

    num_dim_qk = head_dim
    num_dim_v = head_dim
    max_num_blocks = int(num_batch * max_seq_kv // block_size * 1.2)

    q = torch.randn(
        (num_batch * num_seq_q, num_head_q, num_dim_qk), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(num_dim_qk)
    q_scale = q.float().abs().max(-1)[0] / 10
    q = (q / q_scale[:, :, None]).to(torch.float8_e4m3fn)
    k = torch.randn(
        (num_batch * num_seq_q, num_head_kv, num_dim_qk), dtype=torch.bfloat16, device="cuda"
    )
    v = torch.randn(
        (num_batch * num_seq_q, num_head_kv, num_dim_v), dtype=torch.bfloat16, device="cuda"
    )

    v_scale = torch.randn((num_head_kv), dtype=torch.float32, device="cuda")

    if kv_lens is not None:
        num_seq_kvcache = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    else:
        num_seq_kvcache = torch.randint(
            1, max_seq_kv, (num_batch,), dtype=torch.int32, device="cuda"
        )

    task_map_for_cpu = hpc.get_attention_decode_task_workspace_adaptive(
        num_batch, max_seq_kv + num_seq_q, num_head_kv, min_process_len=2048
    )
    task_map_for_cuda = hpc.get_attention_decode_task_workspace_adaptive(
        num_batch, max_seq_kv + num_seq_q, num_head_kv, min_process_len=2048
    )

    hpc.assign_attention_decode_task_adaptive(
        num_seq_kvcache.cpu() + num_seq_q,
        task_map_for_cpu,
        num_head_kv,
        num_seq_q,
        new_kv_included,
        min_process_len=2048,
    )

    hpc.assign_attention_decode_task_adaptive(
        num_seq_kvcache + num_seq_q,
        task_map_for_cuda,
        num_head_kv,
        num_seq_q,
        new_kv_included,
        min_process_len=2048,
    )

    num_total_ctas = task_map_for_cuda[1]

    sched_need_byte_size = (task_map_for_cpu.view(torch.int32)[0] * num_total_ctas + 1) * 48 + (
        num_batch * num_head_kv * 4 + 47
    ) // 48 * 48
    assert torch.allclose(
        task_map_for_cpu[:sched_need_byte_size], task_map_for_cuda[:sched_need_byte_size]
    )

    task_map = task_map_for_cuda

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
            kvcache[block_ids[i, blk_id], 0, slot_id] = k.reshape(
                num_batch, num_seq_q, num_head_kv, num_dim_qk
            )[i, sqi]
            kvcache[block_ids[i, blk_id], 1, slot_id] = v.reshape(
                num_batch, num_seq_q, num_head_kv, num_dim_qk
            )[i, sqi]

    kvcache_fp8 = torch.empty_like(kvcache, dtype=torch.float8_e4m3fn)

    kcache, k_scale = quant_paged_cache_pertoken(kvcache[:, 0, :, :, :], block_size)
    vcache, v_scale = quant_paged_cache_perhead(kvcache[:, 1, :, :, :], block_size)
    p_scale = make_p_scale(p_scale_mode, num_head_q, "cuda")

    kvcache_fp8[:, 0, :, :, :] = kcache
    kvcache_fp8[:, 1, :, :, :] = vcache

    k_scale = kvcache_fp8[:, 0, block_size:, :, :]

    gt = ref_attn_with_paged_kvcache_func(
        q,
        k,
        v,
        kvcache_fp8[:, :, :block_size, :, :],
        block_ids,
        nblocks,
        seqlenq,
        cu_seqlenq,
        num_seq_kvcache,
        q_scale,
        k_scale,
        p_scale,
        v_scale,
    )

    if use_output:
        my = torch.empty_like(q, dtype=torch.bfloat16)
        hpc.attention_decode_fp8_adaptive(
            q,
            kvcache_fp8[:, 0, :block_size, :, :],
            kvcache_fp8[:, 1, :block_size, :, :],
            block_ids,
            num_seq_kvcache + num_seq_q if new_kv_included else num_seq_kvcache,
            q_scale,
            k_scale,
            v_scale,
            mtp=num_seq_q - 1,
            p_scale=p_scale,
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
            task_map=task_map,
            output=my,
        )
    else:
        num_seq_kvcache_input = num_seq_kvcache + num_seq_q if new_kv_included else num_seq_kvcache
        my = hpc.attention_decode_fp8_adaptive(
            q,
            kvcache_fp8[:, 0, :block_size, :, :],
            kvcache_fp8[:, 1, :block_size, :, :],
            block_ids,
            num_seq_kvcache_input,
            q_scale,
            k_scale,
            v_scale,
            mtp=num_seq_q - 1,
            p_scale=p_scale,
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
            task_map=task_map,
        )

    report_cos_atol(
        gt,
        my,
        atol=0.1,
        tag=f"decode[{kvcache_shape},{num_head_kv}x{num_head_q},b{num_batch},"
        f"sq{num_seq_q},kv{max_seq_kv},{p_scale_mode},out{use_output}]",
    )


@pytest.mark.skipif(True, reason="skip on ci!")
@pytest.mark.parametrize("num_batch", [1, 16, 32])
@pytest.mark.parametrize("num_seq_q", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("max_seq_kv", [1024, 4096, 16384])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("kv_head_q_head", [(1, 8), (4, 32)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("new_kv_included", [True])
@pytest.mark.parametrize("use_output", [True, False])
@pytest.mark.parametrize("kvcache_shape", ["NHD", "HND"])
@pytest.mark.parametrize("p_scale_mode", ["256", "linspace"])
def test_attn_fp8_offline(
    num_batch,
    num_seq_q,
    max_seq_kv,
    block_size,
    kv_head_q_head,
    head_dim,
    new_kv_included,
    use_output,
    kvcache_shape,
    p_scale_mode,
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
        kvcache_shape,
        p_scale_mode,
    )


@pytest.mark.parametrize("num_batch", [1])
@pytest.mark.parametrize("num_seq_q", [1])
@pytest.mark.parametrize("max_seq_kv", [256])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("kv_head_q_head", [(1, 8)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("new_kv_included", [True])
@pytest.mark.parametrize("use_output", [True])
@pytest.mark.parametrize("kvcache_shape", ["NHD"])
@pytest.mark.parametrize("p_scale_mode", ["256"])
def test_attn_fp8(
    num_batch,
    num_seq_q,
    max_seq_kv,
    block_size,
    kv_head_q_head,
    head_dim,
    new_kv_included,
    use_output,
    kvcache_shape,
    p_scale_mode,
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
        kvcache_shape,
        p_scale_mode,
    )
