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

        P = BQ @ BK.transpose(-1, -2)

        P = P / math.sqrt(head_dim) * QS[bi][:, None, None] * KS

        causal_mask = torch.ones(
            seqlenq[bi], seqlen - seqlenq[bi], device=Q.device, dtype=torch.bool
        )
        tail_causal_mask = torch.tril(
            torch.ones(seqlenq[bi], seqlenq[bi], device=Q.device, dtype=torch.bool)
        )
        causal_mask = torch.cat([causal_mask, tail_causal_mask], dim=-1).unsqueeze(0)

        P = P.masked_fill(~causal_mask, float("-inf"))

        # Mirror the kernel's online softmax + un-normalised fp8 quant path:
        # the kernel quantises exp(P - row_max) (range (0, 1] with row-max
        # element exactly 1.0 - i.e. the high-resolution end of e4m3) and
        # divides by row_sum AFTER the PV gemm. F.softmax-then-quant would
        # collapse most values into e4m3's low-resolution low end and inflate
        # diff vs kernel.
        attn_weights = torch.exp(P - P.max(dim=-1)[0][:, :, None])
        gSum = attn_weights.sum(dim=-1)[:, :, None]
        attn_weights = attn_weights.to(torch.float8_e4m3fn).float()

        Y = torch.matmul(attn_weights, BV)
        Y = Y / gSum
        Y = Y * VS

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
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)

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

    K = (
        torch.randn(
            (num_batch * num_seq_q, num_head_kv, num_dim_qk), dtype=torch.bfloat16, device="cuda"
        )
        / math.sqrt(num_dim_qk)
    ).to(T)
    V = torch.randn(
        (num_batch * num_seq_q, num_head_kv, num_dim_v), dtype=torch.bfloat16, device="cuda"
    ).to(T)

    KS = torch.randn((1), dtype=torch.float32, device="cuda")
    VS = torch.randn((1), dtype=torch.float32, device="cuda")

    print(QS * KS)

    num_seq_kvcache = torch.randint(1, max_seq_kv, (num_batch,), dtype=torch.int32, device="cuda")

    print(f"num_seq_kvcache: {num_seq_kvcache.sum()}")

    task_map = None
    if use_dynamic_sched:
        # sm90 dynamic path: workspace + assigner at kTileN=64 (sm90 GEMM tile
        # size). Allocate CPU- and CUDA-resident task_maps, populate each via
        # its respective backend of hpc.assign_attention_decode_task
        # and assert byte-for-byte agreement over the used prefix.
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

        # sm90 dynamic path uses a flat bin count: num_sm * kCTAPerSM.
        # Per-task entries are 48 B (SM90DynamicTaskInfo), not 32 B.
        kCTAPerSM = 4
        num_total_ctas = (
            torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
            * kCTAPerSM
        )

        sched_need_byte_size = (task_map_for_cpu.view(torch.int32)[0] * num_total_ctas + 1) * 48 + (
            num_batch * num_head_kv * 4 + 47
        ) // 48 * 48
        assert torch.allclose(
            task_map_for_cpu[:sched_need_byte_size], task_map_for_cuda[:sched_need_byte_size]
        )

        task_map = task_map_for_cuda
        # import pdb; pdb.set_trace()
        # hpc.print_attention_decode_task(task_map)

    nblocks = (num_seq_kvcache + num_seq_q + block_size - 1) // block_size
    total_blocks = sum(nblocks)
    kvcache = (
        torch.randn(
            max_num_blocks,
            2,
            block_size,
            num_head_kv,
            num_dim_qk,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / math.sqrt(num_dim_qk)
    ).to(T)

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

    gt = gt_attention_func(
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
    )

    if use_output:
        my = torch.empty_like(Q, dtype=torch.bfloat16)
        hpc.attention_decode_fp8(
            Q,
            kvcache[:, 0, :, :, :],
            kvcache[:, 1, :, :, :],
            block_ids,
            num_seq_kvcache + num_seq_q if new_kv_included else num_seq_kvcache,
            QS,
            KS,
            VS,
            mtp=num_seq_q - 1,
            new_kv_included=new_kv_included,
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
            splitk=splitk,
            task_map=task_map,
            output=my,
        )
    else:
        for i in range(20):
            my = hpc.attention_decode_fp8(
                Q,
                kvcache[:, 0, :, :, :],
                kvcache[:, 1, :, :, :],
                block_ids,
                num_seq_kvcache + num_seq_q if new_kv_included else num_seq_kvcache,
                QS,
                KS,
                VS,
                mtp=num_seq_q - 1,
                new_kv_included=new_kv_included,
                quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
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
@pytest.mark.parametrize("num_batch", [296])
@pytest.mark.parametrize("num_seq_q", [1])
@pytest.mark.parametrize("max_seq_kv", [1024 * 32])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("kv_head_q_head", [(1, 8)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("new_kv_included", [True])
@pytest.mark.parametrize("use_output", [False])
@pytest.mark.parametrize("splitk", [True])
@pytest.mark.parametrize("use_dynamic_sched", [False, True])
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
# golden below mirrors the original naive_attn_with_paged_kvcache_func and
# applies the same scale/comp pair so that any diff is solely the kernel's.


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
    """P_scale-aware variant of naive_attn_with_paged_kvcache_func.

    When p_scale / p_scale_inv are provided (shape [num_head_q]), softmax
    output is multiplied by p_scale[h] before the (implicit) FP8 quant step,
    and the final V-scale multiplication is replaced by VS * p_scale_inv[h].
    p_scale=None reproduces the original naive exactly.
    """
    num_batch = seqlenq.shape[0]
    num_head_q = Q.shape[1]
    num_head_kv = K.shape[1]
    head_dim = K.shape[2]
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

        P = BQ @ BK.transpose(-1, -2)
        P = P / math.sqrt(head_dim) * QS[bi][:, None, None] * KS

        causal_mask = torch.ones(
            seqlenq[bi], seqlen - seqlenq[bi], device=Q.device, dtype=torch.bool
        )
        tail_causal_mask = torch.tril(
            torch.ones(seqlenq[bi], seqlenq[bi], device=Q.device, dtype=torch.bool)
        )
        causal_mask = torch.cat([causal_mask, tail_causal_mask], dim=-1).unsqueeze(0)
        P = P.masked_fill(~causal_mask, float("-inf"))

        # Mirror the kernel: do NOT normalise (i.e. don't call F.softmax
        # before fp8 quant); the kernel quantises exp(P - row_max) (range
        # (0, 1] with the row-max element exactly == 1) and divides by the
        # row_sum AFTER the PV gemm. Doing softmax-then-quant instead would
        # collapse most values into e4m3's low-resolution low end and
        # exceed the original atol=0.05 even at p_scale=None.
        attn_weights = torch.exp(P - P.max(dim=-1)[0][:, :, None])
        gSum = attn_weights.sum(dim=-1)[:, :, None]

        # P_scale (when provided) is applied per-q-head before fp8 quant.
        if has_ps:
            attn_weights = attn_weights * p_scale[:, None, None]
        attn_weights = attn_weights.to(torch.float8_e4m3fn).float()

        Y = torch.matmul(attn_weights, BV)
        Y = Y / gSum

        if has_ps:
            Y = Y * (VS * p_scale_inv[:, None, None])
        else:
            Y = Y * VS

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


def attention_decode_fp8_pscale_test_func(
    num_batch,
    num_seq_q,
    max_seq_kv,
    block_size,
    kv_head_q_head,
    head_dim,
    p_scale_mode,
    use_dynamic_sched=False,
):
    """Same data construction as attention_decode_fp8_test_func, but with the
    optional p_scale / p_scale_inv kwargs on hpc.attention_decode_fp8 and the
    matching P_scale-aware golden."""
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)

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

    K = (
        torch.randn(
            (num_batch * num_seq_q, num_head_kv, num_dim_qk), dtype=torch.bfloat16, device="cuda"
        )
        / math.sqrt(num_dim_qk)
    ).to(T)
    V = torch.randn(
        (num_batch * num_seq_q, num_head_kv, num_dim_v), dtype=torch.bfloat16, device="cuda"
    ).to(T)

    KS = torch.randn((1), dtype=torch.float32, device="cuda")
    VS = torch.randn((1), dtype=torch.float32, device="cuda")

    num_seq_kvcache = torch.randint(1, max_seq_kv, (num_batch,), dtype=torch.int32, device="cuda")

    nblocks = (num_seq_kvcache + num_seq_q + block_size - 1) // block_size
    total_blocks = sum(nblocks)
    kvcache = (
        torch.randn(
            max_num_blocks,
            2,
            block_size,
            num_head_kv,
            num_dim_qk,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / math.sqrt(num_dim_qk)
    ).to(T)
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

    p_scale, p_scale_inv = _make_pscale(p_scale_mode, num_head_q, device="cuda")

    gt = naive_attn_with_paged_kvcache_pscale_func(
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
        p_scale=p_scale,
        p_scale_inv=p_scale_inv,
    )
    torch.cuda.synchronize()

    task_map = None
    if use_dynamic_sched:
        task_map = hpc.get_attention_decode_task_workspace(
            num_batch, max_seq_kv + num_seq_q, num_head_kv, min_process_len=1024
        )
        hpc.assign_attention_decode_task(
            num_seq_kvcache + num_seq_q,
            task_map,
            num_head_kv,
            num_seq_q,
            True,
            min_process_len=1024,
        )
        torch.cuda.synchronize()

    my = hpc.attention_decode_fp8(
        Q,
        kvcache[:, 0, :, :, :],
        kvcache[:, 1, :, :, :],
        block_ids,
        num_seq_kvcache + num_seq_q,
        QS,
        KS,
        VS,
        mtp=num_seq_q - 1,
        new_kv_included=True,
        quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
        splitk=True,
        task_map=task_map,
        p_scale=p_scale,
        p_scale_inv=p_scale_inv,
    )
    torch.cuda.synchronize()

    # atol = 0.05: golden mirrors the kernel's exp(P-row_max) -> p_scale ->
    # fp8 quant -> matmul -> /gSum -> vscale*p_scale_inv round-trip. Empirical
    # max abs diff over a 30-seed sweep stays under ~0.013; 0.05 leaves
    # ~4x headroom for unseen seeds / future matrix extensions.
    assert allclose(my, gt, atol=0.05), (
        f"[p_scale={p_scale_mode}] decode_fp8_pertensor diverges from golden " f"beyond atol=0.05"
    )
    assert gt.shape == my.shape and gt.dtype == my.dtype and gt.device == my.device


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 9, reason="skip on non sm90!")
@pytest.mark.parametrize("num_batch", [1, 16])
@pytest.mark.parametrize("num_seq_q", [1, 4])
@pytest.mark.parametrize("max_seq_kv", [1024])
@pytest.mark.parametrize("kv_head_q_head", [(2, 8), (4, 32)])
@pytest.mark.parametrize("p_scale_mode", ["none", "all_ones", "all_2", "per_head_random"])
@pytest.mark.parametrize("use_dynamic_sched", [False, True])
def test_attn_fp8_pscale_sm90(
    num_batch,
    num_seq_q,
    max_seq_kv,
    kv_head_q_head,
    p_scale_mode,
    use_dynamic_sched,
):
    attention_decode_fp8_pscale_test_func(
        num_batch=num_batch,
        num_seq_q=num_seq_q,
        max_seq_kv=max_seq_kv,
        block_size=64,
        kv_head_q_head=kv_head_q_head,
        head_dim=128,
        p_scale_mode=p_scale_mode,
        use_dynamic_sched=use_dynamic_sched,
    )
