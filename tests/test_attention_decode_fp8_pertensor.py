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

        attn_weights = F.softmax(P, dim=-1)

        Y = torch.matmul(attn_weights, BV) * VS

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

    print(num_seq_kvcache)

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
        for i in range(1):
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
