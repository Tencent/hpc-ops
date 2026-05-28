import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
from utils import allclose


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

        p = q_batch @ k_batch.transpose(-1, -2)

        p = p / math.sqrt(head_dim) * q_scale[bi][:, None, None] * k_scale

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
        attn_weights = attn_weights * 256.0
        attn_weights = attn_weights.to(torch.float8_e4m3fn).float()

        y = torch.matmul(attn_weights, v_batch)
        y = y / gSum
        y = y * (v_scale / 256.0)

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

    q = torch.randn(
        (num_batch * num_seq_q, num_head_q, num_dim_qk), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(num_dim_qk)
    q_scale = q.float().abs().max(-1)[0] / 10
    q = (q / q_scale[:, :, None]).to(torch.float8_e4m3fn)

    k = (
        torch.randn(
            (num_batch * num_seq_q, num_head_kv, num_dim_qk), dtype=torch.bfloat16, device="cuda"
        )
        / math.sqrt(num_dim_qk)
    ).to(torch.float8_e4m3fn)
    v = torch.randn(
        (num_batch * num_seq_q, num_head_kv, num_dim_v), dtype=torch.bfloat16, device="cuda"
    ).to(torch.float8_e4m3fn)

    k_scale = torch.randn((1), dtype=torch.float32, device="cuda")
    v_scale = torch.randn((1), dtype=torch.float32, device="cuda")

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

        num_total_ctas = task_map_for_cuda[1]

        sched_need_byte_size = (task_map_for_cpu.view(torch.int32)[0] * num_total_ctas + 1) * 48 + (
            num_batch * num_head_kv * 4 + 47
        ) // 48 * 48
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
    ).to(torch.float8_e4m3fn)

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

    gt = ref_attn_with_paged_kvcache_func(
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
    )

    if use_output:
        my = torch.empty_like(q, dtype=torch.bfloat16)
        hpc.attention_decode_fp8(
            q,
            kvcache[:, 0, :, :, :],
            kvcache[:, 1, :, :, :],
            block_ids,
            num_seq_kvcache + num_seq_q if new_kv_included else num_seq_kvcache,
            q_scale,
            k_scale,
            v_scale,
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
                q,
                kvcache[:, 0, :, :, :],
                kvcache[:, 1, :, :, :],
                block_ids,
                num_seq_kvcache + num_seq_q if new_kv_included else num_seq_kvcache,
                q_scale,
                k_scale,
                v_scale,
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

    assert allclose(my, gt, atol=0.2)


@pytest.mark.parametrize("num_batch", [1, 16, 200])
@pytest.mark.parametrize("num_seq_q", [1, 2, 3, 4])
@pytest.mark.parametrize("max_seq_kv", [1024, 4096])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("kv_head_q_head", [(1, 8), (4, 32)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("new_kv_included", [True])
@pytest.mark.parametrize("use_output", [False])
@pytest.mark.parametrize("splitk", [True])
@pytest.mark.parametrize("use_dynamic_sched", [True, False])
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
