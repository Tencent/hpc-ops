"""dim576 MLA decode tests.

Drives ``hpc.mla_decode_with_kvcache_bf16`` directly. The op runs the
three-kernel pipeline (get_scheduler_map → persistent_attn → combine).
Reference is the naive PyTorch MLA.
"""

import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc  # noqa: E402

from utils import allclose  # noqa: E402


# Reference implementation


def _softmax_with_sink(P, sink_logit, *, dim=-1):
    combined = torch.cat([P, sink_logit.expand(*P.shape[:-1], 1)], dim=dim)
    w = F.softmax(combined, dim=dim)
    return w.narrow(dim, 0, P.size(dim))


def _naive_mla_attn(
    Q, kvcache, block_ids, nblocks, cu_seqlenq, num_seq_kv, qk_dim, v_dim, sink_weight=None
):
    num_batch = cu_seqlenq.shape[0] - 1
    num_head_q = Q.shape[1]
    seqlenq = cu_seqlenq[1:] - cu_seqlenq[:-1]
    output = torch.empty((Q.shape[0], num_head_q, v_dim), dtype=Q.dtype, device=Q.device)
    for bi in range(num_batch):
        BQ = Q[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlenq[bi]].permute(1, 0, 2).float()
        blk_ids = block_ids[bi, : nblocks[bi]]
        seqlen = num_seq_kv[bi]
        gathered = (
            kvcache[blk_ids, :, :]
            .reshape(-1, qk_dim)
            .transpose(0, 1)[:, :seqlen]
            .unsqueeze(0)
            .repeat_interleave(num_head_q, dim=0)
            .float()
        )
        P = BQ @ gathered
        P = P / math.sqrt(qk_dim)
        causal_mask = torch.ones(
            seqlenq[bi], seqlen - seqlenq[bi], device=Q.device, dtype=torch.bool
        )
        tail = torch.tril(torch.ones(seqlenq[bi], seqlenq[bi], device=Q.device, dtype=torch.bool))
        causal_mask = torch.cat([causal_mask, tail], dim=-1).unsqueeze(0)
        P = P.masked_fill(~causal_mask, float("-inf"))
        if sink_weight is None:
            attn_weights = F.softmax(P, dim=-1)
        else:
            sink_logit = sink_weight.float().view(num_head_q, 1, 1)
            attn_weights = _softmax_with_sink(P, sink_logit, dim=-1)
        BV = gathered[:, :v_dim, :]
        Y = torch.matmul(attn_weights, BV.transpose(-1, -2))
        output[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlenq[bi]] = Y.transpose(0, 1)
    return output


# Input builder


def _build_inputs(num_batch, per_batch_seq_kv, num_head_q, qk_dim, block_size=64, seed=41):
    """Decode-only inputs: 1 q-token per batch."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    per_batch_seq_q = [1] * num_batch
    num_seq_q = torch.tensor(per_batch_seq_q, dtype=torch.int32, device="cuda")
    num_seq_kv = torch.tensor(per_batch_seq_kv, dtype=torch.int32, device="cuda")
    total_seq_q = int(num_seq_q.sum())

    nblocks_per_batch = [(s + block_size - 1) // block_size for s in per_batch_seq_kv]
    total_blocks = sum(nblocks_per_batch)
    max_num_blocks = max(1, int(total_blocks * 1.2))
    kvcache = torch.randn(
        max_num_blocks, block_size, qk_dim, dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(qk_dim)

    packed_block_ids = torch.randperm(max_num_blocks)[:total_blocks].to(torch.int32).cuda()
    max_nblocks = max(nblocks_per_batch) if nblocks_per_batch else 1
    block_ids = torch.empty(num_batch, max_nblocks, dtype=torch.int32, device="cuda")
    cu = 0
    for i, nb in enumerate(nblocks_per_batch):
        block_ids[i, :nb] = packed_block_ids[cu : cu + nb]
        cu += nb

    cu_seqlenq = torch.cumsum(num_seq_q, dtype=torch.int32, dim=0)
    cu_seqlenq = torch.concat(
        [torch.tensor([0], dtype=torch.int32, device="cuda"), cu_seqlenq], dim=0
    )
    Q = torch.randn(
        total_seq_q, num_head_q, qk_dim, dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(qk_dim)
    nblocks = torch.tensor(nblocks_per_batch, dtype=torch.int32, device="cuda")
    return dict(
        Q=Q,
        kvcache=kvcache,
        block_ids=block_ids,
        cu_seqlenq=cu_seqlenq,
        num_seq_kv=num_seq_kv,
        nblocks=nblocks,
    )


def _run_mla_decode(inputs, num_head_q, sink_weight=None, atol=0.02):
    qk_dim = 576
    v_dim = 512

    gt = _naive_mla_attn(
        inputs["Q"],
        inputs["kvcache"],
        inputs["block_ids"],
        inputs["nblocks"],
        inputs["cu_seqlenq"],
        inputs["num_seq_kv"],
        qk_dim,
        v_dim,
        sink_weight=sink_weight,
    )

    my = hpc.mla_decode_with_kvcache_bf16(
        inputs["Q"],
        inputs["kvcache"],
        inputs["block_ids"],
        inputs["cu_seqlenq"],
        inputs["num_seq_kv"],
        sink_weight=sink_weight,
    )

    assert allclose(gt, my, atol=atol), f"mla_decode mismatch num_head_q={num_head_q}"


# Basic correctness


@pytest.mark.parametrize("num_head_q", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("seq_kv", [256, 1024, 4096, 16384])
def test_mla_decode_b1(num_head_q, seq_kv):
    inputs = _build_inputs(1, [seq_kv], num_head_q, 576)
    _run_mla_decode(inputs, num_head_q)


@pytest.mark.parametrize("num_head_q", [1, 2, 4, 8, 16, 32])
def test_mla_decode_b4_skewed(num_head_q):
    inputs = _build_inputs(4, [256, 256, 256, 8192], num_head_q, 576)
    _run_mla_decode(inputs, num_head_q, atol=0.025)


@pytest.mark.parametrize("num_head_q", [1, 2, 4, 8, 16, 32])
def test_mla_decode_b512_uniform(num_head_q):
    inputs = _build_inputs(512, [256] * 512, num_head_q, 576)
    _run_mla_decode(inputs, num_head_q, atol=0.025)


# Sink weight


@pytest.mark.parametrize("num_head_q", [1, 2, 4, 8, 16, 32])
def test_mla_decode_with_sink(num_head_q):
    inputs = _build_inputs(4, [256, 4096, 256, 8192], num_head_q, 576)
    sink = torch.randn(num_head_q, dtype=torch.float32, device="cuda")
    _run_mla_decode(inputs, num_head_q, sink_weight=sink, atol=0.025)


# Precomputed scheduler map


@pytest.mark.parametrize("num_head_q", [1, 2, 4, 8, 16, 32])
def test_mla_decode_precomputed_scheduler_map_reused_across_layers(num_head_q):
    """Multi-layer: one get_mla_scheduler_map(), N attn calls with different Q/kvcache."""
    seq_kvs = [1024, 2048, 4096, 8192]
    inputs0 = _build_inputs(4, seq_kvs, num_head_q, 576, seed=41)
    task = hpc.get_mla_scheduler_map(inputs0["num_seq_kv"])
    task_initial = task.clone()

    for layer_idx, seed in enumerate([41, 137, 271]):
        inputs = _build_inputs(4, seq_kvs, num_head_q, 576, seed=seed)
        y_inline = hpc.mla_decode_with_kvcache_bf16(
            inputs["Q"],
            inputs["kvcache"],
            inputs["block_ids"],
            inputs["cu_seqlenq"],
            inputs["num_seq_kv"],
        )
        y_amortised = hpc.mla_decode_with_kvcache_bf16(
            inputs["Q"],
            inputs["kvcache"],
            inputs["block_ids"],
            inputs["cu_seqlenq"],
            inputs["num_seq_kv"],
            task_tensor=task,
        )
        assert torch.equal(y_inline, y_amortised), f"layer {layer_idx} mismatched (seed={seed})"
        assert torch.equal(task, task_initial), f"scheduler map mutated by layer {layer_idx}"


@pytest.mark.parametrize("num_head_q", [1, 2, 4, 8, 16, 32])
def test_mla_decode_precomputed_scheduler_map_caller_buffer(num_head_q):
    """Caller-provided pre-allocated buffer is reused (no fresh allocation)."""
    num_batch = 8
    inputs = _build_inputs(num_batch, [1024] * num_batch, num_head_q, 576)
    num_sm = 148  # upper-bound SM count
    expected_elems = (num_batch + num_sm) * 4 + (num_sm + 1) + (num_batch + 1)
    buf = torch.empty(expected_elems, dtype=torch.int32, device="cuda")
    out = hpc.get_mla_scheduler_map(inputs["num_seq_kv"], task_tensor=buf)
    assert out.data_ptr() == buf.data_ptr()
    y = hpc.mla_decode_with_kvcache_bf16(
        inputs["Q"],
        inputs["kvcache"],
        inputs["block_ids"],
        inputs["cu_seqlenq"],
        inputs["num_seq_kv"],
        task_tensor=buf,
    )
    gt = _naive_mla_attn(
        inputs["Q"],
        inputs["kvcache"],
        inputs["block_ids"],
        inputs["nblocks"],
        inputs["cu_seqlenq"],
        inputs["num_seq_kv"],
        576,
        512,
    )
    assert allclose(gt, y, atol=0.02)
