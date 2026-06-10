"""Sparse dim576 MLA (V3.2-DSA-style topk) — decode-mode tests.

Drives ``hpc.sparse_mla_dsa_with_kvcache_bf16`` end-to-end with
``total_seq_q == num_phys_batch`` (decode mode). Reference is a naive
PyTorch gather + masked softmax + V latent product over the topk-selected
KV rows.
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

QK_DIM = 576
V_DIM = 512
BLOCK_SIZE = 64


def _softmax_with_sink(P, sink_logit, *, dim=-1):
    combined = torch.cat([P, sink_logit.expand(*P.shape[:-1], 1)], dim=dim)
    w = F.softmax(combined, dim=dim)
    return w.narrow(dim, 0, P.size(dim))


def _naive_sparse_mla(Q, kvcache, block_ids, topk_ids, sink_weight=None):
    """Reference: per-batch gather K/V from topk_ids → masked softmax → V latent."""
    num_batch = Q.shape[0]
    num_head_q = Q.shape[1]
    num_blocks = kvcache.shape[0]
    out = torch.zeros(num_batch, num_head_q, V_DIM, dtype=Q.dtype, device=Q.device)

    max_kv = num_blocks * BLOCK_SIZE
    for b in range(num_batch):
        topk = topk_ids[b]
        valid = (topk >= 0) & (topk < max_kv)
        idx = topk.clamp(min=0)
        page = block_ids[b, idx // BLOCK_SIZE]
        rows = page * BLOCK_SIZE + (idx % BLOCK_SIZE)
        K = kvcache.reshape(-1, QK_DIM)[rows].float()  # [num_topk, 576]
        V = K[:, :V_DIM]  # latent V == first 512 cols

        BQ = Q[b].float()  # [H, 576]
        P = BQ @ K.t() / math.sqrt(QK_DIM)
        P = P.masked_fill(~valid.unsqueeze(0), float("-inf"))

        if sink_weight is None:
            w = F.softmax(P, dim=-1)
        else:
            w = _softmax_with_sink(P, sink_weight.float().view(num_head_q, 1), dim=-1)
        out[b] = (w @ V).to(Q.dtype)
    return out


def _build_inputs(num_batch, num_head_q, num_max_topk, num_blocks_total, invalid_frac=0.0, seed=41):
    """One q-token per batch; topk indexes random tokens within `num_blocks_total*64`."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    max_blocks = max(num_blocks_total * 2, 1)
    kvcache = torch.randn(
        max_blocks, BLOCK_SIZE, QK_DIM, dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(QK_DIM)

    # Per-batch random page-table.
    block_ids = torch.empty(num_batch, num_blocks_total, dtype=torch.int32, device="cuda")
    for b in range(num_batch):
        block_ids[b] = torch.randperm(max_blocks)[:num_blocks_total].to(torch.int32)

    Q = torch.randn(num_batch, num_head_q, QK_DIM, dtype=torch.bfloat16, device="cuda") / math.sqrt(
        QK_DIM
    )

    max_kv = num_blocks_total * BLOCK_SIZE
    topk_ids = torch.randint(0, max_kv, (num_batch, num_max_topk), dtype=torch.int32, device="cuda")
    if invalid_frac > 0:
        mask = torch.rand(topk_ids.shape, device="cuda") < invalid_frac
        topk_ids = torch.where(mask, torch.full_like(topk_ids, -1), topk_ids)

    cu_seqlens_q = torch.arange(num_batch + 1, dtype=torch.int32, device="cuda")

    return dict(
        Q=Q, kvcache=kvcache, block_ids=block_ids, topk_ids=topk_ids, cu_seqlens_q=cu_seqlens_q
    )


def _run(inputs, num_head_q, sink_weight=None, splitk=True, atol=0.025):
    gt = _naive_sparse_mla(
        inputs["Q"],
        inputs["kvcache"],
        inputs["block_ids"],
        inputs["topk_ids"],
        sink_weight=sink_weight,
    )
    my = hpc.sparse_mla_dsa_with_kvcache_bf16(
        inputs["Q"],
        inputs["kvcache"],
        inputs["block_ids"],
        inputs["topk_ids"],
        inputs["cu_seqlens_q"],
        sink_weight=sink_weight,
        splitk=splitk,
    )
    assert allclose(
        gt, my, atol=atol
    ), f"sparse_mla_dsa mismatch num_head_q={num_head_q} splitk={splitk}"


# Basic correctness
@pytest.mark.parametrize("num_head_q", [6, 64])
def test_with_invalid_topk(num_head_q):
    inputs = _build_inputs(
        num_batch=512,
        num_head_q=num_head_q,
        num_max_topk=2048,
        num_blocks_total=128,
        invalid_frac=0.1,
    )
    _run(inputs, num_head_q)


# Cross-layer scheduler-map amortise
@pytest.mark.parametrize("num_head_q", [8, 64])
@pytest.mark.parametrize("num_max_topk", [2048])
def test_precomputed_scheduler_map_reused_across_layers(num_head_q, num_max_topk):
    num_batch = 512

    # Sparse scheduler-map mode: pass index_topk; num_seq_kv values are
    # ignored, only its size (= num_batch) and device matter.
    nb_buf = torch.empty(num_batch, dtype=torch.int32, device="cuda")
    # Decode: identity cu_seqlens_q = [0,1,...,num_batch] (one q-token per batch).
    cu_seqlens_q = torch.arange(num_batch + 1, dtype=torch.int32, device="cuda")

    inputs = _build_inputs(
        num_batch=num_batch,
        num_head_q=num_head_q,
        num_max_topk=num_max_topk,
        num_blocks_total=64,
        seed=41,
    )
    task = hpc.get_mla_scheduler_map(
        nb_buf,
        cu_seqlens_q=cu_seqlens_q,
        num_actual_tokens=inputs["Q"].size(0),
        index_topk=num_max_topk,
    )
    task_initial = task.clone()
    y_inline = hpc.sparse_mla_dsa_with_kvcache_bf16(
        inputs["Q"],
        inputs["kvcache"],
        inputs["block_ids"],
        inputs["topk_ids"],
        inputs["cu_seqlens_q"],
    )
    y_amortised = hpc.sparse_mla_dsa_with_kvcache_bf16(
        inputs["Q"],
        inputs["kvcache"],
        inputs["block_ids"],
        inputs["topk_ids"],
        inputs["cu_seqlens_q"],
        task_tensor=task,
    )
    assert torch.equal(y_inline, y_amortised), f"amortised path diverged at seed={41}"
    # task tensor should not be mutated across calls.
    assert torch.equal(task, task_initial), "task tensor mutated!"


# Batch-invariant (splitk=False)
def _extract_single_batch_inputs_sparse(inputs, b):
    """Compact a multi-batch sparse-decode inputs dict to a singleton for
    batch ``b``. Pages used by batch b are gathered into a fresh kvcache and
    block_ids is remapped to the contiguous range [0, nblocks); topk_ids
    are batch-local already and stay unchanged."""
    Q_b = inputs["Q"][b : b + 1].contiguous()
    block_ids_full = inputs["block_ids"][b]  # [nblocks]
    nblocks_b = block_ids_full.size(0)
    kvcache_b = inputs["kvcache"][block_ids_full.long()].contiguous()
    block_ids_b = (
        torch.arange(nblocks_b, dtype=torch.int32, device="cuda").view(1, nblocks_b).contiguous()
    )
    topk_ids_b = inputs["topk_ids"][b : b + 1].contiguous()
    cu_seqlens_q_b = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    return dict(
        Q=Q_b,
        kvcache=kvcache_b,
        block_ids=block_ids_b,
        topk_ids=topk_ids_b,
        cu_seqlens_q=cu_seqlens_q_b,
    )


@pytest.mark.parametrize("num_head_q", [64])
@pytest.mark.parametrize("num_max_topk", [2048])
def test_no_splitk_bit_invariant(num_head_q, num_max_topk):
    """splitk=False makes per-batch output bit-invariant: a multi-batch call
    must produce, for each batch b, the exact same bytes as running that
    batch alone (also with splitk=False)."""
    num_batch = 3
    inputs = _build_inputs(
        num_batch=num_batch,
        num_head_q=num_head_q,
        num_max_topk=num_max_topk,
        num_blocks_total=128,
        seed=913,
    )

    y_full = hpc.sparse_mla_dsa_with_kvcache_bf16(
        inputs["Q"],
        inputs["kvcache"],
        inputs["block_ids"],
        inputs["topk_ids"],
        inputs["cu_seqlens_q"],
        splitk=False,
    )

    b = 1
    sub = _extract_single_batch_inputs_sparse(inputs, b)
    y_one = hpc.sparse_mla_dsa_with_kvcache_bf16(
        sub["Q"],
        sub["kvcache"],
        sub["block_ids"],
        sub["topk_ids"],
        sub["cu_seqlens_q"],
        splitk=False,
    )
    assert torch.equal(
        y_full[b : b + 1], y_one
    ), f"batch {b}/{num_batch} (H={num_head_q}): splitk=False not bit-invariant"


# Unified decode/prefill regression tests


@pytest.mark.parametrize("num_head_q", [8, 64])
@pytest.mark.parametrize("splitk", [True, False])
def test_precomputed_scheduler_map_matches_reference(num_head_q, splitk):
    """Sparse decode via a precomputed scheduler map must match the naive
    reference (not just be self-consistent with the inline path)."""
    num_batch = 512
    num_max_topk = 2048
    nb_buf = torch.empty(num_batch, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(num_batch + 1, dtype=torch.int32, device="cuda")

    inputs = _build_inputs(
        num_batch=num_batch,
        num_head_q=num_head_q,
        num_max_topk=num_max_topk,
        num_blocks_total=64,
        seed=53,
    )
    task = hpc.get_mla_scheduler_map(
        nb_buf, cu_seqlens_q, inputs["Q"].size(0), index_topk=num_max_topk, splitk=splitk
    )
    gt = _naive_sparse_mla(inputs["Q"], inputs["kvcache"], inputs["block_ids"], inputs["topk_ids"])
    y_inline = hpc.sparse_mla_dsa_with_kvcache_bf16(
        inputs["Q"],
        inputs["kvcache"],
        inputs["block_ids"],
        inputs["topk_ids"],
        inputs["cu_seqlens_q"],
        splitk=splitk,
    )
    y_map = hpc.sparse_mla_dsa_with_kvcache_bf16(
        inputs["Q"],
        inputs["kvcache"],
        inputs["block_ids"],
        inputs["topk_ids"],
        inputs["cu_seqlens_q"],
        task_tensor=task,
        splitk=splitk,
    )
    assert allclose(
        gt, y_map, atol=0.025
    ), f"sparse precomputed-map != reference (H={num_head_q}, splitk={splitk})"
    assert torch.equal(
        y_inline, y_map
    ), f"sparse precomputed-map != inline (H={num_head_q}, splitk={splitk})"
