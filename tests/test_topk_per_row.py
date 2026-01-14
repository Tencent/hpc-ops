import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
from utils import allclose


@pytest.mark.skipif(bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer")
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_sp_tokens", [2])
@pytest.mark.parametrize("max_seqlen", [1024, 100 * 1024])
@pytest.mark.parametrize("top_k", [2 * 1024])
def test_topk_per_row(batch_size, num_sp_tokens, max_seqlen, top_k):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    ratio = 0.8
    logits = (
        torch.randn(batch_size * num_sp_tokens, max_seqlen, dtype=torch.float, device="cuda") * 100
    )

    seqlens = torch.randint(
        int(max_seqlen * ratio), max_seqlen, (batch_size,), dtype=torch.int32
    ).cuda()
    my = torch.empty((batch_size * num_sp_tokens, top_k), dtype=torch.int32, device="cuda")

    # logits[0, :2048] = 1
    # logits[0, 2048:] = 0
    logits_gt = logits.clone()
    for bi in range(batch_size):
        for ni in range(num_sp_tokens):
            logits_gt[bi * num_sp_tokens + ni, seqlens[bi] - num_sp_tokens + ni + 1 :] = -torch.inf

    if max_seqlen < top_k:
        gt = torch.ones_like(my) * -1
        for bi in range(batch_size):
            for ni in range(num_sp_tokens):
                seqlen = seqlens[bi] - num_sp_tokens + ni + 1
                gt[bi * num_sp_tokens + ni, :seqlen] = torch.arange(seqlen)
    else:
        _, gt = torch.topk(logits_gt, top_k, dim=-1)

    hpc.topk_per_row(logits, seqlens, num_sp_tokens, top_k, my)

    my = my.sort(descending=True)[0]
    gt = gt.sort(descending=True)[0]

    gt = gt.int()

    print(gt)
    print(my)

    assert allclose(gt, my)


@pytest.mark.skipif(bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer")
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("max_seqlen_q", [4 * 1024])
@pytest.mark.parametrize("max_seqlen_kv", [4 * 1024])
@pytest.mark.parametrize("ratio", [4])
@pytest.mark.parametrize("top_k", [512])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("max_context_len", [32 * 1024])
def test_topk_per_row_varlen(
    batch_size, max_seqlen_q, max_seqlen_kv, ratio, top_k, deterministic, max_context_len
):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)

    seqlens_q = torch.randint(1, max_seqlen_q + 1, (batch_size,), dtype=torch.int32, device="cuda")
    seqlen_kvcache = torch.randint(
        1, max_seqlen_kv + 1, (batch_size,), dtype=torch.int32, device="cuda"
    )
    seqlen_kv = seqlens_q + seqlen_kvcache
    cu_seqlenq = torch.cumsum(seqlens_q, dtype=torch.int32, dim=0)
    cu_seqlenq = torch.concat(
        [torch.tensor([0], dtype=torch.int32, device="cuda"), cu_seqlenq], dim=0
    )
    total_seq_q = cu_seqlenq[-1]

    logits = torch.randn(total_seq_q, max_context_len, dtype=torch.float, device="cuda")

    my = torch.empty((total_seq_q, top_k), dtype=torch.int32, device="cuda")

    logits_gt = logits.clone()
    for bi in range(batch_size):
        mask = (
            torch.arange(seqlen_kv[bi] // ratio).repeat(seqlen_kv[bi], 1)
            >= torch.arange(1, seqlen_kv[bi] + 1).unsqueeze(1) // ratio
        )
        mask = torch.where(mask, float("-inf"), 0)
        mask = mask[-seqlens_q[bi] :].cuda()
        logits_gt[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlens_q[bi], seqlen_kv[bi] // ratio :] = (
            -torch.inf
        )
        logits_gt[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlens_q[bi], : seqlen_kv[bi] // ratio] += mask

    if max_seqlen_kv // ratio < top_k:
        gt = torch.ones_like(my) * -1
        for bi in range(batch_size):
            mask = (
                torch.arange(seqlen_kv[bi] // ratio).repeat(seqlen_kv[bi], 1)
                >= torch.arange(1, seqlen_kv[bi] + 1).unsqueeze(1) // ratio
            )
            mulmask = torch.where(mask, float("0"), 1)
            mulmask = mulmask[-seqlens_q[bi] :].cuda()
            addmask = torch.where(mask, float("-1"), 0)
            addmask = addmask[-seqlens_q[bi] :].cuda()
            gt[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlens_q[bi], : seqlen_kv[bi] // ratio] = (
                torch.arange(seqlen_kv[bi] // ratio)
            )
            gt[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlens_q[bi], : seqlen_kv[bi] // ratio] = (
                gt[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlens_q[bi], : seqlen_kv[bi] // ratio]
                * mulmask
                + addmask
            )
    else:
        score, gt = torch.topk(logits_gt, top_k, dim=-1)
        for bi in range(batch_size):
            gt[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlens_q[bi]] = torch.where(
                score[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlens_q[bi]] == -torch.inf,
                float("-1"),
                gt[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlens_q[bi]],
            )

    hpc.topk_per_row_varlen(logits, cu_seqlenq, seqlen_kv, top_k, ratio, deterministic, my)

    my = my.sort(descending=True)[0]
    gt = gt.sort(descending=True)[0]

    gt = gt.int()

    print(gt)
    print(my)

    assert allclose(gt, my)
