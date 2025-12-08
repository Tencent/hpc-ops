import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
from utils import allclose


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
