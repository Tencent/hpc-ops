import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math


def set_penalties_mask_ref(mask, tokens):
    byte_indices = tokens // 8
    bit_offsets = tokens % 8
    bit_masks = torch.bitwise_left_shift(
        torch.ones_like(tokens, dtype=torch.uint8), bit_offsets.to(torch.uint8)
    )
    mask.scatter_add_(0, byte_indices.to(torch.int64), bit_masks)


@pytest.mark.parametrize("batch_size", [1, 16, 32, 64, 200])
@pytest.mark.parametrize("vocab_size", [129024])
@pytest.mark.parametrize("repetition_penalties", [1.05])
@pytest.mark.parametrize("temperature", [0.7])
def test_fused_repetition_penalties_softmax(
    batch_size, vocab_size, repetition_penalties, temperature
):
    max_seq_len = 1024
    logits = torch.rand(batch_size, vocab_size, dtype=torch.float).cuda()
    logits_hpc_input = logits.clone()
    padded_vocab_size = (vocab_size + 8 - 1) // 8 * 8

    seqlens = torch.randint(0, max_seq_len, (batch_size,), dtype=torch.int32).cuda()
    tokens = []
    penalties_masks = []

    penalties_masks_ptrs = torch.empty(batch_size, dtype=torch.uint64).cuda()

    for i in range(batch_size):
        tokens.append(torch.randint(0, vocab_size, (seqlens[i],), dtype=torch.int32).cuda())
        penalties_masks.append(torch.zeros((padded_vocab_size // 8), dtype=torch.uint8).cuda())
        penalties_masks_ptrs[i] = penalties_masks[i].data_ptr()

    for i in range(batch_size):
        set_penalties_mask_ref(penalties_masks[i], torch.unique(tokens[i]))

    for bi in range(batch_size):
        for si in torch.unique(tokens[bi]):
            if logits[bi][si] > 0:
                logits[bi][si] /= repetition_penalties
            else:
                logits[bi][si] *= repetition_penalties
    if temperature > 0:
        logits /= temperature

    gt_y = torch.softmax(logits, dim=-1)

    y = hpc.sampler.fused_repetition_penalties_softmax(
        logits_hpc_input, penalties_masks_ptrs, repetition_penalties, temperature
    )

    assert torch.allclose(gt_y, y)
