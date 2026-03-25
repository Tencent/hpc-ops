import os
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import math

import torch

import hpc
from utils import allclose


def set_penalties_mask_ref(mask, tokens):
    byte_indices = tokens // 8
    bit_offsets = tokens % 8
    bit_masks = torch.bitwise_left_shift(
        torch.ones_like(tokens, dtype=torch.uint8), bit_offsets.to(torch.uint8)
    )
    mask.scatter_add_(0, byte_indices.to(torch.int64), bit_masks)


@pytest.mark.parametrize("batch_size", [1, 18, 35])
@pytest.mark.parametrize(
    "vocab_size",
    [
        129024,  # turbos text to text
        128512,  # turbos image to text
        129280,  # deepseek r1 text to text
        127962,
        127961,
    ],
)
@pytest.mark.parametrize("repetition_penalties", [1.05, 0])
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
        if repetition_penalties > 0:
            for si in torch.unique(tokens[bi]):
                if logits[bi][si] > 0:
                    logits[bi][si] /= repetition_penalties
                else:
                    logits[bi][si] *= repetition_penalties
    if temperature > 0:
        logits /= temperature

    gt_y = torch.softmax(logits, dim=-1)

    if repetition_penalties == 0:
        penalties_masks_ptrs = None
    y = hpc.sampler.fused_repetition_penalties_softmax(
        logits_hpc_input, penalties_masks_ptrs, repetition_penalties, temperature
    )

    assert allclose(gt_y, y)


def apply_top_k_only(
    logits: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits


def ref_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits

        # Avoid sorting vocab for top-k only case.
        return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("vocab_size", [120818, 129024, 128512])
@pytest.mark.parametrize("topk", [20])
@pytest.mark.parametrize("topk_dtype", [torch.int32, torch.int64])
def test_topk_mask_logits(batch_size, vocab_size, topk, topk_dtype):

    logits = torch.randn(batch_size, vocab_size).cuda()

    topk = torch.tensor([topk] * batch_size).to(topk_dtype).cuda()

    my_output_logits = hpc.topk_mask_logits(logits, topk)
    my_probs = my_output_logits.softmax(dim=-1, dtype=torch.float32)

    gt_output_logits = ref_top_k_top_p(logits, topk, None)
    gt_probs = gt_output_logits.softmax(dim=-1, dtype=torch.float32)

    assert allclose(gt_output_logits, my_output_logits)
    assert allclose(gt_probs, my_probs)


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("vocab_size", [120818, 129024, 128512])
@pytest.mark.parametrize("topk", [20])
@pytest.mark.parametrize("topp", [0.9])
@pytest.mark.parametrize("topk_dtype", [torch.int32])
def test_topk_topp_mask_logits(batch_size, vocab_size, topk, topp, topk_dtype):

    logits = torch.randn(batch_size, vocab_size).cuda()

    topk = torch.tensor([topk] * batch_size).to(topk_dtype).cuda()
    topp = torch.tensor([topp] * batch_size).to(torch.float32).cuda()

    my_output_logits = hpc.topk_topp_mask_logits(logits, topk, topp)
    my_probs = my_output_logits.softmax(dim=-1, dtype=torch.float32)

    gt_output_logits = ref_top_k_top_p(logits, topk, topp)
    gt_probs = gt_output_logits.softmax(dim=-1, dtype=torch.float32)

    assert allclose(gt_output_logits, my_output_logits)
    assert allclose(gt_probs, my_probs)
