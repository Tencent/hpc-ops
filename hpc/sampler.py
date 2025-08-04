import torch
from torch import Tensor
from typing import Union, Tuple


def fused_repetition_penalties_softmax(
    logits: Tensor,
    penalties_masks_ptrs: Tensor,
    repetition_penalties: float = 1.0,
    temperature: float = 1.0,
) -> Tensor:
    """Fused repetition_penalties and apply temperature with softmax.
     First, logits[mask] = logits[mask] / repetition_penalties if logits[mask] > 0 else logits[mask] * repetition_penalties.
     Second, do softmax(logits / temperature)

    Args:
        logits: dtype is float, shape is [batch_size, vocab_size]
        penalties_masks_ptrs: Input tensor: dtype uint64, shape [batch_size].
          Each value is the gpu ptr pointer to the mask of each request. And Each mask dtype is uint8, shape is [padded_vocab_size // 8]. each bit indicate one token.
        repetition_penalties:  repetition_penalties factor, dtype is float.
        temperature: temperature factor, dtype is float.
    Return:
        modifed_logits:  dtype is float, shape is [batch_size, vocab_size]
    """
    return torch.ops.hpc.fused_repetition_penalties_softmax(
        logits, penalties_masks_ptrs, repetition_penalties, temperature
    )
