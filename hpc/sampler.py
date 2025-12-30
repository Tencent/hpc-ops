from typing import Optional, Tuple, Union

import torch
from torch import Tensor


def _to_tensor_scalar_tuple(x) -> Tuple[Optional[Tensor], Union[int, float]]:
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.float:
            return (x, 0.0)
        elif x.dtype == torch.int32 or x.dtype == torch.int64:
            return (x, 0)
    else:
        return (None, x)


def fused_repetition_penalties_softmax(
    logits: Tensor,
    penalties_masks_ptrs: Optional[Tensor] = None,
    repetition_penalties: Union[Tensor, float] = 0.0,
    temperature: Union[Tensor, float] = 0.0,
) -> Tensor:
    """Fused repetition_penalties and apply temperature with softmax.
     First, logits[mask] = logits[mask] / repetition_penalties if logits[mask] > 0 else logits[mask] * repetition_penalties.
     Second, do softmax(logits / temperature)

    Args:
        logits: dtype is float, shape is [batch_size, vocab_size]
        penalties_masks_ptrs: Input tensor: dtype uint64, shape [batch_size].
          Each value is the gpu ptr pointer to the mask of each request. And Each mask dtype is uint8, shape is [padded_vocab_size // 8]. each bit indicate one token.
        repetition_penalties:  repetition_penalties factor, dtype is float. if value is 0, means dont repetition penalties.
        temperature: temperature factor, dtype is float. if value is 0, means there dont temperature.
    Return:
        modifed_logits:  dtype is float, shape is [batch_size, vocab_size]
    """

    return torch.ops.hpc.fused_repetition_penalties_softmax(
        logits,
        penalties_masks_ptrs,
        *_to_tensor_scalar_tuple(repetition_penalties),
        *_to_tensor_scalar_tuple(temperature),
    )


def topk_mask_logits(
    logits: Tensor,
    topk: Union[Optional[Tensor], int] = 0,
    reject_threshold: Union[Optional[Tensor], float] = 0.0,
) -> Tensor:
    """TopK Sampling.
    The output logits keep the TopK values in their original positions and set all others to -inf. This operation is NOT in-place.
    Args:
        logits: Input logits
            Shape: [batch_size, vocab_size]
            Dtype: float
        topk: TopK tensor for each batch or int for all batches
            Shape: [batch_size] or int
            Dtype: int
        reject_threshold: reject_threshold is used to filt the low probability tokens
            Shape: [batch_size] or float
            Dtype: float
    Return:
        output_logits: New output logits tensor that keeps TopK logits in original position and set others to -inf
            Shape: [batch_size, vocab_size]
            Dtype: float
    """
    return torch.ops.hpc.topk_mask_logits(
        logits,
        *_to_tensor_scalar_tuple(topk),
        *_to_tensor_scalar_tuple(reject_threshold),
    )
