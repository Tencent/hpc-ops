import torch
from torch import Tensor
from typing import Union, Optional, Tuple


def _to_tensor_scalar_tuple(x) -> Tuple[Optional[Tensor], float]:
    if isinstance(x, torch.Tensor):
        return (x, 0.0)
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
