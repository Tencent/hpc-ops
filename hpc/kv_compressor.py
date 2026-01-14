from typing import Optional, Tuple

import torch
from torch import Tensor


def kv_compressor(
    kv: Tensor,
    score: Tensor,
    cu_seqlens: Tensor,
    cu_compressed_seqlens: Tensor,
    total_compressed_seqlen: int,
    kv_states: Tensor,
    score_states: Tensor,
    state_index: Tensor,
    start_pos: Tensor,
    ape: Tensor,
    ratio: int,
    overlap: bool,
    head_dim: int,
    is_prefill: bool,
) -> Tensor:
    """Applies compressor to kv

    Args:
        kv: kv tensor
            Shape: [total_seqlen, head_dim] or [total_seqlen, head_dim * 2]
            Dtype: float32
        score: score tensor
            Shape: [total_seqlen, head_dim] or [total_seqlen, head_dim * 2]
            Dtype: float32
        cu_seqlens: cumsum of sequence lengths, start from 0
            Shape: [batch_size + 1]
            Dtype: int32
        cu_compressed_seqlens: cumsum of compressed sequence lengths, start from 0
            Shape: [batch_size + 1]
            Dtype: int32
        total_compressed_seqlen: total compressed seqlen, equal to cu_compressed_seqlens[-1] but is a cpu number
            Shape: Scalar
            Dtype: int
        kv_states: all kv_states, if overlap, expanded twice in horizontal and vertical
            Shape: [MAX_BATCH_SIZE, ratio, head_dim] or [MAX_BATCH_SIZE, ratio * 2, head_dim * 2]
            Dtype: float32
        score_states: score_states tensor, same like kv_states
            Shape: [MAX_BATCH_SIZE, ratio, head_dim] or [MAX_BATCH_SIZE, ratio * 2, head_dim * 2]
            Dtype: float32
        state_index: state_index tensor, indicates the position of each batch in kv_states
            Shape: [batch_size]
            Dtype: int32
        start_pos: start_pos tensor, indicates the start position of each batch, if used for chunked prefill, start_pos should be divisible by compress ratio
            Shape: [batch_size]
            Dtype: int32
        ape: ape tensor, bias for score
            Shape: [ratio, head_dim] or [ratio, head_dim * 2]
            Dtype: float32
        ratio: compress ratio
            Shape: Scalar
            Dtype: int
        overlap: overlap or not
            Shape: Scalar
            Dtype: bool
        head_dim: head_dim
            Shape: Scalar
            Dtype: int
        is_prefill: is prefill or not
            Shape: Scalar
            Dtype: bool


    Returns:
        compressed_kv:
            Shape: [total_compressed_seqlen, head_dim]
            Dtype: float32

    """
    if is_prefill:
        return torch.ops.hpc.kv_compressor(
            kv,
            score,
            cu_seqlens,
            cu_compressed_seqlens,
            total_compressed_seqlen,
            kv_states,
            score_states,
            state_index,
            start_pos,
            ape,
            ratio,
            overlap,
            head_dim,
            is_prefill,
        )
    else:
        return torch.ops.hpc.kv_compressor_decode(
            kv,
            score,
            ape,
            kv_states,
            score_states,
            state_index,
            start_pos,
            cu_compressed_seqlens,
            head_dim,
            ratio,
            overlap,
            None,
        )


@torch.library.register_fake("hpc::kv_compressor")
def kv_compressor_fake(
    kv,
    score,
    cu_seqlens,
    cu_compressed_seqlens,
    total_compressed_seqlen,
    kv_states,
    score_states,
    state_index,
    start_pos,
    ape,
    ratio,
    overlap,
    head_dim,
    is_prefill,
):
    num_batch = cu_seqlens.shape[0] - 1
    if is_prefill:
        return torch.empty(total_compressed_seqlen, head_dim, dtype=kv.dtype, device=kv.device)
    else:
        return torch.empty(num_batch, head_dim, dtype=kv.dtype, device=kv.device)
