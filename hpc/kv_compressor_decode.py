import torch
from torch import Tensor


def kv_compressor_decode(
    kv: Tensor,
    score: Tensor,
    ape: Tensor,
    kv_states: Tensor,
    score_states: Tensor,
    state_index: Tensor,
    start_pos: Tensor,
    cu_compress_seqlens: Tensor,
    head_dim: int,
    ratio: int,
    overlap: bool,
    output: Tensor = None,
) -> Tensor:
    """Performs compress kv.
    This API also supports speculative sampling.
    For speculative sampling, currently only mtp=1 is supported.

    coff = 2 if overlap else 1

    Args:
        kv: kv tensor
            Shape: [batch, head_dim * coff]
            Dtype: fp32
        score: score tensor
            Shape: [batch, head_dim * coff]
            Dtype: fp32
        ape: bias tensor
            Shape: [ratio, head_dim * coff]
            Dtype: fp32
        kv_states: the whole kv_state of all batches
            Shape: [max_batch, ratio * coff, head_dim * coff]
            Dtype: fp32
        score_states: the whole score_state of all batches,
            Shape: [max_batch, ratio * coff, head_dim * coff]
            Dtype: fp32
        state_index: index of used kv_state or score_state for each batch
            Shape: [num_batch]
            Dtype: int32
        start_pos: start position of each batch
            Shape: [num_batch]
            Dtype: int32
        cu_compress_seqlens: cumsum of compressed seqlens of all batches
            Shape: [num_batch+1]
            Dtype: int32
        head_dim: head dimension
            Dtype: int
        ratio: compression ratio, 4 or 128
            Dtype: int
        overlap: whether to use overlapped kv_state
            Dtype: int
        output: optional[output tensor after compress kv]
            Shape: [batch, head_dim]
            Dtype: fp32

    Returns:
        Tensor: output tensor after compress kv
            Shape: [batch, head_dim]
            Dtype: fp32

    """
    return torch.ops.hpc.kv_compressor_decode(
        kv,
        score,
        ape,
        kv_states,
        score_states,
        state_index,
        start_pos,
        cu_compress_seqlens,
        head_dim,
        ratio,
        overlap,
        output,
    )
