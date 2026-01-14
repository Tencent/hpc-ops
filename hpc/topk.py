import torch
from torch import Tensor


def topk_per_row(
    logits: Tensor,
    seqlens: Tensor,
    num_sp_tokens: int,
    top_k: int,
    topk_indices=None,
) -> Tensor:
    """Computes approximate topk in logits.

    This function return the indices in logits which have topk values.
    First, we will convert float32 to float16 and drop the last 7 mantissa.
    Secondly, we view the logit as binary and scatter it to 512 bins.
    Thirdly, we calculate the prefix sum for bins count and indicate the bin topk-th val in. We called this bin as threshold_bin.
    Then, we select the value located before threshold_bin and sort the values in threshold_bin to get the topk indices.

    Note: The threshold_bin will located in shared memory. As it has limited size, topk will be not exactly correct when the number values in threshold_bin bigger than the size of threshold_bin.

    Args:
        logits: logits for each request, Topk will use logits as key. Must be 4 floats aligned
            Shape: [num_rows, max_seqlen]
            Dtype: float32
        seqlens: seqlen for each request.
            Shape: [num_batch]
            Dtype: int32
        num_sp_tokens: indicates the number tokens with speculate tokens. For example, 2 for mtp=1.
            Shape: scalar
            Dtype: int32
        top_k:
            Shape: scalar
            Dtype: int
        topk_indices: Topk indices to store the result.
            Shape: [num_rows, top_k]
            Dtype: int32
    Returns:
        Tensor: Topk indices.
            Shape: [num_rows, top_k]
            Dtype: int32

    Raises:
        RuntimeError: If the shapes or dtypes do not satisfy the constraints above.
    """
    return torch.ops.hpc.topk_per_row(logits, num_sp_tokens, top_k, seqlens, topk_indices)


def topk_per_row_varlen(
    logits: Tensor,
    cu_seqlens_q: Tensor,
    seqlens_kv: Tensor,
    top_k: int,
    compress_ratio: int = 1,
    deterministic: bool = False,
    topk_indices=None,
) -> Tensor:
    """Computes approximate topk in logits.

    This function return the indices in logits which have topk values.
    First, we will convert float32 to float16 and drop the last 7 mantissa.
    Secondly, we view the logit as binary and scatter it to 512 bins.
    Thirdly, we calculate the prefix sum for bins count and indicate the bin topk-th val in. We called this bin as threshold_bin.
    Then, we select the value located before threshold_bin and sort the values in threshold_bin to get the topk indices.

    Note: The threshold_bin will located in shared memory. As it has limited size, topk will be not exactly correct when the number values in threshold_bin bigger than the size of threshold_bin.

    Args:
        logits: logits for each request, Topk will use logits as key. Must be 4 floats aligned
            Shape: [num_rows, max_seqlen]
            Dtype: float32
        cu_seqlens_q: start_seq_q for each batch
            Shape: [num_batch + 1]
            Dtype: int32
        seqlens_kv: number tokens in kvcache contain the cur query.
            Shape: [num_batch]
            Dtype: int32
        top_k:
            Shape: scalar
            Dtype: int
        compress_ratio: compress ratio for kvcache
            shape: scalar
            Dtype: int
        topk_indices: Topk indices to store the result.
            Shape: [num_rows, top_k]
            Dtype: int32
    Returns:
        Tensor: Topk indices.
            Shape: [num_rows, top_k]
            Dtype: int32

    Raises:
        RuntimeError: If the shapes or dtypes do not satisfy the constraints above.
    """
    return torch.ops.hpc.topk_per_row_varlen(
        logits, cu_seqlens_q, seqlens_kv, top_k, compress_ratio, deterministic, topk_indices
    )


@torch.library.register_fake("hpc::topk_per_row")
def topk_per_row_fake(logits, seqlens, num_sp_tokens, top_k, topk_indices):
    return torch.empty((logits.shape[0], top_k), dtype=torch.int32, device=logits.device)


@torch.library.register_fake("hpc::topk_per_row_varlen")
def topk_per_row_varlen_fake(logits, cu_seqlens_q, seqlens_kv, top_k, compress_ratio, topk_indices):
    return torch.empty((logits.shape[0], top_k), dtype=torch.int32, device=logits.device)
