from typing import Optional, Tuple

from torch import Tensor
import torch


def topk_filtered(
    logits: Tensor,
    ke: Tensor,
    output: Tensor,
    num_valid_rows: Tensor,
    top_k: int = 2048,
    counters: Optional[Tensor] = None,
    workspace: Optional[Tensor] = None,
) -> Tensor:
    """Compute exact Top-K indices over variable-length FP32 rows.

    Dispatch uses the captured tensor capacities and remains stable across CUDA
    Graph replays. ``num_valid_rows`` controls how many rows are live at replay
    time without changing the host-side launch.

    Args:
        logits: ``[M_cap, N]`` CUDA float32 tensor with contiguous,
            non-overlapping rows.
        ke: ``[M_cap]`` contiguous CUDA int32 tensor. ``ke[r]`` is the valid
            prefix length of row ``r`` and must be in ``[0, N]``.
        output: ``[>=M_cap, >=top_k]`` CUDA int32 tensor written in place.
            Output indices are unordered; ties at the rank boundary may select
            any valid indices carrying the same values.
        num_valid_rows: One-element CUDA int32 tensor containing ``M_live``,
            where ``0 <= M_live <= M_cap``.
        top_k: Supported values are 512 and 2048.
        counters: Optional zero-filled CUDA uint8 persistent-state buffer sized
            by ``topk_filtered_workspace_size(M_cap, N)[0]``. It is left
            zero-filled after every call and can be reused directly.
        workspace: Optional CUDA uint8 scratch buffer sized by
            ``topk_filtered_workspace_size(M_cap, N)[1]``. Its incoming
            contents are ignored. A minimum-sized buffer remains exact but
            disables the KV-split fast path.

    Returns:
        ``output``.
    """
    return torch.ops.hpc.topk_filtered(
        logits, ke, output, top_k, num_valid_rows, counters, workspace
    )


def topk_filtered_workspace_size(num_rows: int, max_kv_len: int) -> Tuple[int, int]:
    """Return recommended ``(counters_bytes, workspace_bytes)`` for a shape."""
    return torch.ops.hpc.topk_filtered_workspace_size(num_rows, max_kv_len)


def topk_filtered_min_workspace_size(max_kv_len: int) -> int:
    """Return minimum scratch bytes that preserve exact execution."""
    return torch.ops.hpc.topk_filtered_min_workspace_size(max_kv_len)


def topk_filtered_peak_workspace_size(max_kv_len: int) -> int:
    """Return peak scratch bytes over every row capacity at a fixed width."""
    return torch.ops.hpc.topk_filtered_peak_workspace_size(max_kv_len)


@torch.library.register_fake("hpc::topk_filtered")
def topk_filtered_fake(logits, ke, output, top_k, num_valid_rows, counters, workspace):
    return output
