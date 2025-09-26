import torch
from torch import Tensor


def has_nan(a: Tensor, tag: str, num_warning_blocks: int = 16) -> None:
    """Checks if a tensor contains NaN values and prints warnings if any are found.

    Performs element-wise check on GPU. When NaN values are detected, the function
    prints warning messages including the provided tag for easy identification.
    The num_warning_blocks parameter limits the maximum number of warning outputs
    to prevent excessive noise in the logs.

    Args:
        a: Input tensor to be checked for NaN values
        tag: Identifier string to distinguish different checkpoints in warning messages
        num_warning_blocks: Maximum number of warning blocks to output (default: 16)
            Used to control output volume and avoid information overload

    Returns:
        None: This function doesn't return any value, only outputs warnings as side effect
    """
    return torch.ops.hpc.has_nan(a, tag, num_warning_blocks)
