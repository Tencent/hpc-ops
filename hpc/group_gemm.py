import torch
from torch import Tensor


def group_gemm_fp8(
    x: Tensor,
    weight: Tensor,
    seqlens: Tensor,
    cu_seqlens: Tensor,
    y_scale: Tensor,
    output: Tensor = None,
) -> Tensor:
    """Performs group GEMM operation with FP8 precision.

    This function executes multiple matrix multiplications in a group manner
    using FP8 precision for improved performance.

    Args:
        x: Input activation tensor
            Shape: [total_seq, hidden_dim]
            Dtype: fp8
        weight: Weight tensor for group matrix multiplication
            Shape: [num_groups, output_dim, hidden_dim]
            Dtype: fp8
        seqlens: Sequence lengths for each group
            Shape: [num_group]
            Dtype: int32
        cu_seqlens: Cumulative sequence lengths indicating start indices in input tensor
            Shape: [num_group + 1]
            Dtype: int32
        y_scale: Scaling factor for FP8 quantization
            Shape: [num_group]
            Dtype: float32

    Returns:
        Tensor: Output tensor after group matrix multiplication
            Shape: [total_seq, output_dim]
            Dtype: bfloat16

    Raises:
        RuntimeError: If the input tensors have incompatible shapes or types,
            or if the CUDA kernel execution fails.

    Note:
        - All input tensors must be on CUDA device

    """
    return torch.ops.hpc.group_gemm_fp8(x, weight, seqlens, cu_seqlens, y_scale, output)


@torch.library.register_fake("hpc::group_gemm_fp8")
def group_gemm_fp8_fake(x, weight, seqlens, cu_seqlens, y_scale, output):
    return torch.empty((x.shape[0], weight.shape[1]), dtype=torch.bfloat16)
