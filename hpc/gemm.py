from typing import Optional

import torch
from torch import Tensor


def get_gemm_bf16xfp32_workspace(max_weight_hidden_size: int, max_tokens: int = 131072) -> Tensor:

    min_tile_m = 16
    min_tile_n = 64
    nm_max = (max_tokens + min_tile_m - 1) // min_tile_m
    nn_max = (max_weight_hidden_size + min_tile_n - 1) // min_tile_n
    return torch.zeros((nm_max, nn_max), dtype=torch.int32, device="cuda")


def gemm_bf16xfp32(
    x: Tensor,
    w_high: Tensor,
    w_low: Tensor,
    scale: Tensor,
    use_fp32_output: bool = False,
    use_splitk: bool = True,
    split_flag: Tensor = None,
) -> Tensor:
    """Performs fp32 GEMM operation with two bf16 gemm.
    Where
        scale = 1 / 256
        w_high = w_fp32.to(torch.bfloat16)
        w_low = ((w_fp32 - w_high.float()) / scale).to(torch.bfloat16)

    Args:
        x: Input activation tensor
            Shape: [m, k]
            Dtype: bfloat16
        w_high: Weight tensor with main precise part of fp32 weight.
            Shape: [n, k]
            Dtype: bfloat16
        w_low: Weight tensor with residual precise part of fp32 weight.
            Shape: [n, k]
            Dtype: bfloat16
        scale: Scaling factor for low weight tensor
            Shape: Scalar
            Dtype: float32
        use_fp32_output: Control Output dtype is float32 or bfloat16
            Shape: Scalar
            Dtype: bfloat16
        use_splitk: Control whether use splitk.
            Shape: Scalar
            Dtype: bool
        split_flag: Optinal Input indicates the split finish state, should be init zero at the beginning.
            Shape: [max_tokens / kTileM, n / kTileN]
            Dtype: int32
    Returns:
        Tensor: Output tensor after matrix multiplication
            Shape: [m, n]
            Dtype: bfloat16 or float32.

    """
    return torch.ops.hpc.gemm_bf16xfp32(
        x, w_high, w_low, scale, use_fp32_output, use_splitk, split_flag
    )


@torch.library.register_fake("hpc::gemm_bf16xfp32")
def gemm_bf16xfp32_fake(
    a: Tensor,
    b_high: Tensor,
    b_low: Tensor,
    scale: Tensor,
    use_fp32_output: bool = False,
    use_splitk: bool = True,
    split_flag: Tensor = None,
):
    if use_fp32_output:
        return torch.empty((a.shape[0], b_high.shape[0]), dtype=torch.float32, device=a.device)
    else:
        return torch.empty((a.shape[0], b_high.shape[0]), dtype=a.dtype, device=a.device)
