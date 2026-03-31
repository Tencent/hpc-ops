import torch
from torch import Tensor
from typing import Union, Tuple, Optional


def fused_rmsnorm_with_scale(
    a: Tensor,
    weight: Tensor,
    eps: float = torch.finfo(torch.float32).eps,
    scale: Tensor = torch.tensor([1], dtype=torch.float32),
    is_moe: bool = False,
) -> Union[Tensor, Tuple[Tensor]]:
    """Perform RMSNorm for input and divide scales, output the fp8_e4m3 results.

    Executes type conversion in a custom GPU kernel for optimized performance.

    Args:
        a: Input tensor. We only support bfloat16 type and hidden_states = 5120/4096/320 now.
            Shape: [batch_size, hidden_states].
            Dtype: torch.bfloat16
        weight: Weight in RMSNorm.
            Shape: [hidden_states].
            Dtype: torch.bfloat16.
        eps: a value added to the denominator for numerical stability.
            Shape: scalar
            Dtype: float
        scale: scales for divide.
            Shape: [1] or [2]
            Dtype: float
        is_moe: Whether the operation after this rmsnorm is moe,
            if is True, the scale shape is [2],
    Returns:
        if is_moe is True, return (RMSNorm(a),  RMSNorm(a) / scale[0], RMSNorm(a) / scale[1])
        else return RMSNorm(a) / scale[0]
    """
    if scale.device != a.device:
        scale = scale.to(a.device)
    output_fp8, output_fp32, output_fp8_scale2 = torch.ops.hpc.fused_rmsnorm_with_scale(
        a, weight, scale, eps, is_moe
    )
    return (output_fp32, output_fp8, output_fp8_scale2) if is_moe else output_fp8


@torch.library.register_fake("hpc::fused_rmsnorm_with_scale")
def fused_rmsnorm_with_scale_fake(a, weight, eps, scale, is_moe):
    if is_moe:
        return (
            torch.empty_like(a, dtype=torch.float32),
            torch.empty_like(a, dtype=torch.float8_e4m3fn),
            torch.empty_like(a, dtype=torch.float8_e4m3fn),
        )
    else:
        return torch.empty_like(a, dtype=torch.float8_e4m3fn)
