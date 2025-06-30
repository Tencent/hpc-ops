import torch
from torch import Tensor
from typing import Union, Tuple


def fused_rms_norm_with_scale(
    a: Tensor,
    weight: Tensor,
    eps: float = torch.finfo(torch.float32).eps,
    scale: Tensor = torch.tensor([1], dtype=torch.float32),
    is_moe: bool = False) -> Union[Tensor, Tuple[Tensor]]:
  """Perform RMSNorm for input and divide scales, output the fp8_e4m3 results.

    Executes type conversion in a custom GPU kernel for optimized performance.

    Args:
        a: Input tensor: [batch_size, hidden_states]. We only support bfloat16 type and hidden_states = 5120 and 320 now.
        weight: [1, hidden_states]. Weight in RMSNorm.
        eps: a value added to the denominator for numerical stability.
        scale: scales for divide.
        output_high_precise: bool. Whether output bfloat16 RMSNorm output.
    Returns:
        New tensor with result RMSNorm(a) / scales in fp8_e4m3 or
        (RMSNorm(a) / scales , RMSNorm(a)) if output_high_precise is True
  """
  if scale.device != a.device:
    scale = scale.to(a.device)
  output_fp8, output_fp32, output_fp8_scale2 = torch.ops.hpc.fused_rms_norm_with_scale(
      a, weight, scale, eps, is_moe)
  return (output_fp32, output_fp8, output_fp8_scale2) if is_moe else output_fp8
