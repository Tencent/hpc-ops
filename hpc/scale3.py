import torch
from torch import Tensor
from typing import Union, Tuple


def scale3(
    a: Tensor,
    scale: Tensor = torch.tensor([1], dtype=torch.float32),
    scale2: Tensor = torch.tensor([1], dtype=torch.float32),
    is_moe: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Perform per-tesor quant, output the fp8_e4m3 results.
    Args:
        a: Input tensor: [batch_size, hidden_states]. We only support bfloat16 type and hidden_states 4096 now.
        scale: scale for divide.
        scale2: another scale for divide.
    Returns:
        Tuple(a / scale, a / scale2, a.to(float32))
    """
    if is_moe:
        assert scale2 is not None
    output_fp8, output_fp8_scale2, output_fp32 = torch.ops.hpc.scale3(a, scale, scale2, is_moe)
    return output_fp8, output_fp8_scale2, output_fp32


@torch.library.register_fake("hpc::scale3")
def scale3_fake(a, scale, scale2, is_moe):
    return (
        torch.empty_like(a, dtype=torch.float8_e4m3fn),
        torch.empty_like(a, dtype=torch.float8_e4m3fn),
        torch.empty_like(a, dtype=torch.float32),
    )
