import torch
from torch import Tensor
from typing import Union, Tuple


def per_token_group_quant(
    x: Tensor,
    group_size: int = 128,
    quant_eps: float = torch.finfo(torch.float32).eps,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Fused Layer Normalization with scale and FP8 quantization.

    Applies FP8 (E4M3) per-token group quantization. The entire
    computation is fused into a custom CUDA kernel for improved performance and
    reduced memory traffic.

    The computation consists of:
        1. FP8 quantization:
           The tensor `x` is quantized into FP8 (E4M3) using per-token,
           group quantization with groups of size `group_size`.
           The function returns the FP8 tensor `output_fp8` and the
           corresponding per-group quantization scales `quant_scale`,
           which can be used for dequantization.

    Args:
        x (Tensor):
            Input tensor of shape `[batch_size, hidden_states]`, with
            `dtype == torch.bfloat16`, located on a CUDA device.
            Currently supports `hidden_states` in `{4096, 5120}`.
        quant_eps (float, optional):
            Minimum quantization scale used to avoid zero or extremely small
            scales during FP8 quantization. Defaults to
            `torch.finfo(torch.float32).eps`.
        group_size (int, optional):
            Number of elements per quantization group for per-token,
            group-wise quantization. Currently only `group_size == 128`
            is supported. Defaults to 128.
    Returns:
        A tuple containing:
            - output_fp8 (Tensor): Quantized output in FP8 E4M3 format with shape
                [batch_size, hidden_states].
            - quant_scale (Tensor): Per-group quantization scales in float32 format
                with shape [batch_size, hidden_states // group_size]. These scales
                can be used for dequantization.

    Raises:
        RuntimeError: If the shapes or dtypes do not satisfy the constraints above.

    Note:
        - All input tensors must be on CUDA device and in bfloat16 format
    """
    output_fp8, quant_scale = torch.ops.hpc.per_token_group_quant(
        x,
        group_size,
        quant_eps,
    )
    return output_fp8, quant_scale


@torch.library.register_fake("hpc::per_token_group_quant")
def per_token_group_quant_fake(
    x,
    group_size,
    quant_eps,
):
    return (
        torch.empty_like(x, dtype=torch.float8_e4m3fn),
        torch.empty(
            (x.shape[0], int(x.shape[1] / group_size)),
            dtype=torch.float32,
            device=x.device,
        ),
    )
