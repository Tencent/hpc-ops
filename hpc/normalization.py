import torch
from torch import Tensor
from typing import Union, Tuple, Optional


def fused_rms_norm_with_scale(
    a: Tensor,
    weight: Tensor,
    eps: float = torch.finfo(torch.float32).eps,
    scale: Tensor = torch.tensor([1], dtype=torch.float32),
    is_moe: bool = False,
) -> Union[Tensor, Tuple[Tensor]]:
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
        a, weight, scale, eps, is_moe
    )
    return (output_fp32, output_fp8, output_fp8_scale2) if is_moe else output_fp8


def fused_layer_norm_with_scale_quant(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    pre_norm_scale1: Tensor,
    pre_norm_scale2: Tensor,
    post_norm_scale: Tensor,
    post_norm_bias_scale: Tensor,
    eps: float = torch.finfo(torch.float32).eps,
    quant_eps: float = torch.finfo(torch.float32).eps,
    group_size: int = 128,
    is_elementwise_affine: bool = False,
    use_pre_norm_scale: bool = False,
    use_post_norm_scale: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Fused Layer Normalization with scale and FP8 quantization.

    Applies Layer Normalization to the input tensor, optionally applies affine
    transformation, pre-normalization scaling, and post-normalization scaling,
    then performs FP8 (E4M3) per-token group quantization. The entire
    computation is fused into a custom CUDA kernel for improved performance and
    reduced memory traffic.

    The computation consists of:
        1. Optional pre-normalization scaling:
           If `use_pre_norm_scale` is True, the input is first transformed as
           `x = x + pre_norm_scale1 * pre_norm_scale2`.
        2. Layer Normalization:
           `normalized = (x - mean(x)) / sqrt(var(x) + eps)`.
        3. Optional affine transformation:
           If `is_elementwise_affine` is True,
           `y = normalized * weight + bias`, otherwise `y = normalized`.
        4. Optional post-normalization scaling:
           If `use_post_norm_scale` is True,
           `y = y * post_norm_scale + post_norm_bias_scale`.
        5. FP8 quantization:
           The tensor `y` is quantized into FP8 (E4M3) using per-token,
           group quantization with groups of size `group_size`.
           The function returns the FP8 tensor `output_fp8` and the
           corresponding per-group quantization scales `quant_scale`,
           which can be used for dequantization.

    Args:
        x (Tensor):
            Input tensor of shape `[batch_size, hidden_states]`, with
            `dtype == torch.bfloat16`, located on a CUDA device.
            Currently supports `hidden_states` in `{4096, 5120}`.
        weight (Tensor):
            LayerNorm scale parameter (gamma) of shape `[hidden_states]`,
            with `dtype == torch.bfloat16`. Used only when
            `is_elementwise_affine` is True.
        bias (Tensor):
            LayerNorm bias parameter (beta) of shape `[hidden_states]`,
            with `dtype == torch.bfloat16`. Used only when
            `is_elementwise_affine` is True.
        pre_norm_scale1 (Tensor):
            First pre-normalization scaling term of shape
            `[batch_size, hidden_states]`, with `dtype == torch.bfloat16`.
            Used only when `use_pre_norm_scale` is True. The pre-norm
            update is `a + pre_norm_scale1 * pre_norm_scale2`.
        pre_norm_scale2 (Tensor):
            Second pre-normalization scaling term of shape
            `[batch_size, hidden_states]`, with `dtype == torch.bfloat16`.
            Used only when `use_pre_norm_scale` is True.
        post_norm_scale (Tensor):
            Post-normalization scale factor of shape
            `[batch_size, hidden_states]`, with `dtype == torch.bfloat16`.
            Used only when `use_post_norm_scale` is True. The post-norm
            update is `y * post_norm_scale + post_norm_bias_scale`.
        post_norm_bias_scale (Tensor):
            Post-normalization bias (or bias scaling) term of shape
            `[batch_size, hidden_states]`, with `dtype == torch.bfloat16`.
            Used only when `use_post_norm_scale` is True.
        eps (float, optional):
            Small constant added to the variance in the LayerNorm denominator
            for numerical stability, i.e. `sqrt(var + eps)`. Defaults to
            `torch.finfo(torch.float32).eps`.
        quant_eps (float, optional):
            Minimum quantization scale used to avoid zero or extremely small
            scales during FP8 quantization. Defaults to
            `torch.finfo(torch.float32).eps`.
        group_size (int, optional):
            Number of elements per quantization group for per-token,
            group-wise quantization. Currently only `group_size == 128`
            is supported. Defaults to 128.
        is_elementwise_affine (bool, optional):
            Whether to apply elementwise affine transformation using `weight`
            and `bias` after normalization. If True, applies
            `normalized * weight + bias`; if False, returns the normalized
            tensor without affine transformation. Defaults to False.
        use_pre_norm_scale (bool, optional):
            Whether to enable the pre-normalization scaling step using
            `pre_norm_scale1` and `pre_norm_scale2`. If False, these
            tensors are ignored. Defaults to False.
        use_post_norm_scale (bool, optional):
            Whether to enable the post-normalization scaling step using
            `post_norm_scale` and `post_norm_bias_scale`. If False, these
            tensors are ignored. Defaults to False.

    Returns:
        A tuple containing:
            - output_fp8 (Tensor): Quantized output in FP8 E4M3 format with shape
                [batch_size, hidden_states].
            - quant_scale (Tensor): Per-group quantization scales in float32 format
                with shape [batch_size, hidden_states // group_size]. These scales
                can be used for dequantization.
            - output_x (Tensor): Output input tensor before layernorm in bfloat16 format with shape [batch_size, hidden_states].

    Raises:
        RuntimeError: If the shapes or dtypes do not satisfy the constraints above.

    Note:
        - All input tensors must be on CUDA device and in bfloat16 format
    """
    output_fp8, quant_scale, output_x = torch.ops.hpc.fused_layer_norm_with_scale_quant(
        x,
        weight,
        bias,
        pre_norm_scale1,
        pre_norm_scale2,
        post_norm_scale,
        post_norm_bias_scale,
        eps,
        quant_eps,
        group_size,
        is_elementwise_affine,
        use_pre_norm_scale,
        use_post_norm_scale,
    )
    return output_fp8, quant_scale, output_x


def fused_rmsnorm_blockwise_quant(
    x: Tensor,
    weight: Tensor,
    eps: float = torch.finfo(torch.float32).eps,
    with_blockwise_quant: bool = False,
    block_size: int = 128,
) -> Tuple[Tensor, Optional[Tensor]]:
    assert block_size == 128, "now only support blockwise == 128"
    return torch.ops.hpc.fused_rmsnorm_blockwise_quant(
        x, weight, eps, with_blockwise_quant, block_size
    )


@torch.library.register_fake("hpc::fused_rms_norm_with_scale")
def fused_rms_norm_with_scale_fake(a, weight, eps, scale, is_moe):
    return (
        torch.empty_like(a, dtype=torch.float8_e4m3fn),
        torch.empty_like(a, dtype=torch.float32),
        torch.empty_like(a, dtype=torch.float8_e4m3fn),
    )


@torch.library.register_fake("hpc::fused_layer_norm_with_scale_quant")
def fused_layer_norm_with_scale_quant_fake(
    x,
    weight,
    bias,
    pre_norm_scale1,
    pre_norm_scale2,
    post_norm_scale,
    post_norm_bias_scale,
    eps,
    quant_eps,
    group_size,
    is_elementwise_affine,
    use_pre_norm_scale,
    use_post_norm_scale,
):
    return (
        torch.empty_like(x, dtype=torch.float8_e4m3fn),
        torch.empty(
            (x.shape[0], int(x.shape[1] / group_size)),
            dtype=torch.float32,
            device=x.device,
        ),
        torch.empty_like(x, dtype=torch.bfloat16),
    )
