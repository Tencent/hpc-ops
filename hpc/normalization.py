import torch
from torch import Tensor
from typing import List, Union, Tuple, Optional


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
    dual_output: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """Perform RMSNorm and BlockWise Quant.
    Args:
        x: Input tensor
            Shape: [num_tokens, dim]
            Dtype: torch.bfloat16
        weight: Weight for RMSNorm
            Shape: [1, dim].
            Dtype: torch.bfloat16
        eps: a value added to the denominator for numerical stability.
            Shape: scalar
            Dtype: float
        with_blockwise_quant: whether quantinize the output of rmsnorm
        block_size: now only support 128
        dual_output: if set to true, will return the output of rmsnorm and its quantinization
    Returns:
        when with_blockwise_quant is True and dual_output is True:
            return [rmsnorm(x), quant(rmsnorm(x)), fp32_scale]
        when only with_blockwise_quant is True:
            return [quant(rmsnorm(x)), fp32_scale]
        else:
            return [rmsnorm(x)]
    """
    assert block_size == 128, "now only support blockwise == 128"

    if with_blockwise_quant and dual_output:
        y_bf16, y_fp8, y_scale = torch.ops.hpc.fused_rmsnorm_blockwise_quant(
            x, weight, eps, with_blockwise_quant, block_size, dual_output
        )
        return y_bf16, y_fp8, y_scale
    elif with_blockwise_quant:
        y_fp8, y_scale, _ = torch.ops.hpc.fused_rmsnorm_blockwise_quant(
            x, weight, eps, with_blockwise_quant, block_size, dual_output
        )
        return y_fp8, y_scale
    else:
        y_bf16, _, _ = torch.ops.hpc.fused_rmsnorm_blockwise_quant(
            x, weight, eps, with_blockwise_quant, block_size, dual_output
        )
        return y_bf16


def fused_rmsnorm_rope(
    positions: Tensor,
    q: Tensor,
    q_weight: Optional[Tensor],
    k: Optional[Tensor],
    k_weight: Optional[Tensor],
    cos_sin_cache: Optional[Tensor],
    eps: float = torch.finfo(torch.float32).eps,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Perform RMSNorm and Rope.
    Args:
        positions: Position indices for each sequence element, used to lookup corresponding rotation angles from cos_sin_cache.
            Shape: [batch]
            Dtype: torch.int64
        q: Input tensor
            Shape: [batch, num_q_heads, dim]
            Dtype: torch.bfloat16
        q_weight: Weight for q in RMSNorm
            Shape: [1, dim].
            Dtype: torch.bfloat16
        k: Input tensor
            Shape: [batch, num_k_heads, dim] or [batch, dim]
            Dtype: torch.bfloat16
        k_weight: Weight for k in RMSNorm
            Shape: [1, dim]
            Dtype: torch.bfloat16
        cos_sin_cache: cos and sin cache for rope, cos_sin_cache should be interleave
            Shape: [1, rope_dim]
            Dtype: torch.bfloat16
        eps: a value added to the denominator for numerical stability.
            Shape: scalar
            Dtype: float

    Returns:
        New tensor with result Rope(RMSNorm(q)) and Rope(RMSNorm(k))
    """
    return torch.ops.hpc.fused_rmsnorm_rope(positions, q, q_weight, k, k_weight, cos_sin_cache, eps)


def fused_qk_rmsnorm_rope(
    qkv: Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    norm_eps: float,
    positions: Tensor,
    cos_sin_cache: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Fused QK RMSNorm + plain RoPE for a single packed QKV tensor.

    This reuses fused_qk_rmsnorm_mrope by treating all tokens as understanding
    tokens, passing an empty generation tensor, and expanding the plain RoPE
    positions to identical T/H/W rows.

    Args:
        qkv: Packed QKV tensor.
            Shape: [num_tokens, num_heads_q*head_dim + num_heads_k*head_dim + num_heads_v*head_dim]
            Dtype: bfloat16
        num_heads_q, num_heads_k, num_heads_v: Head counts.
        head_dim: Head dimension (currently only 128).
        q_norm_weight, k_norm_weight: RMSNorm weights for Q/K. Shape: [head_dim], bf16.
        norm_eps: RMSNorm epsilon.
        positions: Plain RoPE position indices. Shape: [num_tokens], int64.
            If a 2D tensor is passed, the first row is used to match fuse_single_qknorm_rope_hy.
        cos_sin_cache: Precomputed cos/sin table. Shape: [max_pos, head_dim], fp32.

    Returns:
        (q, k, v):
            q: [num_tokens, num_heads_q, head_dim], bf16
            k: [num_tokens, num_heads_k, head_dim], bf16
            v: [num_tokens, num_heads_v, head_dim], bf16
    """
    if positions.dim() == 1:
        rope_positions = positions
    elif positions.dim() == 2:
        rope_positions = positions[0]
    else:
        raise ValueError("positions must be 1D or 2D")

    num_tokens = qkv.shape[0]
    gen_qkv = qkv.new_empty((0, qkv.shape[1]))
    cat_indices = torch.arange(num_tokens, device=qkv.device, dtype=torch.int64)
    gen_indices = torch.empty((0,), device=qkv.device, dtype=torch.int64)
    mrope_positions = rope_positions.unsqueeze(0).expand(3, -1).contiguous()

    return fused_qk_rmsnorm_mrope(
        qkv,
        gen_qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_norm_weight,
        k_norm_weight,
        q_norm_weight,
        k_norm_weight,
        norm_eps,
        mrope_positions,
        [0, 0, 0],
        cos_sin_cache,
        cat_indices,
        gen_indices,
        cat_indices,
    )


def fused_qk_rmsnorm_mrope(
    und_qkv: Tensor,
    gen_qkv: Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    und_q_norm_weight: Tensor,
    und_k_norm_weight: Tensor,
    gen_q_norm_weight: Tensor,
    gen_k_norm_weight: Tensor,
    norm_eps: float,
    positions: Tensor,
    mrope_section: List[int],
    cos_sin_cache: Tensor,
    und_indices: Tensor,
    gen_indices: Tensor,
    cat_indices: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Fused QK RMSNorm + 3D MRoPE + und/gen concat reorder.

    Performs:
        1. Split packed und_qkv/gen_qkv into Q/K/V
        2. RMSNorm on Q/K per-head using respective norm weights
        3. Concat und+gen and reorder by cat_indices
        4. Apply 3D MRoPE (T/H/W) to Q/K using positions and mrope_section
        5. Return reordered Q/K/V

    Args:
        und_qkv: Packed QKV for understanding tokens.
            Shape: [und_len, num_heads_q*head_dim + num_heads_k*head_dim + num_heads_v*head_dim]
            Dtype: bfloat16
        gen_qkv: Packed QKV for generation tokens.
            Shape: [gen_len, same_hidden]
            Dtype: bfloat16
        num_heads_q, num_heads_k, num_heads_v: Head counts.
        head_dim: Head dimension (currently only 128).
        und_q_norm_weight, und_k_norm_weight: RMSNorm weights for und Q/K. Shape: [head_dim], bf16.
        gen_q_norm_weight, gen_k_norm_weight: RMSNorm weights for gen Q/K. Shape: [head_dim], bf16.
        norm_eps: RMSNorm epsilon.
        positions: 3D MRoPE position indices. Shape: [3, total_tokens], int64.
        mrope_section: [T_section, H_section, W_section] controlling frequency axis assignment.
        cos_sin_cache: Precomputed cos/sin table. Shape: [max_pos, head_dim], fp32.
            Layout: [cos_half | sin_half].
        und_indices, gen_indices: Accepted for API parity (unused by kernel).
        cat_indices: Reorder indices. Shape: [total_tokens], int64.

    Returns:
        (q, k, v):
            q: [total_tokens, num_heads_q, head_dim], bf16
            k: [total_tokens, num_heads_k, head_dim], bf16
            v: [total_tokens, num_heads_v, head_dim], bf16
    """
    return torch.ops.hpc.fused_qk_rmsnorm_mrope(
        und_qkv,
        gen_qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        und_q_norm_weight,
        und_k_norm_weight,
        gen_q_norm_weight,
        gen_k_norm_weight,
        norm_eps,
        positions,
        cos_sin_cache,
        cat_indices,
        mrope_section[1],
        mrope_section[2],
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


@torch.library.register_fake("hpc::fused_qk_rmsnorm_mrope")
def fused_qk_rmsnorm_mrope_fake(
    und_qkv,
    gen_qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    und_q_norm_weight,
    und_k_norm_weight,
    gen_q_norm_weight,
    gen_k_norm_weight,
    norm_eps,
    positions,
    cos_sin_cache,
    cat_indices,
    mrope_section_h,
    mrope_section_w,
):
    total_tokens = und_qkv.shape[0] + gen_qkv.shape[0]
    device = und_qkv.device
    return (
        torch.empty(total_tokens, num_heads_q, head_dim, dtype=torch.bfloat16, device=device),
        torch.empty(total_tokens, num_heads_k, head_dim, dtype=torch.bfloat16, device=device),
        torch.empty(total_tokens, num_heads_v, head_dim, dtype=torch.bfloat16, device=device),
    )
