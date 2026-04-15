import torch
from torch import Tensor
from typing import Tuple


def hadamard_transform(
    x: Tensor,
) -> Tensor:
    """Fast Hadamard Transform using power-of-2 decomposition with base matrix.

    Applies a fast Hadamard transform to the last dimension of the input tensor.
    The transform is decomposed as: for n=64, n = 2^4 * 4, so we first perform
    4 levels of unit Hadamard transforms (inter-thread butterfly), then a base-4
    Hadamard transform (intra-thread matrix multiply). The result is scaled by
    1/sqrt(n).

    Args:
        x (Tensor):
            Input tensor of shape ``[..., n]`` where ``n`` is the dimension to
            transform. Currently supports ``n = 64``.
            Must be ``dtype == torch.bfloat16`` and on a CUDA device.

    Returns:
        Tensor: Output tensor of the same shape and dtype as ``x``, containing
        the Hadamard-transformed data scaled by ``1/sqrt(n)``.

    Raises:
        RuntimeError: If the input is not bfloat16, not on CUDA, not contiguous,
            or the last dimension is not a supported size.

    Note:
        - Input tensor must be on CUDA device and in bfloat16 format.
        - The input must be contiguous.
    """
    return torch.ops.hpc.hadamard_transform(x)


def act_mul_hadamard_blockwise_quant(
    gate_up: Tensor, upper_max: float = 448.0, block_size: int = 64, use_pdl: bool = True
) -> Tuple[Tensor, Tensor]:
    """Fused SiLU activation, elementwise multiply, Hadamard transform (n=64 blocks),
    and blockwise FP8 quantization.

    For each row the input ``gate_up`` is split into two equal halves:
    ``gate = gate_up[:, :C]`` and ``up = gate_up[:, C:]``.  The kernel computes:

    1. ``x = silu(gate) * up``          (elementwise, bf16 precision)
    2. ``x = Hadamard64(x)``            (fast Walsh-Hadamard, scaled by 1/sqrt(64),
                                         applied independently to every 64-wide block)
    3. ``out, scale = blockwise_quant(x, block_size=64)``
                                        (one fp32 scale per 64-element group)

    The scale is computed as ``max(|x|) / upper_max`` over the 64-element block.

    Args:
        gate_up (Tensor):
            Input tensor of shape ``[N, 2*C]``, dtype ``bfloat16``, CUDA, contiguous.
            ``C`` must be a multiple of ``block_size``.
        upper_max (float, optional):
            The FP8 dynamic range upper bound used for scale computation.
            ``scale = max(|x|) / upper_max``.  Defaults to ``448.0`` (fp8_e4m3fn max).
        block_size (int, optional):
            Width of each quantization / Hadamard block.  Defaults to ``64``.
            Other values are currently not implemented.
        use_pdl (bool, optional):
            Enable CUDA Programmatic Launch Dependency (PDL) for kernel launch.
            Defaults to ``True``.

    Returns:
        Tuple[Tensor, Tensor]:
            - ``output``:       ``[N, C]``         dtype ``float8_e4m3fn``
            - ``output_scale``: ``[C//64, N]``     dtype ``float32``  (N-major layout)

    Raises:
        RuntimeError: If input dtype is not bfloat16, not on CUDA, not contiguous,
            not 2-dimensional, or ``C`` is not a multiple of 64.
    """
    return torch.ops.hpc.act_mul_hadamard_blockwise_quant(gate_up, upper_max, block_size, use_pdl)


def act_mul_hadamard_per_tensor_quant(
    gate_up: Tensor, scale_inv: Tensor, use_pdl: bool = True
) -> Tensor:
    """Fused SiLU activation, elementwise multiply, Hadamard transform (n=64 blocks),
    and per-tensor FP8 quantization using a pre-computed scale inverse.

    For each row the input ``gate_up`` is split into two equal halves:
    ``gate = gate_up[:, :C]`` and ``up = gate_up[:, C:]``.  The kernel computes:

    1. ``x = silu(gate) * up``          (elementwise, bf16 precision)
    2. ``x = Hadamard64(x)``            (fast Walsh-Hadamard, scaled by 1/sqrt(64),
                                         applied independently to every 64-wide block)
    3. ``out = fp8(x * scale_inv)``     (static per-tensor quantization)

    Unlike the blockwise variant, the scale is not computed inside the kernel.
    The caller provides ``scale_inv`` (the reciprocal of the original scale) so
    that the GPU only needs a single multiply instead of a division.

    Args:
        gate_up (Tensor):
            Input tensor of shape ``[N, 2*C]``, dtype ``bfloat16``, CUDA, contiguous.
            ``C`` must be a multiple of 64.
        scale_inv (Tensor):
            A **single-element** float32 CUDA tensor containing the reciprocal of
            the per-tensor quantization scale, i.e. ``1.0 / scale``.
        use_pdl (bool, optional):
            Enable CUDA Programmatic Launch Dependency (PDL) for kernel launch.
            Defaults to ``True``.

    Returns:
        Tensor: ``[N, C]`` dtype ``float8_e4m3fn`` — the quantized output.

    Raises:
        RuntimeError: If input dtype is not bfloat16, not on CUDA, not contiguous,
            not 2-dimensional, ``C`` is not a multiple of 64, or ``scale_inv`` is
            not a single-element float32 CUDA tensor.
    """
    return torch.ops.hpc.act_mul_hadamard_per_tensor_quant(gate_up, scale_inv, use_pdl)
