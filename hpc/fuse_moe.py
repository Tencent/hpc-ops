import torch
from torch import Tensor
from typing import Tuple


def count_and_gather(
    x: Tensor,
    topk_ids: Tensor,
    num_expert: int,
    rank_ep: int,
    intermediate_size: int,
    num_seq_per_group_avg: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Sorts and aggregates token based on expert assignments for MoE layers.

    This function organizes input token according to their assigned expert IDs
    after top-k expert selection, grouping token by expert for efficient
    parallel computation in mixture-of-experts architectures.

    Args:
        x: Input token features tensor
            Shape: [num_seq, hidden_size]
            Dtype: fp8
        topk_ids: Expert assignment indices for each token
            Shape: [num_seq, num_topk]
            Dtype: int32
        num_expert: Number of experts available on the current device
            Shape: scalar
            DType: int
        rank_ep: Current device rank in expert parallelism
            Shape: scalar
            DType: int

    Returns:
        Tuple containing eight tensors:
        - output: Sorted token features grouped by expert
            Shape: [num_seq * num_topk, hidden_size]
            Dtype: fp8
        - output_for_group_gemm: allocate output for group gemm
            Shape: [num_seq * num_topk, intermediate_size]
            Dtype: bfloat16
        - topk_pos: Position indices mapping each expert output back to original sequence
            Shape: [num_seq * num_topk]
            Dtype: int32
        - seqlens: Number of token assigned to each expert
            Shape: [num_expert]
            Dtype: int32
        - cu_seqlens: Cumulative token counts for expert indexing
            Shape: [num_expert + 1]
            Dtype: int32
        - tiles: Number of tiles assigned to each expert
            Shape: [num_expert]
            Dtype: int32
        - cu_tiles: Cumulative tiles counts for expert indexing
            Shape: [num_expert + 1]
            Dtype: int32
        - tmas: Describe the tma info of output
            Shape: [num_expert * 2 * 128]
            Dtype: int8

    Raises:
        RuntimeError: If input tensors have incompatible shapes or types,
            if expert assignments exceed available expert count,
            or if buffer sizes are insufficient for output tensors.

    Note:
        - This function is designed for expert-parallel distributed training
        - All input and output tensors must reside on the same device
        - The function modifies the output buffers in-place when provided
        - Expert assignments in topk_ids should be in range [0, num_expert-1]
    """
    return torch.ops.hpc.count_and_gather(x, topk_ids, num_expert, rank_ep, intermediate_size)


def reduce(
    x: Tensor,
    topk_pos: Tensor,
    topk_scale: Tensor,
    shared_output: Tensor = None,
) -> Tensor:
    """reduce token based on expert assignments for MoE layers.

    This kernel implements the reduction (scatter-add) operation in the final stage
    of Mixture of Experts (MoE) computation. It aggregates the weighted contributions
    from selected experts back to the original sequence positions.

    Args:
        x: Input token features tensor from expert processing
            Shape: [total_num_seq, hidden_size]
            Dtype: bfloat16
        topk_pos: Position indices mapping each expert output back to original sequence
            Shape: [num_seq, num_topk]
            Dtype: int32
        topk_scale: Scaling factors for expert outputs (typically gating scores)
            Shape: [num_seq, num_topk]
            Dtype: float32
        shared_output: output for shared experts, default is None
            Shape: [num_seq, hidden_size]
            Dtype: bfloat16

    Returns:
        Tensor: Reduced output tensor containing aggregated expert contributions
            Shape: [num_seq, hidden_size]
            Dtype: bfloat16

    Raises:
        RuntimeError: If tensor shapes are incompatible or CUDA kernel execution fails.
        ValueError: If input tensors are on different devices or have unsupported dtypes.

    Note:
        - This is a performance-critical kernel optimized for MoE workloads
        - The operation is equivalent to a weighted scatter-add operation
        - Input tensors must be contiguous and on the same device
        - For best performance, hidden_size should be a multiple of 32
        - topk_pos values must be in range [0, total_num_seq-1]
    """
    return torch.ops.hpc.reduce(x, topk_pos, topk_scale, shared_output)


# Dispatch cp.async backend when intermediate_size is small (TP=8 regime);
# otherwise fall back to the TMA backend.
_CP_ASYNC_N_TP_MAX = 512


def fuse_moe(
    x: Tensor,
    gate_up_weight: Tensor,
    down_weight: Tensor,
    gate_up_scale: Tensor,
    down_scale: Tensor,
    act_and_mul_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    use_bf16_mul: bool = True,
    shared_output: Tensor = None,
    output: Tensor = None,
) -> Tensor:
    """Performs Mixture of Experts (MoE) forward operation with FP8 precision.

    This function executes the MoE computation with all matrix multiplications
    performed in FP8 precision for improved performance and memory efficiency.
    The gate and up projections are fused into a single matrix multiplication.

    Dispatches to the cp.async backend when ``intermediate_size <= 512``,
    ``intermediate_size % 64 == 0``, and ``hidden_size % 64 == 0``;
    otherwise dispatches to the TMA backend.  Call ``fuse_moe_cp_async``
    directly to force the cp.async path.

    Args:
        x: Input activation tensor
            Shape: [num_seq, hidden_size]
            Dtype: fp8
        gate_up_weight: Combined weight tensor for gate and up projections
            Shape: [num_expert_local, intermediate_size * 2, hidden_size]
            Dtype: fp8
        down_weight: Weight tensor for down projection
            Shape: [num_expert_local, hidden_size, intermediate_size]
            Dtype: fp8
        gate_up_scale: Scaling factors for gate-up projection outputs
            Shape: [num_expert_local]
            Dtype: float32
        down_scale: Scaling factors for down projection outputs
            Shape: [num_expert_local]
            Dtype: float32
        act_and_mul_scale: Scaling factor for activation and multiplication
            Shape: [1]
            Dtype: float32
        topk_ids: Token indices assigned to each expert
            Shape: [num_seq, num_topk]
            Dtype: int32
        topk_scale: Weighting factors for each token-expert assignment
            Shape: [num_seq, num_topk]
            Dtype: float32
        rank_ep: Expert parallel rank (for distributed training)
            Dtype: int32
        num_expert_total: the total number of expert
            Dtype: int32
        use_bf16_mul: use bf16 for silu mul or not.
        shared_output: output for shared experts, default is None
            Shape: [num_seq, hidden_size]
            Dtype: bfloat16
        output: specify output tensor.
            Shape: [num_seq, hidden_size]
            Dtype: bfloat16

    Returns:
        torch.Tensor: Output tensor after MoE computation
            Shape: [num_seq, hidden_size]
            Dtype: bfloat16

    Raises:
        RuntimeError: If the input tensors have incompatible shapes or types,
            or if CUDA kernel execution fails.
        ValueError: If the intermediate_size is not divisible by 2 for gate/up split.

    Note:
        - All input tensors must be on CUDA device
        - The gate and up projections are combined into a single matrix multiplication
        - FP8 precision is used for all matrix operations (torch.float8_e4m3fn)
        - Activation function used is SiLU (Swish)
        - Token routing is determined by topk_ids and weighted by topk_scale
        - Output scaling is applied to maintain numerical stability in FP8
    """
    intermediate_size = gate_up_weight.shape[1] // 2
    if (
        intermediate_size <= _CP_ASYNC_N_TP_MAX
        and intermediate_size % 64 == 0
        and x.shape[1] % 64 == 0
    ):
        return torch.ops.hpc.fuse_moe_cp_async(
            x,
            gate_up_weight,
            down_weight,
            gate_up_scale,
            down_scale,
            act_and_mul_scale,
            topk_ids,
            topk_scale,
            shared_output,
            rank_ep,
            num_expert_total,
            use_bf16_mul,
            output,
        )

    return torch.ops.hpc.fuse_moe(
        x,
        gate_up_weight,
        down_weight,
        gate_up_scale,
        down_scale,
        act_and_mul_scale,
        topk_ids,
        topk_scale,
        shared_output,
        rank_ep,
        num_expert_total,
        use_bf16_mul,
        output,
    )


def fuse_moe_blockwise(
    x: Tensor,
    x_scale: Tensor,
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    shared_output: Tensor = None,
    output: Tensor = None,
) -> Tensor:
    """Performs Mixture of Experts (MoE) forward operation with FP8 precision.

    It only supports blockwise quantization of weights and inputs, with a block
    size of 128.

    This function executes the MoE computation with all matrix multiplications
    performed in FP8 precision for improved performance and memory efficiency.
    The gate and up projections are fused into a single matrix multiplication.

    Args:
        x: Input activation tensor
            Shape: [num_tokens, hidden_size]
            Dtype: fp8
        x_scale: Scaling factors for input activation
            Shape: [num_tokens, hidden_size / 128]
            Dtype: fp32
        gate_up_weight: Combined weight tensor for gate and up projections
            Shape: [num_expert_local, intermediate_size * 2, hidden_size]
            Dtype: fp8
        gate_up_weight_scale: Scaling factors for gate_up_weight, should pad the last dim to 64,
                              so hidden_size must be smaller than 64*128
            Shape: [num_expert_local, intermediate_size * 2 / 128, 64]
            Dtype: fp32
        down_weight: Weight tensor for down projection
            Shape: [num_expert_local, hidden_size, intermediate_size]
            Dtype: fp8
        down_weight_scale: Scaling factors for down_weight, should pad the last dim to 64,
                           so intermediate_size must be smaller than 64*128
            Shape: [num_expert_local, hidden_size / 128, 64]
            Dtype: float32
        topk_ids: Token indices assigned to each expert
            Shape: [num_tokens, num_topk]
            Dtype: int32
        topk_scale: Weighting factors for each token-expert assignment
            Shape: [num_tokens, num_topk]
            Dtype: float32
        rank_ep: Expert parallel rank (for distributed training)
            Dtype: int32
        num_expert_total: the total number of expert
            Dtype: int32
        shared_output: output for shared experts, default is None
            Shape: [num_seq, hidden_size]
            Dtype: bfloat16
        output: specify output tensor.
            Shape: [num_seq, hidden_size]
            Dtype: bfloat16
    Returns:
        torch.Tensor: Output tensor after MoE computation
            Shape: [num_tokens, hidden_size]
            Dtype: bfloat16
    """
    return torch.ops.hpc.fuse_moe_blockwise(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        shared_output,
        rank_ep,
        num_expert_total,
        output,
    )


def fuse_moe_groupwise_w4a8(
    x: Tensor,
    gate_up_weight: Tensor,
    gate_up_scale: Tensor,
    down_weight: Tensor,
    down_scale: Tensor,
    act_and_mul_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    gateup_group_size: int,
    down_group_size: int,
    rank_ep: int,
    num_expert_total: int,
    use_hadamard: bool,
    shared_output: Tensor = None,
    output: Tensor = None,
) -> Tensor:
    """Performs Mixture of Experts (MoE) forward operation with weight int4 and activation fp8 precision.

    It only supports static groupwise quantization of weights and static per tensor quantization inputs, with a group size of 64/128.

    This function executes the MoE computation with all matrix multiplications
    performed in weight int4 and activation fp8 precision for improved performance and memory efficiency.
    The gate and up projections are fused into a single matrix multiplication.

    Args:
        x: Input activation tensor
            Shape: [num_tokens, hidden_size]
            Dtype: fp8
        gate_up_weight: Combined weight tensor for gate and up projections
            Shape: [num_expert_local, intermediate_size * 2, hidden_size // 2]
            Dtype: int8
        gate_up_scale: Scaling factors for gate-up projection outputs, which combine input scale with weight scale, and should be pad to 16 bytes.
            Shape: [num_expert_local, intermediate_size * 2, (hidden_size // gate_group_size + 7) // 8 * 8]
            Dtype: bfloat16
        down_weight: Weight tensor for down projection
            Shape: [num_expert_local, hidden_size, intermediate_size // 2]
            Dtype: int8
        down_scale: Scaling factors for down projection outputs, which combine input scale with weight scale, and should be pad to 16 bytes.
            Shape: [num_expert_local, hidden_size, (intermediate_size // down_group_size + 7) // 8 * 8]
            Dtype: bfloat16
        act_and_mul_scale: Scaling factor for activation and multiplication
            Shape: [1]
            Dtype: float32
        topk_ids: Token indices assigned to each expert
            Shape: [num_tokens, num_topk]
            Dtype: int32
        topk_scale: Weighting factors for each token-expert assignment
            Shape: [num_tokens, num_topk]
            Dtype: float32
        gateup_group_size: group size of gate up weight groupwise quant, only support 64/128
            Dtype: int32
        down_group_size: group size of down weight groupwise quant, only support 64/128
            Dtype: int32
        rank_ep: Expert parallel rank (for distributed training)
            Dtype: int32
        num_expert_total: the total number of expert
            Dtype: int32
        use_hadamard: if use hadamard transform for activation output and down projection weight
            Dtype: bool
        shared_output: output for shared experts, default is None
            Shape: [num_tokens, hidden_size]
            Dtype: bfloat16
        output: specify output tensor.
            Shape: [num_tokens, hidden_size]
            Dtype: bfloat16
    Returns:
        torch.Tensor: Output tensor after MoE computation
            Shape: [num_tokens, hidden_size]
            Dtype: bfloat16
    """
    return torch.ops.hpc.fuse_moe_groupwise_w4a8(
        x,
        gate_up_weight,
        gate_up_scale,
        down_weight,
        down_scale,
        act_and_mul_scale,
        topk_ids,
        topk_scale,
        gateup_group_size,
        down_group_size,
        rank_ep,
        num_expert_total,
        use_hadamard,
        shared_output,
        output,
    )


@torch.library.register_fake("hpc::count_and_gather")
def count_and_gather_fake(
    x, topk_ids, num_expert, rank_ep, intermediate_size, num_seq_per_group_avg
):
    return (
        torch.empty((topk_ids.shape[0] * topk_ids.shape[1], x.shape[1]), dtype=torch.float8_e4m3fn),
        torch.empty(
            (topk_ids.shape[0] * topk_ids.shape[1], intermediate_size), dtype=torch.bfloat16
        ),
        torch.empty((topk_ids.shape[0] * topk_ids.shape[1]), dtype=torch.int32),
        torch.empty((num_expert), dtype=torch.int32),
        torch.empty((num_expert + 1), dtype=torch.int32),
        torch.empty((num_expert), dtype=torch.int32),
        torch.empty((num_expert + 1), dtype=torch.int32),
        torch.empty((num_expert * 2 * 128), dtype=torch.int8),
    )


@torch.library.register_fake("hpc::reduce")
def reduce_fake(x, topk_pos, topk_scale):
    return torch.empty((topk_pos.shape[0], x.shape[1]), dtype=torch.bfloat16)


@torch.library.register_fake("hpc::fuse_moe")
def fuse_moe_fake(
    x: Tensor,
    gate_up_weight: Tensor,
    down_weight: Tensor,
    gate_up_scale: Tensor,
    down_scale: Tensor,
    act_and_mul_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    use_bf16_mul: bool = True,
    shared_output: Tensor = None,
    output: Tensor = None,
):
    return torch.empty((x.shape[0], x.shape[1]), dtype=torch.bfloat16, device=x.device)


@torch.library.register_fake("hpc::fuse_moe_blockwise")
def fuse_moe_blockwise_fake(
    x: Tensor,
    x_scale: Tensor,
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    shared_output: Tensor = None,
    output: Tensor = None,
):
    return torch.empty((x.shape[0], x.shape[1]), dtype=torch.bfloat16, device=x.device)


def fuse_moe_cp_async(
    x: Tensor,
    gate_up_weight: Tensor,
    down_weight: Tensor,
    gate_up_scale: Tensor,
    down_scale: Tensor,
    act_and_mul_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int,
    num_expert_total: int,
    use_bf16_mul: bool = False,
    shared_output: Tensor = None,
    output: Tensor = None,
) -> Tensor:
    """Performs Mixture of Experts (MoE) forward operation with FP8 precision,
    using cp.async based group GEMM kernels.

    This is an alternative implementation to `fuse_moe` that issues all
    global-to-shared memory transfers via cp.async instead of TMA.  The
    numerical behaviour matches `fuse_moe` bit-for-bit; the gate and up
    projections are fused into a single matrix multiplication.

    Args:
        x: Input activation tensor
            Shape: [num_seq, hidden_size]
            Dtype: fp8
        gate_up_weight: Combined weight tensor for gate and up projections
            Shape: [num_expert_local, intermediate_size * 2, hidden_size]
            Dtype: fp8
            `intermediate_size * 2` is the fused gate+up dimension; the
            scatter GEMM writes [gate | up] into adjacent N-halves.
        down_weight: Weight tensor for down projection
            Shape: [num_expert_local, hidden_size, intermediate_size]
            Dtype: fp8
        gate_up_scale: Scaling factors for gate-up projection outputs
            Shape: [num_expert_local]
            Dtype: float32
        down_scale: Scaling factors for down projection outputs
            Shape: [num_expert_local]
            Dtype: float32
        act_and_mul_scale: Scaling factor for activation and multiplication
            Shape: [1]
            Dtype: float32
        topk_ids: Token indices assigned to each expert
            Shape: [num_seq, num_topk]
            Dtype: int32
        topk_scale: Weighting factors for each token-expert assignment
            Shape: [num_seq, num_topk]
            Dtype: float32
        rank_ep: Expert parallel rank (for distributed training)
            Dtype: int32
        num_expert_total: the total number of experts across all EP ranks
            Dtype: int32
        use_bf16_mul: whether to perform silu(gate) * up in bf16.  The
            default (False) keeps full fp32 precision through the
            activation; setting True lowers the intermediate precision
            and is slightly faster for activation-bound shapes.
        shared_output: output for shared experts, default is None
            Shape: [num_seq, hidden_size]
            Dtype: bfloat16
        output: specify output tensor.
            Shape: [num_seq, hidden_size]
            Dtype: bfloat16

    Returns:
        torch.Tensor: Output tensor after MoE computation
            Shape: [num_seq, hidden_size]
            Dtype: bfloat16

    Raises:
        RuntimeError: If the input tensors have incompatible shapes or types,
            or if CUDA kernel execution fails.
        ValueError: If the intermediate_size is not divisible by 2 for gate/up split.

    Note:
        - All input tensors must be on CUDA device.
        - The gate and up projections are combined into a single matrix multiplication.
        - FP8 precision is used for all matrix operations (torch.float8_e4m3fn).
        - Activation function used is SiLU (Swish).
        - Token routing is determined by topk_ids and weighted by topk_scale.
        - Output scaling is applied to maintain numerical stability in FP8.
        - num_expert_local must be <= 512 (routing prefix-sum kernel capacity).
    """
    return torch.ops.hpc.fuse_moe_cp_async(
        x,
        gate_up_weight,
        down_weight,
        gate_up_scale,
        down_scale,
        act_and_mul_scale,
        topk_ids,
        topk_scale,
        shared_output,
        rank_ep,
        num_expert_total,
        use_bf16_mul,
        output,
    )
