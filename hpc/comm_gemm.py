import torch
from torch import Tensor, ceil


def get_fuse_gemm_rs_signal_size(m_dim: int, n_dim: int, num_comp_sm: int) -> int:
    """Compute the signal buffer size required by fuse_gemm_reduce_scatter.

    Args:
        m_dim: M dimension of the GEMM (number of rows of input x).
        n_dim: N dimension of the GEMM (number of rows of weight, i.e. output width).
        num_comp_sm: Number of SMs dedicated to computation (total_sm - num_comm_sm).

    Returns:
        int: Required signal buffer size (number of uint64 elements).
    """
    tiles = (m_dim // 64) * (n_dim // 128)
    return 2 * ((tiles + num_comp_sm - 1) // num_comp_sm) + 1


def fuse_gemm_reduce_scatter(
    x: Tensor,
    weight: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    trans_xscale: bool,
    bias: Tensor,
    output: Tensor,
    signal: Tensor,
    multimem_output: Tensor,
    multimem_signal: Tensor,
    num_comp_sm: int,
    num_comm_sm: int,
    rank: int,
    world_size: int,
) -> Tensor:
    """Fused FP8 blockwise GEMM with intra-node reduce-scatter via multicast.

    Overlaps computation and communication by partitioning SMs into compute SMs
    and communication SMs. The compute SMs perform the blockwise FP8 GEMM while
    the communication SMs perform the reduce-scatter using NVLink multicast as
    tiles complete, hiding communication latency behind computation.

    You can refer to the test 'tests/test_fuse_gemm_reduce_scatter.py'.

    Args:
        x: Input activation tensor.
            Shape: [m, k]
            Dtype: fp8 (float8_e4m3fn)
        weight: Weight tensor.
            Shape: [n, k]
            Dtype: fp8 (float8_e4m3fn)
        x_scale: Blockwise scaling factor for x (block_size=128).
            Shape: [m, k // 128]
            Dtype: fp32
        w_scale: Blockwise scaling factor for weight (block_size=128).
            Shape: [n // 128, aligned_to_4(k // 128)]
            Dtype: fp32
        trans_xscale: Whether to transpose x_scale internally.
            When True, the kernel transposes [m, k//128] -> [k//128, aligned_to_4(m)].
        bias: Optional bias tensor.
            Shape: [n] or None
            Dtype: fp32
        output: Pre-allocated output buffer (also used as the multicast symmetric buffer).
            Shape: [m, n]
            Dtype: bfloat16
        signal: Signal buffer for synchronization (local rank's signal).
            Shape: [signal_size]
            Dtype: uint64
            Obtained via MulticastHandle.get_buffer().
        multimem_output: Multicast handle over the output buffer.
            Shape: [m, n]
            Dtype: bfloat16
            Obtained via MulticastHandle.get_multimem_buff().
        multimem_signal: Multicast handle over the signal buffer.
            Shape: [signal_size]
            Dtype: uint64
            Obtained via MulticastHandle.get_multimem_buff().
        num_comp_sm: Number of SMs allocated for GEMM computation.
        num_comm_sm: Number of SMs allocated for reduce-scatter communication.
            num_comp_sm + num_comm_sm should equal the total available SMs (e.g. 78).
        rank: Local rank index within the communication group.
        world_size: Number of ranks in the communication group.

    Returns:
        Tensor: GEMM output after reduce-scatter.
            Shape: [m, n]
            Dtype: bfloat16
            The local rank's reduced chunk is at output[rank * (m // world_size) : (rank+1) * (m // world_size), :].
    """
    return torch.ops.hpc.fuse_gemm_reduce_scatter(
        x,
        weight,
        x_scale,
        w_scale,
        trans_xscale,
        bias,
        output,
        signal,
        multimem_output,
        multimem_signal,
        num_comp_sm,
        num_comm_sm,
        rank,
        world_size,
    )


@torch.library.register_fake("hpc::fuse_gemm_reduce_scatter")
def fuse_gemm_reduce_scatter_fake(
    x,
    weight,
    x_scale,
    w_scale,
    trans_xscale,
    bias,
    output,
    signal,
    multimem_output,
    multimem_signal,
    num_comp_sm,
    num_comm_sm,
    rank,
    world_size,
):
    return torch.empty((x.shape[0], weight.shape[0]), dtype=output.dtype, device=output.device)
