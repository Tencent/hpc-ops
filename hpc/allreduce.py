import torch
from typing import Optional, Any, Sequence, Tuple

from hpc.multicast_handle import MulticastHandle


def fuse_allreduce_rmsnorm_high_throughput(
    x: torch.Tensor,
    multicast_x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    rms_norm_eps: float,
    signal: torch.Tensor,
    rank: int,
    world_size: int,
    num_max_blocks: int,
    output_x: Optional[torch.Tensor] = None,
    output_multicast_x: Optional[torch.Tensor] = None,
    output_residual: Optional[torch.Tensor] = None,
) -> None:
    """High-throughput fused Allreduce + Residual Add + RMSNorm GPU kernel.

    Executes RMSNorm((Allreduce(x)+residual), weight, rms_norm_eps)
    in a custom GPU kernel for optimized performance.

    Args:
        x: input tensor,
            Shape: [batch, hidden_size]
            Dtype: torch.bfloat16
        multicast_x: the multicast ptr of x
            Shape: [batch, hidden_size]
            Dtype: torch.bfloat16
        residual: residual tensor,
            Shape: [batch, hidden_size]
            Dtype: torch.bfloat16
        weight: rmsnorm weight tensor,
            Shape: [hidden_size]
            Dtype: torch.bfloat16
        rms_norm_eps: argument of rmsnorm
        signal: the signal buffer pointer of all rank in device
            Shape: [world_size]
            Dtype: torch.int64
        rank: the idx of parallel group
        world_size: the number of rank in parallel group
        num_max_blocks: the max number of ctas using by kernel
        output_x: output tensor
            Shape: [batch, hidden_size]
            Dtype: torch.bfloat16
        output_multicast_x: the multicast ptr of output_x
            Shape: [batch, hidden_size]
            Dtype: torch.bfloat16
        output_residual: output residual tensor
            Shape: [batch, hidden_size]
            Dtype: torch.bfloat16
    """
    if output_x is None:
        output_x = x
    if output_multicast_x is None:
        output_multicast_x = multicast_x
    if output_residual is None:
        output_residual = residual
    torch.ops.hpc.fuse_allreduce_rmsnorm_high_throughput(
        x,
        multicast_x,
        residual,
        weight,
        signal,
        rank,
        world_size,
        num_max_blocks,
        rms_norm_eps,
        output_x,
        output_multicast_x,
        output_residual,
    )


def fuse_allreduce_rmsnorm_low_latency(
    input_x: torch.Tensor,
    multicast_x: torch.Tensor,
    data_buffer_ptrs: torch.Tensor,
    multinode_x: torch.Tensor,
    buffer_flags: torch.Tensor,
    world_size: int,
    rank: int,
    residual_in: torch.Tensor,
    weight_gamma: torch.Tensor,
    rms_norm_eps: float,
    num_max_blocks: int,
    output_x: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = True,
) -> None:
    # TODO(draken): Update explanation
    """Low-latency fused Allreduce + Residual Add + RMSNorm GPU kernel.

    Executes RMSNorm((Allreduce(x)+residual), weight, rms_norm_eps)
    in a custom GPU kernel for optimized performance.

    Args:
    """
    # TODO(draken): Support limit num_max_blocks
    if output_x is None:
        output_x = input_x
    if residual_out is None:
        residual_out = residual_in
    torch.ops.hpc.fuse_allreduce_rmsnorm_low_latency(
        input_x,
        multicast_x,
        data_buffer_ptrs,
        multinode_x,
        buffer_flags,
        world_size,
        rank,
        True,  # RMSNorm Fusion
        launch_with_pdl,
        True,  # use two-Shot AllReduce
        output_x,
        residual_out,
        residual_in,
        weight_gamma,
        rms_norm_eps,
    )


@torch.library.register_fake("hpc::fuse_allreduce_rmsnorm_high_throughput")
def fuse_allreduce_rmsnorm_high_throughput_fake(
    x: torch.Tensor,
    multicast_x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    rms_norm_eps: float,
    signal: torch.Tensor,
    rank: int,
    world_size: int,
    num_max_blocks: int,
    output_x: Optional[torch.Tensor] = None,
    output_multicast_x: Optional[torch.Tensor] = None,
    output_residual: Optional[torch.Tensor] = None,
) -> None:
    return None


@torch.library.register_fake("hpc::fuse_allreduce_rmsnorm_low_latency")
def fuse_allreduce_rmsnorm_low_latency_fake(
    input_x: torch.Tensor,
    multicast_x: torch.Tensor,
    data_buffer_ptrs: torch.Tensor,
    multinode_x: torch.Tensor,
    buffer_flags: torch.Tensor,
    world_size: int,
    rank: int,
    residual_in: torch.Tensor,
    weight_gamma: torch.Tensor,
    rms_norm_eps: float,
    num_max_blocks: int,
    output_x: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = True,
) -> None:
    return None


def empty_multimem(
    multicomm,
    *size: Any,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, MulticastHandle]:
    """Allocate a symmetric multicast (multimem) buffer over single-node NVLink.

    Returns the local tensor for this rank plus a :class:`MulticastHandle` that
    exposes the multicast pointer and the per-rank buffer pointers needed by the
    fused allreduce kernels. No NVSHMEM dependency.
    """
    if len(size) == 1 and isinstance(size[0], Sequence):
        size = tuple(size[0])
    else:
        size = tuple(size)

    if dtype is None:
        dtype = torch.get_default_dtype()

    if device is None:
        device = torch.get_default_device()

    def device_to_num(device):
        if device.type == "cuda":
            return device.index if device.index is not None else 0
        else:
            return -1

    assert device_to_num(device) == multicomm.GetDeviceId(), (
        f"device(got {device_to_num(device)}) of alloc buffer must be same with "
        f"multicomm(got {multicomm.GetDeviceId()})"
    )

    hdl = MulticastHandle(multicomm, size, dtype)

    return hdl.get_buffer(hdl.rank, size, dtype=dtype), hdl
