import torch
from typing import Tuple, Any, Optional, Sequence
from itertools import accumulate
from operator import mul
from hpc.multicast_handle import MulticastHandle


def fuse_allreduce_rmsnorm(
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
    """Do Allreduce, Residual Add and Res RMSNorm using GPU kernel.

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
    torch.ops.hpc.fuse_allreduce_rmsnorm(
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


def fuse_allreduce_rmsnorm_with_scale(
    x: torch.Tensor,
    multicast_x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    rms_norm_eps: float,
    scale: torch.Tensor,
    output: torch.Tensor,
    signal: torch.Tensor,
    rank: int,
    world_size: int,
    num_max_blocks: int,
    is_moe: bool,
    out_residual: Optional[torch.Tensor] = None,
    scale2: Optional[torch.Tensor] = None,
    output2: Optional[torch.Tensor] = None,
    output_fp32: Optional[torch.Tensor] = None,
):
    """Do Allreduce, Residual Add and Res RMSNorm using GPU kernel.

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
    if out_residual is None:
        out_residual = residual
    return torch.ops.hpc.fuse_allreduce_rmsnorm_with_scale(
        x,
        multicast_x,
        residual,
        weight,
        scale,
        output,
        signal,
        rank,
        world_size,
        num_max_blocks,
        rms_norm_eps,
        is_moe,
        out_residual,
        scale2,
        output2,
        output_fp32,
    )


def empty_multimem(
    multicomm,
    *size: Any,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, MulticastHandle]:
    """
    empty_multimem(multicomm, *size, *, dtype=None, device=None) -> Tensor

    Similar to :func:`torch.empty()`. Return a Tensor which can be access by multimem
    ptx code of other Process.

    Args:
        multicomm(MulticastCommunicator): A multicast communicator for distributed tensor operations
        size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_device`). :attr:`device` will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.
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
            return device.index if device.index is not None else 0  # default is 0
        else:
            return -1  # CPU

    assert (
        device_to_num(device) == multicomm.GetDeviceId()
    ), f"device(got {device_to_num(device)}) of alloc buffer must be same with multicomm(got {multicomm.GetDeviceId()})"

    hdl = MulticastHandle(multicomm, size, dtype)

    return hdl.get_buffer(hdl.rank, size, dtype=dtype), hdl


@torch.library.register_fake("hpc::fuse_allreduce_rmsnorm")
def fuse_allreduce_rmsnorm_fake(
    x,
    mc_input,
    in_residual,
    weight,
    rms_norm_eps,
    signal,
    rank,
    world_size,
    num_max_blocks,
    output,
    mc_output,
    out_residual,
):
    return torch.empty_like(x)


@torch.library.register_fake("hpc::fuse_allreduce_rmsnorm_with_scale")
def fuse_allreduce_rmsnorm_with_scale_fake(
    x,
    multicast_x,
    residual,
    weight,
    rms_norm_eps,
    scale,
    fp8_output,
    signal,
    rank,
    world_size,
    is_moe,
    num_max_blocks,
    out_residual=None,
    scale2=None,
    fp8_output2=None,
    fp32_output=None,
):
    return torch.empty_like(x)
