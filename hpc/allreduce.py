import math
import torch
from typing import Tuple, Any, Optional, Sequence
from hpc.multicast_handle import MulticastHandle
from hpc.multinode_handle import MultiNodeHandle


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


def fuse_allreduce_rmsnorm_v2(
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
    output_x: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = True,
) -> None:
    """Do Allreduce, Residual Add and Res RMSNorm using GPU kernel (v2).

    Executes RMSNorm((Allreduce(input_x)+residual_in), weight_gamma, rms_norm_eps)
    in a custom GPU kernel for optimized performance. Compared to
    ``fuse_allreduce_rmsnorm``, this v2 variant relies on a symmetric Lamport
    workspace (managed by :class:`MultiNodeHandle`) and a set of buffer flags
    instead of an explicit signal buffer, so it supports multi-node scenarios
    and does not require ``num_max_blocks`` as an input.

    Args:
        input_x: input tensor,
            Shape: [batch, hidden_size]
            Dtype: torch.bfloat16
        multicast_x: the multicast ptr of the symmetric workspace tensor
            Shape: [M_pad, hidden_size]
            Dtype: torch.bfloat16
        data_buffer_ptrs: device tensor holding the per-rank unicast pointers
            of the symmetric workspace (``workspace_hdl.data_buffer_ptrs_dev``).
            Shape: [world_size]
            Dtype: torch.int64
        multinode_x: the local view of the symmetric Lamport workspace tensor
            allocated by :func:`create_workspace_for_fuse_ar_rms_v2`.
            Shape: [M_pad, hidden_size]
            Dtype: torch.bfloat16
        buffer_flags: Lamport buffer flags tensor produced by
            :func:`create_workspace_for_fuse_ar_rms_v2`, used by the kernel to
            track the current/dirty buffer indices and the bytes to clear.
            Dtype: torch.uint32
        world_size: the number of ranks in the parallel group
        rank: the idx of the current rank in the parallel group
        residual_in: residual tensor to be added after Allreduce,
            Shape: [batch, hidden_size]
            Dtype: torch.bfloat16
        weight_gamma: rmsnorm weight tensor,
            Shape: [hidden_size]
            Dtype: torch.bfloat16
        rms_norm_eps: epsilon argument of rmsnorm
        output_x: output tensor of the fused op. If ``None``, ``input_x`` is
            reused as the output buffer.
            Shape: [batch, hidden_size]
            Dtype: torch.bfloat16
        residual_out: output residual tensor (Allreduce(input_x) + residual_in).
            If ``None``, ``residual_in`` is reused as the output buffer.
            Shape: [batch, hidden_size]
            Dtype: torch.bfloat16
        launch_with_pdl: whether to launch the kernel with Programmatic
            Dependent Launch (PDL). Default: ``True``.
    """
    if output_x is None:
        output_x = input_x
    if residual_out is None:
        residual_out = residual_in
    torch.ops.hpc.fuse_allreduce_rmsnorm_v2(
        input_x,
        multicast_x,
        data_buffer_ptrs,
        multinode_x,
        buffer_flags,
        residual_in,
        weight_gamma,
        rank,
        world_size,
        rms_norm_eps,
        launch_with_pdl,
        output_x,
        residual_out,
    )


def create_workspace_for_fuse_ar_rms_v2(
    comm,
    N: int,
    H: int,
    world_size: int,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple["MultiNodeHandle", torch.Tensor, torch.Tensor, Sequence[int]]:
    """Create a symmetric workspace handle, buffer and buffer_flags for ``fuse_allreduce_rmsnorm_v2``.

    Args:
        comm: MultiNodeCommunicator instance.
        N: Number of tokens (rows) before padding.
        H: Hidden size (columns).
        world_size: Number of ranks participating in the collective.
        dtype: Tensor dtype of the workspace. Default: ``torch.bfloat16``.

    Returns:
        workspace_hdl: The :class:`MultiNodeHandle` managing the symmetric memory.
        symm_workspace: The symmetric workspace tensor, initialized with ``0x80000000``.
        buffer_flags: The Lamport buffer flags tensor (``torch.uint32``) required by
            ``fuse_allreduce_rmsnorm_v2``.
        workspace_size: The shape of the workspace tensor as ``[M_pad, H]``.
    """
    _NUM_LAMPORT_BUFFERS = 3
    M_pad = 2 * math.ceil(N / world_size) * world_size * _NUM_LAMPORT_BUFFERS
    workspace_size = [M_pad, H]

    workspace_hdl = MultiNodeHandle(comm, workspace_size, dtype=dtype)

    symm_workspace = workspace_hdl.get_multinode_buff(workspace_size, dtype=dtype)
    symm_workspace.view(torch.uint32).fill_(0x80000000)

    elem_size = torch.tensor([], dtype=dtype).element_size()
    workspace_size_bytes = M_pad * H * elem_size
    lamport_buffer_size_bytes = math.floor(workspace_size_bytes / _NUM_LAMPORT_BUFFERS) // 16 * 16
    num_bytes_to_clear = [0] * 4
    buffer_flags = torch.tensor(
        [0, 2, lamport_buffer_size_bytes, 0, *num_bytes_to_clear, 0],
        dtype=torch.uint32,
        device=torch.device("cuda", torch.cuda.current_device()),
    )

    return workspace_hdl, symm_workspace, buffer_flags, workspace_size


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
    if out_residual is None:
        out_residual = residual
    torch.ops.hpc.fuse_allreduce_rmsnorm_with_scale(
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


def fuse_reduce_scatter_rmsnorm(
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
    """Do ReduceScatter, Residual Add and Res RMSNorm using GPU kernel.

    Executes RMSNorm((ReduceScatter(x)+residual), weight, rms_norm_eps)
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

    torch.ops.hpc.fuse_reduce_scatter_rmsnorm(
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


def reduce_scatter(
    x: torch.Tensor,
    multicast_x: torch.Tensor,
    signal: torch.Tensor,
    rank: int,
    world_size: int,
    num_max_blocks: int,
    output_x: Optional[torch.Tensor] = None,
    output_multicast_x: Optional[torch.Tensor] = None,
) -> None:
    """Do ReduceScatter using GPU kernel.

    Executes ReduceScatter(x) in a custom GPU kernel for optimized performance.

    Args:
        x: input tensor,
            Shape: [batch, hidden_size]
            Dtype: torch.bfloat16
        multicast_x: the multicast ptr of x
            Shape: [batch, hidden_size]
            Dtype: torch.bfloat16
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
    """
    if output_x is None:
        output_x = x
    if output_multicast_x is None:
        output_multicast_x = multicast_x

    torch.ops.hpc.reduce_scatter(
        x,
        multicast_x,
        signal,
        rank,
        world_size,
        num_max_blocks,
        output_x,
        output_multicast_x,
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


@torch.library.register_fake("hpc::fuse_allreduce_rmsnorm_v2")
def fuse_allreduce_rmsnorm_v2_fake(
    input_x: torch.Tensor,
    multicast_x: torch.Tensor,
    data_buffer_ptrs: torch.Tensor,
    multinode_x: torch.Tensor,
    buffer_flags: torch.Tensor,
    residual_in: torch.Tensor,
    weight_gamma: torch.Tensor,
    rank: int,
    world_size: int,
    rms_norm_eps: float,
    launch_with_pdl: bool,
    output_x: torch.Tensor,
    residual_out: torch.Tensor,
) -> None:
    return None


@torch.library.register_fake("hpc::fuse_allreduce_rmsnorm_with_scale")
def fuse_allreduce_rmsnorm_with_scale_fake(
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
) -> None:
    return None


@torch.library.register_fake("hpc::fuse_reduce_scatter_rmsnorm")
def fuse_reduce_scatter_rmsnorm_fake(
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


@torch.library.register_fake("hpc::reduce_scatter")
def reduce_scatter_fake(
    x: torch.Tensor,
    multicast_x: torch.Tensor,
    signal: torch.Tensor,
    rank: int,
    world_size: int,
    num_max_blocks: int,
    output_x: Optional[torch.Tensor] = None,
    output_multicast_x: Optional[torch.Tensor] = None,
) -> None:
    return None
