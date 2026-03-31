import torch
from typing import Tuple, Any, Optional, Sequence
from hpc.multinode_handle import MultiNodeHandle


def fuse_allreduce_dispatch(
    x: torch.Tensor,
    multicast_x: torch.Tensor,
    multinode_x: torch.Tensor,
    signal: torch.Tensor,
    rank: int,
    local_size: int,
    world_size: int,
    attn_dp_size: int,
    attn_tp_size: int,
    moe_ep_size: int,
    moe_tp_size: int,
    num_max_blocks: int,
    output_x: torch.Tensor,
    output_multicast_x: torch.Tensor,
    output_multinode_x: torch.Tensor,
    output_multinode_signal: torch.Tensor,
    world_rank: int,
    batch_size: int,
    num_qp: int,
) -> None:
    torch.ops.hpc.fuse_allreduce_dispatch(
        x,
        multicast_x,
        multinode_x,
        signal,
        rank,
        local_size,
        world_size,
        attn_dp_size,
        attn_tp_size,
        moe_ep_size,
        moe_tp_size,
        num_max_blocks,
        output_x,
        output_multicast_x,
        output_multinode_x,
        output_multinode_signal,
        world_rank,
        batch_size,
        num_qp,
    )


def fuse_allreduce_combine(
    x: torch.Tensor,
    multicast_x: torch.Tensor,
    multinode_x: torch.Tensor,
    signal: torch.Tensor,
    rank: int,
    local_size: int,
    world_size: int,
    attn_dp_size: int,
    attn_tp_size: int,
    moe_ep_size: int,
    moe_tp_size: int,
    num_max_blocks: int,
    output_x: torch.Tensor,
    output_multicast_x: torch.Tensor,
    output_multinode_x: torch.Tensor,
    output_multinode_signal: torch.Tensor,
    world_rank: int,
    batch_size: int,
    num_qp: int,
) -> None:
    torch.ops.hpc.fuse_allreduce_combine(
        x,
        multicast_x,
        multinode_x,
        signal,
        rank,
        local_size,
        world_size,
        attn_dp_size,
        attn_tp_size,
        moe_ep_size,
        moe_tp_size,
        num_max_blocks,
        output_x,
        output_multicast_x,
        output_multinode_x,
        output_multinode_signal,
        world_rank,
        batch_size,
        num_qp,
    )


def empty_multinode(
    multicomm,
    *size: Any,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    sub_team: Optional[int] = None,
) -> Tuple[torch.Tensor, MultiNodeHandle]:
    if len(size) == 1 and isinstance(size[0], Sequence):
        size = tuple(size[0])
    else:
        size = tuple(size)

    if dtype is None:
        dtype = torch.get_default_dtype()

    if device is None:
        device = torch.get_default_device()

    if sub_team is None:
        sub_team = 2  # SHMEMX_TEAM_NODE = 2 which means the whole node as a team

    def device_to_num(device):
        if device.type == "cuda":
            return device.index if device.index is not None else 0  # default is 0
        else:
            return -1  # CPU

    assert (
        device_to_num(device) == multicomm.GetDeviceId()
    ), f"device(got {device_to_num(device)}) of alloc buffer must be same with multicomm(got {multicomm.GetDeviceId()})"

    hdl = MultiNodeHandle(multicomm, size, dtype, sub_team)

    return hdl.get_buffer(hdl.rank, size, dtype=dtype), hdl


@torch.library.register_fake("hpc::fuse_allreduce_dispatch")
def fuse_allreduce_dispatch_fake(
    x: torch.Tensor,
    multicast_x: torch.Tensor,
    multinode_x: torch.Tensor,
    signal: torch.Tensor,
    rank: int,
    local_size: int,
    world_size: int,
    attn_dp_size: int,
    attn_tp_size: int,
    moe_ep_size: int,
    moe_tp_size: int,
    num_max_blocks: int,
    output_x: torch.Tensor,
    output_multicast_x: torch.Tensor,
    output_multinode_x: torch.Tensor,
    output_multinode_signal: torch.Tensor,
    world_rank: int,
    batch_size: int,
    num_qp: int,
) -> None:
    return None


@torch.library.register_fake("hpc::fuse_allreduce_combine")
def fuse_allreduce_combine_fake(
    x: torch.Tensor,
    multicast_x: torch.Tensor,
    multinode_x: torch.Tensor,
    signal: torch.Tensor,
    rank: int,
    local_size: int,
    world_size: int,
    attn_dp_size: int,
    attn_tp_size: int,
    moe_ep_size: int,
    moe_tp_size: int,
    num_max_blocks: int,
    output_x: torch.Tensor,
    output_multicast_x: torch.Tensor,
    output_multinode_x: torch.Tensor,
    output_multinode_signal: torch.Tensor,
    world_rank: int,
    batch_size: int,
    num_qp: int,
) -> None:
    return None
