import sys
import os
import math
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import multiprocessing
import torch

from utils import allclose


def rmsnorm(x, w, rms_norm_eps):
    mean_square = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(mean_square + rms_norm_eps)).to(torch.bfloat16) * w.reshape(
        1, -1
    )


def ref_allreduce_rmsnorm(input_list, residual, weight, rms_norm_eps):
    input_sum = torch.zeros_like(input_list[0])
    for x in input_list:
        input_sum += x
    output_residual = input_sum + residual
    output = rmsnorm(output_residual, weight, rms_norm_eps)
    return output_residual, output


def run_task(rank, world_size, N, H, num_max_blocks):
    device = torch.device("cuda", index=rank)
    torch.cuda.set_device(rank)
    torch.set_default_device(device)

    torch.manual_seed(10001)
    N_pad = (N + world_size - 1) // world_size * world_size
    input_list = [
        torch.randn((N_pad, H), dtype=torch.bfloat16).to(device=device) for _ in range(world_size)
    ]
    residual = torch.randn((N_pad, H), dtype=torch.bfloat16).to(device=device)
    weight = torch.randn((H,), dtype=torch.bfloat16).to(device=device)
    rms_norm_eps = 1e-6

    # Single-node NVLink multicast communicator (no NVSHMEM).
    comm = hpc.MulticastCommunicator(
        rank,
        world_size,
        rank,
        f"hpc_ar_ll_ws{world_size}_N{N}_H{H}_blk{num_max_blocks}",
    )

    NUM_LAMPORT_BUFFERS = 3
    M_pad = 2 * math.ceil(N / world_size) * world_size * NUM_LAMPORT_BUFFERS
    workspace_size = [M_pad, H]

    elem_size = torch.tensor([], dtype=torch.bfloat16).element_size()
    workspace_size_bytes = M_pad * H * elem_size

    # multinode_x is this rank's local symmetric buffer; multicast_x is the
    # multimem view of the same workspace; data_buffer_ptrs holds every rank's
    # local buffer pointer (peer P2P over NVLink).
    multinode_x, workspace_hdl = hpc.empty_multimem(
        comm, workspace_size, dtype=torch.bfloat16, device=device
    )
    multinode_x.view(torch.uint32).fill_(0x80000000)
    multicast_x = workspace_hdl.get_multimem_buff(workspace_size, dtype=torch.bfloat16)
    data_buffer_ptrs = workspace_hdl.data_buffer_ptrs_dev

    Lamport_buffer_size_bytes = math.floor(workspace_size_bytes / NUM_LAMPORT_BUFFERS) // 16 * 16
    num_bytes_to_clear = [0] * 4
    buffer_flags = torch.tensor(
        [0, 2, Lamport_buffer_size_bytes, 0, *num_bytes_to_clear, 0],
        dtype=torch.uint32,
        device=device,
    )

    input = input_list[rank]
    output = torch.empty_like(input)
    out_residual = torch.empty_like(residual)

    ref_residual, ref_output = ref_allreduce_rmsnorm(
        [x[:N, :] for x in input_list], residual[:N, :], weight, rms_norm_eps
    )

    comm.Barrier()
    torch.cuda.synchronize()

    atol = 1e-01
    rtol = 1e-01

    hpc.fuse_allreduce_rmsnorm_low_latency(
        input,
        multicast_x,
        data_buffer_ptrs,
        multinode_x,
        buffer_flags,
        world_size,
        rank,
        residual,
        weight,
        rms_norm_eps,
        num_max_blocks,
        output,
        out_residual,
        True,  # launch kernel with PDL
    )
    torch.cuda.synchronize()

    assert allclose(
        ref_residual[:N, :],
        out_residual[:N, :],
        atol=atol,
        rtol=rtol,
    ), "out_residual mismatch"
    assert allclose(ref_output, output[:N, :], atol=atol, rtol=rtol), "output mismatch"


# twoshotAllreduceKernel only supports power-of-two world sizes {2,4,8,16,32,64}.
@pytest.mark.parametrize("world_size", [4, 8])
@pytest.mark.parametrize("N", [128, 77])
@pytest.mark.parametrize("H", [5120, 7168])
@pytest.mark.parametrize("num_max_blocks", [16, 78])
def test_fuse_allreduce_rmsnorm_low_latency(world_size, N, H, num_max_blocks):
    ctx = multiprocessing.get_context("spawn")

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=run_task,
            args=(rank, world_size, N, H, num_max_blocks),
        )
        processes.append(p)

    for p in processes:
        p.start()

    exitcode_list = []
    for p in processes:
        p.join()
        exitcode_list.append(p.exitcode)

    for rank, exitcode in enumerate(exitcode_list):
        assert exitcode == 0, f"rank {rank} subprocess failed"


if __name__ == "__main__":
    test_fuse_allreduce_rmsnorm_low_latency(8, 256, 4096, 78)
    print("fuse_allreduce_rmsnorm_low_latency passed")
