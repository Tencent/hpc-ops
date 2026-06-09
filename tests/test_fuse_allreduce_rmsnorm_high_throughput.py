import sys
import os
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
        f"hpc_ar_ht_ws{world_size}_N{N}_H{H}_blk{num_max_blocks}",
    )

    # in_x / out_x are this rank's local symmetric buffers; the multimem views
    # over the same workspaces are passed to the kernel for the multicast reduce.
    in_x, in_hdl = hpc.empty_multimem(comm, [N_pad, H], dtype=torch.bfloat16, device=device)
    out_x, out_hdl = hpc.empty_multimem(comm, [N_pad, H], dtype=torch.bfloat16, device=device)
    in_x.zero_()
    in_x[:N, :] = input_list[rank][:N, :]
    out_residual = torch.empty_like(residual)

    ref_residual, ref_output = ref_allreduce_rmsnorm(
        [x[:N, :] for x in input_list], residual[:N, :], weight, rms_norm_eps
    )

    start = N_pad // world_size * rank
    end = N_pad // world_size * (rank + 1)
    offset = start * H * in_x.element_size()

    comm.Barrier()
    torch.cuda.synchronize()

    atol = 1e-01
    rtol = 1e-01

    hpc.fuse_allreduce_rmsnorm_high_throughput(
        in_x[start:end, :],
        in_hdl.get_multimem_buff(in_x[start:end, :].shape, dtype=in_x.dtype, storage_offset=offset),
        residual[start:end, :],
        weight,
        rms_norm_eps,
        in_hdl.signal_buffer_ptrs_dev,
        rank,
        world_size,
        num_max_blocks,
        out_x[start:end, :],
        out_hdl.get_multimem_buff(
            out_x[start:end, :].shape, dtype=out_x.dtype, storage_offset=offset
        ),
        out_residual[start:end, :],
    )
    torch.cuda.synchronize()

    assert allclose(
        ref_residual[start : min(end, N), :],
        out_residual[start : min(end, N), :],
        atol=atol,
        rtol=rtol,
    ), "out_residual mismatch"
    assert allclose(ref_output, out_x[:N, :], atol=atol, rtol=rtol), "output mismatch"


@pytest.mark.parametrize("world_size", [4, 8])
@pytest.mark.parametrize("N", [128, 77])
@pytest.mark.parametrize("H", [5120, 7168])
@pytest.mark.parametrize("num_max_blocks", [16, 78])
def test_fuse_allreduce_rmsnorm_high_throughput(world_size, N, H, num_max_blocks):
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
    test_fuse_allreduce_rmsnorm_high_throughput(8, 256, 4096, 78)
    print("fuse_allreduce_rmsnorm_high_throughput passed")
