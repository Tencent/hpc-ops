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
    for input in input_list:
        input_sum += input
    output_residual = input_sum + residual
    output = rmsnorm(output_residual, weight, rms_norm_eps)
    return output_residual, output


def run_task(rank, world_size, N, H):
    device = torch.device("cuda", index=rank)
    torch.cuda.set_device(rank)
    torch.set_default_device(device)

    seed = 10001
    # avoid real communication
    torch.manual_seed(seed)
    N_pad = (N + world_size - 1) // world_size * world_size
    input_list = [
        torch.randn((N_pad, H), dtype=torch.bfloat16).to(device=device) for _ in range(world_size)
    ]
    residual = torch.randn((N_pad, H), dtype=torch.bfloat16).to(device=device)
    weight = torch.randn((H,), dtype=torch.bfloat16).to(device=device)
    rms_norm_eps = 1e-6

    comm = hpc.MultiNodeCommunicator(
        rank,
        world_size,
        rank,
        "127.0.0.1:20088",
    )

    workspace_hdl, symm_workspace, buffer_flags, workspace_size = (
        hpc.create_workspace_for_fuse_ar_rms_v2(comm, N, H, world_size, dtype=torch.bfloat16)
    )

    multinode_x = symm_workspace
    multicast_x = workspace_hdl.get_multimem_buff(workspace_size, dtype=torch.bfloat16)
    data_buffer_ptrs = workspace_hdl.data_buffer_ptrs_dev

    input = input_list[rank]
    output = torch.empty_like(input)
    out_residual = torch.empty_like(residual)

    ref_residual, ref_output = ref_allreduce_rmsnorm(
        [x[:N, :] for x in input_list], residual[:N, :], weight, rms_norm_eps
    )

    atol = 1e-01
    rtol = 1e-01

    hpc.fuse_allreduce_rmsnorm_v2(
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
        output,
        out_residual,
    )

    torch.cuda.synchronize()

    allclose(
        ref_residual[:N, :],
        out_residual[:N, :],
        atol=atol,
        rtol=rtol,
    )

    allclose(ref_output, output[:N, :], atol=atol, rtol=rtol)


@pytest.mark.skipif(os.getenv("NV_SANITIZER_INJECTION_PORT_BASE"), reason="skip sanitizer")
@pytest.mark.parametrize("world_size", [4, 8])
@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("H", [4096, 7168])
def test_fuse_allreduce_rmsnorm_v2(world_size, N, H):
    ctx = multiprocessing.get_context("spawn")

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=run_task,
            args=(rank, world_size, N, H),
        )
        processes.append(p)

    for p in processes:
        p.start()

    exitcode_list = []
    for p in processes:
        p.join()
        exitcode_list.append(p.exitcode)

    for rank, exitcode in enumerate(exitcode_list):
        assert exitcode == 0, f"rank {rank} subprogress failed"
