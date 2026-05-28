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


def run_task(rank, world_size, N, H, num_max_blocks):
    device = torch.device("cuda", index=rank)
    torch.cuda.set_device(rank)
    torch.set_default_device(device)

    seed = 10001
    # avoid real communication
    torch.manual_seed(seed)
    input_list = [
        torch.randn((N, H), dtype=torch.bfloat16).to(device=device) for _ in range(world_size)
    ]
    N_pad = (N + world_size - 1) // world_size * world_size
    residual = torch.randn((N_pad, H), dtype=torch.bfloat16).to(device=device)
    weight = torch.randn((H,), dtype=torch.bfloat16).to(device=device)
    rms_norm_eps = 1e-6

    comm = hpc.MulticastCommunicator(
        rank,
        world_size,
        -1,
        f"group_world_size_{world_size}_N_{N}_H_{H}_num_max_blocks_{num_max_blocks}",
    )
    symm_input, in_hdl = hpc.empty_multimem(comm, [N_pad, H], dtype=torch.bfloat16, device=device)
    input = input_list[rank]
    symm_input[:N, :] = input
    symm_output, out_hdl = hpc.empty_multimem(comm, [N_pad, H], dtype=torch.bfloat16, device=device)
    out_residual = torch.empty_like(residual)

    ref_residual, ref_output = ref_allreduce_rmsnorm(
        input_list, residual[:N, :], weight, rms_norm_eps
    )

    # NOTE(landojiang): copy from vllm/tests/kernels/test_layernorm.py::54
    # LayerNorm operators (including RMS) typically have larger
    # numerical errors than other operators because they involve reductions.
    # Therefore, we use a larger tolerance.
    atol = 1e-2
    rtol = 1e-2

    # NOTE(landojiang): copy from torch/test/distributed/test_symmetric_memory.py::1043
    rtol = 1e-01
    atol = 1e-01

    start = N_pad // world_size * rank
    end = N_pad // world_size * (rank + 1)
    offset = start * H * symm_input.element_size()

    # not inplace
    hpc.fuse_allreduce_rmsnorm(
        symm_input[start:end, :],
        in_hdl.get_multimem_buff(
            symm_input[start:end, :].shape, dtype=symm_input.dtype, storage_offset=offset
        ),
        residual[start:end, :],
        weight,
        rms_norm_eps,
        in_hdl.signal_buffer_ptrs_dev,
        rank,
        world_size,
        num_max_blocks,
        symm_output[start:end, :],
        out_hdl.get_multimem_buff(
            symm_output[start:end, :].shape, dtype=symm_output.dtype, storage_offset=offset
        ),
        out_residual[start:end, :],
    )

    assert allclose(
        ref_residual[start : min(end, N), :],
        out_residual[start : min(end, N), :],
        atol=atol,
        rtol=rtol,
    )

    assert allclose(ref_output, symm_output[:N, :], atol=atol, rtol=rtol)

    # inplace
    hpc.fuse_allreduce_rmsnorm(
        symm_input[start:end, :],
        in_hdl.get_multimem_buff(
            symm_input[start:end, :].shape, dtype=symm_input.dtype, storage_offset=offset
        ),
        residual[start:end, :],
        weight,
        rms_norm_eps,
        in_hdl.signal_buffer_ptrs_dev,
        rank,
        world_size,
        num_max_blocks,
    )

    assert allclose(
        ref_residual[start : min(end, N), :], residual[start : min(end, N), :], atol=atol, rtol=rtol
    )
    assert allclose(ref_output, symm_input[:N, :], atol=atol, rtol=rtol)


@pytest.mark.skipif(os.getenv("NV_SANITIZER_INJECTION_PORT_BASE"), reason="skip sanitizer")
@pytest.mark.skipif(os.getenv("PYTEST_SKIP"), reason="skip pytest as only 1 H20")
@pytest.mark.parametrize("world_size", [3, 8])
@pytest.mark.parametrize("N", [128, 77])
@pytest.mark.parametrize("H", [5120, 7168])
@pytest.mark.parametrize("num_max_blocks", [16, 78])
def test_fuse_allreduce_rmsnorm(world_size, N, H, num_max_blocks):
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
        assert exitcode == 0, f"rank {rank} subprogress failed"
