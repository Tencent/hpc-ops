import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import multiprocessing
import torch

from utils import calculate_errors, errors_to_string


def rmsnorm(x, w, rms_norm_eps):
    mean_square = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(mean_square + rms_norm_eps)) * w.reshape(1, -1).float()


def ref_allreduce_rmsnorm(input_list, residual, weight, scale, scale2, rms_norm_eps):
    input_sum = torch.zeros_like(input_list[0])
    for input in input_list:
        input_sum += input
    output_residual = input_sum + residual
    output = rmsnorm(output_residual, weight, rms_norm_eps)
    fp32_output = output
    inv_scale = 1.0 / scale[0]
    inv_scale2 = 1.0 / scale2[0]
    fp8_output = fp32_output * inv_scale
    fp8_output2 = fp32_output * inv_scale2
    return (
        output_residual,
        fp32_output,
        fp8_output.to(torch.float8_e4m3fn),
        fp8_output2.to(torch.float8_e4m3fn),
    )


def run_task(rank, world_size, N, H, num_max_blocks):
    device = torch.device("cuda", index=rank)
    torch.cuda.set_device(rank)
    torch.set_default_device(device)

    seed = 10001
    # avoid real communication
    torch.manual_seed(seed)
    input_list = [
        torch.rand((N, H), dtype=torch.bfloat16).to(device=device) for _ in range(world_size)
    ]
    N_pad = (N + world_size - 1) // world_size * world_size
    residual = torch.rand((N_pad, H), dtype=torch.bfloat16).to(device=device)
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
    symm_fp8_output, fp8_out_hdl = hpc.empty_multimem(
        comm, [N_pad, H], dtype=torch.float8_e4m3fn, device=device
    )
    symm_fp8_output2, fp8_out2_hdl = hpc.empty_multimem(
        comm, [N_pad, H], dtype=torch.float8_e4m3fn, device=device
    )
    symm_fp32_output, fp32_out_hdl = hpc.empty_multimem(
        comm, [N_pad, H], dtype=torch.float32, device=device
    )

    out_residual = torch.empty_like(residual)
    scale = torch.tensor([0.5], dtype=torch.float32, device=device)
    scale2 = torch.tensor([0.2], dtype=torch.float32, device=device)

    ref_residual, ref_fp32_output, ref_fp8_output, ref_fp8_output2 = ref_allreduce_rmsnorm(
        input_list, residual[:N, :], weight, scale, scale2, rms_norm_eps
    )

    start = N_pad // world_size * rank
    end = N_pad // world_size * (rank + 1)
    input_offset = start * H * symm_input.element_size()
    fp8_output_offset = start * H * symm_fp8_output.element_size()
    fp32_output_offset = start * H * symm_fp32_output.element_size()

    # not inplace
    hpc.fuse_allreduce_rmsnorm_with_scale(
        symm_input[start:end, :],
        in_hdl.get_multimem_buff(
            symm_input[start:end, :].shape, dtype=torch.bfloat16, storage_offset=input_offset
        ),
        residual[start:end, :],
        weight,
        scale,
        fp8_out_hdl.get_multimem_buff(
            symm_input[start:end, :].shape,
            dtype=torch.float8_e4m3fn,
            storage_offset=fp8_output_offset,
        ),
        in_hdl.signal_buffer_ptrs_dev,
        rank,
        world_size,
        num_max_blocks,
        rms_norm_eps,
        True,
        out_residual[start:end, :],
        scale2,
        fp8_out2_hdl.get_multimem_buff(
            symm_input[start:end, :].shape,
            dtype=torch.float8_e4m3fn,
            storage_offset=fp8_output_offset,
        ),
        fp32_out_hdl.get_multimem_buff(
            symm_input[start:end, :].shape, dtype=torch.float32, storage_offset=fp32_output_offset
        ),
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

    assert torch.allclose(
        ref_residual[start : min(end, N), :],
        out_residual[start : min(end, N), :],
        atol=atol,
        rtol=rtol,
    ), errors_to_string(
        calculate_errors(ref_residual[start : min(end, N), :], out_residual[start : min(end, N), :])
    )

    assert torch.allclose(
        ref_fp32_output,
        symm_fp32_output[:N, :],
        atol=atol,
        rtol=rtol,
    ), errors_to_string(calculate_errors(ref_fp32_output, symm_fp32_output[:N, :]))
    assert torch.allclose(
        ref_fp8_output.to(torch.bfloat16),
        symm_fp8_output[:N, :].to(torch.bfloat16),
        atol=2,
        rtol=0.3,
    ), errors_to_string(
        calculate_errors(
            ref_fp8_output.to(torch.bfloat16), symm_fp8_output[:N, :].to(torch.bfloat16)
        )
    )

    assert torch.allclose(
        ref_fp8_output2.to(torch.bfloat16),
        symm_fp8_output2[:N, :].to(torch.bfloat16),
        atol=2,
        rtol=0.3,
    ), errors_to_string(
        calculate_errors(
            ref_fp8_output2.to(torch.bfloat16), symm_fp8_output2[:N, :].to(torch.bfloat16)
        )
    )


@pytest.mark.skipif(os.getenv("NV_SANITIZER_INJECTION_PORT_BASE"), reason="skip sanitizer")
@pytest.mark.parametrize("world_size", [8])
@pytest.mark.parametrize("N", [128, 160, 32, 77])
@pytest.mark.parametrize("H", [5120])
@pytest.mark.parametrize("num_max_blocks", [4, 8, 78])
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
