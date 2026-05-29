import sys
import os
import multiprocessing
from pathlib import Path

import pytest

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import torch.nn.functional as F
from utils import allclose


WORLD_SIZE = 8
NUM_TOTAL_SM = 78


def run_test(local_rank, world_size, comm_name, m, n, k, num_comm_sm, result_queue):
    torch.cuda.set_device(local_rank)
    cuda_device = torch.device("cuda", index=local_rank)

    num_comp_sm = NUM_TOTAL_SM - num_comm_sm

    # Create communicator and allocate buffers
    comm = hpc.MultiNodeCommunicator(local_rank, world_size, local_rank, comm_name)

    signal_size = hpc.get_fuse_gemm_rs_signal_size(m, n, num_comp_sm)

    output_list = comm.CreateTensorSync(int(m * n * torch.bfloat16.itemsize))
    signal_list = comm.CreateTensorSync(int(signal_size * torch.uint64.itemsize))

    output_list[local_rank][:] = 0
    signal_list[local_rank][:] = 0

    output = output_list[local_rank].view(torch.bfloat16).reshape(m, n)
    signal = signal_list[local_rank].view(torch.uint64).reshape(signal_size)
    multimem_output = output_list[-1].view(torch.bfloat16).reshape(m, n)
    multimem_signal = signal_list[-1].view(torch.uint64).reshape(signal_size)

    # Build inputs
    assert k % 128 == 0
    dtype = torch.float8_e4m3fn
    x = torch.randn((m, k), dtype=torch.float, device="cuda").to(dtype)
    w = torch.randn((n, k), dtype=torch.float, device="cuda").to(dtype)
    x_scale = torch.rand((m, k // 128), dtype=torch.float, device="cuda")
    w_scale = torch.rand(((n + 127) // 128, k // 128), dtype=torch.float, device="cuda")

    # Pad w_scale to align k-dim to 4
    ws_pad = (w_scale.size(1) + 3) // 4 * 4
    w_scale_pad = w_scale.clone()
    pad_size = ws_pad - w_scale.size(1)
    w_scale_pad = F.pad(w_scale_pad, (0, pad_size, 0, 0))

    torch.cuda.synchronize()
    comm.BarrierOnStream(torch.cuda.current_stream().cuda_stream)

    # Run fused kernel (our implementation)
    for _ in range(5):
        comm.BarrierOnStream(torch.cuda.current_stream().cuda_stream)
        my = hpc.fuse_gemm_reduce_scatter(
            x,
            w,
            x_scale,
            w_scale_pad,
            True,
            None,
            output,
            signal,
            multimem_output,
            multimem_signal,
            num_comp_sm,
            num_comm_sm,
            local_rank,
            world_size,
        )

    torch.cuda.synchronize()
    comm.BarrierOnStream(torch.cuda.current_stream().cuda_stream)

    # Build ground truth: gemm_blockwise + reduce_scatter (serial)
    num_max_blocks = 16
    ref_comm = hpc.MulticastCommunicator(
        local_rank,
        world_size,
        local_rank,
        f"reduce_scatter_ref_{comm_name}",
    )
    N_pad = (m + world_size - 1) // world_size * world_size
    symm_gemm_out, gemm_hdl = hpc.empty_multimem(
        ref_comm, [N_pad, n], dtype=torch.bfloat16, device=cuda_device
    )
    symm_rs_out, rs_hdl = hpc.empty_multimem(
        ref_comm, [N_pad, n], dtype=torch.bfloat16, device=cuda_device
    )

    gemm_result = hpc.gemm_blockwise(x, w, x_scale, w_scale_pad, True, None)
    symm_gemm_out[:m, :] = gemm_result

    chunk_start = N_pad // world_size * local_rank
    chunk_end = N_pad // world_size * (local_rank + 1)
    offset = chunk_start * n * symm_gemm_out.element_size()

    hpc.reduce_scatter(
        symm_gemm_out[chunk_start:chunk_end, :],
        gemm_hdl.get_multimem_buff(
            symm_gemm_out[chunk_start:chunk_end, :].shape,
            dtype=symm_gemm_out.dtype,
            storage_offset=offset,
        ),
        gemm_hdl.signal_buffer_ptrs_dev,
        local_rank,
        world_size,
        num_max_blocks,
        symm_rs_out[chunk_start:chunk_end, :],
        rs_hdl.get_multimem_buff(
            symm_rs_out[chunk_start:chunk_end, :].shape,
            dtype=symm_rs_out.dtype,
            storage_offset=offset,
        ),
    )

    gt = symm_rs_out[chunk_start : min(chunk_end, m), :]
    chunk = m // world_size
    fused_chunk = my[local_rank * chunk : (local_rank + 1) * chunk]

    passed = allclose(gt, fused_chunk, atol=0.5, rtol=0.01)
    max_diff = (gt.float() - fused_chunk.float()).abs().max().item()

    result_queue.put(
        {
            "rank": local_rank,
            "passed": passed,
            "max_diff": max_diff,
        }
    )
    torch.cuda.synchronize()


@pytest.mark.parametrize("m", [8192, 16384, 32768])
@pytest.mark.parametrize("n", [4096])
@pytest.mark.parametrize("k", [1024])
@pytest.mark.parametrize("num_comm_sm", [4, 8, 12])
def test_fuse_gemm_reduce_scatter(m, n, k, num_comm_sm):
    world_size = WORLD_SIZE
    comm_name = "127.0.0.1:10086"

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for rank in range(world_size):
        p = ctx.Process(
            target=run_test,
            args=(rank, world_size, comm_name, m, n, k, num_comm_sm, result_queue),
        )
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    for rank, p in enumerate(processes):
        assert p.exitcode == 0, f"rank {rank} subprocess failed (exitcode={p.exitcode})"

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    for r in results:
        assert r["passed"], f"rank {r['rank']} failed, max_diff={r['max_diff']:.4f}"
