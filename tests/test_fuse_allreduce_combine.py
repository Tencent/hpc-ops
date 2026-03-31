import sys
import os
import time
import pytest
from pathlib import Path
import tempfile
import json

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import multiprocessing
import torch
from utils import calculate_errors, errors_to_string


def ref_fuse_allreduce_combine_task(input_list, attn_dp_size, attn_tp_size, world_size, world_rank):
    total_N, H = input_list[0].shape
    N = total_N // attn_dp_size
    device = input_list[0].device
    dtype = input_list[0].dtype

    tp_group_idx = world_rank // attn_tp_size

    input_sum = torch.zeros((N, H), dtype=dtype, device=device)
    for i in range(world_size):
        input_sum += input_list[i][tp_group_idx * N : (tp_group_idx + 1) * N, :]

    return input_sum


def run_fuse_allreduce_combine_task(
    rank,
    world_size,
    local_size,
    N,
    H,
    num_max_blocks,
    profile,
    result_queue,
    warmup_iters,
    benchmark_iters,
):
    node_idx = int(os.getenv("UCL_COMM_NODE", 0))
    master_ip = os.getenv("UCL_COMM_MASTER_IP", "127.0.0.1")
    master_port = os.getenv("UCL_COMM_MASTER_PORT", "10086")
    master_addr = f"{master_ip}:{master_port}"

    shmem_ibgda_num_rc_per_pe = int(os.getenv("NVSHMEM_IBGDA_NUM_RC_PER_PE", "40"))
    os.environ["NVSHMEM_IB_ENABLE_IBGDA"] = "1"
    os.environ["NVSHMEM_IBGDA_NUM_RC_PER_PE"] = f"{shmem_ibgda_num_rc_per_pe}"
    os.environ["NVSHMEM_QP_DEPTH"] = os.environ.get("NVSHMEM_QP_DEPTH", "1024")

    node_num = world_size // local_size

    attn_dp_size = 4
    attn_tp_size = 4

    moe_ep_size = 2
    moe_tp_size = 8

    assert world_size == attn_dp_size * attn_tp_size and world_size == moe_ep_size * moe_tp_size
    assert attn_tp_size <= local_size
    assert moe_tp_size <= local_size

    world_rank = node_idx * local_size + rank
    attn_tp_rank = world_rank % attn_tp_size
    local_attn_dp_group_idx = rank // attn_tp_size

    device = torch.device("cuda", index=rank)
    torch.cuda.set_device(rank)
    torch.set_default_device(device)

    seed = 10001
    # avoid real communication
    torch.manual_seed(seed)
    input_list = [
        torch.randn((attn_dp_size * N, H), dtype=torch.bfloat16).to(device=device)
        for _ in range(world_size)
    ]
    N_pad = (N + attn_tp_size - 1) // attn_tp_size * attn_tp_size

    comm = hpc.MultiNodeCommunicator(
        world_rank,
        world_size,
        rank,
        master_addr,
    )

    sub_team = comm.CreateSubTeam(attn_tp_size)
    symm_input, in_hdl = hpc.empty_multinode(
        comm, [attn_dp_size * N_pad, H], dtype=torch.bfloat16, device=device, sub_team=sub_team
    )
    symm_input[: attn_dp_size * N, :] = input_list[world_rank]
    symm_output, out_hdl = hpc.empty_multinode(
        comm, [N_pad, H], dtype=torch.bfloat16, device=device
    )
    symm_output1, out_hdl1 = hpc.empty_multinode(
        comm, [N_pad, H], dtype=torch.bfloat16, device=device
    )

    # ref
    rtol = 1e-01
    atol = 1e-01
    ref_output = ref_fuse_allreduce_combine_task(
        input_list, attn_dp_size, attn_tp_size, world_size, world_rank
    )

    start_idx = N_pad // attn_tp_size * attn_tp_rank
    end_idx = N_pad // attn_tp_size * (attn_tp_rank + 1)

    in_multimem_offset = (local_attn_dp_group_idx * N + start_idx) * H * symm_input.element_size()
    out_multimem_offset = (local_attn_dp_group_idx * N + start_idx) * H * symm_output.element_size()

    in_multinode_offset = local_attn_dp_group_idx * N * H * symm_output.element_size()
    out_multinode_offset = 0

    input_slice = symm_input[start_idx:end_idx, :]
    output_slice = symm_output[:N_pad, :]
    output_slice1 = symm_output1[:N_pad, :]

    in_multimem = in_hdl.get_multimem_buff(
        input_slice.shape, dtype=symm_input.dtype, storage_offset=in_multimem_offset
    )
    in_multinode = in_hdl.get_multinode_buff(
        output_slice.shape, dtype=symm_output.dtype, storage_offset=in_multinode_offset
    )
    out_subteam_multimem = in_hdl.get_subteam_multimem_buff(
        input_slice.shape, dtype=symm_output.dtype, storage_offset=out_multimem_offset
    )
    out_multinode = out_hdl.get_multinode_buff(
        output_slice.shape, dtype=symm_output.dtype, storage_offset=out_multinode_offset
    )
    in_signal_ptrs = in_hdl.signal_buffer_ptrs_dev
    out_multinode_signal_ptr = out_hdl.get_multinode_signal()

    # Two output buffer
    in_multinode1 = in_hdl.get_multinode_buff(
        output_slice1.shape, dtype=symm_output1.dtype, storage_offset=in_multinode_offset
    )
    out_multinode1 = out_hdl1.get_multinode_buff(
        output_slice1.shape, dtype=symm_output1.dtype, storage_offset=out_multinode_offset
    )
    out_multinode_signal_ptr1 = out_hdl1.get_multinode_signal()

    # (Warning!) When continuously call combine kernel, the output buffer should be swapped between two buffers
    # If only use one output buffer, the output buffer may be overwritten by the next combine kernel
    out_variants = [
        (output_slice, out_multinode, out_multinode_signal_ptr, in_multinode, symm_output),
        (output_slice1, out_multinode1, out_multinode_signal_ptr1, in_multinode1, symm_output1),
    ]

    torch.cuda.synchronize()

    flip = 0
    for accuracy_iter in range(warmup_iters):
        (
            cur_output_slice,
            cur_out_multinode,
            cur_out_multinode_signal_ptr,
            cur_in_multinode,
            cur_symm_output,
        ) = out_variants[flip]

        comm.Barrier()
        hpc.fuse_allreduce_combine(
            input_slice,
            in_multimem,
            cur_in_multinode,
            in_signal_ptrs,
            rank,
            local_size,
            world_size,
            attn_dp_size,
            attn_tp_size,
            moe_ep_size,
            moe_tp_size,
            num_max_blocks,
            cur_output_slice,
            out_subteam_multimem,
            cur_out_multinode,
            cur_out_multinode_signal_ptr,
            world_rank,
            N,
            shmem_ibgda_num_rc_per_pe,
        )
        torch.cuda.synchronize()

        assert torch.allclose(
            ref_output, cur_symm_output[:N, :], atol=atol, rtol=rtol
        ), f"Accuracy check failed at iteration {accuracy_iter}: " + errors_to_string(
            calculate_errors(ref_output, cur_symm_output[:N, :])
        )

        if accuracy_iter < warmup_iters - 1:
            # During each iter, the input_list is changed by adding 1
            for i in range(world_size):
                input_list[i] += 1
            ref_output = ref_fuse_allreduce_combine_task(
                input_list, attn_dp_size, attn_tp_size, world_size, world_rank
            )
        else:
            # Restore data to initial state before accuracy check
            for i in range(world_size):
                input_list[i] -= warmup_iters - 1

        symm_input[: attn_dp_size * N, :] = input_list[world_rank]
        symm_output.zero_()
        symm_output1.zero_()
        flip = 1 - flip

    torch.cuda.synchronize()

    if not profile:
        return 0

    comm.Barrier()
    g0 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g0):
        hpc.fuse_allreduce_combine(
            input_slice,
            in_multimem,
            in_multinode,
            in_signal_ptrs,
            rank,
            local_size,
            world_size,
            attn_dp_size,
            attn_tp_size,
            moe_ep_size,
            moe_tp_size,
            num_max_blocks,
            output_slice,
            out_subteam_multimem,
            out_multinode,
            out_multinode_signal_ptr,
            world_rank,
            N,
            shmem_ibgda_num_rc_per_pe,
        )

    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        hpc.fuse_allreduce_combine(
            input_slice,
            in_multimem,
            in_multinode1,
            in_signal_ptrs,
            rank,
            local_size,
            world_size,
            attn_dp_size,
            attn_tp_size,
            moe_ep_size,
            moe_tp_size,
            num_max_blocks,
            output_slice1,
            out_subteam_multimem,
            out_multinode1,
            out_multinode_signal_ptr1,
            world_rank,
            N,
            shmem_ibgda_num_rc_per_pe,
        )

    graphs = [g0, g1]

    torch.cuda.synchronize()

    # Use torch.profiler.profile for performance analysis
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        # Warm up graph when profiler
        flip = 0
        for wi in range(warmup_iters):
            graphs[flip].replay()
            flip = 1 - flip
        torch.cuda.synchronize()

        # Benchmark
        flip = 0
        for i in range(benchmark_iters):
            # Synchronize all ranks every 10 iterations to minimize cumulative error
            if i % 10 == 0:
                comm.Barrier()
            graphs[flip].replay()
            flip = 1 - flip

    torch.cuda.synchronize()

    # Export profiler trace and extract all_reduce_inplace kernel durations
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        prof.export_chrome_trace(tmp.name)
        profile_data = json.loads(Path(tmp.name).read_text())

    # Filter events for all_reduce_inplace kernel
    events = [
        event
        for event in profile_data["traceEvents"]
        if "fuse_allreduce_combine_kernel" in event["name"]
    ]
    events = sorted(events, key=lambda event: event["ts"])
    # skip profiler warm-up phase
    events = events[warmup_iters:]

    # Extract durations in milliseconds
    durations = [event["dur"] for event in events]  # Convert microseconds to milliseconds

    # Calculate statistics
    if durations:
        n = len(durations)
        result = {
            "rank": rank,
            "avg_time_us": sum(durations) / n,
        }
    else:
        result = {
            "rank": rank,
            "avg_time_us": 0.0,
        }

    result_queue.put(result)


@pytest.mark.skip(reason="Need 2 nodes to test, use python3 to run")
def test_fuse_allreduce_combine(
    world_size,
    local_size,
    N,
    H,
    num_max_blocks,
    profile=False,
    results=None,
    warmup_iters=10,
    benchmark_iters=100,
):
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    for rank in range(local_size):
        p = ctx.Process(
            target=run_fuse_allreduce_combine_task,
            args=(
                rank,
                world_size,
                local_size,
                N,
                H,
                num_max_blocks,
                profile,
                result_queue,
                warmup_iters,
                benchmark_iters,
            ),
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

    if profile:
        while not result_queue.empty():
            results.append(result_queue.get())
        results.sort(key=lambda x: x["rank"])


if __name__ == "__main__":
    N = 20
    H = 4096
    world_size = 16
    local_size = 8
    num_max_blocks = 5
    warmup_iters = 10
    benchmark_iters = 100
    results = []

    test_fuse_allreduce_combine(
        world_size, local_size, N, H, num_max_blocks, True, results, warmup_iters, benchmark_iters
    )

    print(f"\n{'='*70}")
    print(f"fuse_allreduce_combine Benchmark Results")
    print(
        f"Config: N={N}, H={H}, world_size={world_size}, local_size={local_size}, num_max_blocks={num_max_blocks}"
    )
    print(f"Warmup: {warmup_iters} iters, Benchmark: {benchmark_iters} iters")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Avg Time (us)':<15}")
    print(f"{'-'*70}")
    avg_times = []
    for r in results:
        print(f"{r['rank']:<6} {r['avg_time_us']:<15.2f}")
        avg_times.append(r["avg_time_us"])
    print(f"{'-'*70}")
    print(f"Mean Avg Time:    {sum(avg_times)/len(avg_times):.2f} us")
    print(f"{'='*70}\n")
