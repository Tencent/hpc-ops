import sys
import os
import pytest
import ctypes
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import multiprocessing
import torch


def run_task(node_id, local_rank, world_rank, local_size, world_size, device, bytes, comm_name):
    comm = hpc.MulticastCommunicator(local_rank, local_size)
    tensors = comm.CreateTensorSync(bytes)
    print("rank={}".format(local_rank))
    for r, tensor in tensors.items():
        assert tensor.shape == torch.Size([bytes])
        assert tensor.stride() == (1,)
        assert tensor.dtype == torch.uint8
        assert tensor.device == torch.device("cuda", device)
        if r != -1:
            # write
            tensor[:] = 0
            # read
            print(tensor)

    comm = hpc.MulticastCommunicator(local_rank, local_size, -1, "another hpc multicomm")
    tensors = comm.CreateTensorSync(bytes)
    print("rank={}".format(local_rank))
    for r, tensor in tensors.items():
        assert tensor.shape == torch.Size([bytes])
        assert tensor.stride() == (1,)
        assert tensor.dtype == torch.uint8
        assert tensor.device == torch.device("cuda", device)
        if r != -1:
            # write
            tensor[:] = 0
            # read
            print(tensor)
    comm.Barrier()

    comm = hpc.MultiNodeCommunicator(world_rank, world_size, device, comm_name)
    tensors = comm.CreateTensorSync(bytes)
    print("rank={}".format(world_rank))
    for r, tensor in tensors.items():
        if r in range(node_id * local_size, (node_id + 1) * local_size):
            assert tensor.shape == torch.Size([bytes])
            assert tensor.stride() == (1,)
            assert tensor.dtype == torch.uint8
            assert tensor.device == torch.device("cuda", device)
            # write
            tensor[:] = 0
            # read
            print(tensor)
    comm.Barrier()


@pytest.mark.skipif(os.getenv("NV_SANITIZER_INJECTION_PORT_BASE"), reason="skip sanitizer")
@pytest.mark.skipif(os.getenv("PYTEST_SKIP"), reason="skip pytest as only 1 H20")
@pytest.mark.parametrize("local_size", [3, 8])
@pytest.mark.parametrize("bytes", [5, 1024 * 1024 * 600])
def test_communicator(local_size, bytes):
    node_num = 1
    node_id = 0
    world_size = node_num * local_size
    comm_name = "127.0.0.1:10086"

    ctx = multiprocessing.get_context("spawn")
    processes = []
    for local_rank in range(local_size):
        p = ctx.Process(
            target=run_task,
            args=(
                node_id,
                local_rank,
                node_id * local_size + local_rank,
                local_size,
                world_size,
                local_rank,
                bytes,
                comm_name,
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


if __name__ == "__main__":
    test_communicator(3, 5)
