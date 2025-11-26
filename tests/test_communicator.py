import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import multiprocessing
import torch


def run_task(rank, world_size, bytes):
    comm = hpc.MulticastCommunicator(rank, world_size)
    tensors = comm.CreateTensorSync(bytes)
    print("rank={}".format(rank))
    for r, tensor in tensors.items():
        assert tensor.shape == torch.Size([bytes])
        assert tensor.stride() == (1,)
        assert tensor.dtype == torch.uint8
        assert tensor.device == torch.device("cuda", rank)
        if r != -1:
            # write
            tensor[:] = 0
            # read
            print(tensor)

    comm = hpc.MulticastCommunicator(rank, world_size, -1, "another hpc multicomm")
    tensors = comm.CreateTensorSync(bytes)
    print("rank={}".format(rank))
    for r, tensor in tensors.items():
        assert tensor.shape == torch.Size([bytes])
        assert tensor.stride() == (1,)
        assert tensor.dtype == torch.uint8
        assert tensor.device == torch.device("cuda", rank)
        if r != -1:
            # write
            tensor[:] = 0
            # read
            print(tensor)
    comm.Barrier()


@pytest.mark.skipif(os.getenv("NV_SANITIZER_INJECTION_PORT_BASE"), reason="skip sanitizer")
@pytest.mark.parametrize("world_size", [3, 8])
@pytest.mark.parametrize("bytes", [5, 1024 * 1024 * 600])
def test_communicator(world_size, bytes):
    ctx = multiprocessing.get_context("spawn")

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=run_task,
            args=(
                rank,
                world_size,
                bytes,
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
