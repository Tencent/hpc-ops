import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import multiprocessing
import torch


def run_task(rank, world_size, bytes):
    comm = hpc.MulticastCommunicator(rank, world_size, -1)
    multi_tensor, local_tensor = comm.CreateTensorSync(bytes)

    print("rank={}".format(rank))
    print(multi_tensor.data_ptr(), multi_tensor.dtype, multi_tensor.shape, multi_tensor.device)

    local_tensor[:] = 1.0
    print(local_tensor)

    assert multi_tensor.shape == torch.Size([bytes])
    assert local_tensor.shape == torch.Size([bytes])

    assert multi_tensor.stride() == (1,)
    assert local_tensor.stride() == (1,)

    assert multi_tensor.dtype == torch.uint8
    assert local_tensor.dtype == torch.uint8

    assert multi_tensor.device == torch.device("cuda", 0)
    assert local_tensor.device == torch.device("cuda", rank)


@pytest.mark.skipif(os.getenv("NV_SANITIZER_INJECTION_PORT_BASE"), reason="skip sanitizer")
@pytest.mark.parametrize("world_size", [3, 8])
@pytest.mark.parametrize("bytes", [5, 1024 * 1024 * 600])
def test_communicator(world_size, bytes):

    processes = []
    for rank in range(world_size):
        p = multiprocessing.Process(
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

    for p in processes:
        p.join()
