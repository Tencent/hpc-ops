import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc

import multiprocessing


def run_task(rank, world_size):
    comm = hpc.MulticastComm(rank, world_size, -1, 0)
    multi_tensor, local_tensor = comm.CreateTensorSync(1024)

    print("rank={}".format(rank))
    print(multi_tensor.data_ptr(), multi_tensor.dtype, multi_tensor.shape, multi_tensor.device)
    print(local_tensor)


@pytest.mark.skipif(os.getenv("NV_SANITIZER_INJECTION_PORT_BASE"), reason="skip sanitizer")
@pytest.mark.parametrize("world_size", [3, 8])
def test_communicator(world_size):

    processes = []
    for rank in range(world_size):
        p = multiprocessing.Process(
            target=run_task,
            args=(
                rank,
                world_size,
            ),
        )
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()
