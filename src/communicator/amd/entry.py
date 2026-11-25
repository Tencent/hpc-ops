# Copyright (C) 2026 Tencent.


class MulticastCommunicator:
    def __init__(
        self,
        rank: int,
        world_size: int,
        device_id: int = -1,
        comm_name: str = "hpc-comm.sock",
    ):
        # TODO(amd): implement communication buffer management and synchronization.
        raise NotImplementedError("TODO(amd): multicast communicator is not implemented")
