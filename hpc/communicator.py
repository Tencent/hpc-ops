import torch
from torch import Tensor
from typing import Dict, Tuple

import os
import fcntl
from datetime import datetime


class MulticastCommunicator:
    """A multicast communication handler for distributed tensor operations.

    This class provides functionality for creating synchronized tensors and
    performing barrier synchronization in a distributed environment. It enables
    coordinated tensor operations across multiple GPUs in a distributed group.

    Attributes:
        rank (int): The rank of the current process in the distributed group.
        world_size (int): The total number of processes in the distributed group.
        device_id (int): The device ID for tensor operations (-1 for CPU).

    Example:
        >>> # Initialize communication for rank 0 in a 4-GPU setup
        >>> comm = MulticastCommunicator(rank=0, world_size=4, device_id=-1)
        >>> # With device_id=-1, rank 0 will use GPU 0, rank 1 will use GPU 1, etc.
        >>> # Create synchronized tensors of size 1024
        >>> tensor_map = comm.CreateTensorSync(1024)
        >>> # Access the multicast tensor (shared across all processes)
        >>> multi_tensor = tensor_map[-1]
        >>> # Access the local tensor for this specific rank
        >>> local_tensor = tensor_map[0]
        >>> # Perform barrier synchronization
        >>> comm.Barrier()

    Note:
        This class is typically used in distributed training scenarios
        where multiple processes need to synchronize tensor operations,
        such as model parallelism or distributed data loading.
    """

    def __init__(self, rank: int, world_size: int, device_id: int = -1, comm_name="hpc-comm.sock"):
        """Initializes the MulticastCommunicator instance.

        Args:
            rank: The rank of the current process in the distributed group.
                Must be in range [0, world_size-1].
            world_size: The total number of processes in the distributed group.
                Must be a positive integer.
            device_id: The device ID for tensor operations. Use -1 to automatically
                use the rank as the GPU device ID. Otherwise, specifies a specific
                GPU device ID for all processes. Defaults to -1.
            group_name: the unique id of a group name

        Raises:
            ValueError: If rank is not in [0, world_size-1] or world_size < 1.
            RuntimeError: If communication backend initialization fails or
                device_id is invalid for the current system.

        Example:
            >>> # Each rank uses its rank as GPU ID (default behavior)
            >>> comm1 = MulticastCommunicator(rank=0, world_size=4, device_id=-1)
            >>> # Rank 0 will use GPU 0, rank 1 will use GPU 1, etc.
            >>>
        """
        pass

    def CreateTensorSync(self, size: int) -> Dict[int, Tensor]:
        """Creates synchronized tensors for distributed operations.

        This method creates a coordinated set of tensors across all processes
        in the distributed group. It returns a dictionary mapping process ranks
        to their respective tensors, with a special key for the multicast tensor.

        Args:
            size: The size (number of elements) of the tensor to create.
                Must be a positive integer.

        Returns:
            A dictionary where:
                - Keys are integers representing process ranks (0 to world_size-1)
                - Values are torch.Tensor objects on their respective devices
                - Special key -1 maps to the multicast tensor that is synchronized
                  across all processes

        Raises:
            ValueError: If size is not a positive integer.
            RuntimeError: If tensor creation fails due to memory allocation issues
                or communication errors during synchronization.

        Example:
            >>> # With device_id=-1 (default), each rank uses its rank as GPU ID
            >>> comm = MulticastCommunicator(rank=0, world_size=2, device_id=-1)
            >>> tensor_map = comm.CreateTensorSync(512)
            >>>
            >>> # Access different tensors in the map
            >>> multicast_tensor = tensor_map[-1]  # Shared multicast tensor
            >>> local_tensor = tensor_map[0]       # Tensor for rank 0 (on GPU 0)
            >>> remote_tensor = tensor_map[1]      # Tensor for rank 1 (on GPU 1)
            >>>
            >>> print(f"Local tensor device: {local_tensor.device}")
            >>> # For rank 0, this will output: device(type='cuda', index=0)
            >>> # For rank 1, this will output: device(type='cuda', index=1)

        Note:
            The multicast tensor (key=-1) is particularly useful for operations
            that need to be broadcasted or synchronized across all processes,
            while rank-specific tensors are used for local computations.
        """
        pass

    def Barrier(self) -> None:
        """Performs a barrier synchronization across all processes.

        This method blocks the calling process until all processes in the
        distributed group have reached this barrier. It ensures that all
        preceding operations have completed before any process continues.

        Raises:
            RuntimeError: If barrier synchronization fails due to
                communication errors or process disconnection.

        Example:
            >>> comm = MulticastCommunicator(rank=0, world_size=4, device_id=-1)
            >>>
            >>> # Perform some distributed operations
            >>> tensor_map = comm.CreateTensorSync(1024)
            >>> # ... some computations ...
            >>>
            >>> # Synchronize all processes before proceeding
            >>> comm.Barrier()
            >>>
            >>> # All processes continue execution simultaneously after this point
            >>> print("All processes synchronized!")

        Note:
            Barrier synchronization is crucial for ensuring consistency
            in distributed computations, especially when operations depend
            on results from other processes.
        """
        pass

    def GetRank(self) -> int:
        pass

    def GetWorldSize(self) -> int:
        pass

    def GetDeviceId(self) -> int:
        pass


class MultiNodeCommunicator:
    def __init__(
        self, rank: int, world_size: int, device_id: int = -1, comm_name="127.0.0.1:10086"
    ):
        pass

    def CreateTensorSync(self, size: int, sub_team: int = -1) -> Dict[int, Tensor]:
        pass

    def Barrier(self) -> None:
        pass

    def BarrierOnStream(self, stream: torch.cuda.Stream) -> None:
        """Performs a barrier synchronization on the specified CUDA stream.

        Unlike Barrier() which synchronizes on the default stream, this method
        performs barrier synchronization on the specified CUDA stream, allowing
        for more fine-grained stream-level synchronization.

        Args:
            stream (torch.cuda.Stream): The CUDA stream on which to perform
                the barrier synchronization.

        Raises:
            RuntimeError: If barrier synchronization fails due to
                communication errors or process disconnection.

        Example:
            >>> comm = MultiNodeCommunicator(rank=0, world_size=16, device_id=0, comm_name="192.168.1.1:10086")
            >>> # Synchronize all processes on the specified CUDA stream
            >>> stream = torch.cuda.current_stream().cuda_stream
            >>> comm.BarrierOnStream(stream)
        """
        pass

    def GetRank(self) -> int:
        pass

    def GetWorldSize(self) -> int:
        pass

    def GetDeviceId(self) -> int:
        pass

    def CreateSubTeam(self, subgroup_size: int) -> int:
        """Creates a new subgroup team with the specified size.

        This method partitions the world into subgroups of the given size
        and returns a team handle for the subgroup that this process belongs to.

        Args:
            subgroup_size: The number of processes in each subgroup.
                Must be a divisor of world_size.

        Returns:
            The team idx handle for the newly created subgroup as an integer.
            The underlying type is shmem_team_t (int32_t).

        Example:
            >>> comm = MultiNodeCommunicator(rank=0, world_size=16, device_id=0, comm_name="192.168.1.1:10086")
            >>> # Create subgroups of 4 processes each (4 subgroups total)
            >>> team = comm.CreateSubTeam(4)
        """
        pass


# Alternative interface using torch C++ classes
_RawMulticastCommunicator = torch.classes.hpc.MulticastCommunicator
_RawMultiNodeCommunicator = torch.classes.hpc.MultiNodeCommunicator


_HPC_UCL_VERSION = "R01C01"
_HPC_UCL_TARGET_DIR = "/dockerdata/.trmt"
_HPC_UCL_CONFIG_FILE = os.path.join(_HPC_UCL_TARGET_DIR, "uclop.config.json")
_HPC_UCL_LOCK_FILE = os.path.join(_HPC_UCL_TARGET_DIR, ".__uclop.config.lock")


def _record_hpc_ucl_version_info() -> None:
    try:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        config_data = (
            "{\n"
            f'    "TIMESTAMP": "{timestamp}",\n'
            f'    "KERNEL_VERSION": "{_HPC_UCL_VERSION}"\n'
            "}"
        )

        try:
            os.makedirs(_HPC_UCL_TARGET_DIR, mode=0o755, exist_ok=True)
        except OSError:
            return

        lock_fd = os.open(_HPC_UCL_LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                with open(_HPC_UCL_CONFIG_FILE, "w") as f:
                    f.write(config_data)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
    except Exception:
        return


def MulticastCommunicator(*args, **kwargs):
    _record_hpc_ucl_version_info()
    return _RawMulticastCommunicator(*args, **kwargs)


def MultiNodeCommunicator(*args, **kwargs):
    _record_hpc_ucl_version_info()
    return _RawMultiNodeCommunicator(*args, **kwargs)
