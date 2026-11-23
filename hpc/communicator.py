import torch

# Single-node NVLink multicast communicator, registered as a torch C++ class
# in src/communicator/entry.cc. No NVSHMEM dependency.
if not torch.version.hip:
    MulticastCommunicator = torch.classes.hpc.MulticastCommunicator
else:
    # TODO(amd): implement communication without NVIDIA NVLink/multicast primitives.
    class MulticastCommunicator:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("TODO(amd): NVLink multicast communicator is not ported")
