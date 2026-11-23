import torch

if not torch.version.hip:
    # Single-node NVLink multicast communicator, registered as a torch C++ class
    # in src/communicator/entry.cc. No NVSHMEM dependency.
    MulticastCommunicator = torch.classes.hpc.MulticastCommunicator
else:
    from ._amd.communicator.entry import MulticastCommunicator
