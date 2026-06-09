import torch

# Single-node NVLink multicast communicator, registered as a torch C++ class
# in src/communicator/entry.cc. No NVSHMEM dependency.
MulticastCommunicator = torch.classes.hpc.MulticastCommunicator
