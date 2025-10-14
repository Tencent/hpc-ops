// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include "src/communicator/multicast_comm.h"

namespace hpc {
namespace communicator {}  // namespace communicator
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.class_<hpc::communicator::MulticastComm>("MulticastComm")
      .def(torch::init<int64_t, int64_t, int64_t, int64_t>())
      .def("CreateTensorSync", &hpc::communicator::MulticastComm::CreateTensorSync);
}
