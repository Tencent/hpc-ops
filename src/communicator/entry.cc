// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <memory>
#include <tuple>

#include "src/communicator/multicast_communicator.h"
#include "src/communicator/type.h"

namespace hpc {
namespace communicator {

class IMulticastCommunicator : public torch::CustomClassHolder {
 public:
  IMulticastCommunicator(int64_t rank, int64_t world_size, int64_t device_id = -1) {
    multicomm_ = std::make_unique<MulticastCommunicator>(rank, world_size, device_id);
  }

  ~IMulticastCommunicator() { multicomm_.reset(); }

  std::tuple<torch::Tensor, torch::Tensor> CreateTensorSync(int64_t bytes) {
    auto tensors = multicomm_->CreateTensorSync(bytes);

    torch::TensorOptions options;
    auto multi_opt = options.dtype(torch::kUInt8).device(torch::kCUDA, tensors.multi_device);
    auto local_opt = options.dtype(torch::kUInt8).device(torch::kCUDA, tensors.local_device);

    auto multi_deleter = [keeper = tensors.multi_ptr](void *ptr) {};
    auto local_deleter = [keeper = tensors.local_ptr](void *ptr) {};

    torch::Tensor multi =
        torch::from_blob(tensors.multi_ptr.get(), {bytes}, multi_deleter, multi_opt);
    torch::Tensor local =
        torch::from_blob(tensors.local_ptr.get(), {bytes}, local_deleter, local_opt);

    return std::make_tuple(multi, local);
  }

  void Barrier() { multicomm_->Barrier(); }

 private:
  std::unique_ptr<MulticastCommunicator> multicomm_;
};

}  // namespace communicator
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.class_<hpc::communicator::IMulticastCommunicator>("MulticastCommunicator")
      .def(torch::init<int64_t, int64_t, int64_t>())
      .def("CreateTensorSync", &hpc::communicator::IMulticastCommunicator::CreateTensorSync)
      .def("Barrier", &hpc::communicator::IMulticastCommunicator::Barrier);
}
