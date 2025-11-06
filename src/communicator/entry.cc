// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <memory>
#include <vector>

#include "src/communicator/multicast_communicator.h"

namespace hpc {
namespace communicator {

class IMulticastCommunicator : public torch::CustomClassHolder {
 public:
  IMulticastCommunicator(int64_t rank, int64_t world_size, int64_t device_id = -1) {
    multicomm_ = std::make_unique<MulticastCommunicator>(rank, world_size, device_id);
  }

  ~IMulticastCommunicator() { multicomm_.reset(); }

  auto CreateTensorSync(int64_t bytes) {
    std::vector<std::shared_ptr<void>> sptrs;
    std::vector<int> devices;
    std::shared_ptr<void> multi_sptr;
    int multi_device;

    auto target_device = torch::Device(torch::kCUDA, multicomm_->GetDeviceId());
    torch::TensorOptions options;
    auto opt = options.dtype(torch::kUInt8).device(target_device);

    TORCH_CHECK(multicomm_->CreateTensorSync(bytes, &sptrs, &devices, &multi_sptr, &multi_device),
                "create tensor sync fail");

    c10::Dict<int64_t, torch::Tensor> tensors;

    // multi tensor
    {
      auto deleter = [keeper = multi_sptr](void *ptr) {};
      // use at:: apis to specify target device
      auto t = at::from_blob(multi_sptr.get(), {bytes}, deleter, opt, target_device);

      tensors.insert(-1, t);
    }

    // remote tensor and local tensor
    for (uint32_t rank = 0; rank < sptrs.size(); ++rank) {
      auto deleter = [keeper = sptrs[rank]](void *ptr) {};
      // use at:: apis to specify target device
      auto t = at::from_blob(sptrs[rank].get(), {bytes}, deleter, opt, target_device);

      tensors.insert(rank, t);
    }

    return tensors;
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
