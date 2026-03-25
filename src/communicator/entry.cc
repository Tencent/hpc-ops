// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <memory>
#include <string>
#include <vector>

#include "src/communicator/multicast_communicator.h"
#include "src/communicator/multinode_communicator.h"

namespace hpc {
namespace communicator {

class IMulticastCommunicator : public torch::CustomClassHolder {
 public:
  IMulticastCommunicator(int64_t rank, int64_t world_size, int64_t device_id = -1,
                         const std::string &comm_name = "hpc-comm.sock") {
    multicomm_ = std::make_unique<MulticastCommunicator>(rank, world_size, device_id, comm_name);
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
      // Although a tensor is returned here, users should never manipulate
      // this tensor using torch's ops; it is merely a container of multimem pointers.
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

  int64_t GetRank() { return multicomm_->GetRank(); }

  int64_t GetWorldSize() { return multicomm_->GetWorldSize(); }

  int64_t GetDeviceId() { return multicomm_->GetDeviceId(); }

 private:
  std::unique_ptr<MulticastCommunicator> multicomm_;
};

class IMultiNodeCommunicator : public torch::CustomClassHolder {
 public:
  IMultiNodeCommunicator(int64_t rank, int64_t world_size, int64_t device_id,
                         const std::string &comm_name) {
    multinode_comm_ =
        std::make_unique<MultiNodeCommunicator>(rank, world_size, device_id, comm_name);
  }

  ~IMultiNodeCommunicator() { multinode_comm_.reset(); }

  auto CreateTensorSync(int64_t bytes, int64_t sub_team = -1) {
    std::vector<std::shared_ptr<void>> sptrs;
    std::vector<int> devices;
    std::shared_ptr<void> multinode_sptr;
    std::shared_ptr<void> multicast_sptr;
    std::shared_ptr<void> subgroup_multicast_sptr = nullptr;
    int multi_device;

    auto target_device = torch::Device(torch::kCUDA, multinode_comm_->GetDeviceId());
    torch::TensorOptions options;
    auto opt = options.dtype(torch::kUInt8).device(target_device);

    TORCH_CHECK(
        multinode_comm_->CreateTensorSync(bytes, &sptrs, &devices, &multinode_sptr, &multicast_sptr,
                                          &multi_device, sub_team, &subgroup_multicast_sptr),
        "create tensor sync fail");

    c10::Dict<int64_t, torch::Tensor> tensors;

    // multicast tensor
    {
      // Although a tensor is returned here, users should never manipulate
      // this tensor using torch's ops; it is merely a container of multimem pointers.
      auto deleter = [keeper = multicast_sptr](void *ptr) {};
      // use at:: apis to specify target device
      auto t = at::from_blob(multicast_sptr.get(), {bytes}, deleter, opt, target_device);
      tensors.insert(-1, t);
    }

    // multinode tensor
    {
      // Although a tensor is returned here, users should never manipulate
      // this tensor using torch's ops; it is merely a container of multinode pointers.
      auto deleter = [keeper = multinode_sptr](void *ptr) {};
      // use at:: apis to specify target device
      auto t = at::from_blob(multinode_sptr.get(), {bytes}, deleter, opt, target_device);
      tensors.insert(-2, t);
    }

    // subgroup multicast tensor
    {
      // Although a tensor is returned here, users should never manipulate
      // this tensor using torch's ops; it is merely a container of multimem pointers.
      auto deleter = [keeper = subgroup_multicast_sptr](void *ptr) {};
      // use at:: apis to specify target device
      auto t = at::from_blob(subgroup_multicast_sptr.get(), {bytes}, deleter, opt, target_device);
      tensors.insert(-3, t);
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

  void Barrier() { multinode_comm_->Barrier(); }

  int64_t GetRank() { return multinode_comm_->GetRank(); }

  int64_t GetWorldSize() { return multinode_comm_->GetWorldSize(); }

  int64_t GetDeviceId() { return multinode_comm_->GetDeviceId(); }

  int64_t CreateSubTeam(int64_t subgroup_size) {
    int64_t team = static_cast<int64_t>(multinode_comm_->CreateSubTeam(subgroup_size));
    return team;
  }

 private:
  std::unique_ptr<MultiNodeCommunicator> multinode_comm_;
};

}  // namespace communicator
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.class_<hpc::communicator::IMulticastCommunicator>("MulticastCommunicator")
      .def(torch::init<int64_t, int64_t, int64_t, const std::string &>(),
           "initialize multicast communcommunicator",
           {torch::arg("rank"), torch::arg("world_size"), torch::arg("device_id") = -1,
            torch::arg("comm_name") = "hpc-comm.sock"})
      .def("CreateTensorSync", &hpc::communicator::IMulticastCommunicator::CreateTensorSync)
      .def("Barrier", &hpc::communicator::IMulticastCommunicator::Barrier)
      .def("GetRank", &hpc::communicator::IMulticastCommunicator::GetRank)
      .def("GetWorldSize", &hpc::communicator::IMulticastCommunicator::GetWorldSize)
      .def("GetDeviceId", &hpc::communicator::IMulticastCommunicator::GetDeviceId);
  m.class_<hpc::communicator::IMultiNodeCommunicator>("MultiNodeCommunicator")
      .def(torch::init<int64_t, int64_t, int64_t, const std::string &>(),
           "initialize multinode communcommunicator",
           {torch::arg("rank"), torch::arg("world_size"), torch::arg("device_id"),
            torch::arg("comm_name")})
      .def("CreateTensorSync", &hpc::communicator::IMultiNodeCommunicator::CreateTensorSync,
           "create tensor sync", {torch::arg("bytes"), torch::arg("sub_team") = -1})
      .def("Barrier", &hpc::communicator::IMultiNodeCommunicator::Barrier)
      .def("GetRank", &hpc::communicator::IMultiNodeCommunicator::GetRank)
      .def("GetWorldSize", &hpc::communicator::IMultiNodeCommunicator::GetWorldSize)
      .def("GetDeviceId", &hpc::communicator::IMultiNodeCommunicator::GetDeviceId)
      .def("CreateSubTeam", &hpc::communicator::IMultiNodeCommunicator::CreateSubTeam);
}
