// Copyright 2025 hpc-ops authors

#include "src/communicator/multicast_comm.h"

#include <torch/custom_class.h>

#include <cstdio>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "src/communicator/communicator.h"

namespace hpc {
namespace communicator {

MulticastComm::MulticastComm(int64_t rank, int64_t world_size, int64_t device_id, int64_t root) {
  rank_ = rank;
  world_size_ = world_size;

  if (device_id == -1) {
    device_id_ = rank;
  } else {
    device_id_ = device_id;
  }

  comm_ = std::make_unique<Communicator>(rank, world_size);
  multiobj_ = std::make_unique<MulticastObjectManager>(device_id_, world_size_);
}

MulticastComm::~MulticastComm() {
  multiobj_.reset();
  comm_.reset();
}

void MulticastComm::Barrier() { comm_->Barrier(); }

std::tuple<torch::Tensor, torch::Tensor> MulticastComm::CreateTensorSync(int64_t bytes) {
  if (rank_ == 0) {
    // get multimem fd and broadcast it
    int fd = -1;
    if (!multiobj_->CreateMulticastObjAndExportFd(&fd, bytes)) {
      throw std::runtime_error("hpc multimem obj create multicast obj and export fd fail");
    }

    std::string data = std::to_string(bytes);
    int rfd = -1;
    if (!comm_->BroadcastFd(fd, &rfd, data, &data, 0)) {
      throw std::runtime_error("hpc comm broadcast fail");
    }
  } else {
    int fd = -1;
    int rfd = -1;
    std::string data;

    if (!comm_->BroadcastFd(fd, &rfd, data, &data, 0)) {
      throw std::runtime_error("hpc comm broadcast fail");
    }
    if (!multiobj_->CreateMulticastObjByImportFd(rfd)) {
      throw std::runtime_error("hpc multimem obj create multicast obj by import fd fail");
    }
  }

  void *multi_ptr = nullptr;
  void *local_ptr = nullptr;
  if (!multiobj_->AllocateMemoryAndBindToMulticastObj(&multi_ptr, &local_ptr)) {
    throw std::runtime_error("hpc multimem obj allocate memory and bind to multicast obj fail");
  }

  cudaPointerAttributes multi_attr;
  cudaPointerGetAttributes(&multi_attr, multi_ptr);

  cudaPointerAttributes local_attr;
  cudaPointerGetAttributes(&local_attr, local_ptr);

  if (local_attr.device != device_id_) {
    throw std::runtime_error("hpc multimem obj local_ptr.device != device_id");
  }

  torch::TensorOptions options;
  auto multi_opt = options.dtype(torch::kUInt8).device(torch::kCUDA, multi_attr.device);
  auto local_opt = options.dtype(torch::kUInt8).device(torch::kCUDA, local_attr.device);

  torch::Tensor multi = torch::from_blob(multi_ptr, {bytes}, multi_opt);
  torch::Tensor local = torch::from_blob(local_ptr, {bytes}, local_opt);

  return std::make_tuple(multi, local);
}

}  // namespace communicator
}  // namespace hpc
