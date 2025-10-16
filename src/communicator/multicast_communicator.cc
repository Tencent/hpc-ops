// Copyright 2025 hpc-ops authors

#include "src/communicator/multicast_communicator.h"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "src/communicator/communicator.h"
#include "src/communicator/type.h"

namespace hpc {
namespace communicator {

MulticastCommunicator::MulticastCommunicator(int64_t rank, int64_t world_size, int64_t device_id) {
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

MulticastCommunicator::~MulticastCommunicator() {
  multiobj_.reset();
  comm_.reset();
}

void MulticastCommunicator::Barrier() { comm_->Barrier(); }

MulticastTensors MulticastCommunicator::CreateTensorSync(int64_t bytes) {
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

    int64_t bytes_from_root = std::stoll(data);
    if (bytes_from_root != bytes) {
      throw std::runtime_error("hpc comm broadcast bytes not consistent");
    }

    if (!multiobj_->CreateMulticastObjByImportFd(rfd, bytes)) {
      throw std::runtime_error("hpc multimem obj create multicast obj by import fd fail");
    }
  }

  auto tensors = multiobj_->AllocateMemoryAndBindToMulticastObj();
  if (!tensors.ok) {
    throw std::runtime_error("hpc multimem obj allocate memory and bind to multicast obj fail");
  }

  return tensors;
}

}  // namespace communicator
}  // namespace hpc
