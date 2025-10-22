// Copyright 2025 hpc-ops authors

#include "src/communicator/multicast_communicator.h"

#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "src/communicator/communicator.h"

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

bool MulticastCommunicator::CreateTensorSync(int64_t bytes,
                                             std::vector<std::shared_ptr<void>> *sptrs,
                                             std::vector<int> *devices,
                                             std::shared_ptr<void> *multi_sptr, int *multi_device) {
  sptrs->clear();
  sptrs->resize(world_size_);
  devices->clear();
  devices->resize(world_size_);

  // 1. create memory object and export fd
  int fd = -1;
  std::shared_ptr<void> obj;
  int device = -1;
  if (!multiobj_->CreateMemoryObjAndExportFd(&fd, bytes, &obj, &device)) {
    throw std::runtime_error("hpc create memory obj and export fd fail");
  }

  std::vector<int> recv_fds;
  std::string send_data = "MemoryObj-From-Rank-" + std::to_string(rank_);
  std::vector<std::string> recv_datas;
  if (!comm_->AllgatherFd(fd, &recv_fds, send_data, &recv_datas)) {
    throw std::runtime_error("hpc all gather fd fail");
  }

  (*sptrs)[rank_] = obj;
  (*devices)[rank_] = device;

  // 2. create memory by import fd
  for (int rank = 0; rank < world_size_; ++rank) {
    if (rank == rank_) {
      continue;
    }

    int fd = recv_fds[rank];
    std::shared_ptr<void> obj;
    if (!multiobj_->CreateMemoryObjByImportFd(fd, bytes, &obj, &device)) {
      throw std::runtime_error("hpc create memory obj by import fd fail");
    }

    (*sptrs)[rank] = obj;
    (*devices)[rank] = device;
  }

  // 3. create multicast obj
  // i. root create multi object and export fd, then broadcast it to non-root
  // ii. non-root broadcast get the fd and import it

  constexpr int root_ = 0;
  std::shared_ptr<void> multi_handle;
  if (rank_ == root_) {
    // get multimem fd and broadcast it
    int fd = -1;
    if (!multiobj_->CreateMulticastObjAndExportFd(&fd, bytes, &multi_handle)) {
      throw std::runtime_error("hpc multimem obj create multicast obj and export fd fail");
    }

    std::string data = std::to_string(bytes);
    int rfd = -1;
    if (!comm_->BroadcastFd(fd, &rfd, data, &data, root_)) {
      throw std::runtime_error("hpc comm broadcast fail");
    }
  } else {
    int fd = -1;
    int rfd = -1;
    std::string data;

    if (!comm_->BroadcastFd(fd, &rfd, data, &data, root_)) {
      throw std::runtime_error("hpc comm broadcast fail");
    }

    int64_t bytes_from_root = std::stoll(data);
    if (bytes_from_root != bytes) {
      throw std::runtime_error("hpc comm broadcast bytes not consistent");
    }

    if (!multiobj_->CreateMulticastObjByImportFd(rfd, bytes, &multi_handle)) {
      throw std::runtime_error("hpc multimem obj create multicast obj by import fd fail");
    }
  }

  std::shared_ptr<void> multi_obj;
  if (!multiobj_->MapHandleToAddresableObj(multi_handle, &multi_obj, multi_device, bytes)) {
    throw std::runtime_error("hpc multimem map handle to addressable obj fail");
  }

  // 4. bind multicast obj with local memory obj
  if (!multiobj_->BindLocalMemoryObjToMulticastObj(obj, device, multi_obj, *multi_device, bytes)) {
    throw std::runtime_error("hpc multimem obj create multicast obj by import fd fail");
  }

  *multi_sptr = multi_obj;

  return true;
}

}  // namespace communicator
}  // namespace hpc
