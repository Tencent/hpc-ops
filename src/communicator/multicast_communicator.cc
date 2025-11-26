// Copyright 2025 hpc-ops authors

#include "src/communicator/multicast_communicator.h"

#include <unistd.h>

#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "src/communicator/communicator.h"
#include "src/communicator/multicast_object_manager.h"

namespace hpc {
namespace communicator {

MulticastCommunicator::MulticastCommunicator(int rank, int world_size, int device_id,
                                             const std::string &comm_name) {
  rank_ = rank;
  world_size_ = world_size;

  if (device_id == -1) {
    device_id_ = rank;
  } else {
    device_id_ = device_id;
  }

  comm_ = std::make_unique<Communicator>(rank, world_size, comm_name);
  multimgr_ = std::make_unique<MulticastObjectManager>(device_id_, world_size_);
}

MulticastCommunicator::~MulticastCommunicator() {
  multimgr_.reset();
  comm_.reset();
}

void MulticastCommunicator::Barrier() { comm_->Barrier(); }

int64_t MulticastCommunicator::GetRank() { return rank_; }

int64_t MulticastCommunicator::GetWorldSize() { return world_size_; }

int64_t MulticastCommunicator::GetDeviceId() { return device_id_; }

bool MulticastCommunicator::CreateTensorSync(int64_t bytes,
                                             std::vector<std::shared_ptr<void>> *sptrs,
                                             std::vector<int> *devices,
                                             std::shared_ptr<void> *multi_sptr, int *multi_device) {
  sptrs->clear();
  sptrs->resize(world_size_);
  devices->clear();
  devices->resize(world_size_);

  std::set<int> all_fds;

  // 1. create memory object and export fd
  int fd = -1;
  std::shared_ptr<void> obj;
  int device = -1;
  if (!multimgr_->CreateMemoryObjAndExportFd(&fd, bytes, &obj, &device)) {
    throw std::runtime_error("hpc create memory obj and export fd fail");
  }
  all_fds.insert(fd);

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
    int fd = recv_fds[rank];
    all_fds.insert(fd);

    if (rank == rank_) {
      continue;
    }

    std::shared_ptr<void> obj;
    if (!multimgr_->CreateMemoryObjByImportFd(fd, bytes, &obj, &device)) {
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
    if (!multimgr_->CreateMulticastHandleAndExportFd(&fd, bytes, &multi_handle)) {
      throw std::runtime_error("hpc multimem obj create multicast obj and export fd fail");
    }
    all_fds.insert(fd);

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

    if (!multimgr_->CreateMulticastHandleByImportFd(rfd, bytes, &multi_handle)) {
      throw std::runtime_error("hpc multimem obj create multicast obj by import fd fail");
    }
    all_fds.insert(rfd);
  }

  std::shared_ptr<void> multi_obj;
  if (!multimgr_->MapHandleToMulticastObj(multi_handle, &multi_obj, multi_device, bytes)) {
    throw std::runtime_error("hpc multimem map handle to addressable obj fail");
  }

  // 4. bind multicast obj with local memory obj
  if (!multimgr_->BindLocalMemoryObjToMulticastObj(obj, device, multi_obj, *multi_device, bytes)) {
    throw std::runtime_error("hpc multimem obj create multicast obj by import fd fail");
  }
  *multi_sptr = multi_obj;

  for (const auto &fd : all_fds) {
    ::close(fd);
  }

  return true;
}

}  // namespace communicator
}  // namespace hpc
