// Copyright 2025 hpc-ops authors
#include "src/communicator/multinode_communicator.h"

#include <nvshmem.h>
#include <nvshmemx.h>
#include <unistd.h>

#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "src/communicator/communicator.h"

namespace hpc {
namespace communicator {

MultinodeCommunicator::MultinodeCommunicator(int rank, int world_size, int device_id,
                                             const std::string &comm_name) {
  rank_ = rank;
  world_size_ = world_size;

  if (device_id == -1) {
    device_id_ = rank;
  } else {
    device_id_ = device_id;
  }
  const std::string url = std::string("tcp://") + comm_name;
  comm_ = std::make_unique<Communicator>(rank, world_size, url);

  std::string send_data;
  std::string recv_data;
  nvshmemx_uniqueid_t send_uid = NVSHMEMX_UNIQUEID_INITIALIZER;
  nvshmemx_uniqueid_t recv_uid = NVSHMEMX_UNIQUEID_INITIALIZER;

  if (rank_ == 0) {
    nvshmemx_get_uniqueid(&send_uid);
    send_data = std::string(reinterpret_cast<char *>(&send_uid), sizeof(nvshmemx_uniqueid_t));
  }
  comm_->Broadcast(send_data, &recv_data, rank);

  memcpy(&recv_uid, recv_data.data(), sizeof(nvshmemx_uniqueid_t));

  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  nvshmemx_set_attr_uniqueid_args(rank_, world_size_, &recv_uid, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
}

MultinodeCommunicator::~MultinodeCommunicator() {
  for (auto ptr : shmem_ptrs_) {
    nvshmem_free(ptr);
  }

  comm_.reset();
  nvshmem_finalize();
}

void MultinodeCommunicator::Barrier() {}

int64_t MultinodeCommunicator::GetRank() { return rank_; }

int64_t MultinodeCommunicator::GetWorldSize() { return world_size_; }

int64_t MultinodeCommunicator::GetDeviceId() { return device_id_; }

bool MultinodeCommunicator::CreateShmemSync(int64_t bytes, void *&ptrs) {
  ptrs = nvshmem_malloc(bytes);
  if (ptrs == nullptr) {
    throw std::runtime_error("CreateShmemSync fail");
  }
  shmem_ptrs_.push_back(ptrs);
  return true;
}

}  // namespace communicator
}  // namespace hpc
