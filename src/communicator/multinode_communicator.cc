// Copyright 2025 hpc-ops authors
#include "src/communicator/multinode_communicator.h"

#include <unistd.h>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "src/communicator/communicator.h"
#include "ucl/shmem.h"

namespace hpc {
namespace communicator {

MultiNodeCommunicator::MultiNodeCommunicator(int rank, int world_size, int device_id,
                                             const std::string &comm_name) {
  rank_ = rank;
  world_size_ = world_size;

  if (device_id == -1) {
    device_id_ = rank;
  } else {
    device_id_ = device_id;
  }
  const std::string url = std::string("tcp://") + comm_name;
  comm_ = std::make_shared<Communicator>(rank, world_size, url);

  std::string send_data;
  std::string recv_data;
  ucl::shmemx_uniqueid_t send_uid = ucl::SHMEMX_UNIQUEID_INITIALIZER;
  ucl::shmemx_uniqueid_t recv_uid = ucl::SHMEMX_UNIQUEID_INITIALIZER;

  cudaSetDevice(device_id_);
  if (rank_ == 0) {
    ucl::shmemx_get_uniqueid(&send_uid);
    send_data = std::string(reinterpret_cast<char *>(&send_uid), sizeof(ucl::shmemx_uniqueid_t));
  }
  comm_->Broadcast(send_data, &recv_data, rank);

  memcpy(&recv_uid, recv_data.data(), sizeof(ucl::shmemx_uniqueid_t));

  ucl::shmemx_init_attr_t attr = ucl::SHMEMX_INIT_ATTR_INITIALIZER;
  ucl::shmemx_set_attr_uniqueid_args(rank_, world_size_, &recv_uid, &attr);
  ucl::shmemx_init_attr(ucl::SHMEMX_INIT_WITH_UNIQUEID, &attr);
}

ucl::shmem_team_t MultiNodeCommunicator::CreateSubTeam(int subgroup_size) {
  int local_rank = ucl::shmem_team_my_pe(ucl::SHMEMX_TEAM_NODE);
  int local_size = ucl::shmem_team_n_pes(ucl::SHMEMX_TEAM_NODE);

  int subgroup_id_ = local_rank / subgroup_size;
  int subgroup_start = subgroup_id_ * subgroup_size;
  int subgroup_actual_size = std::min(subgroup_size, local_size - subgroup_start);

  ucl::shmem_team_t subgroup_team_;
  int ret = ucl::shmem_team_split_strided(ucl::SHMEMX_TEAM_NODE, subgroup_start, 1,
                                          subgroup_actual_size, nullptr, 0, &subgroup_team_);

  if (ret != 0 || subgroup_team_ == ucl::SHMEM_TEAM_INVALID) {
    throw std::runtime_error("Failed to create subgroup team");
  }

  return subgroup_team_;
}

MultiNodeCommunicator::~MultiNodeCommunicator() {
  comm_.reset();
  ucl::shmem_finalize();
}

void MultiNodeCommunicator::Barrier() { ucl::shmem_barrier_all(); }

void MultiNodeCommunicator::BarrierOnStream(cudaStream_t stream) {
  ucl::shmemx_barrier_all_on_stream(stream);
}

int64_t MultiNodeCommunicator::GetRank() { return rank_; }

int64_t MultiNodeCommunicator::GetWorldSize() { return world_size_; }

int64_t MultiNodeCommunicator::GetDeviceId() { return device_id_; }

bool MultiNodeCommunicator::CreateTensorSync(
    int64_t bytes, std::vector<std::shared_ptr<void>> *sptrs, std::vector<int> *devices,
    std::shared_ptr<void> *multinode_sptr, std::shared_ptr<void> *multicast_sptr, int *multi_device,
    int64_t sub_team, std::shared_ptr<void> *subgroup_multicast_sptr) {
  if (sub_team == -1) {
    sub_team = ucl::SHMEM_TEAM_WORLD;
  }

  sptrs->clear();
  sptrs->resize(world_size_);
  devices->clear();
  devices->resize(world_size_);

  *multi_device = device_id_;
  void *ptr = ucl::shmem_align(ALIGNMENT_BYTES, bytes);
  if (ptr == nullptr) {
    throw std::runtime_error("CreateTensorSync fail");
  }

  cudaMemset(ptr, 0, bytes);
  cudaDeviceSynchronize();

  auto comm_weak = std::weak_ptr<Communicator>(comm_);
  *multicast_sptr =
      std::shared_ptr<void>(ucl::shmemx_mc_ptr(ucl::SHMEMX_TEAM_NODE, ptr), [](void *p) {});

  ucl::shmem_team_t subgroup_team_ = static_cast<ucl::shmem_team_t>(sub_team);
  if (subgroup_team_ != ucl::SHMEM_TEAM_WORLD) {
    void *subgroup_mc_ptr = ucl::shmemx_mc_ptr(subgroup_team_, ptr);
    *subgroup_multicast_sptr = std::shared_ptr<void>(subgroup_mc_ptr, [](void *p) {});
  } else {
    *subgroup_multicast_sptr = nullptr;
  }

  *multinode_sptr = std::shared_ptr<void>(ptr, [comm_weak](void *p) {
    if (!comm_weak.expired()) {
      ucl::shmem_free(p);
    }
  });

  for (int i = 0; i < world_size_; ++i) {
    if (i == rank_) {
      (*sptrs)[i] = *multinode_sptr;
    } else {
      void *remote_ptr = ucl::shmem_ptr(ptr, i);
      (*sptrs)[i] = std::shared_ptr<void>(remote_ptr, [](void *p) {});
    }
    (*devices)[i] = i;
  }
  return true;
}
}  // namespace communicator
}  // namespace hpc
