// Copyright 2025 hpc-ops authors

#ifndef SRC_COMMUNICATOR_MULTINODE_COMMUNICATOR_H_
#define SRC_COMMUNICATOR_MULTINODE_COMMUNICATOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "src/communicator/communicator.h"

namespace hpc {
namespace communicator {

class MultinodeCommunicator {
 public:
  MultinodeCommunicator(int rank, int world_size, int device_id, const std::string &comm_name);
  ~MultinodeCommunicator();

  bool CreateShmemSync(int64_t bytes, void *&ptrs);
  void Barrier();
  int64_t GetRank();

  int64_t GetWorldSize();

  int64_t GetDeviceId();

 private:
  int rank_;
  int world_size_;
  int device_id_;
  std::vector<void *> shmem_ptrs_;

  std::unique_ptr<Communicator> comm_;
};

}  // namespace communicator
}  // namespace hpc

#endif  // SRC_COMMUNICATOR_MULTINODE_COMMUNICATOR_H_
