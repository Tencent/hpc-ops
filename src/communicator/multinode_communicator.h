// Copyright 2025 hpc-ops authors

#ifndef SRC_COMMUNICATOR_MULTINODE_COMMUNICATOR_H_
#define SRC_COMMUNICATOR_MULTINODE_COMMUNICATOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "src/communicator/communicator.h"
#include "ucl/shmem.h"

#define ALIGNMENT_BYTES 128

namespace hpc {
namespace communicator {

class MultiNodeCommunicator {
 public:
  MultiNodeCommunicator(int rank, int world_size, int device_id, const std::string &comm_name);
  ~MultiNodeCommunicator();

  void Barrier();
  void BarrierOnStream(cudaStream_t stream);

  int64_t GetRank();
  int64_t GetWorldSize();
  int64_t GetDeviceId();

  ucl::shmem_team_t CreateSubTeam(int subgroup_size);

  bool CreateTensorSync(int64_t bytes, std::vector<std::shared_ptr<void>> *sptrs,
                        std::vector<int> *devices, std::shared_ptr<void> *multinode_sptr,
                        std::shared_ptr<void> *multicast_sptr, int *multi_device, int64_t sub_team,
                        std::shared_ptr<void> *subgroup_multicast_sptr);

 private:
  int rank_;
  int world_size_;
  int device_id_;

  std::shared_ptr<Communicator> comm_;
};

}  // namespace communicator
}  // namespace hpc

#endif  // SRC_COMMUNICATOR_MULTINODE_COMMUNICATOR_H_
