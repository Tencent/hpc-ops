// Copyright 2025 hpc-ops authors

#ifndef SRC_COMMUNICATOR_MULTICAST_COMMUNICATOR_H_
#define SRC_COMMUNICATOR_MULTICAST_COMMUNICATOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "src/communicator/communicator.h"
#include "src/communicator/multicast_object_manager.h"

namespace hpc {
namespace communicator {

class MulticastCommunicator {
 public:
  MulticastCommunicator(int64_t rank, int64_t world_size, int64_t device_id = -1);
  ~MulticastCommunicator();

  bool CreateTensorSync(int64_t bytes, std::vector<std::shared_ptr<void>> *sptrs,
                        std::vector<int> *devices, std::shared_ptr<void> *multi_ptr,
                        int *multi_device);
  void Barrier();

 private:
  int rank_;
  int world_size_;
  int device_id_;

  std::unique_ptr<Communicator> comm_;
  std::unique_ptr<MulticastObjectManager> multimgr_;
};

}  // namespace communicator
}  // namespace hpc

#endif  // SRC_COMMUNICATOR_MULTICAST_COMMUNICATOR_H_
