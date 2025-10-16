// Copyright 2025 hpc-ops authors

#ifndef SRC_COMMUNICATOR_MULTICAST_COMMUNICATOR_H_
#define SRC_COMMUNICATOR_MULTICAST_COMMUNICATOR_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>

#include "src/communicator/communicator.h"
#include "src/communicator/multicast_object_manager.h"
#include "src/communicator/type.h"

namespace hpc {
namespace communicator {

class MulticastCommunicator {
 public:
  MulticastCommunicator(int64_t rank, int64_t world_size, int64_t device_id = -1);
  ~MulticastCommunicator();

  MulticastTensors CreateTensorSync(int64_t bytes);
  void Barrier();

 private:
  int rank_;
  int world_size_;
  int device_id_;

  std::unique_ptr<Communicator> comm_;
  std::unique_ptr<MulticastObjectManager> multiobj_;
};

}  // namespace communicator
}  // namespace hpc

#endif  // SRC_COMMUNICATOR_MULTICAST_COMMUNICATOR_H_
