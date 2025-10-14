// Copyright 2025 hpc-ops authors

#ifndef SRC_COMMUNICATOR_MULTICAST_COMM_H_
#define SRC_COMMUNICATOR_MULTICAST_COMM_H_

#include <torch/all.h>
#include <torch/custom_class.h>

#include <map>
#include <memory>
#include <string>
#include <tuple>

#include "src/communicator/communicator.h"
#include "src/communicator/multicast_object_manager.h"

namespace hpc {
namespace communicator {

class MulticastComm : public torch::CustomClassHolder {
 public:
  MulticastComm(int64_t rank, int64_t world_size, int64_t device_id = -1, int64_t root = 0);
  ~MulticastComm();

  std::tuple<torch::Tensor, torch::Tensor> CreateTensorSync(int64_t size);
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

#endif  // SRC_COMMUNICATOR_MULTICAST_COMM_H_
