// Copyright 2025 hpc-ops authors

#ifndef SRC_COMMUNICATOR_TYPE_H_
#define SRC_COMMUNICATOR_TYPE_H_

#include <memory>
#include <tuple>

namespace hpc {
namespace communicator {

struct MulticastTensors {
  std::shared_ptr<void> multi_ptr;
  std::shared_ptr<void> local_ptr;
  int multi_device;
  int local_device;
  bool ok;
};

}  // namespace communicator
}  // namespace hpc

#endif  // SRC_COMMUNICATOR_TYPE_H_
