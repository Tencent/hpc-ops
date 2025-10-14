// Copyright 2025 hpc-ops authors

#ifndef SRC_COMMUNICATOR_CONNECTOR_H_
#define SRC_COMMUNICATOR_CONNECTOR_H_

#include <memory>
#include <string>

#include "src/communicator/channel.h"

namespace hpc {
namespace communicator {

class Connector {
 public:
  Connector();
  ~Connector();

  static std::shared_ptr<Channel> Connect(const std::string& file);
  static std::shared_ptr<Channel> ConnectMayFail(const std::string& file);

 private:
};

}  // namespace communicator
}  // namespace hpc

#endif  // SRC_COMMUNICATOR_CONNECTOR_H_
