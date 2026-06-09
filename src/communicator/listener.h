// Copyright 2025 hpc-ops authors

#ifndef SRC_COMMUNICATOR_LISTENER_H_
#define SRC_COMMUNICATOR_LISTENER_H_

#include <memory>
#include <string>

#include "src/communicator/channel.h"

namespace hpc {
namespace communicator {

class Listener {
 public:
  Listener();
  ~Listener();

  bool Listen(const std::string &url);
  std::shared_ptr<Channel> Accept();
  void Close();

 private:
  int socket_;
};

}  // namespace communicator
}  // namespace hpc

#endif  // SRC_COMMUNICATOR_LISTENER_H_
