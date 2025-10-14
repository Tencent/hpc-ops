// Copyright 2025 hpc-ops authors

#ifndef SRC_COMMUNICATOR_COMMUNICATOR_H_
#define SRC_COMMUNICATOR_COMMUNICATOR_H_

#include <map>
#include <memory>
#include <string>

#include "src/communicator/channel.h"

namespace hpc {
namespace communicator {

class Communicator {
 public:
  Communicator(int rank, int world_size);
  ~Communicator();

  bool Broadcast(const std::string &send_data, std::string *recv_data, int root = 0);
  bool BroadcastFd(const int send_fd, int *recv_fd, const std::string &send_data,
                   std::string *recv_data, int root = 0);

  void Barrier();

 private:
  const std::string kRegistery_ = "hpc-comm.sock";

  int rank_;
  int world_size_;

  int root_;

  std::shared_ptr<Channel> channel_;
  std::map<int, std::shared_ptr<Channel>> channels_;
};

}  // namespace communicator
}  // namespace hpc

#endif  // SRC_COMMUNICATOR_COMMUNICATOR_H_
