// Copyright 2025 hpc-ops authors

#ifndef SRC_COMMUNICATOR_COMMUNICATOR_H_
#define SRC_COMMUNICATOR_COMMUNICATOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "src/communicator/channel.h"

namespace hpc {
namespace communicator {

class Communicator {
 public:
  Communicator(int rank, int world_size, const std::string &group_name = "hpc-comm.sock");
  ~Communicator();

  bool Broadcast(const std::string &send_data, std::string *recv_data, int root = 0);
  bool BroadcastFd(const int send_fd, int *recv_fd, const std::string &send_data,
                   std::string *recv_data, int root = 0);

  bool Allgather(const std::string &send_data, std::vector<std::string> *recv_datas);
  bool AllgatherFd(const int send_fd, std::vector<int> *recv_fds, const std::string &send_data,
                   std::vector<std::string> *recv_datas);

  void Barrier();

 private:
  const std::string kRegistery_;

  int rank_;
  int world_size_;

  int root_;

  std::shared_ptr<Channel> channel_;
  std::map<int, std::shared_ptr<Channel>> channels_;
};

}  // namespace communicator
}  // namespace hpc

#endif  // SRC_COMMUNICATOR_COMMUNICATOR_H_
