// Copyright 2025 hpc-ops authors

#ifndef SRC_COMMUNICATOR_CHANNEL_H_
#define SRC_COMMUNICATOR_CHANNEL_H_

#include <string>

namespace hpc {
namespace communicator {

class Channel {
 public:
  Channel();
  explicit Channel(int sock);
  ~Channel();

  bool Send(const std::string &data);
  bool Recv(std::string *data);

  bool SendFd(int fd, const std::string &data);
  bool RecvFd(int *fd, std::string *data);

 private:
  int socket_;
};

}  // namespace communicator
}  // namespace hpc

#endif  // SRC_COMMUNICATOR_CHANNEL_H_
