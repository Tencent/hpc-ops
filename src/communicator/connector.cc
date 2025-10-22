// Copyright 2025 hpc-ops authors

#include "src/communicator/connector.h"

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <memory>
#include <string>

#include "src/communicator/channel.h"

namespace hpc {
namespace communicator {

Connector::Connector() {}

Connector::~Connector() {}

std::shared_ptr<Channel> Connector::Connect(const std::string& file) {
  int sock = socket(AF_UNIX, SOCK_STREAM, 0);

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;

  addr.sun_path[0] = 0;
  strncpy(addr.sun_path + 1, file.c_str(), sizeof(addr.sun_path) - 2);

  socklen_t addr_len = offsetof(struct sockaddr_un, sun_path) + 1 + file.size();

  constexpr int kTimeWait = 50000;  // 50ms
  while (true) {
    int r = connect(sock, (struct sockaddr*)&addr, addr_len);
    if (r == 0) {
      break;
    }
    usleep(kTimeWait);
  }

  auto channel = std::make_shared<Channel>(sock);

  return channel;
}

std::shared_ptr<Channel> Connector::ConnectMayFail(const std::string& file) {
  int sock = socket(AF_UNIX, SOCK_STREAM, 0);

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;

  addr.sun_path[0] = 0;
  strncpy(addr.sun_path + 1, file.c_str(), sizeof(addr.sun_path) - 2);

  socklen_t addr_len = offsetof(struct sockaddr_un, sun_path) + 1 + file.size();
  int r = connect(sock, (struct sockaddr*)&addr, addr_len);
  if (r != 0) {
    return nullptr;
  }

  auto channel = std::make_shared<Channel>(sock);

  return channel;
}

}  // namespace communicator
}  // namespace hpc
