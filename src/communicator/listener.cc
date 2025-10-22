// Copyright 2025 hpc-ops authors

#include "src/communicator/listener.h"

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
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

Listener::Listener() { socket_ = -1; }

Listener::~Listener() {
  if (socket_ != -1) {
    Close();
  }
}

bool Listener::Listen(const std::string &file) {
  socket_ = socket(AF_UNIX, SOCK_STREAM, 0);

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;

  addr.sun_path[0] = 0;
  strncpy(addr.sun_path + 1, file.c_str(), sizeof(addr.sun_path) - 2);

  socklen_t addr_len = offsetof(struct sockaddr_un, sun_path) + 1 + file.size();
  int r = bind(socket_, (struct sockaddr *)&addr, addr_len);
  if (r != 0) {
    return false;
  }

  r = listen(socket_, 128);
  if (r != 0) {
    return false;
  }

  return true;
}

std::shared_ptr<Channel> Listener::Accept() {
  int c = accept(socket_, NULL, NULL);
  if (c < 0) {
    return nullptr;
  }

  auto channel = std::make_shared<Channel>(c);
  return channel;
}

void Listener::Close() {
  ::shutdown(socket_, SHUT_RD);
  ::close(socket_);
  socket_ = -1;
}

}  // namespace communicator
}  // namespace hpc
