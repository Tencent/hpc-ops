// Copyright 2025 hpc-ops authors

#include "src/communicator/listener.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "src/communicator/channel.h"
#include "src/communicator/protocol.h"

namespace hpc {
namespace communicator {

static bool listen_tcp(int *sock, const std::string &ip, int port) {
  int s = socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0) {
    return false;
  }

  int opt = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);

  if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0) {
    close(s);
    return false;
  }

  int r = bind(s, (struct sockaddr *)&addr, sizeof(addr));
  if (r != 0) {
    close(s);
    return false;
  }

  r = listen(s, 128);
  if (r != 0) {
    close(s);
    return false;
  }

  *sock = s;
  return true;
}

static bool listen_unix(int *sock, const std::string &file) {
  int s = socket(AF_UNIX, SOCK_STREAM, 0);

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;

  addr.sun_path[0] = 0;
  strncpy(addr.sun_path + 1, file.c_str(), sizeof(addr.sun_path) - 2);

  socklen_t addr_len = offsetof(struct sockaddr_un, sun_path) + 1 + file.size();
  int r = bind(s, (struct sockaddr *)&addr, addr_len);
  if (r != 0) {
    return false;
  }

  r = listen(s, 128);
  if (r != 0) {
    return false;
  }

  *sock = s;
  return true;
}

// =======================
// class implementation
// =======================

Listener::Listener() { socket_ = -1; }

Listener::~Listener() {
  if (socket_ != -1) {
    Close();
  }
}

bool Listener::Listen(const std::string &url) {
  auto proto = parse_proto(url);

  if (proto == kTcp) {
    std::string ip;
    int port;
    if (parse_tcp(url, &ip, &port)) {
      return listen_tcp(&socket_, ip, port);
    }
  } else if (proto == kUnix) {
    std::string file;
    if (parse_unix(url, &file)) {
      return listen_unix(&socket_, file);
    }
  } else {
    std::string err =
        "unknown protocol: " + url + ", we only support tcp://1.1.1.1:8080 and unix://a/b/c form";
    throw std::invalid_argument(err);
  }

  return false;
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
