// Copyright 2025 hpc-ops authors

#include "src/communicator/connector.h"

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

Connector::Connector() {}

Connector::~Connector() {}

static std::shared_ptr<Channel> connect_unix(const std::string& file) {
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

static std::shared_ptr<Channel> connect_tcp(const std::string& ip, int port) {
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    return nullptr;
  }

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);

  if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0) {
    close(sock);
    return nullptr;
  }

  constexpr int kTimeWait = 50000;
  while (true) {
    int r = connect(sock, (struct sockaddr*)&addr, sizeof(addr));
    if (r == 0) {
      break;
    }
    usleep(kTimeWait);
  }

  auto channel = std::make_shared<Channel>(sock);
  return channel;
}

std::shared_ptr<Channel> Connector::Connect(const std::string& url) {
  auto proto = parse_proto(url);

  if (proto == kTcp) {
    std::string ip;
    int port;
    if (parse_tcp(url, &ip, &port)) {
      return connect_tcp(ip, port);
    }
  } else if (proto == kUnix) {
    std::string file;
    if (parse_unix(url, &file)) {
      return connect_unix(file);
    }
  } else {
    std::string err =
        "unknown protocol: " + url + ", we only support tcp://1.1.1.1:8080 and unix://a/b/c form";
    throw std::invalid_argument(err);
  }

  return nullptr;
}

}  // namespace communicator
}  // namespace hpc
