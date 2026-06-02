// Copyright 2025 hpc-ops authors

#include "src/communicator/protocol.h"

#include <string>

namespace hpc {
namespace communicator {

Protocol parse_proto(const std::string &url) {
  const std::string kTcpPrefix = "tcp://";
  const std::string kUnixPrefix = "unix://";

  if (url.compare(0, kTcpPrefix.length(), kTcpPrefix) == 0) {
    return kTcp;
  } else if (url.compare(0, kUnixPrefix.length(), kUnixPrefix) == 0) {
    return kUnix;
  }

  return kUnknown;
}

bool parse_tcp(const std::string &url, std::string *ip, int *port) {
  const std::string kTcpPrefix = "tcp://";

  if (url.compare(0, kTcpPrefix.length(), kTcpPrefix) != 0) {
    return false;
  }

  const std::string ip_and_port = url.substr(kTcpPrefix.length());

  auto colon_pos = ip_and_port.find(":");
  if (colon_pos == std::string::npos) {
    return false;
  }

  const std::string sip = ip_and_port.substr(0, colon_pos);
  const std::string sport = ip_and_port.substr(colon_pos + 1);

  int lport = -1;
  try {
    lport = std::stoi(sport);
  } catch (const std::exception &e) {
    return false;
  }

  if (lport < 1 || lport > 65535) {
    return false;
  }

  *ip = sip;
  *port = lport;

  return true;
}

bool parse_unix(const std::string &url, std::string *file) {
  const std::string kUnixPrefix = "unix://";

  if (url.compare(0, kUnixPrefix.length(), kUnixPrefix)) {
    return false;
  }

  const std::string name = url.substr(kUnixPrefix.length());

  *file = name;

  return true;
}

}  // namespace communicator
}  // namespace hpc
