// Copyright 2025 hpc-ops authors

#ifndef SRC_COMMUNICATOR_PROTOCOL_H_
#define SRC_COMMUNICATOR_PROTOCOL_H_

#include <string>

namespace hpc {
namespace communicator {

enum Protocol { kUnix, kTcp, kUnknown };

// supported protocol example:
// tcp://192.2.1.100:10010
// unix://a/bc/d.sock
Protocol parse_proto(const std::string &url);
bool parse_tcp(const std::string &url, std::string *ip, int *port);
bool parse_unix(const std::string &url, std::string *file);

}  // namespace communicator
}  // namespace hpc

#endif  // SRC_COMMUNICATOR_PROTOCOL_H_
