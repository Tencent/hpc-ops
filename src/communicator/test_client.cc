// Copyright 2025 hpc-ops authors

#include <stdio.h>

#include <string>

#include "src/communicator/channel.h"
#include "src/communicator/connector.h"

int main() {
  auto channel = hpc::communicator::Connector::Connect("tcp://0.0.0.0:10086");

  if (!channel) {
    printf("channel is none\n");
    return 0;
  }

  {
    std::string data;
    bool ok = channel->Recv(&data);
    printf("recv.ok = %d, .data = [%s]\n", ok, data.c_str());
  }

  {
    int fd = -1;
    std::string data;
    bool ok = channel->RecvFd(&fd, &data);
    printf("recv.ok = %d, .data = [%s] .fd = %d\n", ok, data.c_str(), fd);
  }

  {
    std::string data;
    bool ok = channel->Recv(&data);
    printf("recv.ok = %d, .data =[%s]\n", ok, data.c_str());
  }

  return 0;
}
