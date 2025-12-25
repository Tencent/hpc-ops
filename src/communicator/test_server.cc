// Copyright 2025 hpc-ops authors

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#include "src/communicator/channel.h"
#include "src/communicator/listener.h"

int main() {
  hpc::communicator::Listener listener;

  listener.Listen("tcp://0.0.0.0:10086");

  auto channel = listener.Accept();
  if (!channel) {
    printf("channel is none\n");
    return 0;
  }

  {
    bool ok = channel->Send("1234567890");
    printf("send.ok = %d\n", ok);
  }

  {
    int fd = open("/tmp/a.txt", O_CREAT | O_RDWR, 0666);
    bool ok = channel->SendFd(fd, "a file descriptor meta");
    printf("send.ok = %d fd = %d\n", ok, fd);
  }

  {
    bool ok = channel->Send("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    printf("send.ok = %d\n", ok);
  }

  while (1) {
    usleep(1000000);
  }

  return 0;
}
