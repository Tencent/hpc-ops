// Copyright 2025 hpc-ops authors

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#include <string>

#include "src/communicator/communicator.h"

int main(int argc, char *argv[]) {
  int rank = std::stoi(argv[1]);
  int world_size = std::stoi(argv[2]);

  hpc::communicator::Communicator comm(rank, world_size);

  {
    std::string data = "broadcast-data-from-rank" + std::to_string(rank);
    bool ok = comm.Broadcast(data, &data, 0);
    printf("rank = %d, broadcast.ok = %d, broadcast.data = %s\n", rank, ok, data.c_str());
  }

  {
    int fd = open("/tmp/a.txt", O_CREAT | O_RDWR, 0666);
    std::string data = "file descriptor meta from rank-" + std::to_string(rank);
    int recv_fd = -1;
    bool ok = comm.BroadcastFd(fd, &recv_fd, data, &data, 0);
    printf("rank = %d, broadcast.ok = %d, broadcast.fd = %d, broadcast.data = %s\n", rank, ok,
           recv_fd, data.c_str());
  }

  return 0;
}
