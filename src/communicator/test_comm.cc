// Copyright 2025 hpc-ops authors

#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "src/communicator/communicator.h"

int main(int argc, char *argv[]) {
  int rank = std::stoi(argv[1]);
  int world_size = std::stoi(argv[2]);

  hpc::communicator::Communicator comm(rank, world_size);

  printf("\nTest Broadcast\n");
  {
    std::string data = "broadcast-data-from-rank" + std::to_string(rank);
    bool ok = comm.Broadcast(data, &data, 1);
    printf("rank = %d, broadcast.ok = %d, broadcast.data = %s\n", rank, ok, data.c_str());
  }

  printf("\nTest BroadcastFd\n");
  for (int root = 0; root < world_size; ++root) {
    std::string file = "/tmp/rank-" + std::to_string(rank) + ".txt";
    int fd = open(file.c_str(), O_CREAT | O_RDWR, 0666);
    write(fd, file.data(), file.size());
    struct stat lst;
    fstat(fd, &lst);

    std::string data = "file descriptor meta from rank-" + std::to_string(rank);
    int recv_fd = -1;
    bool ok = comm.BroadcastFd(fd, &recv_fd, data, &data, root);

    struct stat rst;
    fstat(recv_fd, &rst);
    printf("local-inode = %lld, remote-inode = %lld\n", lst.st_ino, rst.st_ino);
    printf("rank = %d, broadcast.ok = %d, broadcast.fd = %d , broadcast.data = %s\n", rank, ok,
           recv_fd, data.c_str());
  }

  printf("\nTest Allgather\n");
  {
    std::string data = "allgather-data-rank-" + std::to_string(rank);
    std::vector<std::string> recv_datas;
    comm.Allgather(data, &recv_datas);

    printf("rank %d, send: %s\n", rank, data.c_str());
    printf("recvs = [");
    for (auto &s : recv_datas) {
      printf("%s, ", s.c_str());
    }
    printf("]\n");
  }

  printf("\nTest AllgatherFd\n");
  {
    std::string send_data = "/tmp/allgather-data-rank-" + std::to_string(rank);
    int send_fd = open(send_data.c_str(), O_CREAT | O_RDWR, 0666);
    std::vector<std::string> recv_datas;
    std::vector<int> recv_fds;

    comm.AllgatherFd(send_fd, &recv_fds, send_data, &recv_datas);

    printf("rank %d, send_fd: %d, send_data: %s\n", rank, send_fd, send_data.c_str());

    printf("recv_fds = [");
    for (auto &fd : recv_fds) {
      printf("%d, ", fd);
    }
    printf("]\n");

    printf("recv_datas = [");
    for (auto &s : recv_datas) {
      printf("%s, ", s.c_str());
    }
    printf("]\n");
  }

  return 0;
}
