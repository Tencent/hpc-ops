// Copyright 2025 hpc-ops authors

#include <stdio.h>

#include <memory>
#include <string>

#include "src/communicator/multicast_communicator.h"

int main(int argc, char *argv[]) {
  int rank = std::stoi(argv[1]);
  int world_size = std::stoi(argv[2]);

  auto comm = std::make_shared<hpc::communicator::MulticastCommunicator>(rank, world_size);

  {
    int multi_device = -1;
    int local_device = -1;

    int64_t bytes = 1024;
    auto tensors = comm->CreateTensorSync(bytes);

    printf("rank = %d, ok = %d\n", rank, tensors.ok);
  }

  return 0;
}
