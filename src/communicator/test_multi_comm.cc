// Copyright 2025 hpc-ops authors

#include <stdio.h>

#include <memory>
#include <string>
#include <vector>

#include "src/communicator/multicast_communicator.h"

int main(int argc, char *argv[]) {
  int rank = std::stoi(argv[1]);
  int world_size = std::stoi(argv[2]);

  auto comm = std::make_shared<hpc::communicator::MulticastCommunicator>(rank, world_size);

  {
    int local_device = -1;

    int64_t bytes = 1024;

    std::vector<std::shared_ptr<void>> sptrs;
    std::vector<int> devices;
    std::shared_ptr<void> multi_sptr;
    int multi_device = -1;

    bool ok = comm->CreateTensorSync(bytes, &sptrs, &devices, &multi_sptr, &multi_device);

    printf("rank = %d, ok = %d\n", rank, ok);
    printf(" multi: %p@CUDA%d\n", multi_sptr.get(), multi_device);
    for (int rank = 0; rank < sptrs.size(); ++rank) {
      printf("  %d: %p@CUDA%d\n", rank, sptrs[rank].get(), devices[rank]);
    }
  }

  return 0;
}
