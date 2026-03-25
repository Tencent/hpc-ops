// Copyright 2025 hpc-ops authors

#include <stdio.h>

#include <memory>
#include <string>
#include <vector>

#include "src/communicator/multinode_communicator.h"

__global__ void simple_shift(int *destination) {
  int mype = ucl::shmem_my_pe();
  int npes = ucl::shmem_n_pes();
  int peer = (mype + 1) % npes;

  ucl::shmem_int_p(destination, mype, peer);
}

int main(int argc, char *argv[]) {
  int rank = std::stoi(argv[1]);
  int world_size = std::stoi(argv[2]);
  int device_id = std::stoi(argv[3]);
  const std::string comm_name = std::string(argv[4]);

  int msg;
  cudaStream_t stream;

  cudaSetDevice(device_id);
  cudaStreamCreate(&stream);

  auto comm =
      std::make_shared<hpc::communicator::MultiNodeCommunicator>(rank, world_size, rank, comm_name);

  std::vector<std::shared_ptr<void>> sptrs;
  std::vector<int> devices;
  std::shared_ptr<void> multinode_sptr;
  std::shared_ptr<void> multicast_sptr;
  int multi_device = -1;
  int64_t sub_team = -1;
  std::shared_ptr<void> subgroup_multicast_sptr;

  comm->CreateTensorSync(sizeof(msg), &sptrs, &devices, &multinode_sptr, &multicast_sptr,
                         &multi_device, sub_team, &subgroup_multicast_sptr);
  cudaMemset((int *)multinode_sptr.get(), 0, sizeof(int));

  simple_shift<<<1, 1, 0, stream>>>((int *)multinode_sptr.get());
  comm->BarrierOnStream(stream);
  cudaMemcpyAsync(&msg, multinode_sptr.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);
  printf("%d: received message %d\n", ucl::shmem_my_pe(), msg);

  return 0;
}
