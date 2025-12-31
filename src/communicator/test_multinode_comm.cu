// Copyright 2025 hpc-ops authors

#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdio.h>

#include <memory>
#include <string>

#include "src/communicator/multinode_communicator.h"

#define CUDA_CHECK(stmt)                                                    \
  do {                                                                      \
    cudaError_t result = (stmt);                                            \
    if (cudaSuccess != result) {                                            \
      fprintf(stderr, "[%s:%d] CUDA failed with %s \n", __FILE__, __LINE__, \
              cudaGetErrorString(result));                                  \
      exit(-1);                                                             \
    }                                                                       \
  } while (0)

__global__ void simple_shift(int *destination) {
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int peer = (mype + 1) % npes;

  nvshmem_int_p(destination, mype, peer);
}

int main(int argc, char *argv[]) {
  int rank = std::stoi(argv[1]);
  int world_size = std::stoi(argv[2]);
  int device_id = std::stoi(argv[3]);
  const std::string comm_name = std::string(argv[4]);

  int msg;
  cudaStream_t stream;

  CUDA_CHECK(cudaSetDevice(device_id));
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto comm =
      std::make_shared<hpc::communicator::MultinodeCommunicator>(rank, world_size, rank, comm_name);

  void *destination = nullptr;
  comm->CreateShmemSync(sizeof(int), destination);
  CUDA_CHECK(cudaMemset((int *)destination, 0, sizeof(int)));

  simple_shift<<<1, 1, 0, stream>>>((int *)destination);
  nvshmemx_barrier_all_on_stream(stream);
  CUDA_CHECK(cudaMemcpyAsync(&msg, destination, sizeof(int), cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));
  printf("%d: received message %d\n", nvshmem_my_pe(), msg);

  return 0;
}
