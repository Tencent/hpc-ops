// Copyright 2025 hpc-ops authors

#include <stdio.h>

#include <memory>
#include <string>
#include <vector>

#include "src/communicator/multicast_communicator.h"

__global__ void write(int *ptr, int rank) {
  ptr[0] = rank;
  const char str1[] = "[I am rank ";
  const char str2[] = " data in gpu memory]";

  int i = 0;
  for (; i < sizeof(str1) - 1; ++i) {
    ptr[i] = str1[i];
  }

  ptr[i] = rank + '0';
  ++i;

  for (int j = 0; j < sizeof(str2); ++j) {
    ptr[i] = str2[j];
    ++i;
  }
}

__global__ void read(int *ptr, int exec_rank, int data_rank) {
  printf("exec_rank = %d, data_rank = %d, content = ", exec_rank, data_rank);
  for (int i = 0; i < 100; ++i) {
    int v = ptr[i];
    if (v != 0) {
      printf("%c", v);
    } else {
      printf("\n");
      break;
    }
  }
}

__device__ __forceinline__ void multimem_store(void *ptr, int v) {
  asm volatile("multimem.st.relaxed.gpu.global.s32 [%0], %1;\n" ::"l"(ptr), "r"(v));
}

__global__ void multi_write(int *multi_ptr, int rank) {
  const char str1[] = "[I am the multicast message write by rank ";
  const char str2[] = "]";

  int i = 0;
  for (; i < sizeof(str1) - 1; ++i) {
    multimem_store(multi_ptr + i, str1[i]);
  }
  multimem_store(multi_ptr + i, rank + '0');
  ++i;

  for (int j = 0; j < sizeof(str2); ++j) {
    multimem_store(multi_ptr + i, str2[j]);
    ++i;
  }
}

__global__ void local_set_value(int *ptr, int rank) { ptr[0] = 1 << rank; }

__device__ __forceinline__ int multimem_ld_reduce(const void *ptr) {
  int r;
  asm volatile("multimem.ld_reduce.relaxed.gpu.global.add.s32 %0, [%1];\n" : "=r"(r) : "l"(ptr));
  return r;
}

__global__ void multi_ld_reduce(int *ptr, int rank) {
  int v = multimem_ld_reduce(ptr);

  printf("exec_rank = %d, ld_reduce = ", rank);
  for (int i = 0; i < sizeof(v) * 8; ++i) {
    bool ok = (v >> (31 - i)) & 0x1;
    if (ok) {
      printf("1");
    } else {
      printf("0");
    }
  }
  printf("\n");
}

__device__ __forceinline__ void multimem_reduce(void *ptr, int v) {
  asm volatile("multimem.red.relaxed.gpu.global.add.s32 [%0], %1;\n" ::"l"(ptr), "r"(v));
}

__global__ void multi_reduce(int *ptr) { multimem_reduce(ptr, 1 << 20); }

__global__ void local_read(int *ptr, int rank) {
  int v = ptr[0];

  printf("exec_rank = %d, reduced = ", rank);
  for (int i = 0; i < sizeof(v) * 8; ++i) {
    bool ok = (v >> (31 - i)) & 0x1;
    if (ok) {
      printf("1");
    } else {
      printf("0");
    }
  }
  printf("\n");
}

int main(int argc, char *argv[]) {
  int rank = std::stoi(argv[1]);
  int world_size = std::stoi(argv[2]);

  auto comm = std::make_shared<hpc::communicator::MulticastCommunicator>(rank, world_size);

  cudaSetDevice(rank);

  int64_t bytes = 1024;

  std::vector<std::shared_ptr<void>> sptrs;
  std::vector<int> devices;
  std::shared_ptr<void> multi_sptr;
  int multi_device = -1;

  bool ok = comm->CreateTensorSync(bytes, &sptrs, &devices, &multi_sptr, &multi_device);

  printf("rank = %d, ok = %d\n", rank, ok);
  printf(" multi: %p@CUDA%d\n", multi_sptr.get(), multi_device);
  for (int r = 0; r < sptrs.size(); ++r) {
    printf("  %d: %p@CUDA%d\n", r, sptrs[r].get(), devices[r]);
  }

  printf("== ACCESS TEST ==\n");
  {
    // write self to rank
    int *ptr = (int *)(sptrs[rank].get());
    write<<<1, 1>>>(ptr, rank);
    cudaDeviceSynchronize();
    comm->Barrier();

    // read self and others
    for (int r = 0; r < sptrs.size(); ++r) {
      int *ptr = (int *)(sptrs[r].get());
      read<<<1, 1>>>(ptr, rank, r);
    }

    cudaDeviceSynchronize();
    comm->Barrier();
  }

  printf("== MULTICAST STORE TEST ==\n");
  {
    // multicast write with rank 0
    printf("rank %d multicast data\n", 0);
    if (rank == 0) {
      int *ptr = (int *)(multi_sptr.get());
      multi_write<<<1, 1>>>(ptr, rank);
    }

    cudaDeviceSynchronize();
    comm->Barrier();

    // read the multicasted data from each rank
    {
      int *ptr = (int *)(sptrs[rank].get());
      read<<<1, 1>>>(ptr, rank, rank);
    }

    cudaDeviceSynchronize();
    comm->Barrier();

    // multicast write with rank 1
    printf("rank %d multicast data\n", 1);
    if (rank == 1) {
      int *ptr = (int *)(multi_sptr.get());
      multi_write<<<1, 1>>>(ptr, rank);
    }

    cudaDeviceSynchronize();
    comm->Barrier();

    // read the multicasted data from each rank
    {
      int *ptr = (int *)(sptrs[rank].get());
      read<<<1, 1>>>(ptr, rank, rank);
    }

    cudaDeviceSynchronize();
    comm->Barrier();
  }

  printf("== LOAD REDUCE TEST ==\n");
  {
    // each rank set the value to its rank
    printf("each rank set the value to its rank\n");
    {
      int *ptr = (int *)(sptrs[rank].get());
      local_set_value<<<1, 1>>>(ptr, rank);
    }
    cudaDeviceSynchronize();
    comm->Barrier();

    printf("each rank ld reduce the value to its local register\n");
    {
      int *ptr = (int *)(multi_sptr.get());
      multi_ld_reduce<<<1, 1>>>(ptr, rank);
    }
    cudaDeviceSynchronize();
    comm->Barrier();
  }

  printf("== REDUCE TEST ==\n");
  {
    printf("multimem reduce the value to all the pointer data\n");
    {
      int *ptr = (int *)(multi_sptr.get());
      multi_reduce<<<1, 1>>>(ptr);
    }
    cudaDeviceSynchronize();
    comm->Barrier();

    printf("each rank load the reduced value\n");
    {
      int *ptr = (int *)(sptrs[rank].get());
      local_read<<<1, 1>>>(ptr, rank);
    }
    cudaDeviceSynchronize();
    comm->Barrier();
  }

  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  comm->Barrier();

  return 0;
}
