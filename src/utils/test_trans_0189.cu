// Copyright (C) 2026 Tencent.

#include <cuda.h>
#include <stdio.h>

#include "cute/tensor.hpp"
#include "src/utils/utils.cuh"

template <typename T>
__device__ __forceinline__ void pprint(T& t) {
  using namespace cute;  // NOLINT

  for (int irow = 0; irow < size<0>(t); ++irow) {
    for (int icol = 0; icol < size<1>(t); ++icol) {
      int v = t(irow, icol);
      printf("%-3d ", v);
    }
    printf("\n");
  }
}

__global__ void transpose_and_interleave_row0189() {
  using namespace cute;  // NOLINT
  using T = uint8_t;

  constexpr int kRow = 64;
  constexpr int kCol = 64;

  __shared__ T smem_data[kRow * kCol];
  __shared__ T smem_data_t[kRow * kCol];

  int idx = threadIdx.x;

  auto sV = make_tensor(make_smem_ptr(smem_data), make_shape(Int<kRow>{}, Int<kCol>{}),
                        make_stride(Int<kCol>{}, Int<1>{}));
  auto sVt = make_tensor(make_smem_ptr(smem_data_t), make_shape(Int<kRow>{}, Int<kCol>{}),
                         make_stride(Int<1>{}, Int<kRow>{}));

  int ix = 0;
  for (int icol = 0; icol < kCol; ++icol) {
    for (int irow = 0; irow < kRow; ++irow) {
      uint8_t v = ix;
      sV(irow, icol) = v;
      ++ix;
    }
  }

  if (idx == 0) {
    print("sV\n");
    print(sV);
    print("\n");
    pprint(sV);
  }

  int ilane = threadIdx.x % 32;

  auto sV_tile = local_tile(sV, make_tile(Int<16>{}, Int<32>{}), make_coord(_, _));
  auto sVt_tile = local_tile(sVt, make_tile(Int<16>{}, Int<32>{}), make_coord(_, _));

  hpc::smem_trans_and_interleave0189_mn_mn(sV_tile(_, _, 0, 0), sVt_tile(_, _, Int<0>{}, Int<0>{}),
                                           ilane);
  __syncthreads();

  if (idx == 0) {
    print("sVt\n");
    print(sVt);
    print("\n");
    pprint(sVt);
  }
}

int main() {
  transpose_and_interleave_row0189<<<1, 32>>>();

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  return 0;
}
