// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {

__global__ void bfloat16_to_float(float *output, const __nv_bfloat16 *input) {
  auto fx8 = to<float>(load<__half2, 4>(input));

#pragma unroll
  for (int i = 0; i < fx8.num; ++i) {
    output[threadIdx.x + i] = fx8[i];
  }
}

__global__ void floatx2(float *output, const float *input) {
  auto x2 = load<float, 2>(input);
  store(output, x2[1], x2[0]);
}

__global__ void floatx4(float *output, const float *input) {
  auto x4 = load<float, 4>(input);

  store(output, x4[3], x4[2], x4[1], x4[0]);
}

__global__ void bfloat16x4(__nv_bfloat16 *output, const __nv_bfloat16 *input) {
  auto x4 = load<__nv_bfloat16, 4>(input);

  store(output, x4);
}

__global__ void bfloat16x8(__nv_bfloat16 *output, const __nv_bfloat16 *input) {
  auto x8 = load<__nv_bfloat16, 8>(input);

  store(output, x8);
}

template <typename T>
__global__ void set_value(T *data, int num) {
  for (int i = 0; i < num; ++i) {
    data[i] = T{i};
  }
}

template <typename T, typename TmaA>
__global__ void update_tma_desc(const T *ptr, __grid_constant__ const TmaA tma_tmplt,
                                cute::TmaDescriptor *gmem_tma_desc, int m, int k) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int ilane = idx % 32;

  __shared__ TmaDescriptor smem_tma_desc;

  auto gA = make_tensor(make_gmem_ptr(ptr), make_shape(m - 1, k), make_stride(k, Int<1>{}));

  if (ilane == 0) {
    smem_tma_desc = *tma_tmplt.get_tma_descriptor();
    hpc::update_tma_gtensor<TmaA>(smem_tma_desc, gA);
  }

  __syncwarp();

  tma_descriptor_cp_fence_release(gmem_tma_desc, smem_tma_desc);
}

template <typename T, typename Tma, typename CPBox>
__global__ void copy_with_tma(cute::TmaDescriptor *tma_desc, const __grid_constant__ Tma tmax,
                              int m, int k) {
  using namespace cute;  // NOLINT

  __shared__ uint8_t shm_data[cosize(CPBox{}) * sizeof(T)] alignas(128);
  __shared__ uint64_t bar;

  tma_descriptor_fence_acquire(tma_desc);

  int idx = threadIdx.x;

  auto sA = make_tensor(make_smem_ptr((T *)shm_data), CPBox{});

  if (idx == 0) {
    initialize_barrier(bar, 1);
    set_barrier_transaction_bytes(bar, sizeof(T) * cosize(CPBox{}));
  }

  Tma tma;
  auto btma = tma.get_slice(0);

  auto gA = tma.get_tma_tensor(make_shape(m, k));

  auto tS = btma.partition_S(gA);
  auto tD = btma.partition_D(sA);

  if (idx == 0) {
    cute::copy(tma.with(tma_desc, bar), tS(_, 0, 0), tD(_, 0, 0));
  }

  __syncthreads();
  wait_barrier(bar, 0);

  if (idx == 0) {
    print_tensor(sA);
  }
}

}  // namespace hpc

int main() {
  using namespace cute;  // NOLINT
  using T = cute::bfloat16_t;

  T *input = nullptr;
  cute::TmaDescriptor *tma_desc;

  int m = 5;
  int m_view = 3;
  int k = 16;
  cudaMalloc(&input, sizeof(T) * m * k);
  cudaMalloc(&tma_desc, sizeof(cute::TmaDescriptor));

  auto gtensor = make_tensor(make_gmem_ptr((T *)input), make_shape(m, k), make_stride(k, Int<1>{}));
  auto cpbox = make_layout(make_shape(Int<8>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{}));

  auto tma = make_tma_copy(SM90_TMA_LOAD{}, gtensor, cpbox);

  print(tma);

  hpc::set_value<<<1, 1>>>(input, m * k);
  hpc::update_tma_desc<<<1, 32>>>(input, tma, tma_desc, m_view, k);
  hpc::copy_with_tma<T, decltype(tma), decltype(cpbox)><<<1, 32>>>(tma_desc, tma, m, k);

  cudaDeviceSynchronize();

  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
}
