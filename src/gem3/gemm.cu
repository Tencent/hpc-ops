// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/gem3/gemm.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace gem3 {

namespace kernels {

template <typename TiledMma, typename TmaA, typename TmaB, typename TmaD, int kTileM, int kTileN,
          int kStage, typename Tin, typename Tout, typename SLayoutA, typename SLayoutB,
          typename SLayoutC>
__global__ void gemm(const __grid_constant__ TmaA tma_a, const __grid_constant__ TmaB tma_b,
                     const __grid_constant__ TmaD tma_d, int m, int n, int k) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int itile_n = blockIdx.x;
  int itile_m = blockIdx.y;

  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  int elected = cute::elect_one_sync();
  bool is_leader_in_block = (iwarp == 0) && elected;

  __shared__ uint64_t bar_ab[kStage];

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_a = (Tin *)shm_data;
  auto *shm_b = (Tin *)shm_a + cosize(SLayoutA{});

  auto gA = tma_a.get_tma_tensor(make_shape(m, k));
  auto gB = tma_b.get_tma_tensor(make_shape(n, k));
  auto gC = make_tensor(make_gmem_ptr((Tout *)(nullptr)), make_shape(Int<kTileM>{}, Int<kTileN>{}),
                        make_stride(Int<kTileN>{}, Int<1>{}));

  auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});

  auto btma_a = tma_a.get_slice(0);
  auto btma_b = tma_b.get_slice(0);

  auto tAg = btma_a.partition_S(gA);  // (TMA, TMA_M, TMA_K)
  auto tAs = btma_a.partition_D(sA);  // (TMA, _1, _1, kStage)

  auto tBg = btma_b.partition_S(gB);  // (TMA, TMA_N, TMA_K)
  auto tBs = btma_b.partition_D(sB);  // (TMA, _1, _1, kStage)

  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(bar_ab[i], 1);
    }
  }
  // sync to avoid ahead thread use(wait) bar_ab when it is not initizlized yet
  __syncthreads();

  TiledMma tiled_mma;

  auto thr_mma = tiled_mma.get_slice(idx);
  auto tAs4r = thr_mma.partition_A(sA);
  auto tBs4r = thr_mma.partition_B(sB);

  auto tAr = thr_mma.make_fragment_A(tAs4r);  // (MMA, MMA_M, MMA_K, kStage)
  auto tBr = thr_mma.make_fragment_B(tBs4r);  // (MMA, MMA_N, MMA_K, kStage)

  auto tCr = thr_mma.partition_fragment_C(gC);

  constexpr int kTransactionBytes =
      sizeof(Tin) * cosize(SLayoutA{}(_, _, 0)) + sizeof(Tin) * cosize(SLayoutB{}(_, _, 0));
  int ismem_write = 0;
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    if (is_leader_in_block) {
      cute::copy(tma_a.with(bar_ab[istage]), tAg(_, itile_m, istage), tAs(_, 0, 0, istage));
      cute::copy(tma_b.with(bar_ab[istage]), tBg(_, itile_n, istage), tBs(_, 0, 0, istage));

      set_barrier_transaction_bytes(bar_ab[istage], kTransactionBytes);

      ++ismem_write;
    }
  }

  int ismem_read = 0;
  int phase = 0;

  // set scale to zero to avoid clear(tCr)
  tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
  {
    wait_barrier(bar_ab[ismem_read], phase);

    // mma
    warpgroup_fence_operand(tCr);
    warpgroup_arrive();
    for (int ik = 0; ik < size<2>(tAr); ++ik) {
      cute::gemm(tiled_mma, tAr(_, _, ik, ismem_read), tBr(_, _, ik, ismem_read), tCr(_, _, _));
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    warpgroup_fence_operand(tCr);
    warpgroup_wait<0>();

    ++ismem_read;
  }

  int itile_write_k = ismem_write;
  int ntile_k = size<2>(tAg);
  int ntile_todo = ntile_k - 1;
#pragma unroll 1
  for (; ntile_todo > 0; --ntile_todo) {
    // load a, b
    if (itile_write_k < ntile_k) {
      if (is_leader_in_block) {
        cute::copy(tma_a.with(bar_ab[ismem_write]), tAg(_, itile_m, itile_write_k),
                   tAs(_, 0, 0, ismem_write));
        cute::copy(tma_b.with(bar_ab[ismem_write]), tBg(_, itile_n, itile_write_k),
                   tBs(_, 0, 0, ismem_write));

        set_barrier_transaction_bytes(bar_ab[ismem_write], kTransactionBytes);

        ismem_write = (ismem_write + 1) % kStage;
      }
    }
    ++itile_write_k;

    wait_barrier(bar_ab[ismem_read], phase);

    // mma
    warpgroup_fence_operand(tCr);
    warpgroup_arrive();
    cute::gemm(tiled_mma, tAr(_, _, _, ismem_read), tBr(_, _, _, ismem_read), tCr(_, _, _));
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    ++ismem_read;
    if (ismem_read == kStage) {
      phase ^= 1;
      ismem_read = 0;
    }

    __syncthreads();
  }

  warpgroup_fence_operand(tCr);

  // float32 -> bfloat16
  auto tCrh = make_tensor_like<cute::bfloat16_t>(tCr);

#pragma unroll
  for (int i = 0; i < size(tCr); ++i) {
    tCrh(i) = (Tout)(tCr(i));
  }

  // Epilogue
  auto sC = make_tensor(make_smem_ptr((Tout *)shm_data), SLayoutC{});
  using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>;
  auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto thr_copy_c = tiled_copy_c.get_slice(idx);

  auto tCr4s = thr_copy_c.retile_S(tCrh);
  auto tCs4r = thr_copy_c.partition_D(sC);

  cute::copy(tiled_copy_c, tCr4s, tCs4r);
  __syncthreads();
  cute::tma_store_fence();

  if (is_leader_in_block) {
    auto gD = tma_d.get_tma_tensor(make_shape(m, n));
    auto btma_d = tma_d.get_slice(0);

    auto tDs = btma_d.partition_S(sC);  // (TMA, _1, _1)
    auto tDg = btma_d.partition_D(gD);  // (TMA, TMA_M, TMA_N)

    cute::copy(tma_d, tDs(_, 0, 0), tDg(_, itile_m, itile_n));
  }
}

}  // namespace kernels

void gemm_async(void *y_ptr, const void *x_ptr, const void *w_ptr, int m, int n, int k,
                cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  constexpr int kTileM = 128;
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;
  constexpr int kStage = 5;

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto W = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)), make_shape(n, k),
                       make_stride(k, Int<1>{}));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(m, n),
                       make_stride(n, Int<1>{}));

  auto slayout_x = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_w = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_y =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tout>{}, make_shape(Int<kTileM>{}, Int<kTileN>{}));

  int shm_xw = sizeof(Tin) * (cosize(slayout_x) + cosize(slayout_w));
  int shm_y = sizeof(Tout) * cosize(slayout_y);
  int shm_size = std::max(shm_xw, shm_y);

  auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, X, slayout_x(_, _, 0));
  auto tma_w = make_tma_copy(SM90_TMA_LOAD{}, W, slayout_w(_, _, 0));
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, slayout_y);

  auto warpgroup_layout = make_layout(make_shape(Int<2>{}, Int<1>{}, Int<1>{}));
  auto tiled_mma = make_tiled_mma(SM90_64x128x32_F32E4M3E4M3_SS_TN<>{}, warpgroup_layout);

  auto kernel = kernels::gemm<decltype(tiled_mma), decltype(tma_x), decltype(tma_w),
                              decltype(tma_y), kTileM, kTileN, kStage, Tin, Tout,
                              decltype(slayout_x), decltype(slayout_w), decltype(slayout_y)>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  dim3 block(size(tiled_mma));
  dim3 grid((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM);
  kernel<<<grid, block, shm_size, stream>>>(tma_x, tma_w, tma_y, m, n, k);

  printf("m = %d, n = %d, k = %d\n", m, n, k);
  printf("grid = [%d, %d, %d], block = [%d, %d, %d], shm = %d\n", grid.x, grid.y, grid.z, block.x,
         block.y, block.z, shm_size);
}

}  // namespace gem3
}  // namespace hpc
