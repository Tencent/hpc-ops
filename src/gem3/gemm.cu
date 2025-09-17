// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "src/gem3/gemm.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace gem3 {

namespace kernels {

__device__ __forceinline__ auto get_next_tile(int iblock) {
  /*
   */
  int irow0 = iblock / 2;
  int icol0 = iblock % 2;

  int irow = irow0 % 4;
  int icol = irow0 / 4;

  int itile_m = irow;
  int itile_n = icol0 + icol * 2;

  /*
  int irow = iblock % 4;
  int icol = iblock / 4;

  int itile_m = irow;
  int itile_n = icol;
  */

  return cute::make_tuple(itile_m, itile_n);
}

template <typename TiledMma, typename TmaA, typename TmaB, typename TmaD, int kTileM, int kTileN,
          int kStage, typename Tin, typename Tout, typename SLayoutA, typename SLayoutB,
          typename SLayoutC>
__global__ void __launch_bounds__(384, 1)
    gemm(const __grid_constant__ TmaA tma_a, const __grid_constant__ TmaB tma_b,
         const __grid_constant__ TmaD tma_d, int m, int n, int k) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;

  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  int iwarpgroup = __shfl_sync(0xFFFFFFFF, idx / 128, 0);
  int elected = cute::elect_one_sync();
  bool is_leader_in_block = (iwarp == 0) && elected;

  __shared__ uint64_t writable[kStage];
  __shared__ uint64_t readable[kStage];

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_a = (Tin *)shm_data;
  auto *shm_b = (Tin *)shm_a + cosize(SLayoutA{});
  auto *shm_c = (Tout *)(shm_b + cosize(SLayoutB{}));

  auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});

  auto gA = tma_a.get_tma_tensor(make_shape(m, k));
  auto gB = tma_b.get_tma_tensor(make_shape(n, k));
  auto gC = make_tensor(make_gmem_ptr((Tout *)(nullptr)), make_shape(Int<kTileM>{}, Int<kTileN>{}),
                        make_stride(Int<kTileN>{}, Int<1>{}));

  auto btma_a = tma_a.get_slice(0);
  auto btma_b = tma_b.get_slice(0);

  auto tAg = btma_a.partition_S(gA);  // (TMA, TMA_M, TMA_K)
  auto tAs = btma_a.partition_D(sA);  // (TMA, _1, _1, kStage)

  auto tBg = btma_b.partition_S(gB);  // (TMA, TMA_N, TMA_K)
  auto tBs = btma_b.partition_D(sB);  // (TMA, _1, _1, kStage)

  int num_tile_m = size<1>(tAg);
  int num_tile_n = size<1>(tBg);

  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable[i], 1);
      initialize_barrier(writable[i], size(TiledMma{}) / 128);
    }
  }

  // we can also use the following code to initialize the barrier
  /*
  if (idx < kStage) {
    readable[idx] = 0x7ffff800001ffffe;  // initialize_barrier(1);
    writable[idx] = 0x7ffff000001ffffc;  // initialize_barrier(2);
  }
   */

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  // load warpgroup
  if (idx >= 256) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= 256;
    constexpr int kTransactionBytes =
        sizeof(Tin) * cosize(SLayoutA{}(_, _, 0)) + sizeof(Tin) * cosize(SLayoutB{}(_, _, 0));

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    int phase = 1;  // start with ok
    int ismem_write = 0;
    int ntile_k = size<2>(tAg);
    int iblock = blockIdx.x;
    if (is_leader_in_load) {
      while (true) {
        auto [itile_m, itile_n] = get_next_tile(iblock);
        if ((itile_m >= num_tile_m) || (itile_n >= num_tile_n)) {
          break;
        }

        iblock += gridDim.x;

#pragma unroll 1
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          // load a, b
          wait_barrier(writable[ismem_write], phase);

          cute::copy(tma_a.with(readable[ismem_write]), tAg(_, itile_m, itile_k),
                     tAs(_, 0, 0, ismem_write));

          cute::copy(tma_b.with(readable[ismem_write]), tBg(_, itile_n, itile_k),
                     tBs(_, 0, 0, ismem_write));

          set_barrier_transaction_bytes(readable[ismem_write], kTransactionBytes);

          ++ismem_write;
          if (ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }  // ntile_todo
      }  // while
    }  // if idx == 0

  } else {
    // math warpgroup
    cutlass::arch::warpgroup_reg_alloc<168>();

    int idx_in_warpgroup = idx % 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    TiledMma tiled_mma;

    auto thr_mma = tiled_mma.get_slice(idx);
    auto tAs4r = thr_mma.partition_A(sA);
    auto tBs4r = thr_mma.partition_B(sB);

    auto tAr = thr_mma.make_fragment_A(tAs4r);  // (MMA, MMA_M, MMA_K, kStage)
    auto tBr = thr_mma.make_fragment_B(tBs4r);  // (MMA, MMA_N, MMA_K, kStage)

    auto tCr = thr_mma.partition_fragment_C(gC);

    int ismem_read = 0;
    int phase = 0;

    int iblock = blockIdx.x;
    while (true) {
      auto [itile_m, itile_n] = get_next_tile(iblock);

      if ((itile_m >= num_tile_m) || (itile_n >= num_tile_n)) {
        break;
      }
      iblock += gridDim.x;

      auto tDr = make_tensor_like(tCr);
      clear(tDr);

      int ntile_k = size<2>(tAg);
      int ntile_todo = ntile_k;
#pragma unroll 1
      for (; ntile_todo > 0; --ntile_todo) {
        if (elected_idx_in_warpgroup) {
          wait_barrier(readable[ismem_read], phase);
        }

        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        // mma
        warpgroup_fence_operand(tCr);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tAr); ++ik) {
          cute::gemm(tiled_mma, tAr(_, _, ik, ismem_read), tBr(_, _, ik, ismem_read), tCr(_, _, _));
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCr);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable[ismem_read]);
        }

#pragma unroll
        for (int i = 0; i < size(tCr); ++i) {
          tDr(i) += tCr(i);
        }

        ++ismem_read;
        if (ismem_read == kStage) {
          phase ^= 1;
          ismem_read = 0;
        }
      }

      // float32 -> bfloat16
      auto tCrh = make_tensor_like<cute::bfloat16_t>(tCr);

#pragma unroll
      for (int i = 0; i < size(tCr); ++i) {
        tCrh(i) = (Tout)(tDr(i));
      }

      // Epilogue
      auto sC = make_tensor(make_smem_ptr((Tout *)shm_c), SLayoutC{});
      using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>;
      auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
      auto thr_copy_c = tiled_copy_c.get_slice(idx);

      auto tCr4s = thr_copy_c.retile_S(tCrh);
      auto tCs4r = thr_copy_c.partition_D(sC);

      cute::copy(tiled_copy_c, tCr4s, tCs4r);
      asm volatile("bar.sync %0, 256;\n" ::"r"(2) : "memory");
      cute::tma_store_fence();

      if (is_leader_in_block) {
        auto gD = tma_d.get_tma_tensor(make_shape(m, n));
        auto btma_d = tma_d.get_slice(0);

        auto tDs = btma_d.partition_S(sC);  // (TMA, _1, _1)
        auto tDg = btma_d.partition_D(gD);  // (TMA, TMA_M, TMA_N)

        cute::copy(tma_d, tDs(_, 0, 0), tDg(_, itile_m, itile_n));
      }
    }
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
  constexpr int kStage = 6;

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
  int shm_size = shm_xw + shm_y;

  auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, X, slayout_x(_, _, 0));
  auto tma_w = make_tma_copy(SM90_TMA_LOAD{}, W, slayout_w(_, _, 0));
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, slayout_y);

  auto warpgroup_layout = make_layout(make_shape(Int<2>{}, Int<1>{}, Int<1>{}));
  auto tiled_mma = make_tiled_mma(SM90_64x128x32_F32E4M3E4M3_SS_TN<>{}, warpgroup_layout);

  auto kernel = kernels::gemm<decltype(tiled_mma), decltype(tma_x), decltype(tma_w),
                              decltype(tma_y), kTileM, kTileN, kStage, Tin, Tout,
                              decltype(slayout_x), decltype(slayout_w), decltype(slayout_y)>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  dim3 block(size(tiled_mma) + 128);
  dim3 grid(78);

  kernel<<<grid, block, shm_size, stream>>>(tma_x, tma_w, tma_y, m, n, k);
}

}  // namespace gem3
}  // namespace hpc
