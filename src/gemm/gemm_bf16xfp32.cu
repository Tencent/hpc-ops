// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "src/gemm/gemm.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace gemm {

namespace kernels {

template <int kBlockSwizzle>
__device__ __forceinline__ auto get_next_tile(int iblock, int num_tile_m, int num_tile_n,
                                              cutlass::FastDivmod swizzle_divider,
                                              cutlass::FastDivmod flat_divider) {
  int itile_m, itile_n;
  int num_tile_bxn = kBlockSwizzle * num_tile_n;
  int total_sizzle_blocks = num_tile_m / kBlockSwizzle * num_tile_bxn;

  if (iblock >= total_sizzle_blocks) {
    flat_divider(itile_m, itile_n, iblock);
  } else {
    int i_bxn, i_bxn_res;
    swizzle_divider(i_bxn, i_bxn_res, iblock);

    itile_m = i_bxn * kBlockSwizzle + i_bxn_res % kBlockSwizzle;
    itile_n = i_bxn_res / kBlockSwizzle;
  }

  return cute::make_tuple(itile_m, itile_n);
}

template <typename Tin, typename Tout, typename TiledMma, typename TmaX, typename TmaWH,
          typename TmaWL, typename TmaY, int kTileM, int kTileN, int kTileK, int kStage,
          int kWarpGroupN, typename SLayoutX, typename SLayoutW, typename SLayoutY,
          int kBlockSwizzle>
__global__ void __launch_bounds__(384, 1)
    gemm_bf16xfp32_kernel(const __grid_constant__ TmaX tma_x, const __grid_constant__ TmaWH tma_wh,
                          const __grid_constant__ TmaWL tma_wl, const __grid_constant__ TmaY tma_y,
                          int m, int n, int k, float scale, cutlass::FastDivmod swizzle_divider,
                          cutlass::FastDivmod flat_divider) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;

  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  int elected = cute::elect_one_sync();
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  constexpr int kWLIdx = 0;
  constexpr int kWHIdx = 1;

  __shared__ uint64_t writable_x[kStage];
  __shared__ uint64_t readable_x[kStage];

  __shared__ uint64_t writable_w[kStage][kWarpGroupN][2];
  __shared__ uint64_t readable_w[kStage][kWarpGroupN][2];

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_x = (Tin *)shm_data;
  auto *shm_w = (Tin *)shm_x + cosize(SLayoutX{});
  auto *shm_y = (Tout *)(shm_w + cosize(SLayoutW{}));

  auto sX = make_tensor(make_smem_ptr(shm_x), SLayoutX{});
  auto sW = make_tensor(make_smem_ptr(shm_w), SLayoutW{});

  auto gX = tma_x.get_tma_tensor(make_shape(m, k));
  auto gWH = tma_wh.get_tma_tensor(make_shape(n, k));
  auto gWL = tma_wl.get_tma_tensor(make_shape(n, k));

  auto gY = make_tensor(make_gmem_ptr((Tout *)(nullptr)), make_shape(Int<kTileN>{}, Int<kTileM>{}),
                        make_stride(Int<kTileM>{}, Int<1>{}));

  auto btma_x = tma_x.get_slice(0);
  auto btma_wh = tma_wh.get_slice(0);
  auto btma_wl = tma_wl.get_slice(0);

  auto tXg = btma_x.partition_S(gX);  // (TMA, TMA_M, TMA_K)
  auto tXs = btma_x.partition_D(sX);  // (TMA, _1, _1, kStage)

  auto tWHg = btma_wh.partition_S(gWH);  // (TMA, TMA_N, TMA_K)
  auto tWHs = btma_wh.partition_D(sW);   // (TMA, _1, _1, kStage)

  auto tWLg = btma_wl.partition_S(gWL);  // (TMA, TMA_N, TMA_K)
  auto tWLs = btma_wl.partition_D(sW);   // (TMA, _1, _1, kStage)

  int num_tile_m = size<1>(tXg);
  int num_tile_n = size<1>(tWHg) / kWarpGroupN;

  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable_x[i], 1);
      initialize_barrier(writable_x[i], 2);
    }
#pragma unroll
    for (int istage = 0; istage < kStage; ++istage) {
#pragma unroll
      for (int j = 0; j < kWarpGroupN; ++j) {
        initialize_barrier(readable_w[istage][j][kWLIdx], 1);
        initialize_barrier(readable_w[istage][j][kWHIdx], 1);
        initialize_barrier(writable_w[istage][j][kWLIdx], 1);
        initialize_barrier(writable_w[istage][j][kWHIdx], 1);
      }
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  // load warpgroup
  if (idx >= kWarpGroupN * 128) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= kWarpGroupN * 128;
    constexpr int kTransactionBytesX = sizeof(Tin) * kTileK * kTileM;
    constexpr int kTransactionBytesW = sizeof(Tin) * kTileK * kTileN;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;  // start with ok
      int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);
      int iblock = blockIdx.x;
      int ntile_k = size<2>(tXg);

      while (true) {
        auto [itile_m, itile_n] = get_next_tile<kBlockSwizzle>(iblock, num_tile_m, num_tile_n,
                                                               swizzle_divider, flat_divider);

        if (itile_m >= num_tile_m) {
          break;
        }

        iblock += gridDim.x;

#pragma unroll 1
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          // load a
          wait_barrier(writable_x[ismem_write], phase);
          cute::copy(tma_x.with(readable_x[ismem_write]), tXg(_, itile_m, itile_k),
                     tXs(_, 0, 0, ismem_write));
          set_barrier_transaction_bytes(readable_x[ismem_write], kTransactionBytesX);

          // load wg0 b low
          wait_barrier(writable_w[ismem_write][0][kWLIdx], phase);
          cute::copy(tma_wl.with(readable_w[ismem_write][0][kWLIdx]), tWLg(_, 2 * itile_n, itile_k),
                     tWLs(_, 0, 0, 0, kWLIdx, ismem_write));
          set_barrier_transaction_bytes(readable_w[ismem_write][0][kWLIdx], kTransactionBytesW);

          // load wg1 b low
          wait_barrier(writable_w[ismem_write][1][kWLIdx], phase);
          cute::copy(tma_wl.with(readable_w[ismem_write][1][kWLIdx]),
                     tWLg(_, 2 * itile_n + 1, itile_k), tWLs(_, 0, 0, 1, kWLIdx, ismem_write));
          set_barrier_transaction_bytes(readable_w[ismem_write][1][kWLIdx], kTransactionBytesW);

          // load wg0 b high
          wait_barrier(writable_w[ismem_write][0][kWHIdx], phase);
          cute::copy(tma_wh.with(readable_w[ismem_write][0][kWHIdx]), tWHg(_, 2 * itile_n, itile_k),
                     tWHs(_, 0, 0, 0, kWHIdx, ismem_write));
          set_barrier_transaction_bytes(readable_w[ismem_write][0][kWHIdx], kTransactionBytesW);

          // load wg1 b high
          wait_barrier(writable_w[ismem_write][1][kWHIdx], phase);
          cute::copy(tma_wh.with(readable_w[ismem_write][1][kWHIdx]),
                     tWHg(_, 2 * itile_n + 1, itile_k), tWHs(_, 0, 0, 1, kWHIdx, ismem_write));
          set_barrier_transaction_bytes(readable_w[ismem_write][1][kWHIdx], kTransactionBytesW);

          ++ismem_write;
          if (ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else {
    // math warpgroup
    cutlass::arch::warpgroup_reg_alloc<168>();

    int idx_in_warpgroup = idx % 128;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    TiledMma tiled_mma;

    auto thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    auto tWs4r = thr_mma.partition_A(sW);
    auto tXs4r = thr_mma.partition_B(sX);

    auto tWr = thr_mma.make_fragment_A(tWs4r);  // (MMA, MMA_M, MMA_K, kStage)
    auto tXr = thr_mma.make_fragment_B(tXs4r);  // (MMA, MMA_N, MMA_K, kStage)

    auto tYr_low = thr_mma.partition_fragment_C(gY);
    auto tYr_high = make_tensor_like(tYr_low);

    int ismem_read = 0;
    int phase = 0;

    int iblock = blockIdx.x;
    while (true) {
      auto [itile_m, itile_n] = get_next_tile<kBlockSwizzle>(iblock, num_tile_m, num_tile_n,
                                                             swizzle_divider, flat_divider);
      if (itile_m >= num_tile_m) {
        break;
      }
      iblock += gridDim.x;

      clear(tYr_low);
      clear(tYr_high);

      int ntile_k = size<2>(tXg);

      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
#pragma unroll 1
      for (int itilek = 0; itilek < ntile_k; ++itilek) {
        wait_barrier(readable_x[ismem_read], phase);

        // mma low
        wait_barrier(readable_w[ismem_read][iwarpgroup][kWLIdx], phase);
        warpgroup_fence_operand(tYr_low);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tXr); ++ik) {
          cute::gemm(tiled_mma, tWr(_, _, ik, iwarpgroup, kWLIdx, ismem_read),
                     tXr(_, _, ik, ismem_read), tYr_low(_, _, _));
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tYr_low);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_w[ismem_read][iwarpgroup][kWLIdx]);
        }

        // mma high
        wait_barrier(readable_w[ismem_read][iwarpgroup][kWHIdx], phase);
        warpgroup_fence_operand(tYr_high);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tXr); ++ik) {
          cute::gemm(tiled_mma, tWr(_, _, ik, iwarpgroup, kWHIdx, ismem_read),
                     tXr(_, _, ik, ismem_read), tYr_high(_, _, _));
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tYr_high);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_x[ismem_read]);
          arrive_barrier(writable_w[ismem_read][iwarpgroup][kWHIdx]);
        }

        ++ismem_read;
        if (ismem_read == kStage) {
          phase ^= 1;
          ismem_read = 0;
        }
      }

      // float32 -> bfloat16
      auto tYrh = make_tensor_like<Tout>(tYr_low);

#pragma unroll
      for (int i = 0; i < size(tYr_low); ++i) {
        tYrh(i) = (Tout)(tYr_low(i) * scale + tYr_high(i));
      }

      using STSM_ATOM =
          std::conditional_t<kTileM == 8, cute::SM90_U16x4_STSM_T, cute::SM90_U16x8_STSM_T>;
      using STS_ATOM =
          std::conditional_t<std::is_same_v<Tout, float>, UniversalCopy<uint32_t>, STSM_ATOM>;
      // Epilogue
      auto sY = make_tensor(make_smem_ptr((Tout *)shm_y), SLayoutY{});  // (M, N)
      using R2SCopyAtomY = Copy_Atom<STS_ATOM, Tout>;
      auto tiled_copy_y = make_tiled_copy_C(R2SCopyAtomY{}, tiled_mma);
      auto thr_copy_y = tiled_copy_y.get_slice(idx_in_warpgroup);

      auto tYr4s = thr_copy_y.retile_S(tYrh);
      auto tYs4r = thr_copy_y.partition_D(sY);

      cute::tma_store_wait<0>();
      syncwarpgroup(iwarpgroup);

      cute::copy(tiled_copy_y, tYr4s, tYs4r(_, _, _, iwarpgroup));

      syncwarpgroup(iwarpgroup);
      cute::tma_store_fence();

      if (is_leader_in_warpgroup) {
        auto gY = tma_y.get_tma_tensor(make_shape(n, m));
        auto btma_y = tma_y.get_slice(0);

        auto tYs = btma_y.partition_S(sY);  // (TMA, _2, _1)
        auto tYg = btma_y.partition_D(gY);  // (TMA, TMA_M, TMA_N)

        cute::copy(tma_y, tYs(_, 0, 0, iwarpgroup), tYg(_, 2 * itile_n + iwarpgroup, itile_m));
        tma_store_arrive();
      }
    }
  }
}

}  // namespace kernels

template <typename Tin, typename Tout, int kTileM, int kTileN, int kTileK, int kStage,
          int kWarpGroupN>
void launch_gemm_bf16xfp32_kernel(void *y_ptr, const void *x_ptr, const void *w_high_ptr,
                                  const void *w_low_ptr, int m, int n, int k, float scale,
                                  cudaStream_t stream) {
  using namespace cute;  // NOLINT

  constexpr int kBlockSwizzle = 4;

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto W_HIGH = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_high_ptr)),
                            make_shape(n, k), make_stride(k, Int<1>{}));
  auto W_LOW = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_low_ptr)),
                           make_shape(n, k), make_stride(k, Int<1>{}));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(n, m),
                       make_stride(Int<1>{}, n));

  auto slayout_x = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_w = tile_to_shape(
      GMMA::Layout_K_SW128_Atom<Tin>{},
      make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kWarpGroupN>{}, Int<2>{}, Int<kStage>{}));
  auto slayout_y = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tout>{},
                                 make_shape(Int<kTileN>{}, Int<kTileM>{}, Int<kWarpGroupN>{}));

  int shm_xw = sizeof(Tin) * (cosize(slayout_x) + cosize(slayout_w));
  int shm_y = sizeof(Tout) * cosize(slayout_y);
  int shm_size = shm_xw + shm_y;

  auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, X, take<0, 2>(slayout_x));
  auto tma_wh = make_tma_copy(SM90_TMA_LOAD{}, W_HIGH, take<0, 2>(slayout_w));
  auto tma_wl = make_tma_copy(SM90_TMA_LOAD{}, W_LOW, take<0, 2>(slayout_w));
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, take<0, 2>(slayout_y));

  using MMA_ATOM =
      std::conditional_t<kTileM == 64, SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>,
                         SM90_64x8x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>>;

  auto tiled_mma = make_tiled_mma(MMA_ATOM{});

  auto kernel =
      kernels::gemm_bf16xfp32_kernel<Tin, Tout, decltype(tiled_mma), decltype(tma_x),
                                     decltype(tma_wh), decltype(tma_wl), decltype(tma_y), kTileM,
                                     kTileN, kTileK, kStage, kWarpGroupN, decltype(slayout_x),
                                     decltype(slayout_w), decltype(slayout_y), kBlockSwizzle>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  int num_tile_m = (m + kTileM - 1) / kTileM;
  int num_tile_n = (n + (kTileN * kWarpGroupN) - 1) / (kTileN * kWarpGroupN);
  int num_tile = num_tile_m * num_tile_n;
  int num_tile_bxn = kBlockSwizzle * num_tile_n;
  cutlass::FastDivmod swizzle_divider(num_tile_bxn);
  cutlass::FastDivmod flat_divider(num_tile_n);

  dim3 block(size(tiled_mma) * kWarpGroupN + 128);
  dim3 grid(std::min(get_sm_count(), num_tile));

  kernel<<<grid, block, shm_size, stream>>>(tma_x, tma_wh, tma_wl, tma_y, m, n, k, scale,
                                            swizzle_divider, flat_divider);
}

bool gemm_bf16xfp32_async(void *y_ptr, const void *x_ptr, const void *w_high_ptr,
                          const void *w_low_ptr, int m, int n, int k, float scale,
                          bool use_fp32_output, cudaStream_t stream) {
  if (use_fp32_output) {
    if (m > 128) {
      launch_gemm_bf16xfp32_kernel<cute::bfloat16_t, float, 64, 64, 64, 2, 2>(
          y_ptr, x_ptr, w_high_ptr, w_low_ptr, m, n, k, scale, stream);
    } else {
      launch_gemm_bf16xfp32_kernel<cute::bfloat16_t, float, 8, 64, 128, 3, 2>(
          y_ptr, x_ptr, w_high_ptr, w_low_ptr, m, n, k, scale, stream);
    }
  } else {
    if (m > 128) {
      launch_gemm_bf16xfp32_kernel<cute::bfloat16_t, cute::bfloat16_t, 64, 64, 64, 2, 2>(
          y_ptr, x_ptr, w_high_ptr, w_low_ptr, m, n, k, scale, stream);
    } else {
      launch_gemm_bf16xfp32_kernel<cute::bfloat16_t, cute::bfloat16_t, 8, 64, 128, 3, 2>(
          y_ptr, x_ptr, w_high_ptr, w_low_ptr, m, n, k, scale, stream);
    }
  }
  return true;
}

}  // namespace gemm
}  // namespace hpc
