// Copyright 2025 hpc-ops authors

#ifndef SRC_GEMM_SM100_GEMM_BF16XFP32_KERNELS_CUH_
#define SRC_GEMM_SM100_GEMM_BF16XFP32_KERNELS_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace gemm {
namespace kernels {

template <int kBlockSwizzle, int kMmaSM>
__device__ __forceinline__ auto get_next_tile(int iblock, cutlass::FastDivmod flat_divider,
                                              cutlass::FastDivmod swizzle_divider) {
  int itile_m;
  int itile_n;

  if constexpr (kBlockSwizzle > 1) {
    int i_bxm, i_bxm_res;
    swizzle_divider(i_bxm, i_bxm_res, iblock);

    itile_m = (i_bxm * kBlockSwizzle + i_bxm_res % kBlockSwizzle) / kMmaSM;
    itile_n = i_bxm_res / kBlockSwizzle;
  } else {
    flat_divider(itile_n, itile_m, iblock);
    itile_m /= kMmaSM;
  }

  return cute::make_tuple(itile_m, itile_n);
}

template <int kBlockSwizzle, int kSplitK>
__device__ __forceinline__ auto get_next_splitk_tile(int iblock, cutlass::FastDivmod flat_divider,
                                                     cutlass::FastDivmod swizzle_divider) {
  int itile_m, itile_n;
  flat_divider(itile_m, itile_n, iblock);

  int ichunk = itile_n % kSplitK;
  itile_n = itile_n / kSplitK;

  return cute::make_tuple(itile_m, itile_n, ichunk);
}

template <int kTileM, int kTileN, int kSplitK, typename Tout, typename TmaY, typename TensorSplitY,
          typename TensorSY>
__device__ __forceinline__ void splitk_reduce_to_gmem(TmaY &tma_y, TensorSplitY &sSplitY,
                                                      TensorSY &sY, const int &m, const int &n,
                                                      const int &itile_m, const int &itile_n,
                                                      const int &ichunk, const int &iwarp,
                                                      const int &idx) {
  using namespace cute;  // NOLINT
  constexpr int kRowsPerBlock = kTileM / kSplitK;

  auto tiled_copy_s2r =
      make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                      make_layout(make_shape(Int<128 / (kTileN / 4)>{}, Int<kTileN / 4>{}),
                                  make_stride(Int<kTileN / 4>{}, Int<1>{})),
                      make_layout(make_shape(Int<kRowsPerBlock / (128 / (kTileN / 4))>{}, Int<4>{}),
                                  make_stride(Int<4>{}, Int<1>{})));

  using R2S_ATOM = std::conditional_t<std::is_same_v<Tout, float>, UniversalCopy<uint128_t>,
                                      UniversalCopy<uint64_t>>;
  auto tiled_copy_r2s =
      make_tiled_copy(Copy_Atom<R2S_ATOM, Tout>{},
                      make_layout(make_shape(Int<128 / (kTileN / 4)>{}, Int<kTileN / 4>{}),
                                  make_stride(Int<kTileN / 4>{}, Int<1>{})),
                      make_layout(make_shape(Int<kRowsPerBlock / (128 / (kTileN / 4))>{}, Int<4>{}),
                                  make_stride(Int<4>{}, Int<1>{})));

  auto thr_copy_s2r = tiled_copy_s2r.get_slice(idx);
  auto tIs4r = thr_copy_s2r.partition_S(sSplitY[0]);
  auto tIr4s = make_tensor_like(tIs4r(_, 0, 0));

  auto thr_copy_r2s = tiled_copy_r2s.get_slice(idx);
  auto tOs4r = thr_copy_r2s.partition_D(sY);
  auto tOr4s = make_tensor_like(tOs4r);
  auto tAcc = make_tensor_like(tIr4s);

  copy(tiled_copy_s2r, tIs4r(_, ichunk, 0), tAcc);
#pragma unroll
  for (int isplit = 1; isplit < kSplitK; isplit++) {
    auto tIs4r_remote = thr_copy_s2r.partition_S(sSplitY[isplit]);

    copy(tiled_copy_s2r, tIs4r_remote(_, ichunk, 0), tIr4s);
#pragma unroll
    for (int i = 0; i < size(tIr4s); i++) {
      tAcc(i) += tIr4s(i);
    }
  }

  if constexpr (std::is_same_v<Tout, cute::bfloat16_t>) {
    auto tAccr_fp2 = recast<float2>(tAcc);
    auto tOr_bf162 = recast<__nv_bfloat162>(tOr4s);
#pragma unroll
    for (int i = 0; i < size(tAccr_fp2); i++) {
      tOr_bf162(i) = __float22bfloat162_rn(tAccr_fp2(i));
    }
  } else {
#pragma unroll
    for (int i = 0; i < size(tAcc); i++) {
      tOr4s(i) = tAcc(i);
    }
  }
  copy(tiled_copy_r2s, tOr4s, tOs4r);

  tma_store_fence();
  cutlass::arch::NamedBarrier::sync(128, 0);

  if (iwarp == 4) {
    auto gYY = tma_y.get_tma_tensor(make_shape(m, n));
    auto btma_y = tma_y.get_slice(0);

    auto tYs = btma_y.partition_S(sY);   // (TMA, _2, _1)
    auto tYg = btma_y.partition_D(gYY);  // (TMA, TMA_M, TMA_N)

    cute::copy(tma_y, tYs(_, 0, 0), tYg(_, itile_m * kSplitK + ichunk, itile_n));
    tma_store_arrive();
  }
}

template <typename Tin, typename TY, typename Tout, typename TiledMma, typename TmaX,
          typename TmaWH, typename TmaWL, typename TmaY, typename SLayoutX, typename SLayoutW,
          typename SLayoutY, int kTileM, int kTileN, int kTileK, int kStageK, int kClusterM,
          int kClusterN, int kClusterK, int kMmaSM, int kEpiTileN, int kStageTile,
          int kBlockSwizzle, int kSplitK>
__global__ void __launch_bounds__(256, 1)
    gemm_bf16xfp32_2sm_kernel(const __grid_constant__ TmaX tma_x,
                              const __grid_constant__ TmaWH tma_wh,
                              const __grid_constant__ TmaWL tma_wl,
                              const __grid_constant__ TmaY tma_y, Tout *y_ptr, float *splitk_y_ptr,
                              int *split_flag_ptr, int m, int n, int k, float scale, int num_tile_m,
                              int num_tile_n, cutlass::FastDivmod swizzle_divider,
                              cutlass::FastDivmod flat_divider,
                              cutlass::FastDivmod reduce_flat_divider) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int elected = cute::elect_one_sync();

  int iblock = blockIdx.x;

  constexpr int kWLIdx = 0;
  constexpr int kWHIdx = 1;

  constexpr int kStageClc = 2;
  constexpr int kCtaTileM = kTileM / kMmaSM;
  constexpr int kClusterSize = kClusterM * kClusterN * kClusterK;

  __shared__ uint64_t tma_readable_x[kStageK];
  __shared__ uint64_t tma_writable_x[kStageK];

  __shared__ uint64_t tma_readable_w[kStageK][2];
  __shared__ uint64_t tma_writable_w[kStageK][2];

  __shared__ uint64_t tmem_readable[kStageTile];
  __shared__ uint64_t tmem_writable[kStageTile];

  __shared__ uint64_t clc_readable[kStageClc];
  __shared__ uint64_t clc_writable[kStageClc];
  __shared__ uint4 clc_resp[kStageClc] alignas(128);

  __shared__ uint32_t tmem_base_ptr;

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_x = reinterpret_cast<Tin *>(shm_data);
  auto *shm_w = reinterpret_cast<Tin *>(shm_x + cosize(SLayoutX{}));
  auto *shm_y = reinterpret_cast<TY *>(shm_w + cosize(SLayoutW{}));

  auto sX = make_tensor(make_smem_ptr(shm_x), SLayoutX{});
  auto sW = make_tensor(make_smem_ptr(shm_w), SLayoutW{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});

  auto cluster_layout =
      tiled_divide(make_layout(make_shape(Int<kClusterM>{}, Int<kClusterN>{}, Int<kClusterK>{})),
                   make_tile(Int<kMmaSM>{}, Int<1>{}, Int<1>{}));
  int block_rank_in_cluster = cute::block_rank_in_cluster();
  auto cluster_coord = cluster_layout.get_flat_coord(block_rank_in_cluster);
  bool elected_cta = get<0>(cluster_coord) == Int<0>{};

  auto gX = tma_x.get_tma_tensor(make_shape(m, k));
  auto gWH = tma_wh.get_tma_tensor(make_shape(n, k));
  auto gWL = tma_wl.get_tma_tensor(make_shape(n, k));
  auto gY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));

  // TMA partition
  auto btma_gx = tma_x.get_slice(get<2>(cluster_coord) + get<0>(cluster_coord) * kClusterN);
  auto btma_gwh =
      tma_wh.get_slice(get<1>(cluster_coord) + get<0>(cluster_coord) * kClusterM / kMmaSM);
  auto btma_gwl =
      tma_wl.get_slice(get<1>(cluster_coord) + get<0>(cluster_coord) * kClusterM / kMmaSM);

  auto btma_sx = tma_x.get_slice(get<2>(cluster_coord));
  auto btma_swh = tma_wh.get_slice(get<1>(cluster_coord));
  auto btma_swl = tma_wl.get_slice(get<1>(cluster_coord));

  auto tXg = btma_gx.partition_S(gX);  // (TMA, TMA_M, TMA_K)
  auto tXs = btma_sx.partition_D(sX);  // (TMA, _1, _1, kStage)

  auto tWHg = btma_gwh.partition_S(gWH);  // (TMA, TMA_N, TMA_K)
  auto tWHs = btma_swh.partition_D(sW);   // (TMA, -1, -1, stage)

  auto tWLg = btma_gwl.partition_S(gWL);  // (TMA, TMA_N, TMA_K)
  auto tWLs = btma_swl.partition_D(sW);   // (TMA, -1, -1, stage)

  // UMMA partition
  TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(get<0>(cluster_coord));

  auto tXs4r = cta_mma.partition_A(sX);
  auto tWs4r = cta_mma.partition_B(sW);
  auto tYgY = cta_mma.partition_C(gY);

  auto tXr = cta_mma.make_fragment_A(tXs4r);
  auto tWr = cta_mma.make_fragment_B(tWs4r);
  auto tYt = cta_mma.make_fragment_C(tYgY);

  uint16_t mcast_mask_c = 3;
  using TmemAllocator = TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};

  __syncthreads();

  if (iwarp == 0 && elected) {
#pragma unroll
    for (int ik = 0; ik < kStageK; ++ik) {
      initialize_barrier(tma_readable_x[ik], 2);
      initialize_barrier(tma_writable_x[ik], 1);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        initialize_barrier(tma_readable_w[ik][j], 2);
        initialize_barrier(tma_writable_w[ik][j], 1);
      }
    }

#pragma unroll
    for (int i = 0; i < kStageTile; ++i) {
      initialize_barrier(tmem_readable[i], 1);
      initialize_barrier(tmem_writable[i], 2);
    }

    constexpr int kMmaThreads = 32;
    constexpr int kEpiThreads = 128;
    constexpr int kTmaThreads = 1;
    constexpr int kSchedThreads = 1;
#pragma unroll
    for (int i = 0; i < kStageClc; i++) {
      initialize_barrier(clc_readable[i], 1);
      initialize_barrier(clc_writable[i],
                         kSchedThreads + kMmaThreads + kClusterSize * (kTmaThreads + kEpiThreads));
    }

    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 1) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &tmem_base_ptr);
  }

  cluster_relaxed_sync();

  int ntile_k = size<2>(tXg);

  constexpr int kTransactionBytesA = kMmaSM * sizeof(Tin) * (cosize(SLayoutX{}(_, _, 0)));
  constexpr int kTransactionBytesB = kMmaSM * sizeof(Tin) * (cosize(SLayoutW{}(_, _, 0, 0)));

  if (iwarp == 0 && elected) {
    // TMA warp
    // cutlass::arch::warpgroup_reg_dealloc<48>();
    int phase = 1;
    int istage_k = 0;

    int phase_clc = 0;
    int istage_clc = 0;

    while (true) {
      auto [itile_m, itile_n] =
          get_next_tile<kBlockSwizzle, kMmaSM>(iblock, flat_divider, swizzle_divider);
      if (itile_m >= num_tile_m || itile_n >= num_tile_n) {
        break;
      }
#pragma unroll 1
      for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
        // load a
        wait_barrier(tma_writable_x[istage_k], phase);
        copy(tma_x.with(tma_readable_x[istage_k]), tXg(_, itile_m, itile_k),
             tXs(_, 0, 0, istage_k));
        if (elected_cta) {
          set_barrier_transaction_bytes(tma_readable_x[istage_k], kTransactionBytesA);
        } else {
          arrive_cluster_barrier(tma_readable_x[istage_k]);
        }

        // load b low
        wait_barrier(tma_writable_w[istage_k][kWLIdx], phase);
        copy(tma_wl.with(tma_readable_w[istage_k][kWLIdx]), tWLg(_, itile_n, itile_k),
             tWLs(_, 0, 0, kWLIdx, istage_k));
        if (elected_cta) {
          set_barrier_transaction_bytes(tma_readable_w[istage_k][kWLIdx], kTransactionBytesB);
        } else {
          arrive_cluster_barrier(tma_readable_w[istage_k][kWLIdx]);
        }

        // load b high
        wait_barrier(tma_writable_w[istage_k][kWHIdx], phase);
        copy(tma_wh.with(tma_readable_w[istage_k][kWHIdx]), tWHg(_, itile_n, itile_k),
             tWHs(_, 0, 0, kWHIdx, istage_k));
        if (elected_cta) {
          set_barrier_transaction_bytes(tma_readable_w[istage_k][kWHIdx], kTransactionBytesB);
        } else {
          arrive_cluster_barrier(tma_readable_w[istage_k][kWHIdx]);
        }

        istage_k++;
        if (istage_k == kStageK) {
          phase ^= 1;
          istage_k = 0;
        }
      }

      wait_barrier(clc_readable[istage_clc], phase_clc);
      auto [icluster_first_cta, valid] = get_next_block(&clc_resp[istage_clc]);
      arrive_cluster_barrier(clc_writable[istage_clc]);

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }

      if (!valid) {
        break;
      }
      iblock = icluster_first_cta + block_rank_in_cluster;
    }
  } else if (iwarp == 1 && elected_cta) {
    // UMMA warp
    int phase = 0;
    int phase_tile = 1;
    int phase_clc = 0;
    int istage_k = 0;
    int istage_tile = 0;
    int istage_clc = 0;

    while (true) {
      auto [itile_m, itile_n] =
          get_next_tile<kBlockSwizzle, kMmaSM>(iblock, flat_divider, swizzle_divider);
      if (itile_m >= num_tile_m || itile_n >= num_tile_n) {
        break;
      }

      tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
      wait_barrier(tmem_writable[istage_tile], phase_tile);
#pragma unroll 1
      for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
        wait_barrier(tma_readable_x[istage_k], phase);

        tYt.data() = tmem_base_ptr + istage_tile * kTileN * 2;
        wait_barrier(tma_readable_w[istage_k][kWLIdx], phase);
        for (int ik = 0; ik < size<2>(tXr); ++ik) {
          cute::gemm(tiled_mma, tXr(_, _, ik, istage_k), tWr(_, _, ik, kWLIdx, istage_k), tYt);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive_multicast_2x1SM(&tma_writable_w[istage_k][kWLIdx], mcast_mask_c);

        if (itile_k == 0) {
          tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
        }
        tYt.data() = tmem_base_ptr + istage_tile * kTileN * 2 + kTileN;
        wait_barrier(tma_readable_w[istage_k][kWHIdx], phase);
        for (int ik = 0; ik < size<2>(tXr); ++ik) {
          cute::gemm(tiled_mma, tXr(_, _, ik, istage_k), tWr(_, _, ik, kWHIdx, istage_k), tYt);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }

        cutlass::arch::umma_arrive_multicast_2x1SM(&tma_writable_w[istage_k][kWHIdx], mcast_mask_c);
        cutlass::arch::umma_arrive_multicast_2x1SM(&tma_writable_x[istage_k], mcast_mask_c);

        istage_k++;
        if (istage_k == kStageK) {
          phase ^= 1;
          istage_k = 0;
        }
      }

      cutlass::arch::umma_arrive_multicast_2x1SM(&tmem_readable[istage_tile], mcast_mask_c);

      istage_tile++;
      if (istage_tile == kStageTile) {
        phase_tile ^= 1;
        istage_tile = 0;
      }

      wait_barrier(clc_readable[istage_clc], phase_clc);
      auto [icluster_first_cta, valid] = get_next_block(&clc_resp[istage_clc]);

      if (!valid) {
        break;
      }

      arrive_cluster_barrier(clc_writable[istage_clc]);

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }

      iblock = icluster_first_cta + block_rank_in_cluster;
    }
  } else if (iwarp == 2 && (block_rank_in_cluster == 0) && elected) {
    int phase_clc_read = 0;
    int phase_clc_write = 1;
    int istage_clc = 0;
    do {
      wait_barrier(clc_writable[istage_clc], phase_clc_write);
      find_next_block(&clc_resp[istage_clc], &clc_readable[istage_clc]);
      set_barrier_transaction_bytes(clc_readable[istage_clc], sizeof(uint4));
#pragma unroll
      for (int i = 1; i < kClusterSize; i++) {
        set_barrier_transaction_bytes_cluster(clc_readable[istage_clc], sizeof(uint4), i);
      }
      wait_barrier(clc_readable[istage_clc], phase_clc_read);
      auto [icluster_first_cta, valid] = get_next_block(&clc_resp[istage_clc]);

      if (!valid) {
        break;
      }

      arrive_cluster_barrier(clc_writable[istage_clc]);

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc_write ^= 1;
        phase_clc_read ^= 1;
      }
    } while (true);
  } else if (idx >= 128) {
    idx -= 128;
    // cutlass::arch::warpgroup_reg_alloc<232>();

    tYt.data() = tmem_base_ptr;
    auto tmem_epi_tiler = make_tile(Int<kCtaTileM>{}, Int<kTileN>{});
    auto tYt_epi = zipped_divide(tYt, make_tile(tmem_epi_tiler));
    auto sY_epi = zipped_divide(sY, tmem_epi_tiler);

    // TiledCopy TMEM -> RMEM
    auto tiled_copy_t2r = make_tmem_copy(SM100_TMEM_LOAD_16dp256b4x{}, tYt_epi(_, 0));
    auto thr_copy_t2r = tiled_copy_t2r.get_slice(idx);
    auto tYLt4r = thr_copy_t2r.partition_S(tYt_epi);
    auto tYLr4t = make_tensor_like<float>(thr_copy_t2r.partition_D(sY_epi));

    tYt.data() = tmem_base_ptr + kTileN;
    auto tYt_epi2 = zipped_divide(tYt, make_tile(tmem_epi_tiler));
    auto tYHt4r = thr_copy_t2r.partition_S(tYt_epi2);
    auto tYHr4t = make_tensor_like<float>(tYLr4t);

    // TiledCopy RMEM -> SMEM
    using R2S_ATOM =
        std::conditional_t<std::is_same_v<TY, float>, UniversalCopy<uint64_t>, SM90_U32x4_STSM_N>;
    auto tiled_copy_r2s = make_tiled_copy_D(Copy_Atom<R2S_ATOM, TY>{}, tiled_copy_t2r);
    auto thr_copy_r2s = tiled_copy_r2s.get_slice(idx);
    auto tYr4s = make_tensor_like<TY>(thr_copy_r2s.partition_S(sY_epi));
    auto tYs4r = thr_copy_r2s.partition_D(sY_epi);

    // epi warpgroup
    int phase_tile = 0;
    int phase_clc = 0;

    int istage_tile = 0;
    int istage_clc = 0;

    bool is_leader = elected && (iwarp == 4);

    auto tYLt4r_base_ptr = tYLt4r.data();
    auto tYHt4r_base_ptr = tYHt4r.data();

    while (true) {
      auto [itile_m, itile_n] =
          get_next_tile<kBlockSwizzle, kMmaSM>(iblock, flat_divider, swizzle_divider);
      if (itile_m >= num_tile_m || itile_n >= num_tile_n) {
        break;
      }

      tYLt4r.data() = tYLt4r_base_ptr + istage_tile * kTileN * 2;
      tYHt4r.data() = tYHt4r_base_ptr + istage_tile * kTileN * 2;

      wait_barrier(tmem_readable[istage_tile], phase_tile);
      copy(tiled_copy_t2r, tYLt4r, tYLr4t);
      copy(tiled_copy_t2r, tYHt4r, tYHr4t);
      if (is_leader) {
        arrive_cluster_barrier(tmem_writable[istage_tile]);
      }

#pragma unroll
      for (int i = 0; i < size(tYLr4t); i++) {
        tYLr4t(i) = tYLr4t(i) * scale + tYHr4t(i);
      }

      if constexpr (std::is_same_v<TY, cute::bfloat16_t>) {
        auto tYr_fp2 = recast<float2>(tYLr4t);
        auto tYr_bf162 = recast<__nv_bfloat162>(tYr4s);

        // cast
#pragma unroll
        for (int i = 0; i < cute::size(tYr_bf162); i++) {
          tYr_bf162(i) = __float22bfloat162_rn(tYr_fp2(i));
        }
      } else {
#pragma unroll
        for (int i = 0; i < cute::size(tYr4s); i++) {
          tYr4s(i) = tYLr4t(i);
        }
      }

      tma_store_wait<0>();
      cutlass::arch::NamedBarrier::sync(128, 0);

      // RMEM -> SMEM
      copy(tiled_copy_r2s, tYr4s, tYs4r);

      // SMEM -> GMEM
      tma_store_fence();
      cutlass::arch::NamedBarrier::sync(128, 0);

      if (iwarp == 4) {
        auto gYY = tma_y.get_tma_tensor(make_shape(m, n, kSplitK));
        auto btma_y = tma_y.get_slice(0);

        auto tYs = btma_y.partition_S(sY);   // (TMA, _2, _1)
        auto tYg = btma_y.partition_D(gYY);  // (TMA, TMA_M, TMA_N)

        cute::copy(tma_y, tYs(_, 0, 0),
                   tYg(_, itile_m * kMmaSM + get<0>(cluster_coord), itile_n, 0));
        tma_store_arrive();
      }

      istage_tile++;
      if (istage_tile == kStageTile) {
        phase_tile ^= 1;
        istage_tile = 0;
      }

      wait_barrier(clc_readable[istage_clc], phase_clc);
      auto [icluster_first_cta, valid] = get_next_block(&clc_resp[istage_clc]);

      if (!valid) {
        break;
      }

      arrive_cluster_barrier(clc_writable[istage_clc]);

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }

      iblock = icluster_first_cta + block_rank_in_cluster;
    }
  }

  __syncthreads();
  cluster_relaxed_sync();
  // Release the right to allocate before deallocations so that the next CTA can rasterize
  // Then deallocate TMEM
  if (iwarp == 1) {
    // tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

template <typename Tin, typename Tout, typename TiledMma, typename TmaX, typename TmaWH,
          typename TmaWL, typename TmaY, typename SLayoutX, typename SLayoutW,
          typename SLayoutSplitY, typename SLayoutY, int kTileM, int kTileN, int kTileK,
          int kStageK, int kClusterM, int kClusterN, int kClusterK, int kEpiTileN, int kStageTile,
          int kBlockSwizzle, int kSplitK>
__global__ void __launch_bounds__(256, 1)
    gemm_bf16xfp32_1sm_cluster_splitk_kernel(const __grid_constant__ TmaX tma_x,
                                             const __grid_constant__ TmaWH tma_wh,
                                             const __grid_constant__ TmaWL tma_wl,
                                             const __grid_constant__ TmaY tma_y, int m, int n,
                                             int k, float scale, int num_tile_m, int num_tile_n,
                                             cutlass::FastDivmod swizzle_divider,
                                             cutlass::FastDivmod flat_divider) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int elected = cute::elect_one_sync();

  int iblock = blockIdx.x;

  constexpr int kWLIdx = 0;
  constexpr int kWHIdx = 1;

  constexpr int kStageClc = kStageTile;
  constexpr int kClusterSize = kClusterM * kClusterN * kClusterK;

  __shared__ uint64_t tma_readable_x[kStageK];
  __shared__ uint64_t tma_writable_x[kStageK];

  __shared__ uint64_t tma_readable_w[kStageK][2];
  __shared__ uint64_t tma_writable_w[kStageK][2];

  __shared__ uint64_t tmem_readable[kStageTile];
  __shared__ uint64_t tmem_writable[kStageTile];

  __shared__ uint64_t clc_readable[kStageClc];
  __shared__ uint64_t clc_writable[kStageClc];

  __shared__ uint64_t splitk_readable;
  __shared__ uint64_t splitk_writable;

  __shared__ uint4 clc_resp[kStageClc] alignas(128);

  __shared__ uint32_t tmem_base_ptr;

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_x = reinterpret_cast<Tin *>(shm_data);
  auto *shm_w = reinterpret_cast<Tin *>(shm_x + cosize(SLayoutX{}));
  auto *shm_y = reinterpret_cast<Tout *>(shm_w + cosize(SLayoutW{}));
  using TSplitSY = std::conditional_t<(kSplitK > 1), float, Tout>;
  constexpr int shm_y_size = kSplitK > 1 ? cosize(SLayoutY{}) : 0;
  auto *shm_splity = reinterpret_cast<TSplitSY *>(shm_y + shm_y_size);

  auto sX = make_tensor(make_smem_ptr(shm_x), SLayoutX{});
  auto sW = make_tensor(make_smem_ptr(shm_w), SLayoutW{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});

  auto sSplitY = make_tensor(make_smem_ptr(shm_splity), SLayoutSplitY{});
  using TensorsSplitY = decltype(sSplitY);

  auto cluster_layout =
      tiled_divide(make_layout(make_shape(Int<kClusterM>{}, Int<kClusterN>{}, Int<kClusterK>{})),
                   make_tile(Int<1>{}, Int<1>{}, Int<1>{}));
  int block_rank_in_cluster = cute::block_rank_in_cluster();
  auto cluster_coord = cluster_layout.get_flat_coord(block_rank_in_cluster);

  vec_t<TensorsSplitY, kSplitK> sSplitY_cluster;

#pragma unroll
  for (int isplit = 0; isplit < kSplitK; isplit++) {
    if (isplit != block_rank_in_cluster) {
      auto *dsmem_ptr = map_shared_rank(shm_splity, isplit);
      sSplitY_cluster[isplit] = make_tensor(make_smem_ptr(dsmem_ptr), SLayoutSplitY{});
    } else {
      sSplitY_cluster[isplit] = sSplitY;
    }
  }

  auto gX = tma_x.get_tma_tensor(make_shape(m, k));
  auto gWH = tma_wh.get_tma_tensor(make_shape(n, k));
  auto gWL = tma_wl.get_tma_tensor(make_shape(n, k));
  auto gY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));

  // TMA partition
  auto btma_x = tma_x.get_slice(0);
  auto btma_wh = tma_wh.get_slice(0);
  auto btma_wl = tma_wl.get_slice(0);

  auto tXg = btma_x.partition_S(gX);  // (TMA, TMA_M, TMA_K)
  auto tXs = btma_x.partition_D(sX);  // (TMA, _1, _1, kStage)

  auto tWHg = btma_wh.partition_S(gWH);  // (TMA, TMA_N, TMA_K)
  auto tWHs = btma_wh.partition_D(sW);   // (TMA, -1, -1, stage)

  auto tWLg = btma_wl.partition_S(gWL);  // (TMA, TMA_N, TMA_K)
  auto tWLs = btma_wl.partition_D(sW);   // (TMA, -1, -1, stage)

  // UMMA partition
  TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(0);

  auto tXs4r = cta_mma.partition_A(sX);
  auto tWs4r = cta_mma.partition_B(sW);
  auto tYgY = cta_mma.partition_C(gY);

  auto tXr = cta_mma.make_fragment_A(tXs4r);
  auto tWr = cta_mma.make_fragment_B(tWs4r);
  auto tYt = cta_mma.make_fragment_C(tYgY);

  using TmemAllocator = TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  if (iwarp == 0 && elected) {
#pragma unroll
    for (int ik = 0; ik < kStageK; ++ik) {
      initialize_barrier(tma_readable_x[ik], 1);
      initialize_barrier(tma_writable_x[ik], 1);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        initialize_barrier(tma_readable_w[ik][j], 1);
        initialize_barrier(tma_writable_w[ik][j], 1);
      }
    }

#pragma unroll
    for (int i = 0; i < kStageTile; ++i) {
      initialize_barrier(tmem_readable[i], 1);
      initialize_barrier(tmem_writable[i], 1);
    }

    initialize_barrier(splitk_readable, kSplitK);
    initialize_barrier(splitk_writable, kSplitK);

    constexpr int kMmaThreads = 32;
    constexpr int kEpiThreads = 128;
    constexpr int kTmaThreads = 1;
    constexpr int kSchedThreads = 1;
#pragma unroll
    for (int i = 0; i < kStageClc; i++) {
      initialize_barrier(clc_readable[i], 1);
      initialize_barrier(clc_writable[i],
                         kSchedThreads + kClusterSize * (kMmaThreads + kTmaThreads + kEpiThreads));
    }

    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 1) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &tmem_base_ptr);
  }

  cluster_relaxed_sync();

  int ntile_k = size<2>(tXg);

  constexpr int kTransactionBytesA = sizeof(Tin) * (cosize(SLayoutX{}(_, _, 0)));
  constexpr int kTransactionBytesB = sizeof(Tin) * (cosize(SLayoutW{}(_, _, 0, 0)));

  if (iwarp == 0 && elected) {
    // TMA warp
    int phase = 1;
    int istage_k = 0;

    int phase_clc = 0;
    int istage_clc = 0;

    while (true) {
      auto [itile_m, itile_n, ichunk] =
          get_next_splitk_tile<kBlockSwizzle, kSplitK>(iblock, flat_divider, swizzle_divider);
      if (itile_m >= num_tile_m || itile_n >= num_tile_n) {
        break;
      }

      for (int itile_k = ichunk; itile_k < ntile_k; itile_k += kSplitK) {
        // load a
        wait_barrier(tma_writable_x[istage_k], phase);
        copy(tma_x.with(tma_readable_x[istage_k]), tXg(_, itile_m, itile_k),
             tXs(_, 0, 0, istage_k));
        set_barrier_transaction_bytes(tma_readable_x[istage_k], kTransactionBytesA);

        // load b low
        wait_barrier(tma_writable_w[istage_k][kWLIdx], phase);
        copy(tma_wl.with(tma_readable_w[istage_k][kWLIdx]), tWLg(_, itile_n, itile_k),
             tWLs(_, 0, 0, kWLIdx, istage_k));
        set_barrier_transaction_bytes(tma_readable_w[istage_k][kWLIdx], kTransactionBytesB);

        // load b high
        wait_barrier(tma_writable_w[istage_k][kWHIdx], phase);
        copy(tma_wh.with(tma_readable_w[istage_k][kWHIdx]), tWHg(_, itile_n, itile_k),
             tWHs(_, 0, 0, kWHIdx, istage_k));
        set_barrier_transaction_bytes(tma_readable_w[istage_k][kWHIdx], kTransactionBytesB);

        istage_k++;
        if (istage_k == kStageK) {
          phase ^= 1;
          istage_k = 0;
        }
      }

      wait_barrier(clc_readable[istage_clc], phase_clc);
      auto [icluster_first_cta, valid] = get_next_block(&clc_resp[istage_clc]);

      if (!valid) {
        break;
      }

      arrive_cluster_barrier(clc_writable[istage_clc]);

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }

      iblock = icluster_first_cta + block_rank_in_cluster;
    }
  } else if (iwarp == 1) {
    // UMMA warp
    int phase = 0;
    int phase_tile = 1;
    int phase_clc = 0;
    int istage_k = 0;
    int istage_tile = 0;
    int istage_clc = 0;

    while (true) {
      auto [itile_m, itile_n, ichunk] =
          get_next_splitk_tile<kBlockSwizzle, kSplitK>(iblock, flat_divider, swizzle_divider);
      if (itile_m >= num_tile_m || itile_n >= num_tile_n) {
        break;
      }

      tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
      wait_barrier(tmem_writable[istage_tile], phase_tile);

      for (int itile_k = ichunk; itile_k < ntile_k; itile_k += kSplitK) {
        wait_barrier(tma_readable_x[istage_k], phase);

        tYt.data() = tmem_base_ptr + istage_tile * kTileN * 2;
        wait_barrier(tma_readable_w[istage_k][kWLIdx], phase);
        for (int ik = 0; ik < size<2>(tXr); ++ik) {
          cute::gemm(tiled_mma, tXr(_, _, ik, istage_k), tWr(_, _, ik, kWLIdx, istage_k), tYt);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(&tma_writable_w[istage_k][kWLIdx]);

        if (itile_k == ichunk) {
          tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
        }
        tYt.data() = tmem_base_ptr + istage_tile * kTileN * 2 + kTileN;
        wait_barrier(tma_readable_w[istage_k][kWHIdx], phase);
        for (int ik = 0; ik < size<2>(tXr); ++ik) {
          cute::gemm(tiled_mma, tXr(_, _, ik, istage_k), tWr(_, _, ik, kWHIdx, istage_k), tYt);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }

        cutlass::arch::umma_arrive(&tma_writable_w[istage_k][kWHIdx]);
        cutlass::arch::umma_arrive(&tma_writable_x[istage_k]);

        istage_k++;
        if (istage_k == kStageK) {
          phase ^= 1;
          istage_k = 0;
        }
      }

      cutlass::arch::umma_arrive(&tmem_readable[istage_tile]);

      istage_tile++;
      if (istage_tile == kStageTile) {
        phase_tile ^= 1;
        istage_tile = 0;
      }

      wait_barrier(clc_readable[istage_clc], phase_clc);
      auto [icluster_first_cta, valid] = get_next_block(&clc_resp[istage_clc]);

      if (!valid) {
        break;
      }

      arrive_cluster_barrier(clc_writable[istage_clc]);

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }

      iblock = icluster_first_cta + block_rank_in_cluster;
    }
  } else if (iwarp == 2 && (block_rank_in_cluster == 0) && elected) {
    int phase_clc_read = 0;
    int phase_clc_write = 1;
    int istage_clc = 0;
    do {
      wait_barrier(clc_writable[istage_clc], phase_clc_write);
      find_next_block(&clc_resp[istage_clc], &clc_readable[istage_clc]);
      set_barrier_transaction_bytes(clc_readable[istage_clc], sizeof(uint4));
#pragma unroll
      for (int i = 1; i < kClusterSize; i++) {
        set_barrier_transaction_bytes_cluster(clc_readable[istage_clc], sizeof(uint4), i);
      }
      wait_barrier(clc_readable[istage_clc], phase_clc_read);
      auto [icluster_first_cta, valid] = get_next_block(&clc_resp[istage_clc]);

      if (!valid) {
        break;
      }

      arrive_cluster_barrier(clc_writable[istage_clc]);

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc_write ^= 1;
        phase_clc_read ^= 1;
      }
    } while (true);
  } else if (idx >= 128) {
    idx -= 128;

    tYt.data() = tmem_base_ptr;
    auto tmem_epi_tiler = make_tile(Int<kTileM>{}, Int<kTileN>{});
    auto tYt_epi = zipped_divide(tYt, make_tile(tmem_epi_tiler));
    auto sY_epi = zipped_divide(sSplitY, tmem_epi_tiler);

    // TiledCopy TMEM -> RMEM
    auto tiled_copy_t2r = make_tmem_copy(SM100_TMEM_LOAD_16dp256b4x{}, tYt_epi(_, 0));
    auto thr_copy_t2r = tiled_copy_t2r.get_slice(idx);
    auto tYLt4r = thr_copy_t2r.partition_S(tYt_epi(_, 0));
    auto tYLr4t = make_tensor_like<float>(thr_copy_t2r.partition_D(sY_epi(_, 0)));

    tYt.data() = tmem_base_ptr + kTileN;
    auto tYt_epi2 = zipped_divide(tYt, make_tile(tmem_epi_tiler));
    auto tYHt4r = thr_copy_t2r.partition_S(tYt_epi2(_, 0));
    auto tYHr4t = make_tensor_like<float>(tYLr4t);

    // TiledCopy RMEM -> SMEM
    using TSY = std::conditional_t<(kSplitK > 1) || std::is_same_v<Tout, float>, float, Tout>;
    using R2S_ATOM =
        std::conditional_t<std::is_same_v<TSY, float>, UniversalCopy<uint64_t>, SM90_U32x4_STSM_N>;
    auto tiled_copy_r2s = make_tiled_copy_D(Copy_Atom<R2S_ATOM, TSY>{}, tiled_copy_t2r);
    auto thr_copy_r2s = tiled_copy_r2s.get_slice(idx);
    auto tYr4s = make_tensor_like<TSY>(thr_copy_r2s.partition_S(sY_epi(_, 0)));
    auto tYs4r = thr_copy_r2s.partition_D(sY_epi);

    // epi warpgroup
    int phase_tile = 0;
    int phase_clc = 0;
    int phase_splitk_write = 0;
    int phase_splitk_read = 0;

    int istage_tile = 0;
    int istage_clc = 0;

    bool is_leader = elected && (iwarp == 4);

    auto tYLt4r_base_ptr = tYLt4r.data();
    auto tYHt4r_base_ptr = tYHt4r.data();

    while (true) {
      auto [itile_m, itile_n, ichunk] =
          get_next_splitk_tile<kBlockSwizzle, kSplitK>(iblock, flat_divider, swizzle_divider);
      if (itile_m >= num_tile_m || itile_n >= num_tile_n) {
        break;
      }
      tYLt4r.data() = tYLt4r_base_ptr + istage_tile * kTileN * 2;
      tYHt4r.data() = tYHt4r_base_ptr + istage_tile * kTileN * 2;

      wait_barrier(tmem_readable[istage_tile], phase_tile);
      copy(tiled_copy_t2r, tYLt4r, tYLr4t);
      copy(tiled_copy_t2r, tYHt4r, tYHr4t);

      cutlass::arch::fence_view_async_tmem_load();

      if (is_leader) {
        arrive_barrier(tmem_writable[istage_tile]);
      }

#pragma unroll
      for (int i = 0; i < size(tYLr4t); i++) {
        tYLr4t(i) = tYLr4t(i) * scale + tYHr4t(i);
      }

      if constexpr (std::is_same_v<TSY, cute::bfloat16_t>) {
        auto tYr_fp2 = recast<float2>(tYLr4t);
        auto tYr_bf162 = recast<__nv_bfloat162>(tYr4s);

        // cast
#pragma unroll
        for (int i = 0; i < cute::size(tYr_bf162); i++) {
          tYr_bf162(i) = __float22bfloat162_rn(tYr_fp2(i));
        }
      } else {
#pragma unroll
        for (int i = 0; i < cute::size(tYr4s); i++) {
          tYr4s(i) = tYLr4t(i);
        }
      }

      tma_store_wait<0>();
      cutlass::arch::NamedBarrier::sync(128, 0);

      if constexpr (kSplitK > 1) {
        if (idx < kSplitK) {
          arrive_cluster_barrier(splitk_writable, idx);
        }
        wait_barrier(splitk_writable, phase_splitk_write);
        phase_splitk_write ^= 1;
      }
      // RMEM -> SMEM
      copy(tiled_copy_r2s, tYr4s, tYs4r(_, _, ichunk));

      // SMEM -> GMEM
      tma_store_fence();
      asm volatile("fence.proxy.async.shared::cluster;");
      cutlass::arch::NamedBarrier::sync(128, 0);

      if constexpr (kSplitK > 1) {
        if (idx < kSplitK) {
          arrive_cluster_barrier(splitk_readable, idx);
        }
        wait_barrier(splitk_readable, phase_splitk_read);
        phase_splitk_read ^= 1;
        asm volatile("fence.proxy.async.shared::cluster;");
        splitk_reduce_to_gmem<kTileM, kTileN, kSplitK, Tout>(tma_y, sSplitY_cluster, sY, m, n,
                                                             itile_m, itile_n, ichunk, iwarp, idx);
      } else {
        if (iwarp == 4) {
          auto gYY = tma_y.get_tma_tensor(make_shape(m, n));
          auto btma_y = tma_y.get_slice(0);

          auto tYs = btma_y.partition_S(sSplitY);  // (TMA, _2, _1)
          auto tYg = btma_y.partition_D(gYY);      // (TMA, TMA_M, TMA_N)

          cute::copy(tma_y, tYs(_, 0, 0), tYg(_, itile_m, itile_n));
          tma_store_arrive();
        }
      }

      istage_tile++;
      if (istage_tile == kStageTile) {
        phase_tile ^= 1;
        istage_tile = 0;
      }

      wait_barrier(clc_readable[istage_clc], phase_clc);
      auto [icluster_first_cta, valid] = get_next_block(&clc_resp[istage_clc]);

      if (!valid) {
        break;
      }

      arrive_cluster_barrier(clc_writable[istage_clc]);

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }

      iblock = icluster_first_cta + block_rank_in_cluster;
    }
  }

  __syncthreads();
  cluster_sync();
  // cluster_relaxed_sync();
  // Release the right to allocate before deallocations so that the next CTA can rasterize
  // Then deallocate TMEM
  if (iwarp == 1) {
    // tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

}  // namespace kernels
}  // namespace gemm
}  // namespace hpc

#endif  // SRC_GEMM_SM100_GEMM_BF16XFP32_KERNELS_CUH_
