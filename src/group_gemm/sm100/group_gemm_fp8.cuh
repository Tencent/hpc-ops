// Copyright 2025 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM100_GROUP_GEMM_FP8_CUH_
#define SRC_GROUP_GEMM_SM100_GROUP_GEMM_FP8_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace group_gemm {

namespace kernels {

template <int kMmaSM>
__device__ __forceinline__ void get_next_tile_horizon(const int *tiles_ptr, int iblock,
                                                      int num_group, int &igroup, int &itile_n,
                                                      int &itile_m, int &sum_tile_m,
                                                      cutlass::FastDivmod flat_divider) {
  int num_tile_m, itile_m_total;

  flat_divider(itile_m_total, itile_n, iblock);
  itile_n /= kMmaSM;
  for (int i = igroup; i < num_group; i++) {
    num_tile_m = tiles_ptr[i];
    sum_tile_m += num_tile_m;
    if (itile_m_total < sum_tile_m) {
      igroup = i;
      sum_tile_m = sum_tile_m - num_tile_m;
      itile_m = itile_m_total - sum_tile_m;
      return;
    }
  }
  igroup = -1;
}

template <typename Tin, typename Tout, typename TmaX, typename TmaY, int kTileM,
          int kGroupPerThread, int kThreadPerBlock, bool kUsePDL = false>
__global__ void update_grouped_tma(const vec_t<cute::TmaDescriptor, 2> td_xy,
                                   cute::TmaDescriptor *tma_xy, const Tin *x_ptr, const Tout *y_ptr,
                                   const int *seqlens_ptr, const int *cu_seqlens_ptr,
                                   int *tiles_ptr, int *cu_tiles_ptr, int num_group, int m, int n,
                                   int k) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int igroup = blockIdx.x;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  if (igroup == num_group) {
    int tiles[kGroupPerThread];
#pragma unroll
    for (int i = 0; i < kGroupPerThread; i++) {
      int igroup = idx * kGroupPerThread + i;
      if (igroup < num_group) {
        tiles[i] = (seqlens_ptr[igroup] + kTileM - 1) / kTileM;
        tiles_ptr[igroup] = tiles[i];
      } else {
        tiles[i] = 0;
      }
    }

    using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    int block_aggregate;
    BlockScan(temp_storage).ExclusiveSum(tiles, tiles, block_aggregate);

#pragma unroll
    for (int i = 0; i < kGroupPerThread; i++) {
      int igroup = idx * kGroupPerThread + i;
      if (igroup < num_group) {
        cu_tiles_ptr[igroup] = tiles[i];
      }
    }
    if (idx == 0) {
      cu_tiles_ptr[num_group] = block_aggregate;
    }
  } else {
    __shared__ cute::TmaDescriptor smem_tma_desc[2];

    int num_seq = seqlens_ptr[igroup];
    uint64_t cu_seqlen = cu_seqlens_ptr[igroup];
    auto *x_ibatch_ptr = x_ptr + cu_seqlen * k;
    auto *y_ibatch_ptr = y_ptr + cu_seqlen * n;

    if (idx < 2) {
      smem_tma_desc[idx] = td_xy[idx];
    }
    __syncwarp();

    // X
    if (idx == 0) {
      auto gX = make_tensor(make_gmem_ptr(x_ibatch_ptr), make_shape(num_seq, k),
                            make_stride(k, Int<1>{}));
      update_tma_gtensor<TmaX, decltype(gX), true, true>(smem_tma_desc[idx], gX);
    }

    // K
    if (idx == 1) {
      auto gY = make_tensor(make_gmem_ptr(y_ibatch_ptr), make_shape(n, num_seq),
                            make_stride(Int<1>{}, n));
      update_tma_gtensor<TmaY, decltype(gY), true, true>(smem_tma_desc[idx], gY);
    }

#pragma unroll
    for (int i = 0; i < 2; i++) {
      __syncwarp();
      if (cute::elect_one_sync()) {
        cute::tma_desc_commit_group();
        cute::tma_desc_wait_group();
      }
      tma_descriptor_cp_fence_release(tma_xy + igroup * 2 + i, smem_tma_desc[i]);
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <typename GemmConfig, typename TmaX, typename TmaW, typename TmaY, int kTaskLoopPolicy,
          bool kUsePDL = false>
__global__ void __launch_bounds__(256, 1)
    group_gemm_2sm_fp8_kernel(const __grid_constant__ TmaW tma_w, cute::TmaDescriptor *td_xy,
                              int *seqlens_ptr, float *yscale_ptr, int *tiles_ptr,
                              int *cu_tiles_ptr, int num_group, int m, int n, int k,
                              cutlass::FastDivmod flat_divider) {
  using namespace cute;  // NOLINT

  using Tin = typename GemmConfig::Tin;
  using Tout = typename GemmConfig::Tout;
  using SLayoutX = typename GemmConfig::SLayoutX;
  using SLayoutW = typename GemmConfig::SLayoutW;
  using SLayoutY = typename GemmConfig::SLayoutY;

  constexpr int kTileM = GemmConfig::kTileM;
  constexpr int kTileN = GemmConfig::kTileN;
  constexpr int kTileK = GemmConfig::kTileK;
  constexpr int kStageK = GemmConfig::kStageK;
  constexpr int kClusterM = GemmConfig::kClusterM;
  constexpr int kClusterN = GemmConfig::kClusterN;
  constexpr int kClusterK = GemmConfig::kClusterK;
  constexpr int kMmaSM = GemmConfig::kMmaSM;
  constexpr int kEpiTileM = GemmConfig::kEpiTileM;
  constexpr int kStageTile = GemmConfig::kStageTile;
  constexpr int kStageTMA = GemmConfig::kStageTMA;
  constexpr int kCtaTileM = GemmConfig::kCtaTileM;
  constexpr int kCtaTileN = GemmConfig::kCtaTileN;

  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int elected = cute::elect_one_sync();
  // int ilane = idx % 32;

  int iblock = blockIdx.x;

  constexpr int kStageClc = 4;

  __shared__ uint64_t tma_readable[kStageK];
  __shared__ uint64_t tma_writable[kStageK];
  __shared__ uint64_t tmem_readable[kStageTile];
  __shared__ uint64_t tmem_writable[kStageTile];

  __shared__ uint64_t task_readable[kStageClc];
  __shared__ uint64_t task_writable[kStageClc];

  __shared__ int task_shm[kStageClc][4];

  __shared__ uint32_t tmem_base_ptr;

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_w = reinterpret_cast<Tin *>(shm_data);
  auto *shm_x = shm_w + cosize(SLayoutW{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_x + cosize(SLayoutX{}));
  int *shm_tiles = reinterpret_cast<int *>(shm_y + cosize(SLayoutY{}));

  TmaX tma_x;
  TmaY tma_y;

  auto sW = make_tensor(make_smem_ptr(shm_w), SLayoutW{});
  auto sX = make_tensor(make_smem_ptr(shm_x), SLayoutX{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});

  auto cluster_layout =
      tiled_divide(make_layout(make_shape(Int<kClusterM>{}, Int<kClusterN>{}, Int<kClusterK>{})),
                   make_tile(Int<1>{}, Int<kMmaSM>{}, Int<1>{}));
  int block_rank_in_cluster = cute::block_rank_in_cluster();
  auto cluster_coord = cluster_layout.get_flat_coord(block_rank_in_cluster);
  bool elected_cta = get<0>(cluster_coord) == Int<0>{};

  auto gW = tma_w.get_tma_tensor(make_shape(n, k, num_group));
  auto gX = tma_x.get_tma_tensor(make_shape(m, k));
  auto gY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));

  // TMA partition
  auto btma_gw = tma_w.get_slice(get<2>(cluster_coord) + get<0>(cluster_coord) * kClusterM);
  auto btma_gx =
      tma_x.get_slice(get<1>(cluster_coord) + get<0>(cluster_coord) * kClusterN / kMmaSM);
  auto btma_sw = tma_w.get_slice(get<2>(cluster_coord));
  auto btma_sx = tma_x.get_slice(get<1>(cluster_coord));

  auto tWg = btma_gw.partition_S(gW);  // (TMA, TMA_M, TMA_K)
  auto tWs = btma_sw.partition_D(sW);  // (TMA, _1, _1, kStage)

  auto tXg = btma_gx.partition_S(gX);  // (TMA, TMA_N, TMA_K)
  auto tXs = btma_sx.partition_D(sX);  // (TMA, _1, _1, stage)

  // UMMA partition
  typename GemmConfig::TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(get<0>(cluster_coord));

  auto tWs4r = cta_mma.partition_A(sW);
  auto tXs4r = cta_mma.partition_B(sX);
  auto tCgC = cta_mma.partition_C(gY);

  auto tWr = cta_mma.make_fragment_A(tWs4r);
  auto tXr = cta_mma.make_fragment_B(tXs4r);
  auto tCt = cta_mma.make_fragment_C(tCgC);

  uint16_t mcast_mask_c = 3;
  using TmemAllocator = TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};

  // __syncthreads();
  if (iwarp == 0 && elected) {
#pragma unroll
    for (int ik = 0; ik < kStageK; ++ik) {
      initialize_barrier(tma_readable[ik], 2);
      initialize_barrier(tma_writable[ik], 1);
    }

#pragma unroll
    for (int i = 0; i < kStageTile; ++i) {
      initialize_barrier(tmem_readable[i], 1);
      initialize_barrier(tmem_writable[i], 2);
    }

    constexpr int kMmaThreads = 32;
    constexpr int kEpiThreads = 128;
    constexpr int kTmaThreads = 1;
    if (elected_cta) {
#pragma unroll
      for (int i = 0; i < kStageClc; i++) {
        initialize_barrier(task_readable[i], 1);
        initialize_barrier(task_writable[i], kMmaThreads + kTmaThreads + kEpiThreads);
      }
    } else {
#pragma unroll
      for (int i = 0; i < kStageClc; i++) {
        initialize_barrier(task_readable[i], 1);
        initialize_barrier(task_writable[i], kTmaThreads + kEpiThreads);
      }
    }

    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 1) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &tmem_base_ptr);
  }

  // cluster_relaxed_sync();

  int ntile_k = size<2>(tXg);

  constexpr int kTransactionBytes =
      kMmaSM * sizeof(Tin) * (kCtaTileM * kTileK + kCtaTileN * kTileK);

  int total_m = 0;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  // int num_total_warps = blockDim.x / 32;
  // for (int i = iwarp; i < num_group * 2; i += num_total_warps) {
  //   tma_descriptor_fence_acquire(td_xy + i);
  // }

  if constexpr (kTaskLoopPolicy == 1) {
    for (int i = idx; i < num_group; i += blockDim.x) {
      shm_tiles[i] = tiles_ptr[i];
    }
  } else if constexpr (kTaskLoopPolicy == 2) {
    total_m = cu_tiles_ptr[num_group];
    for (int i = idx; i < (num_group + 1); i += blockDim.x) {
      shm_tiles[i] = cu_tiles_ptr[i];
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  // __syncthreads();
  cluster_relaxed_sync();

  if (iwarp == 0 && elected) {
    // TMA warp
    int phase = 1;
    int istage_k = 0;

    int phase_clc = 0;
    int istage_clc = 0;

    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                  sum_tile_m, flat_divider);

    while (true) {
      if (igroup >= 0) {
        auto *td_x = td_xy + igroup * 2;
        prefetch_tma_descriptor(td_x);

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(tma_writable[istage_k], phase);
          copy(tma_w.with(tma_readable[istage_k], 0, TMA::CacheHintSm100::EVICT_FIRST),
               tWg(_, itile_n, itile_k, igroup), tWs(_, 0, 0, istage_k));
          copy(tma_x.with(td_x, tma_readable[istage_k], 0, TMA::CacheHintSm100::EVICT_FIRST),
               tXg(_, itile_m, itile_k), tXs(_, 0, 0, istage_k));
          if (elected_cta) {
            set_barrier_transaction_bytes(tma_readable[istage_k], kTransactionBytes);
          } else {
            arrive_cluster_barrier(tma_readable[istage_k]);
          }

          istage_k++;
          if (istage_k == kStageK) {
            phase ^= 1;
            istage_k = 0;
          }
        }
      }

      wait_barrier(task_readable[istage_clc], phase_clc);
      igroup = task_shm[istage_clc][0];
      itile_m = task_shm[istage_clc][1];
      itile_n = task_shm[istage_clc][2];
      arrive_barrier(task_writable[istage_clc]);

      if (igroup < 0) {
        break;
      }

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }
    }
  } else if (iwarp == 1 && elected_cta) {
    // UMMA warp
    int phase = 0;
    int phase_tile = 1;
    int phase_clc = 0;
    int istage_k = 0;
    int istage_tile = 0;
    int istage_clc = 0;

    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                  sum_tile_m, flat_divider);

    while (true) {
      if (igroup >= 0) {
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
        wait_barrier(tmem_writable[istage_tile], phase_tile);
        tCt.data() = tmem_base_ptr + istage_tile * kTileM;

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(tma_readable[istage_k], phase);

          // Execute a MmaTile_M x MmaTile_N x MmaTile_K GEMM
          for (int ik = 0; ik < size<2>(tWr); ++ik) {
            gemm(tiled_mma, tWr(_, _, ik, istage_k), tXr(_, _, ik, istage_k), tCt);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }

          cutlass::arch::umma_arrive_multicast_2x1SM(&tma_writable[istage_k], mcast_mask_c);

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
      }

      wait_barrier(task_readable[istage_clc], phase_clc);
      igroup = task_shm[istage_clc][0];
      itile_m = task_shm[istage_clc][1];
      itile_n = task_shm[istage_clc][2];
      arrive_barrier(task_writable[istage_clc]);

      if (igroup < 0) {
        break;
      }

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }
    }
  } else if (iwarp == 3 && elected) {
    int phase_clc_read = 0;
    int phase_clc_write = 1;
    int istage_clc = 0;

    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    while (true) {
      wait_barrier(task_writable[istage_clc], phase_clc_write);
      iblock += gridDim.x;
      get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                    sum_tile_m, flat_divider);
      task_shm[istage_clc][0] = igroup;
      task_shm[istage_clc][1] = itile_m;
      task_shm[istage_clc][2] = itile_n;
      arrive_barrier(task_readable[istage_clc]);

      if (igroup < 0) {
        break;
      }

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc_write ^= 1;
        phase_clc_read ^= 1;
      }
    }

  } else if (idx >= 128) {
    idx -= 128;

    tCt.data() = tmem_base_ptr;
    auto epi_tiler = make_tile(Int<kCtaTileN>{}, Int<kEpiTileM>{});
    auto tCt_epi = zipped_divide(tCt, make_tile(epi_tiler));
    auto rC_epi = zipped_divide(gY, epi_tiler);
    auto sC_epi = zipped_divide(sY, epi_tiler);

    // TiledCopy TMEM -> RMEM
    auto tiled_copy_t2r = make_tmem_copy(SM100_TMEM_LOAD_16dp256b4x{}, tCt_epi(_, _0{}));
    auto thr_copy_t2r = tiled_copy_t2r.get_slice(idx);
    auto tCt4r = thr_copy_t2r.partition_S(tCt_epi);
    auto tCr4t = make_tensor_like<float>(thr_copy_t2r.partition_D(rC_epi(_, 0)));

    // TiledCopy RMEM -> SMEM
    auto tiled_copy_r2s_t =
        make_tiled_copy_D(Copy_Atom<cute::SM90_U16x8_STSM_T, Tout>{}, tiled_copy_t2r);

    auto thr_copy_r2s_t = tiled_copy_r2s_t.get_slice(idx);
    auto tCr4s = make_tensor_like<Tout>(thr_copy_r2s_t.partition_S(rC_epi(_, 0)));
    auto tCs4r = thr_copy_r2s_t.partition_D(sC_epi);

    auto nepi_tile = size<2>(tCt4r);

    // epi warpgroup
    int phase_tile = 0;
    int phase_clc = 0;

    int istage_tile = 0;
    int istage_clc = 0;

    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    int istage_tma = 0;

    bool is_leader = elected && (iwarp == 4);

    auto tCt4r_base_ptr = tCt4r.data();

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                  sum_tile_m, flat_divider);

    auto gYY = tma_y.get_tma_tensor(make_shape(n, m));
    auto btma_y = tma_y.get_slice(0);

    auto tDs = btma_y.partition_S(sY);   // (TMA, _2, _1)
    auto tDg = btma_y.partition_D(gYY);  // (TMA, TMA_M, TMA_N)

    while (true) {
      if (igroup >= 0) {
        tCt4r.data() = tCt4r_base_ptr + istage_tile * kTileM;
        float yscale = yscale_ptr[igroup];
        auto *td_y = td_xy + igroup * 2 + 1;
        prefetch_tma_descriptor(td_y);
        wait_barrier(tmem_readable[istage_tile], phase_tile);
        // per-group output scale (applied before fp32->bf16 cast in epilogue)
#pragma unroll
        for (int iepi = 0; iepi < nepi_tile; iepi++) {
          // TMEM -> RMEM
          copy(tiled_copy_t2r, tCt4r(_, _, iepi), tCr4t);
          cutlass::arch::fence_view_async_tmem_load();

          auto tCr_fp2 = recast<float2>(tCr4t);
          auto tCr_bf162 = recast<__nv_bfloat162>(tCr4s);

          // cast (with per-group output scale)
#pragma unroll
          for (int i = 0; i < cute::size(tCr_bf162); i++) {
            float2 v = tCr_fp2(i);
            v.x *= yscale;
            v.y *= yscale;
            tCr_bf162(i) = __float22bfloat162_rn(v);
          }

          tma_store_wait<kStageTMA - 1>();
          cutlass::arch::NamedBarrier::sync(128, 0);

          // RMEM -> SMEM
          copy(tiled_copy_r2s_t, tCr4s, tCs4r(_, _, istage_tma));

          // SMEM -> GMEM
          tma_store_fence();
          cutlass::arch::NamedBarrier::sync(128, 0);

          if (is_leader) {
            cute::copy(
                tma_y.with(td_y), tDs(_, 0, 0, istage_tma),
                tDg(_, itile_n * kMmaSM + get<0>(cluster_coord), itile_m * nepi_tile + iepi));
            tma_store_arrive();
          }

          istage_tma = (istage_tma + 1) % kStageTMA;
        }

        if (is_leader) {
          arrive_cluster_barrier(tmem_writable[istage_tile]);
        }

        istage_tile++;
        if (istage_tile == kStageTile) {
          phase_tile ^= 1;
          istage_tile = 0;
        }
      }

      wait_barrier(task_readable[istage_clc], phase_clc);
      igroup = task_shm[istage_clc][0];
      itile_m = task_shm[istage_clc][1];
      itile_n = task_shm[istage_clc][2];
      arrive_barrier(task_writable[istage_clc]);

      if (igroup < 0) {
        break;
      }

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }
    }
  }

  cluster_relaxed_sync();
  // Release the right to allocate before deallocations so that the next CTA can rasterize
  // Then deallocate TMEM
  if (iwarp == 1) {
    // tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <typename GemmConfig, typename TmaX, typename TmaW, typename TmaY, int kTaskLoopPolicy,
          bool kUsePDL = false>
__global__ void __launch_bounds__(256, 1)
    group_gemm_1sm_fp8_kernel(const __grid_constant__ TmaW tma_w, cute::TmaDescriptor *td_xy,
                              int *seqlens_ptr, float *yscale_ptr, int *tiles_ptr,
                              int *cu_tiles_ptr, int num_group, int m, int n, int k,
                              cutlass::FastDivmod flat_divider) {
  using namespace cute;  // NOLINT

  using Tin = typename GemmConfig::Tin;
  using Tout = typename GemmConfig::Tout;
  using SLayoutX = typename GemmConfig::SLayoutX;
  using SLayoutW = typename GemmConfig::SLayoutW;
  using SLayoutY = typename GemmConfig::SLayoutY;

  constexpr int kTileM = GemmConfig::kTileM;
  constexpr int kTileN = GemmConfig::kTileN;
  constexpr int kTileK = GemmConfig::kTileK;
  constexpr int kStageK = GemmConfig::kStageK;
  constexpr int kClusterM = GemmConfig::kClusterM;
  constexpr int kClusterN = GemmConfig::kClusterN;
  constexpr int kClusterK = GemmConfig::kClusterK;
  constexpr int kMmaSM = GemmConfig::kMmaSM;
  constexpr int kEpiTileM = GemmConfig::kEpiTileM;
  constexpr int kStageTile = GemmConfig::kStageTile;
  constexpr int kStageTMA = GemmConfig::kStageTMA;
  constexpr int kCtaTileM = GemmConfig::kCtaTileM;
  constexpr int kCtaTileN = GemmConfig::kCtaTileN;

  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int elected = cute::elect_one_sync();
  // int ilane = idx % 32;

  int iblock = blockIdx.x;

  constexpr int kStageClc = 5;
  constexpr int kClusterSize = kClusterM * kClusterN * kClusterK;

  __shared__ uint64_t tma_readable[kStageK];
  __shared__ uint64_t tma_writable[kStageK];
  __shared__ uint64_t tmem_readable[kStageTile];
  __shared__ uint64_t tmem_writable[kStageTile];

  __shared__ uint64_t task_readable[kStageClc];
  __shared__ uint64_t task_writable[kStageClc];

  __shared__ int task_shm[kStageClc][4];

  __shared__ uint32_t tmem_base_ptr;

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_w = reinterpret_cast<Tin *>(shm_data);
  auto *shm_x = shm_w + cosize(SLayoutW{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_x + cosize(SLayoutX{}));
  int *shm_tiles = reinterpret_cast<int *>(shm_y + cosize(SLayoutY{}));

  TmaX tma_x;
  TmaY tma_y;

  auto sW = make_tensor(make_smem_ptr(shm_w), SLayoutW{});
  auto sX = make_tensor(make_smem_ptr(shm_x), SLayoutX{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});

  auto cluster_layout =
      tiled_divide(make_layout(make_shape(Int<kClusterM>{}, Int<kClusterN>{}, Int<kClusterK>{})),
                   make_tile(Int<kMmaSM>{}, Int<1>{}, Int<1>{}));
  int block_rank_in_cluster = cute::block_rank_in_cluster();
  auto cluster_coord = cluster_layout.get_flat_coord(block_rank_in_cluster);
  bool elected_cta = get<0>(cluster_coord) == Int<0>{};

  auto gW = tma_w.get_tma_tensor(make_shape(n, k, num_group));
  auto gX = tma_x.get_tma_tensor(make_shape(m, k));
  auto gY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));

  // TMA partition
  auto btma_w = tma_w.get_slice(0);
  auto btma_x = tma_x.get_slice(0);

  auto tWg = btma_w.partition_S(gW);  // (TMA, TMA_M, TMA_K)
  auto tWs = btma_w.partition_D(sW);  // (TMA, _1, _1, kStage)

  auto tXg = btma_x.partition_S(gX);  // (TMA, TMA_N, TMA_K)
  auto tXs = btma_x.partition_D(sX);  // (TMA, _1, _1, stage)

  // UMMA partition
  typename GemmConfig::TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(0);

  auto tWs4r = cta_mma.partition_A(sW);
  auto tXs4r = cta_mma.partition_B(sX);
  auto tCgC = cta_mma.partition_C(gY);

  auto tWr = cta_mma.make_fragment_A(tWs4r);
  auto tXr = cta_mma.make_fragment_B(tXs4r);
  auto tCt = cta_mma.make_fragment_C(tCgC);

  using TmemAllocator = TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  auto next_power2 = [&](auto tmem_cols) {
    constexpr int x = decltype(tmem_cols)::value;
    if constexpr (x <= 32) {
      return 32;
    } else if constexpr (x <= 64) {
      return 64;
    } else if constexpr (x <= 128) {
      return 128;
    } else if constexpr (x <= 256) {
      return 256;
    } else {
      return 512;
    }
  };

  constexpr int kTmemCols = next_power2(std::integral_constant<int, kTileM * kStageTile>{});

  // __syncthreads();
  if (iwarp == 0 && elected) {
#pragma unroll
    for (int ik = 0; ik < kStageK; ++ik) {
      initialize_barrier(tma_readable[ik], 1);
      initialize_barrier(tma_writable[ik], 1);
    }

#pragma unroll
    for (int i = 0; i < kStageTile; ++i) {
      initialize_barrier(tmem_readable[i], 1);
      initialize_barrier(tmem_writable[i], 1);
    }

    constexpr int kMmaThreads = 32;
    constexpr int kEpiThreads = 128;
    constexpr int kTmaThreads = 1;
#pragma unroll
    for (int i = 0; i < kStageClc; i++) {
      initialize_barrier(task_readable[i], 1);
      initialize_barrier(task_writable[i],
                         kMmaThreads + kClusterSize * (kTmaThreads + kEpiThreads));
    }

    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 1) {
    // tmem_allocator.allocate(kTileN * kStageTile, &tmem_base_ptr);
    tmem_allocator.allocate(kTmemCols, &tmem_base_ptr);
    tmem_allocator.release_allocation_lock();
  }

  int ntile_k = size<2>(tWg);

  constexpr int kTransactionBytes =
      kMmaSM * sizeof(Tin) * (kCtaTileM * kTileK + kCtaTileN * kTileK);

  int total_m = 0;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  // int num_total_warps = blockDim.x / 32;
  // for (int i = iwarp; i < num_group * 2; i += num_total_warps) {
  //   tma_descriptor_fence_acquire(td_xy + i);
  // }

  if constexpr (kTaskLoopPolicy == 1) {
    for (int i = idx; i < num_group; i += blockDim.x) {
      shm_tiles[i] = tiles_ptr[i];
    }
  } else if constexpr (kTaskLoopPolicy == 2) {
    total_m = cu_tiles_ptr[num_group];
    for (int i = idx; i < (num_group + 1); i += blockDim.x) {
      shm_tiles[i] = cu_tiles_ptr[i];
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  if (iwarp == 0 && elected) {
    // TMA warp
    int phase = 1;
    int istage_k = 0;

    int phase_clc = 0;
    int istage_clc = 0;

    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                  sum_tile_m, flat_divider);

    while (true) {
      if (igroup >= 0) {
        auto *td_x = td_xy + igroup * 2;
        prefetch_tma_descriptor(td_x);
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(tma_writable[istage_k], phase);
          copy(tma_w.with(tma_readable[istage_k], 0, TMA::CacheHintSm90::EVICT_FIRST),
               tWg(_, itile_n, itile_k, igroup), tWs(_, 0, 0, istage_k));
          copy(tma_x.with(td_x, tma_readable[istage_k], 0, TMA::CacheHintSm90::EVICT_FIRST),
               tXg(_, itile_m, itile_k), tXs(_, 0, 0, istage_k));
          set_barrier_transaction_bytes(tma_readable[istage_k], kTransactionBytes);

          istage_k++;
          if (istage_k == kStageK) {
            phase ^= 1;
            istage_k = 0;
          }
        }
      }

      wait_barrier(task_readable[istage_clc], phase_clc);
      igroup = task_shm[istage_clc][0];
      itile_m = task_shm[istage_clc][1];
      itile_n = task_shm[istage_clc][2];
      arrive_barrier(task_writable[istage_clc]);

      if (igroup < 0) {
        break;
      }

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }
    }
  } else if (iwarp == 1) {
    // UMMA warp
    int phase = 0;
    int phase_tile = 1;
    int phase_clc = 0;
    int istage_k = 0;
    int istage_tile = 0;
    int istage_clc = 0;

    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                  sum_tile_m, flat_divider);

    while (true) {
      if (igroup >= 0) {
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
        wait_barrier(tmem_writable[istage_tile], phase_tile);
        tCt.data() = tmem_base_ptr + istage_tile * kTileM;

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(tma_readable[istage_k], phase);

          // Execute a MmaTile_M x MmaTile_N x MmaTile_K GEMM
          for (int ik = 0; ik < size<2>(tWr); ++ik) {
            gemm(tiled_mma, tWr(_, _, ik, istage_k), tXr(_, _, ik, istage_k), tCt);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }

          cutlass::arch::umma_arrive(&tma_writable[istage_k]);

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
      }

      wait_barrier(task_readable[istage_clc], phase_clc);
      igroup = task_shm[istage_clc][0];
      itile_m = task_shm[istage_clc][1];
      itile_n = task_shm[istage_clc][2];
      arrive_barrier(task_writable[istage_clc]);

      if (igroup < 0) {
        break;
      }

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }
    }
  } else if (iwarp == 3 && elected) {
    int phase_clc_read = 0;
    int phase_clc_write = 1;
    int istage_clc = 0;

    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    while (true) {
      wait_barrier(task_writable[istage_clc], phase_clc_write);
      iblock += gridDim.x;
      get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                    sum_tile_m, flat_divider);
      task_shm[istage_clc][0] = igroup;
      task_shm[istage_clc][1] = itile_m;
      task_shm[istage_clc][2] = itile_n;
      task_shm[istage_clc][3] = igroup;
      arrive_barrier(task_readable[istage_clc]);

      if (igroup < 0) {
        break;
      }

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc_write ^= 1;
        phase_clc_read ^= 1;
      }
    }
  } else if (idx >= 128) {
    idx -= 128;

    tCt.data() = tmem_base_ptr;
    auto epi_tiler = make_tile(Int<kCtaTileN>{}, Int<kEpiTileM>{});
    auto tCt_epi = zipped_divide(tCt, make_tile(epi_tiler));
    auto rC_epi = zipped_divide(gY, epi_tiler);
    auto sC_epi = zipped_divide(sY, epi_tiler);

    // TiledCopy TMEM -> RMEM
    auto tiled_copy_t2r = make_tmem_copy(SM100_TMEM_LOAD_16dp256b2x{}, tCt_epi(_, _0{}));
    auto thr_copy_t2r = tiled_copy_t2r.get_slice(idx);
    auto tCt4r = thr_copy_t2r.partition_S(tCt_epi);
    auto tCr4t = make_tensor_like<float>(thr_copy_t2r.partition_D(rC_epi(_, 0)));

    // TiledCopy RMEM -> SMEM
    auto tiled_copy_r2s_t =
        make_tiled_copy_D(Copy_Atom<cute::SM90_U16x8_STSM_T, Tout>{}, tiled_copy_t2r);

    auto thr_copy_r2s_t = tiled_copy_r2s_t.get_slice(idx);
    auto tCr4s = make_tensor_like<Tout>(thr_copy_r2s_t.partition_S(rC_epi(_, 0)));
    auto tCs4r = thr_copy_r2s_t.partition_D(sC_epi);

    auto nepi_tile = size<2>(tCt4r);

    // epi warpgroup
    int phase_tile = 0;
    int phase_clc = 0;

    int istage_tile = 0;
    int istage_clc = 0;

    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    int istage_tma = 0;

    bool is_leader = elected && (iwarp == 4);

    auto tCt4r_base_ptr = tCt4r.data();

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                  sum_tile_m, flat_divider);

    while (true) {
      if (igroup >= 0) {
        tCt4r.data() = tCt4r_base_ptr + istage_tile * kTileM;
        float yscale = yscale_ptr[igroup];
        auto *td_y = td_xy + igroup * 2 + 1;
        prefetch_tma_descriptor(td_y);
        wait_barrier(tmem_readable[istage_tile], phase_tile);
#pragma unroll
        for (int iepi = 0; iepi < nepi_tile; iepi++) {
          // TMEM -> RMEM
          copy(tiled_copy_t2r, tCt4r(_, _, iepi), tCr4t);
          cutlass::arch::fence_view_async_tmem_load();

          auto tCr_fp2 = recast<float2>(tCr4t);
          auto tCr_bf162 = recast<__nv_bfloat162>(tCr4s);

          // cast (with per-group output scale)
#pragma unroll
          for (int i = 0; i < cute::size(tCr_bf162); i++) {
            float2 v = tCr_fp2(i);
            v.x *= yscale;
            v.y *= yscale;
            tCr_bf162(i) = __float22bfloat162_rn(v);
          }

          tma_store_wait<kStageTMA - 1>();
          cutlass::arch::NamedBarrier::sync(128, 0);

          // RMEM -> SMEM
          copy(tiled_copy_r2s_t, tCr4s, tCs4r(_, _, istage_tma));

          // SMEM -> GMEM
          tma_store_fence();
          cutlass::arch::NamedBarrier::sync(128, 0);

          if (iwarp == 4 && elected) {
            auto gYY = tma_y.get_tma_tensor(make_shape(n, m));
            auto btma_y = tma_y.get_slice(0);

            auto tDs = btma_y.partition_S(sY);   // (TMA, _2, _1)
            auto tDg = btma_y.partition_D(gYY);  // (TMA, TMA_M, TMA_N)

            cute::copy(tma_y.with(td_y), tDs(_, 0, 0, istage_tma),
                       tDg(_, itile_n, itile_m * nepi_tile + iepi));
            tma_store_arrive();
          }

          istage_tma = (istage_tma + 1) % kStageTMA;
        }

        if (is_leader) {
          arrive_barrier(tmem_writable[istage_tile]);
        }

        istage_tile++;
        if (istage_tile == kStageTile) {
          phase_tile ^= 1;
          istage_tile = 0;
        }
      }

      wait_barrier(task_readable[istage_clc], phase_clc);
      igroup = task_shm[istage_clc][0];
      itile_m = task_shm[istage_clc][1];
      itile_n = task_shm[istage_clc][2];
      arrive_barrier(task_writable[istage_clc]);

      if (igroup < 0) {
        break;
      }

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }
    }
  }

  __syncthreads();
  // Release the right to allocate before deallocations so that the next CTA can rasterize
  // Then deallocate TMEM
  if (iwarp == 1) {
    // tmem_allocator.release_allocation_lock();
    // tmem_allocator.free(tmem_base_ptr, kTileN * kStageTile);
    tmem_allocator.free(tmem_base_ptr, kTmemCols);
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

}  // namespace group_gemm
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM100_GROUP_GEMM_FP8_CUH_
