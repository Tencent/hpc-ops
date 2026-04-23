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
                                                      int num_group, int &igroup, int &itile_m,
                                                      int &itile_n, int &sum_tile_n,
                                                      cutlass::FastDivmod flat_divider) {
  int num_tile_n, itile_n_total;
  // if(igroup < 0) {
  //   return;
  // }

  flat_divider(itile_n_total, itile_m, iblock);
  itile_m /= kMmaSM;
  for (int i = igroup; i < num_group; i++) {
    num_tile_n = tiles_ptr[i];
    sum_tile_n += num_tile_n;
    if (itile_n_total < sum_tile_n) {
      igroup = i;
      sum_tile_n = sum_tile_n - num_tile_n;
      itile_n = itile_n_total - sum_tile_n;
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
      update_tma_gtensor<TmaX>(smem_tma_desc[idx], gX);
    }

    // K
    if (idx == 1) {
      auto gY = make_tensor(make_gmem_ptr(y_ibatch_ptr), make_shape(n, num_seq),
                            make_stride(Int<1>{}, n));
      update_tma_gtensor<TmaY>(smem_tma_desc[idx], gY);
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

template <typename Config, typename TiledMma, typename TmaA, typename TmaB, typename TmaDT,
          int kTaskLoopPolicy, bool kUsePDL = false>
__global__ void __launch_bounds__(256, 1)
    group_gemm_2sm_fp8_kernel(const __grid_constant__ TmaA tma_a, cute::TmaDescriptor *td_xy,
                              int *seqlens_ptr, float *yscale_ptr, int *tiles_ptr,
                              int *cu_tiles_ptr, int num_group, int m, int n, int k,
                              cutlass::FastDivmod flat_divider) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using SLayoutA = typename Config::SLayoutX;
  using SLayoutB = typename Config::SLayoutW;
  using SLayoutC = typename Config::SLayoutY;
  using SLayoutCT = typename Config::SLayoutYT;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStageK = Config::kStage;
  constexpr int kClusterM = Config::kClusterM;
  constexpr int kClusterN = Config::kClusterN;
  constexpr int kClusterK = Config::kClusterK;
  constexpr int kMmaSM = Config::kMmaSM;
  constexpr int kEpiTileN = Config::kEpiTileN;
  constexpr int kStageTile = Config::kStageTile;
  constexpr int kStageTMA = Config::kStageTMA;
  constexpr int kCtaTileM = Config::kCtaTileM;
  constexpr int kCtaTileN = Config::kCtaTileN;

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
  auto *shm_a = reinterpret_cast<Tin *>(shm_data);
  auto *shm_b = shm_a + cosize(SLayoutA{});
  auto *shm_c = reinterpret_cast<Tout *>(shm_b + cosize(SLayoutB{}));

  int *shm_tiles = reinterpret_cast<int *>(shm_c + cosize(SLayoutCT{}));

  TmaB tma_b;
  TmaDT tma_dt;

  auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
  auto sC = make_tensor(make_smem_ptr(shm_c), SLayoutC{});
  auto sCT = make_tensor(make_smem_ptr(shm_c), SLayoutCT{});

  auto cluster_layout =
      tiled_divide(make_layout(make_shape(Int<kClusterM>{}, Int<kClusterN>{}, Int<kClusterK>{})),
                   make_tile(Int<kMmaSM>{}, Int<1>{}, Int<1>{}));
  int block_rank_in_cluster = cute::block_rank_in_cluster();
  auto cluster_coord = cluster_layout.get_flat_coord(block_rank_in_cluster);
  bool elected_cta = get<0>(cluster_coord) == Int<0>{};

  auto gA = tma_a.get_tma_tensor(make_shape(m, k, num_group));
  auto gB = tma_b.get_tma_tensor(make_shape(n, k));
  auto gC =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));

  // TMA partition
  auto btma_ga = tma_a.get_slice(get<2>(cluster_coord) + get<0>(cluster_coord) * kClusterN);
  auto btma_gb =
      tma_b.get_slice(get<1>(cluster_coord) + get<0>(cluster_coord) * kClusterM / kMmaSM);
  auto btma_sa = tma_a.get_slice(get<2>(cluster_coord));
  auto btma_sb = tma_b.get_slice(get<1>(cluster_coord));

  auto tAg = btma_ga.partition_S(gA);  // (TMA, TMA_M, TMA_K)
  auto tAs = btma_sa.partition_D(sA);  // (TMA, _1, _1, kStage)

  auto tBg = btma_gb.partition_S(gB);  // (TMA, TMA_N, TMA_K)
  auto tBs = btma_sb.partition_D(sB);  // (TMA, _1, _1, stage)

  // UMMA partition
  TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(get<0>(cluster_coord));

  auto tAs4r = cta_mma.partition_A(sA);
  auto tBs4r = cta_mma.partition_B(sB);
  auto tCgC = cta_mma.partition_C(gC);

  auto tAr = cta_mma.make_fragment_A(tAs4r);
  auto tBr = cta_mma.make_fragment_B(tBs4r);
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

  int ntile_k = size<2>(tAg);

  constexpr int kTransactionBytes =
      // kMmaSM * sizeof(Tin) * (cosize(SLayoutA{}(_, _, 0)) + cosize(SLayoutB{}(_, _, 0)));
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
    int sum_tile_n = 0;
    int itile_m, itile_n;

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_m, itile_n,
                                  sum_tile_n, flat_divider);

    while (true) {
      if (igroup >= 0) {
        auto *td_x = td_xy + igroup * 2;

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(tma_writable[istage_k], phase);
          copy(tma_a.with(tma_readable[istage_k]), tAg(_, itile_m, itile_k, igroup),
               tAs(_, 0, 0, istage_k));
          copy(tma_b.with(td_x, tma_readable[istage_k]), tBg(_, itile_n, itile_k),
               tBs(_, 0, 0, istage_k));
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
    int sum_tile_n = 0;
    int itile_m, itile_n;

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_m, itile_n,
                                  sum_tile_n, flat_divider);

    while (true) {
      if (igroup >= 0) {
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
        wait_barrier(tmem_writable[istage_tile], phase_tile);
        tCt.data() = tmem_base_ptr + istage_tile * kTileN;

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(tma_readable[istage_k], phase);

          // Execute a MmaTile_M x MmaTile_N x MmaTile_K GEMM
          for (int ik = 0; ik < size<2>(tAr); ++ik) {
            gemm(tiled_mma, tAr(_, _, ik, istage_k), tBr(_, _, ik, istage_k), tCt);
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
    int sum_tile_n = 0;
    int itile_m, itile_n;

    while (true) {
      wait_barrier(task_writable[istage_clc], phase_clc_write);
      iblock += gridDim.x;
      get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_m, itile_n,
                                    sum_tile_n, flat_divider);
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

    auto epi_tiler = make_tile(Int<kCtaTileM>{}, Int<kEpiTileN>{});
    auto tCt_epi = zipped_divide(tCt, make_tile(epi_tiler));
    auto sC_epi = zipped_divide(sC, epi_tiler);
    auto sCT_epi = zipped_divide(sCT, epi_tiler);

    // TiledCopy TMEM -> RMEM
    auto tiled_copy_t2r = make_tmem_copy(SM100_TMEM_LOAD_16dp256b4x{}, tCt_epi(_, _0{}));
    auto thr_copy_t2r = tiled_copy_t2r.get_slice(idx);
    auto tCt4r = thr_copy_t2r.partition_S(tCt_epi);
    auto tCr4t = make_tensor_like<float>(thr_copy_t2r.partition_D(sC_epi(_, 0)));

    // TiledCopy RMEM -> SMEM
    auto tiled_copy_r2s_t =
        make_tiled_copy_D(Copy_Atom<cute::SM90_U16x8_STSM_T, Tout>{}, tiled_copy_t2r);

    auto thr_copy_r2s_t = tiled_copy_r2s_t.get_slice(idx);
    auto tCTr4s = make_tensor_like<Tout>(thr_copy_r2s_t.partition_S(sC_epi(_, 0)));
    auto tCTs4r = thr_copy_r2s_t.partition_D(sCT_epi);

    auto &tCr = tCr4t;
    auto &tCTr = tCTr4s;

    auto nepi_tile = size<2>(tCt4r);

    // epi warpgroup
    int phase_tile = 0;
    int phase_clc = 0;

    int istage_tile = 0;
    int istage_clc = 0;

    int igroup = 0;
    int sum_tile_n = 0;
    int itile_m, itile_n;

    int istage_tma = 0;

    bool is_leader = elected && (iwarp == 4);

    auto tCt4r_base_ptr = tCt4r.data();

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_m, itile_n,
                                  sum_tile_n, flat_divider);

    auto gDT = tma_dt.get_tma_tensor(make_shape(m, n));
    auto btma_dt = tma_dt.get_slice(0);

    auto tDs = btma_dt.partition_S(sCT);  // (TMA, _2, _1)
    auto tDg = btma_dt.partition_D(gDT);  // (TMA, TMA_M, TMA_N)

    while (true) {
      if (igroup >= 0) {
        tCt4r.data() = tCt4r_base_ptr + istage_tile * kTileN;
        wait_barrier(tmem_readable[istage_tile], phase_tile);
        // per-group output scale (applied before fp32->bf16 cast in epilogue)
        float yscale = yscale_ptr[igroup];
#pragma unroll
        for (int iepi = 0; iepi < nepi_tile; iepi++) {
          // TMEM -> RMEM
          copy(tiled_copy_t2r, tCt4r(_, _, iepi), tCr4t);
          cutlass::arch::fence_view_async_tmem_load();

          auto tCr_fp2 = recast<float2>(tCr);
          auto tCr_bf162 = recast<__nv_bfloat162>(tCTr);

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
          copy(tiled_copy_r2s_t, tCTr4s, tCTs4r(_, _, istage_tma));

          // SMEM -> GMEM
          tma_store_fence();
          cutlass::arch::NamedBarrier::sync(128, 0);

          if (is_leader) {
            auto *td_y = td_xy + igroup * 2 + 1;
            cute::copy(
                tma_dt.with(td_y), tDs(_, 0, 0, istage_tma),
                tDg(_, itile_m * kMmaSM + get<0>(cluster_coord), itile_n * nepi_tile + iepi));
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

template <typename Config, typename TiledMma, typename TmaA, typename TmaB, typename TmaDT,
          int kTaskLoopPolicy, bool kUsePDL = false>
__global__ void __launch_bounds__(256, 1)
    group_gemm_1sm_fp8_kernel(const __grid_constant__ TmaA tma_a, cute::TmaDescriptor *td_xy,
                              int *seqlens_ptr, float *yscale_ptr, int *tiles_ptr,
                              int *cu_tiles_ptr, int num_group, int m, int n, int k,
                              cutlass::FastDivmod flat_divider) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using SLayoutA = typename Config::SLayoutX;
  using SLayoutB = typename Config::SLayoutW;
  using SLayoutC = typename Config::SLayoutY;
  using SLayoutCT = typename Config::SLayoutYT;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStageK = Config::kStage;
  constexpr int kClusterM = Config::kClusterM;
  constexpr int kClusterN = Config::kClusterN;
  constexpr int kClusterK = Config::kClusterK;
  constexpr int kMmaSM = Config::kMmaSM;
  constexpr int kEpiTileN = Config::kEpiTileN;
  constexpr int kStageTile = Config::kStageTile;
  constexpr int kStageTMA = Config::kStageTMA;
  constexpr int kCtaTileM = Config::kCtaTileM;
  constexpr int kCtaTileN = Config::kCtaTileN;

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
  auto *shm_a = reinterpret_cast<Tin *>(shm_data);
  auto *shm_b = shm_a + cosize(SLayoutA{});
  auto *shm_c = reinterpret_cast<Tout *>(shm_b + cosize(SLayoutB{}));

  int *shm_tiles = reinterpret_cast<int *>(shm_c + cosize(SLayoutCT{}));

  TmaB tma_b;
  TmaDT tma_dt;

  auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
  auto sC = make_tensor(make_smem_ptr(shm_c), SLayoutC{});
  auto sCT = make_tensor(make_smem_ptr(shm_c), SLayoutCT{});

  auto cluster_layout =
      tiled_divide(make_layout(make_shape(Int<kClusterM>{}, Int<kClusterN>{}, Int<kClusterK>{})),
                   make_tile(Int<kMmaSM>{}, Int<1>{}, Int<1>{}));
  int block_rank_in_cluster = cute::block_rank_in_cluster();
  auto cluster_coord = cluster_layout.get_flat_coord(block_rank_in_cluster);
  bool elected_cta = get<0>(cluster_coord) == Int<0>{};

  auto gA = tma_a.get_tma_tensor(make_shape(m, k, num_group));
  auto gB = tma_b.get_tma_tensor(make_shape(n, k));
  auto gC =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));

  // TMA partition
  auto btma_a = tma_a.get_slice(0);
  auto btma_b = tma_b.get_slice(0);

  auto tAg = btma_a.partition_S(gA);  // (TMA, TMA_M, TMA_K)
  auto tAs = btma_a.partition_D(sA);  // (TMA, _1, _1, kStage)

  auto tBg = btma_b.partition_S(gB);  // (TMA, TMA_N, TMA_K)
  auto tBs = btma_b.partition_D(sB);  // (TMA, _1, _1, stage)

  // UMMA partition
  TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(0);

  auto tAs4r = cta_mma.partition_A(sA);
  auto tBs4r = cta_mma.partition_B(sB);
  auto tCgC = cta_mma.partition_C(gC);

  auto tAr = cta_mma.make_fragment_A(tAs4r);
  auto tBr = cta_mma.make_fragment_B(tBs4r);
  auto tCt = cta_mma.make_fragment_C(tCgC);

  using TmemAllocator = TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

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
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &tmem_base_ptr);
  }

  int ntile_k = size<2>(tAg);

  constexpr int kTransactionBytes =
      // kMmaSM * sizeof(Tin) * (cosize(SLayoutA{}(_, _, 0)) + cosize(SLayoutB{}(_, _, 0)));
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
    int sum_tile_n = 0;
    int itile_m, itile_n;

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_m, itile_n,
                                  sum_tile_n, flat_divider);

    while (true) {
      if (igroup >= 0) {
        auto *td_x = td_xy + igroup * 2;

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(tma_writable[istage_k], phase);
          copy(tma_a.with(tma_readable[istage_k]), tAg(_, itile_m, itile_k, igroup),
               tAs(_, 0, 0, istage_k));
          copy(tma_b.with(td_x, tma_readable[istage_k]), tBg(_, itile_n, itile_k),
               tBs(_, 0, 0, istage_k));
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
    int sum_tile_n = 0;
    int itile_m, itile_n;

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_m, itile_n,
                                  sum_tile_n, flat_divider);

    while (true) {
      if (igroup >= 0) {
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
        wait_barrier(tmem_writable[istage_tile], phase_tile);
        tCt.data() = tmem_base_ptr + istage_tile * kTileN;

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(tma_readable[istage_k], phase);

          // Execute a MmaTile_M x MmaTile_N x MmaTile_K GEMM
          for (int ik = 0; ik < size<2>(tAr); ++ik) {
            gemm(tiled_mma, tAr(_, _, ik, istage_k), tBr(_, _, ik, istage_k), tCt);
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
    int sum_tile_n = 0;
    int itile_m, itile_n;

    while (true) {
      wait_barrier(task_writable[istage_clc], phase_clc_write);
      iblock += gridDim.x;
      get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_m, itile_n,
                                    sum_tile_n, flat_divider);
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

    auto epi_tiler = make_tile(Int<kCtaTileM>{}, Int<kEpiTileN>{});
    auto tCt_epi = zipped_divide(tCt, make_tile(epi_tiler));
    auto sC_epi = zipped_divide(sC, epi_tiler);
    auto sCT_epi = zipped_divide(sCT, epi_tiler);

    // TiledCopy TMEM -> RMEM
    auto tiled_copy_t2r = make_tmem_copy(SM100_TMEM_LOAD_16dp256b2x{}, tCt_epi(_, _0{}));
    auto thr_copy_t2r = tiled_copy_t2r.get_slice(idx);
    auto tCt4r = thr_copy_t2r.partition_S(tCt_epi);
    auto tCr4t = make_tensor_like<float>(thr_copy_t2r.partition_D(sC_epi(_, 0)));

    // TiledCopy RMEM -> SMEM
    auto tiled_copy_r2s_t =
        make_tiled_copy_D(Copy_Atom<cute::SM90_U16x8_STSM_T, Tout>{}, tiled_copy_t2r);

    auto thr_copy_r2s_t = tiled_copy_r2s_t.get_slice(idx);
    auto tCTr4s = make_tensor_like<Tout>(thr_copy_r2s_t.partition_S(sC_epi(_, 0)));
    auto tCTs4r = thr_copy_r2s_t.partition_D(sCT_epi);

    auto &tCr = tCr4t;
    auto &tCTr = tCTr4s;

    auto nepi_tile = size<2>(tCt4r);

    // epi warpgroup
    int phase_tile = 0;
    int phase_clc = 0;

    int istage_tile = 0;
    int istage_clc = 0;

    int igroup = 0;
    int sum_tile_n = 0;
    int itile_m, itile_n;

    int istage_tma = 0;

    bool is_leader = elected && (iwarp == 4);

    auto tCt4r_base_ptr = tCt4r.data();

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_m, itile_n,
                                  sum_tile_n, flat_divider);

    while (true) {
      if (igroup >= 0) {
        tCt4r.data() = tCt4r_base_ptr + istage_tile * kTileN;
        wait_barrier(tmem_readable[istage_tile], phase_tile);
        float yscale = yscale_ptr[igroup];
#pragma unroll
        for (int iepi = 0; iepi < nepi_tile; iepi++) {
          // TMEM -> RMEM
          copy(tiled_copy_t2r, tCt4r(_, _, iepi), tCr4t);
          cutlass::arch::fence_view_async_tmem_load();

          auto tCr_fp2 = recast<float2>(tCr);
          auto tCr_bf162 = recast<__nv_bfloat162>(tCTr);

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
          copy(tiled_copy_r2s_t, tCTr4s, tCTs4r(_, _, istage_tma));

          // SMEM -> GMEM
          tma_store_fence();
          cutlass::arch::NamedBarrier::sync(128, 0);

          if (iwarp == 4 && elected) {
            auto gDT = tma_dt.get_tma_tensor(make_shape(m, n));
            auto btma_dt = tma_dt.get_slice(0);

            auto tDs = btma_dt.partition_S(sCT);  // (TMA, _2, _1)
            auto tDg = btma_dt.partition_D(gDT);  // (TMA, TMA_M, TMA_N)

            auto *td_y = td_xy + igroup * 2 + 1;
            cute::copy(tma_dt.with(td_y), tDs(_, 0, 0, istage_tma),
                       tDg(_, itile_m, itile_n * nepi_tile + iepi));
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
    tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

}  // namespace group_gemm
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM100_GROUP_GEMM_FP8_CUH_
