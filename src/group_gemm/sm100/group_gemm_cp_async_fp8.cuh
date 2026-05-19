// Copyright 2025 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM100_GROUP_GEMM_CP_ASYNC_FP8_CUH_
#define SRC_GROUP_GEMM_SM100_GROUP_GEMM_CP_ASYNC_FP8_CUH_

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
  // if(igroup < 0) {
  //   return;
  // }

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

template <typename GemmConfig, typename TmaX, typename TmaW, typename TmaY, int kTaskLoopPolicy,
          bool kUsePDL = false>
__global__ void __launch_bounds__(384, 1)
    group_gemm_1sm_cp_async_fp8_kernel(const __grid_constant__ TmaW tma_w,
                                       cute::TmaDescriptor *td_xy, typename GemmConfig::Tin *x_ptr,
                                       int *cu_seqlens_ptr, int *seqlens_ptr, float *yscale_ptr,
                                       int *tiles_ptr, int *cu_tiles_ptr, int num_group, int m,
                                       int n, int k, cutlass::FastDivmod flat_divider,
                                       const int *x_row_map_ptr = nullptr, int x_num_rows = 0) {
  using namespace cute;  // NOLINT

  using Tin = typename GemmConfig::Tin;
  using Tout = typename GemmConfig::Tout;
  using SLayoutX = typename GemmConfig::SLayoutX;
  using SLayoutW = typename GemmConfig::SLayoutW;
  using SLayoutY = typename GemmConfig::SLayoutY;
  using G2SCopy = typename GemmConfig::G2SCopy;

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
  constexpr int kCtaTileN = GemmConfig::kCtaTileN;

  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int elected = cute::elect_one_sync();
  // int ilane = idx % 32;
  // bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  int iblock = blockIdx.x;

  constexpr int kStageClc = 5;
  constexpr int kClusterSize = kClusterM * kClusterN * kClusterK;

  __shared__ uint64_t cp_async_readable[kStageK];
  __shared__ uint64_t cp_async_writable[kStageK];
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
  auto gY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto gX = local_tile(X, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(0, _));
  auto IX = make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileK>{}));

  // TMA partition
  auto btma_w = tma_w.get_slice(0);
  auto btma_x = tma_x.get_slice(0);

  auto tWg = btma_w.partition_S(gW);  // (TMA, TMA_M, TMA_K)
  auto tWs = btma_w.partition_D(sW);  // (TMA, _1, _1, kStage)

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

  // __syncthreads();
  if (iwarp == 0 && elected) {
#pragma unroll
    for (int ik = 0; ik < kStageK; ++ik) {
      initialize_barrier(cp_async_readable[ik], 128);
      initialize_barrier(cp_async_writable[ik], 1);
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
    constexpr int kTmaThreads = 128;  // 1;
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

  int ntile_k = size<2>(tWg);

  constexpr int kTransactionBytesW = kMmaSM * sizeof(Tin) * (kCtaTileN * kTileK);

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

  // Optimized: A via TMA (tma_readable barrier), B via CP.ASYNC (cp_async_readable barrier)
  // UMMA waits on both barriers — TMA barrier is transaction-based (near-zero overhead)
  if (idx >= 256) {
    idx -= 256;

    int phase = 1;
    int istage_k = 0;
    int phase_clc = 0;
    int istage_clc = 0;

    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                  sum_tile_m, flat_divider);

    // B partitions only — A uses TMA
    G2SCopy g2s_tiled_copy;
    auto g2s_thr_copy = g2s_tiled_copy.get_slice(idx);
    auto tXg = g2s_thr_copy.partition_S(gX);
    auto tXs = g2s_thr_copy.partition_D(sX);
    auto tIX = g2s_thr_copy.partition_S(IX);

    constexpr int kCopyM = size<1>(tXs);
    int b_row_offsets[kCopyM];

    // When x_row_map_ptr is non-null, B's rows are gathered on-the-fly from
    // the original un-permuted x tensor via the inverse-permutation index
    // map.  This eliminates the physical gather kernel.
    //
    // Addressing derivation:
    //   tBg1(_, ir, _, itile_k).data() = B_raw + ir_in_tile*k + <thread_k_offset> + itile_k*kTileK
    //   We want to access row `src_row` (either `abs_row` or `row_map[abs_row]`).
    //   So the per-row fix-up offset in elements is:
    //     b_row_offsets[ir] = (src_row - ir_in_tile) * k
    //   Without row map this simplifies to (cu_seq + itile_n*kTileN) * k, i.e. `base_row_k`.
    const bool kUseRowMap = (x_row_map_ptr != nullptr);

    bool is_tma_leader = (idx == 0);

    while (true) {
      if (igroup >= 0) {
        // Precompute per-row B fix-up offsets for this (igroup, itile_n) tile.
        int cu_seq = cu_seqlens_ptr[igroup];
        int tile_base_row = cu_seq + itile_m * kTileM;
        int remaining_m = seqlens_ptr[igroup] - itile_m * kTileM;
#pragma unroll
        for (int ir = 0; ir < kCopyM; ir++) {
          int ir_in_tile = get<0>(tIX(0, ir, 0));
          bool valid = (ir_in_tile < remaining_m);
          int abs_row = tile_base_row + ir_in_tile;
          int src_row;
          if (kUseRowMap) {
            src_row = valid ? x_row_map_ptr[abs_row] : ir_in_tile;
          } else {
            src_row = valid ? abs_row : ir_in_tile;
          }
          b_row_offsets[ir] = (src_row - ir_in_tile) * k;
        }

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(cp_async_writable[istage_k], phase);

          // A (weight): TMA — single thread, hardware prefetch, tma_readable barrier
          if (is_tma_leader) {
            // copy(tma_a.with(tma_readable[istage_k]), tAg(_, itile_m, itile_k, igroup),
            //      tAs(_, 0, 0, istage_k));
            copy(tma_w.with(tma_readable[istage_k], 0, TMA::CacheHintSm90::EVICT_FIRST),
                 tWg(_, itile_n, itile_k, igroup), tWs(_, 0, 0, istage_k));
            set_barrier_transaction_bytes(tma_readable[istage_k], kTransactionBytesW);
          }

          // B (activation): CP.ASYNC — all 128 threads, per-row indirect
          // addressing via b_row_offsets[].
#pragma unroll
          for (int ir = 0; ir < kCopyM; ir++) {
            auto tXg_src = make_tensor(tXg(_, ir, _, itile_k).data() + b_row_offsets[ir],
                                       tXg(_, ir, _, itile_k).layout());
            cute::copy(g2s_tiled_copy, tXg_src, tXs(_, ir, _, istage_k));
          }
          cpasync_barrier_arrive_noinc(reinterpret_cast<uint64_t *>(&cp_async_readable[istage_k]));

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
    // UMMA warp — wait both tma_readable (A) and cp_async_readable (B)
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
          wait_barrier(cp_async_readable[istage_k], phase);

          for (int ik = 0; ik < size<2>(tWr); ++ik) {
            gemm(tiled_mma, tWr(_, _, ik, istage_k), tXr(_, _, ik, istage_k), tCt);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }

          cutlass::arch::umma_arrive(&cp_async_writable[istage_k]);

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
  } else if (idx >= 128 && idx < 256) {
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
        wait_barrier(tmem_readable[istage_tile], phase_tile);
        float yscale = yscale_ptr[igroup];
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

            auto *td_y = td_xy + igroup * 2 + 1;
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
    tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

}  // namespace group_gemm
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM100_GROUP_GEMM_CP_ASYNC_FP8_CUH_
