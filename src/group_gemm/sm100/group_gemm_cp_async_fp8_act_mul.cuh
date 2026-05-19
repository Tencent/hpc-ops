// Copyright 2025 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM100_GROUP_GEMM_CP_ASYNC_FP8_ACT_MUL_CUH_
#define SRC_GROUP_GEMM_SM100_GROUP_GEMM_CP_ASYNC_FP8_ACT_MUL_CUH_

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
                                                      cutlass::FastDivmod flat_divider);

template <typename GemmConfig, typename TmemForStore, typename TmaGate, typename TmaUp,
          typename TmaX, typename TmaY, int kTaskLoopPolicy, bool kUsePDL = false>
__global__ void __launch_bounds__(384, 1) group_gemm_1sm_cp_async_fp8_act_mul_kernel(
    const __grid_constant__ TmaGate tma_gate, const __grid_constant__ TmaUp tma_up,
    cute::TmaDescriptor *td_xy, typename GemmConfig::Tin *x_ptr, int *cu_seqlens_ptr,
    int *seqlens_ptr, const float *gate_up_scale_ptr, const float *act_mul_scale_ptr,
    int *tiles_ptr, int *cu_tiles_ptr, int num_group, int m, int n, int k,
    cutlass::FastDivmod flat_divider, const int *x_row_map_ptr = nullptr, int x_num_rows = 0) {
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

  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int elected = cute::elect_one_sync();

  int iblock = blockIdx.x;

  constexpr int kStageClc = 5;
  constexpr int kClusterSize = kClusterM * kClusterN * kClusterK;

  __shared__ uint64_t cp_async_readable[kStageK];
  __shared__ uint64_t cp_async_writable[kStageK];
  // Single tma_readable barrier tracks BOTH gate and up A loads for a stage
  // (2 * kTransactionBytesA bytes total).
  __shared__ uint64_t tma_readable[kStageK];
  __shared__ uint64_t tma_writable[kStageK];
  __shared__ uint64_t tmem_readable[kStageTile];
  __shared__ uint64_t tmem_writable[kStageTile];

  __shared__ uint64_t task_readable[kStageClc];
  __shared__ uint64_t task_writable[kStageClc];

  __shared__ int task_shm[kStageClc][4];

  __shared__ uint32_t tmem_base_ptr;

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_w_gateup = reinterpret_cast<Tin *>(shm_data);
  auto *shm_x = shm_w_gateup + cosize(SLayoutW{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_x + cosize(SLayoutX{}));
  int *shm_tiles = reinterpret_cast<int *>(shm_y + cosize(SLayoutY{}));

  TmaX tma_x;
  TmaY tma_y;

  auto sW = make_tensor(make_smem_ptr(shm_w_gateup), SLayoutW{});
  auto sX = make_tensor(make_smem_ptr(shm_x), SLayoutX{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});

  auto cluster_layout =
      tiled_divide(make_layout(make_shape(Int<kClusterM>{}, Int<kClusterN>{}, Int<kClusterK>{})),
                   make_tile(Int<kMmaSM>{}, Int<1>{}, Int<1>{}));
  int block_rank_in_cluster = cute::block_rank_in_cluster();
  auto cluster_coord = cluster_layout.get_flat_coord(block_rank_in_cluster);
  bool elected_cta = get<0>(cluster_coord) == Int<0>{};

  auto gW_gate = tma_gate.get_tma_tensor(make_shape(n / 2, k, num_group));
  auto gW_up = tma_up.get_tma_tensor(make_shape(n / 2, k, num_group));

  auto gY = make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                        make_shape(Int<kTileN * 2>{}, Int<kTileM>{}),
                        make_stride(Int<kTileM>{}, Int<1>{}));

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto gX = local_tile(X, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(0, _));
  auto IX = make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileK>{}));

  // TMA partition
  auto btma_gate = tma_gate.get_slice(0);
  auto btma_up = tma_up.get_slice(0);
  auto btma_x = tma_x.get_slice(0);

  auto tWg_gate = btma_gate.partition_S(gW_gate);  // (TMA, TMA_M, TMA_K, num_group)
  auto tWg_up = btma_up.partition_S(gW_up);
  auto tWs_gate = btma_gate.partition_D(sW);  // (TMA, _1, _1, kStage)
  auto tWs_up = btma_up.partition_D(sW);      // (TMA, _1, _1, kStage)

  // UMMA partition
  typename GemmConfig::TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(0);

  auto tWs4r = cta_mma.partition_A(sW);
  auto tXs4r = cta_mma.partition_B(sX);
  auto tYgY = cta_mma.partition_C(gY);

  auto tWr = cta_mma.make_fragment_A(tWs4r);
  auto tXr = cta_mma.make_fragment_B(tXs4r);
  auto tCt = cta_mma.make_fragment_C(tYgY);

  using TmemAllocator = TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

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
    constexpr int kTmaThreads = 128;
#pragma unroll
    for (int i = 0; i < kStageClc; i++) {
      initialize_barrier(task_readable[i], 1);
      initialize_barrier(task_writable[i],
                         kMmaThreads + kClusterSize * (kTmaThreads + kEpiThreads + 1));
    }

    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 1) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &tmem_base_ptr);
  }

  int ntile_k = size<2>(tWg_gate);

  constexpr int kTransactionBytesW = kMmaSM * sizeof(Tin) * kTileN * kTileK * 2;

  int total_m = 0;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

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

  __syncthreads();

  // =====================================================================
  // Warpgroup 2 (idx 256..383): TMA loader for A (gate+up) + cp.async B
  // =====================================================================
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

    const bool kUseRowMap = (x_row_map_ptr != nullptr);

    while (true) {
      if (igroup >= 0) {
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
  } else if (iwarp == 1 && elected) {
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
    while (true) {
      if (igroup >= 0) {
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(cp_async_writable[istage_k], phase);
          constexpr int kLoadPerTileN = kTileN / 16;
#pragma unroll
          for (int i = 0; i < kLoadPerTileN; i++) {
            copy(tma_gate.with(tma_readable[istage_k], 0, TMA::CacheHintSm90::EVICT_FIRST),
                 tWg_gate(_, itile_n * kLoadPerTileN + i, itile_k, igroup),
                 tWs_gate(_, 2 * i, 0, istage_k));
            copy(tma_up.with(tma_readable[istage_k], 0, TMA::CacheHintSm90::EVICT_FIRST),
                 tWg_up(_, itile_n * kLoadPerTileN + i, itile_k, igroup),
                 tWs_up(_, 2 * i + 1, 0, istage_k));
          }
          set_barrier_transaction_bytes(tma_readable[istage_k], kTransactionBytesW);
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
  } else if (iwarp == 2) {
    // =====================================================================
    // UMMA warp: 2 sequential MMAs per K-tile (gate acc, then up acc).
    // =====================================================================
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
        wait_barrier(tmem_writable[istage_tile], phase_tile);
        tCt.data() = tmem_base_ptr + istage_tile * kTileM;

        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

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
    // =====================================================================
    // CLC warp (task dispatcher)
    // =====================================================================
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
    // =====================================================================
    // Epilogue warpgroup (idx 128..255): load gate+up from TMEM, compute
    // silu(gate)*up*scale, cast to fp8, STSM -> SMEM, TMA STORE -> GMEM.
    // =====================================================================
    idx -= 128;

    tCt.data() = tmem_base_ptr;

    auto epi_tiler = make_tile(Int<kTileN * 2>{}, Int<kEpiTileM>{});
    auto tCt_epi = zipped_divide(tCt, make_tile(epi_tiler));
    auto rC_epi = zipped_divide(gY, epi_tiler);
    auto tiled_copy_t2r = make_tmem_copy(SM100_TMEM_LOAD_16dp256b2x{}, tCt_epi(_, _0{}));
    auto thr_copy_t2r = tiled_copy_t2r.get_slice(idx);
    auto tCt4r = thr_copy_t2r.partition_S(tCt_epi);
    auto tCr = make_tensor_like<float>(thr_copy_t2r.partition_D(rC_epi(_, 0)));

    auto make_flatten_tCr = [&]() {
      if constexpr (kEpiTileM == 16) {
        return make_tensor(
            tCr.data(), make_layout(append(tCr.shape(), Int<1>{}), append(tCr.stride(), Int<0>{})));
      } else {
        auto tCr_flatten_dim1_layout =
            make_layout(get<0>(tCr.layout()), get<0>(flatten(get<1>(tCr.layout()))),
                        get<1>(flatten(get<1>(tCr.layout()))));
        return make_tensor(tCr.data(), tCr_flatten_dim1_layout);
      }
    };

    auto tCr_flatten = make_flatten_tCr();

    auto gO =
        make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                    make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));
    auto epi_store_tiler = make_tile(Int<kTileN>{}, Int<kEpiTileM>{});
    auto tOt_epi = zipped_divide(TmemForStore{}, make_tile(epi_store_tiler));
    auto rO_epi = zipped_divide(gO, epi_store_tiler);
    auto sC_epi = zipped_divide(sY, epi_store_tiler);
    auto tiled_copy_r2s =
        make_tiled_copy_D(Copy_Atom<SM100_U8x8_STSM_T, Tout>{},
                          make_tmem_copy(SM100_TMEM_LOAD_16dp256b2x{}, tOt_epi(_, _0{})));
    auto thr_copy_r2s = tiled_copy_r2s.get_slice(idx);
    auto tOs4r = thr_copy_r2s.partition_D(sC_epi);
    auto tOr4s = make_tensor_like<Tout>(thr_copy_r2s.partition_S(rO_epi(_, _0{})));

    auto nepi_tile = size<2>(tCt4r);

    auto r4s_store_layout = make_layout(make_shape(Int<2>{}, Int<2>{}, Int<2>{}),
                                        make_stride(Int<1>{}, Int<4>{}, Int<2>{}));

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

    float am_scale = act_mul_scale_ptr[0];

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                  sum_tile_m, flat_divider);

    while (true) {
      if (igroup >= 0) {
        tCt4r.data() = tCt4r_base_ptr + istage_tile * kTileM;
        float gu_scale = gate_up_scale_ptr[igroup];
        auto *td_y = td_xy + igroup * 2 + 1;
        prefetch_tma_descriptor(td_y);
        wait_barrier(tmem_readable[istage_tile], phase_tile);
#pragma unroll
        for (int iepi = 0; iepi < nepi_tile; iepi++) {
          // TMEM -> RMEM for gate, then up.
          copy(tiled_copy_t2r, tCt4r(_, _, iepi), tCr);
          cutlass::arch::fence_view_async_tmem_load();

          auto tCr_gate = tCr_flatten(_, 0, _);
          auto tCr_up = tCr_flatten(_, 1, _);

#pragma unroll
          for (int i = 0; i < cute::size(tCr_gate) / 8; i++) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
              int in_i = i * 8 + j;
              int out_i = i * 8 + r4s_store_layout(j);
              float g = tCr_gate(in_i) * gu_scale;
              float u = tCr_up(in_i) * gu_scale;
              float silu_g = g * rcpf_ftz(1.f + expf_ftz(-g));
              float m = silu_g * u;
              float v = m * am_scale;
              tOr4s(out_i) = static_cast<Tout>(v);
            }
          }

          tma_store_wait<kStageTMA - 1>();
          cutlass::arch::NamedBarrier::sync(128, 0);
          copy(tiled_copy_r2s, tOr4s, tOs4r(_, _, istage_tma));

          // SMEM -> GMEM (FP8 STORE)
          tma_store_fence();
          cutlass::arch::NamedBarrier::sync(128, 0);

          if (is_leader) {
            auto gYY = tma_y.get_tma_tensor(make_shape(n, m));
            auto btma_y = tma_y.get_slice(0);

            auto tDs = btma_y.partition_S(sY);
            auto tDg = btma_y.partition_D(gYY);

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
  if (iwarp == 1) {
    tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

}  // namespace group_gemm
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM100_GROUP_GEMM_CP_ASYNC_FP8_ACT_MUL_CUH_
