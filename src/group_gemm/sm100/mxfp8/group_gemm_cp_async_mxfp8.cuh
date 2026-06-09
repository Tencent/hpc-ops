// Copyright 2026 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM100_MXFP8_GROUP_GEMM_CP_ASYNC_MXFP8_CUH_
#define SRC_GROUP_GEMM_SM100_MXFP8_GROUP_GEMM_CP_ASYNC_MXFP8_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/algorithm/gemm.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm100.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_sm100.hpp"
#include "cute/atom/copy_traits_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm100.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/float8.h"
#include "cutlass/numeric_types.h"
#include "src/group_gemm/sm100/mxfp8/config.h"
#include "src/group_gemm/sm100/mxfp8/group_gemm_mxfp8.cuh"  // for get_next_tile_horizon_mxfp8
#include "src/group_gemm/sm100/mxfp8/utils.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace group_gemm {

using namespace cute;  // NOLINT

namespace kernels {

template <typename GemmConfig, typename TmaB, typename TmaY, typename TmaSFA, typename TmaSFB,
          bool kUsePDL = false>
__global__ __launch_bounds__(384, 1) void group_gemm_1sm_cp_async_mxfp8_kernel(
    const __grid_constant__ TmaB tma_b, const __grid_constant__ TmaSFB tma_sfb,
    const __grid_constant__ TmaSFA tma_sfa, cute::TmaDescriptor *td_ay,
    const typename GemmConfig::Tin *__restrict__ x_ptr, const int *__restrict__ x_row_map_ptr,
    int x_num_rows, int *seqlens_ptr, int *cu_seqlens_ptr, int *tiles_ptr, int *cu_tiles_ptr,
    int num_group, int m, int n, int k, cutlass::FastDivmod flat_divider) {
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  using namespace cute;  // NOLINT

  using Tin = typename GemmConfig::Tin;
  using Tout = typename GemmConfig::Tout;
  using Tsf = typename GemmConfig::Tsf;
  using SLayoutA = typename GemmConfig::SLayoutA;
  using SLayoutB = typename GemmConfig::SLayoutB;
  using SLayoutSFA = typename GemmConfig::SLayoutSFA;
  using SLayoutSFB = typename GemmConfig::SLayoutSFB;
  using SLayoutY = typename GemmConfig::SLayoutY;
  using SLayoutYT = typename GemmConfig::SLayoutYT;
  using TiledMma = typename GemmConfig::TiledMma;
  using G2SCopy = typename GemmConfig::G2SCopy;

  constexpr int kTileM = GemmConfig::kTileM;
  constexpr int kTileN = GemmConfig::kTileN;
  constexpr int kTileK = GemmConfig::kTileK;
  constexpr int kEpiTileM = GemmConfig::kEpiTileM;
  constexpr int kStage = GemmConfig::kStage;
  constexpr int kStageTMA = GemmConfig::kStageTMA;
  constexpr int kStageTile = GemmConfig::kStageTile;
  constexpr int kStageTask = 5;
  constexpr int kSfVec = GemmConfig::kSfVec;
  constexpr bool kSmallTM = GemmConfig::kSmallTM;
  constexpr int kSfxRows = GemmConfig::kSfxRows;
  constexpr int kMmaSM = GemmConfig::kMmaSM;
  constexpr int kScaleColsPerTile = GemmConfig::kScaleColsPerTile;

  static_assert(kMmaSM == 1, "cp_async mxfp8 kernel is 1SM only");

  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int elected = cute::elect_one_sync();

  // Smem
  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_a = reinterpret_cast<Tin *>(shm_data);
  auto *shm_b = shm_a + cosize(SLayoutA{});
  auto *shm_sfa = reinterpret_cast<Tsf *>(shm_b + cosize(SLayoutB{}));
  auto *shm_sfb = shm_sfa + cosize(SLayoutSFA{});
  auto *shm_yt = reinterpret_cast<Tout *>(shm_sfb + cosize(SLayoutSFB{}));
  int *shm_tiles = reinterpret_cast<int *>(shm_yt + cosize(SLayoutYT{}));
  int *shm_cu_tiles = shm_tiles + (num_group + 1);

  Tensor sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
  Tensor sSFA = make_tensor(make_smem_ptr(shm_sfa), SLayoutSFA{});
  Tensor sSFB = make_tensor(make_smem_ptr(shm_sfb), SLayoutSFB{});
  Tensor sY = make_tensor(make_smem_ptr(shm_yt), SLayoutY{});
  Tensor sYT = make_tensor(make_smem_ptr(shm_yt), SLayoutYT{});

  // Global tensors
  TmaY tma_y;
  auto gB = tma_b.get_tma_tensor(make_shape(n, k, num_group));
  int sfx_max_tiles = m / kTileM + num_group;
  int Ksf_tiles_runtime = (k / kSfVec + 3) / 4;
  auto gSFA = tma_sfa.get_tma_tensor(
      make_shape(Int<kSfxRows>{}, Int<16>{}, sfx_max_tiles, Ksf_tiles_runtime));
  auto gSFB = tma_sfb.get_tma_tensor(
      make_shape(Int<32>{}, Int<16>{}, num_group * (n / 128), Ksf_tiles_runtime));
  auto gY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));

  auto gA_full = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                             make_stride(k, Int<1>{}));
  auto gA_tiled = local_tile(gA_full, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(0, _));
  // Identity tensor for recovering the in-tile (row, k) coord per cp.async element.
  auto IA = make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileK>{}));

  // TMA partition (B, SFA, SFB only)
  auto btma_b = tma_b.get_slice(0);
  auto btma_sfa = tma_sfa.get_slice(0);
  auto btma_sfb = tma_sfb.get_slice(0);
  auto tBg = btma_b.partition_S(gB);
  auto tBs = btma_b.partition_D(sB);
  auto tSFAg = btma_sfa.partition_S(gSFA);
  auto tSFAs = btma_sfa.partition_D(sSFA);
  auto tSFBg = btma_sfb.partition_S(gSFB);
  auto tSFBs = btma_sfb.partition_D(sSFB);

  int ntile_k = size<2>(tBg);

  // TiledMma partition
  TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(0);
  auto tBr = cta_mma.make_fragment_A(cta_mma.partition_A(sB));
  auto tAr = cta_mma.make_fragment_B(cta_mma.partition_B(sA));
  auto tCt = cta_mma.make_fragment_C(cta_mma.partition_C(gY));

  // SF UTCCP descriptors (pre-built for all SMEM SF stages).
  __shared__ uint64_t sfb_desc[kStage];
  __shared__ uint64_t sfa_low_desc[kStage];
  __shared__ uint64_t sfa_high_desc[kStage];  // only used when !kSmallTM
#pragma unroll
  for (int s = 0; s < kStage; s++) {
    auto *sfb_ptr = shm_sfb + s * 32 * 16;
    auto *sfa_ptr = shm_sfa + s * kSfxRows * 16;
    Tensor t_sfb = make_tensor(
        make_smem_ptr(sfb_ptr),
        make_layout(make_shape(Int<32>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));
    Tensor t_sfa_lo = make_tensor(
        make_smem_ptr(sfa_ptr),
        make_layout(make_shape(Int<32>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));
    sfb_desc[s] = static_cast<uint64_t>(UMMA::make_umma_desc<UMMA::Major::K>(t_sfb));
    sfa_low_desc[s] = static_cast<uint64_t>(UMMA::make_umma_desc<UMMA::Major::K>(t_sfa_lo));
    if constexpr (!kSmallTM) {
      Tensor t_sfa_hi = make_tensor(
          make_smem_ptr(sfa_ptr + 32 * 16),
          make_layout(make_shape(Int<32>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));
      sfa_high_desc[s] = static_cast<uint64_t>(UMMA::make_umma_desc<UMMA::Major::K>(t_sfa_hi));
    }
  }

  // TMEM allocation
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
  constexpr int kTmemCols =
      next_power2(std::integral_constant<int, kStageTile * kTileM + kStage * kScaleColsPerTile>{});

  __shared__ uint32_t s_tmem_base;
  __shared__ uint64_t tma_readable[kStage];
  __shared__ uint64_t cp_async_readable[kStage];
  __shared__ uint64_t cp_async_writable[kStage];
  __shared__ uint64_t tma_sf_readable[kStage];
  __shared__ uint64_t utccp_done[kStage];
  __shared__ uint64_t tmem_readable[kStageTile];
  __shared__ uint64_t tmem_writable[kStageTile];

  __shared__ uint64_t task_readable[kStageTask];
  __shared__ uint64_t task_writable[kStageTask];
  __shared__ int task_shm[kStageTask][4];

  if (iwarp == 0 && elected) {
#pragma unroll
    for (int i = 0; i < kStage; i++) {
      initialize_barrier(tma_readable[i], 1);         // 1 transaction (B)
      initialize_barrier(cp_async_readable[i], 128);  // 128 cp.async producers (W8-W11)
      initialize_barrier(cp_async_writable[i], 1);    // 1 arrive (W2 MMA)
      initialize_barrier(tma_sf_readable[i], 1);
      initialize_barrier(utccp_done[i], 1);
    }
#pragma unroll
    for (int i = 0; i < kStageTile; i++) {
      initialize_barrier(tmem_readable[i], 1);
      initialize_barrier(tmem_writable[i], 1);
    }

    constexpr int kMmaThreads = 32;    // W2
    constexpr int kEpiThreads = 128;   // W4-W7
    constexpr int kTmaThreads = 128;   // W8-W11 (cp.async + TMA leader)
    constexpr int kUtccpThreads = 32;  // W1
#pragma unroll
    for (int i = 0; i < kStageTask; i++) {
      initialize_barrier(task_readable[i], 1);
      initialize_barrier(task_writable[i], kMmaThreads + kTmaThreads + kUtccpThreads + kEpiThreads);
    }
    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 1) {
    tmem_allocator.allocate(kTmemCols, &s_tmem_base);
    tmem_allocator.release_allocation_lock();
  }

  // Load shm_tiles
  for (int i = idx; i < num_group; i += blockDim.x) {
    shm_tiles[i] = tiles_ptr[i];
  }
  for (int i = idx; i <= num_group; i += blockDim.x) {
    shm_cu_tiles[i] = cu_tiles_ptr[i];
  }

  // fp4 weight
  if constexpr (GemmConfig::kNeedPreZeroB) {
    for (int i = idx; i < static_cast<int>(cosize(SLayoutB{})); i += blockDim.x) {
      reinterpret_cast<uint8_t *>(shm_b)[i] = 0;
    }
  }
  __syncthreads();

  tCt.data() = make_tmem_ptr<float>(s_tmem_base);

  constexpr uint32_t kExpectedBytesB = GemmConfig::kExpectedBytesB;
  constexpr uint32_t kExpectedBytesSF = 32 * 16 + kSfxRows * 16;

  if (idx >= 256) {
    // W8-W11 (128 threads): cp.async A + TMA leader (idx==256, elected) for B/SFB/SFA.
    int local_idx = idx - 256;
    bool is_tma_leader = (idx == 256);

    int phase = 1;  // cp_async_writable phase (whole-stage reusable signal)
    int istage_k = 0;

    int phase_task = 0;
    int istage_task = 0;

    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;
    int iblock = blockIdx.x;

    get_next_tile_horizon_mxfp8<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                        sum_tile_m, flat_divider);

    // cp.async partition (A side). The G2SCopy maps thread id → element.
    G2SCopy g2s_tiled_copy;
    auto g2s_thr_copy = g2s_tiled_copy.get_slice(local_idx);  // [0, 128)
    auto tAg = g2s_thr_copy.partition_S(gA_tiled);            // (CPY, COPY_M, COPY_K, K_tile)
    auto tAs = g2s_thr_copy.partition_D(sA);                  // (CPY, COPY_M, COPY_K, kStage)
    auto tIA = g2s_thr_copy.partition_S(IA);                  // (CPY, COPY_M, COPY_K)

    constexpr int kCopyM = size<1>(tAg);
    int a_row_offsets[kCopyM];

    const bool kUseRowMap = (x_row_map_ptr != nullptr);

    while (true) {
      if (igroup >= 0) {
        if (is_tma_leader) {
          auto *td_y_g = td_ay + igroup * 2 + 1;
          prefetch_tma_descriptor(td_y_g);
        }

        // Precompute per-row A fix-up offsets for this (igroup, itile_m) tile.
        int cu_seq = cu_seqlens_ptr[igroup];
        int tile_base_row = cu_seq + itile_m * kTileM;
        int rows_in_group = seqlens_ptr[igroup];
        int remaining_m = rows_in_group - itile_m * kTileM;
#pragma unroll
        for (int ir = 0; ir < kCopyM; ir++) {
          int ir_in_tile = get<0>(tIA(0, ir, 0));
          bool valid = (ir_in_tile < remaining_m);
          int abs_row = tile_base_row + ir_in_tile;
          int src_row;
          if (kUseRowMap) {
            src_row = valid ? x_row_map_ptr[abs_row] : ir_in_tile;
          } else {
            src_row = valid ? abs_row : ir_in_tile;
          }
          a_row_offsets[ir] = (src_row - ir_in_tile) * k;
        }

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(cp_async_writable[istage_k], phase);

          if (is_tma_leader) {
            int sfb_flat_n = igroup * (n / 128) + itile_n;
            int sfa_tile_global = shm_cu_tiles[igroup] + itile_m;
            copy(tma_sfb.with(tma_sf_readable[istage_k], 0, TMA::CacheHintSm90::EVICT_FIRST),
                 tSFBg(_, 0, 0, sfb_flat_n, itile_k), tSFBs(_, 0, 0, istage_k));
            copy(tma_sfa.with(tma_sf_readable[istage_k], 0, TMA::CacheHintSm90::EVICT_FIRST),
                 tSFAg(_, 0, 0, sfa_tile_global, itile_k), tSFAs(_, 0, 0, istage_k));
            set_barrier_transaction_bytes(tma_sf_readable[istage_k], kExpectedBytesSF);

            copy(tma_b.with(tma_readable[istage_k], 0, TMA::CacheHintSm90::EVICT_FIRST),
                 tBg(_, itile_n, itile_k, igroup), tBs(_, 0, 0, istage_k));
            set_barrier_transaction_bytes(tma_readable[istage_k], kExpectedBytesB);
          }

          // === cp.async path: A (all 128 lanes in this warp group) ===
#pragma unroll
          for (int ir = 0; ir < kCopyM; ir++) {
            auto tAg_src = make_tensor(tAg(_, ir, _, itile_k).data() + a_row_offsets[ir],
                                       tAg(_, ir, _, itile_k).layout());
            cute::copy(g2s_tiled_copy, tAg_src, tAs(_, ir, _, istage_k));
          }
          cpasync_barrier_arrive_noinc(reinterpret_cast<uint64_t *>(&cp_async_readable[istage_k]));

          istage_k++;
          if (istage_k == kStage) {
            phase ^= 1;
            istage_k = 0;
          }
        }
      }

      wait_barrier(task_readable[istage_task], phase_task);
      igroup = task_shm[istage_task][0];
      itile_m = task_shm[istage_task][1];
      itile_n = task_shm[istage_task][2];
      arrive_barrier(task_writable[istage_task]);

      if (igroup < 0) {
        break;
      }

      istage_task++;
      if (istage_task == kStageTask) {
        istage_task = 0;
        phase_task ^= 1;
      }
    }
  } else if (iwarp == 1) {
    // W1: SF SMEM -> SF TMEM via UTCCP. Same as the TMA-only kernel.
    int phase_sf = 0;
    int phase_utccp_w = 1;
    int istage_k = 0;

    int phase_task = 0;
    int istage_task = 0;

    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;
    int iblock = blockIdx.x;

    uint32_t sf_base = s_tmem_base + kStageTile * kTileM;

    get_next_tile_horizon_mxfp8<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                        sum_tile_m, flat_divider);

    while (true) {
      if (igroup >= 0) {
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(cp_async_writable[istage_k], phase_utccp_w);
          wait_barrier(tma_sf_readable[istage_k], phase_sf);

          uint32_t sf_slot = sf_base + istage_k * kScaleColsPerTile;

          if (elected) {
            SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(sfb_desc[istage_k], sf_slot);
            SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(sfa_low_desc[istage_k],
                                                                    sf_slot + 4);
            if constexpr (!kSmallTM) {
              SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(sfa_high_desc[istage_k],
                                                                      sf_slot + 4 + 4);
            }
          }

          cutlass::arch::umma_arrive(&utccp_done[istage_k]);

          istage_k++;
          if (istage_k == kStage) {
            phase_sf ^= 1;
            phase_utccp_w ^= 1;
            istage_k = 0;
          }
        }
      }

      wait_barrier(task_readable[istage_task], phase_task);
      igroup = task_shm[istage_task][0];
      itile_m = task_shm[istage_task][1];
      itile_n = task_shm[istage_task][2];
      arrive_barrier(task_writable[istage_task]);

      if (igroup < 0) {
        break;
      }

      istage_task++;
      if (istage_task == kStageTask) {
        istage_task = 0;
        phase_task ^= 1;
      }
    }
  } else if (iwarp == 2) {
    // W2: MMA warp
    int phase = 0;
    int phase_cpasync = 0;
    int phase_utccp_done = 0;
    int istage_k = 0;
    int istage_tile = 0;
    int phase_tile = 1;

    int phase_task = 0;
    int istage_task = 0;

    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;
    int iblock = blockIdx.x;

    get_next_tile_horizon_mxfp8<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                        sum_tile_m, flat_divider);

    while (true) {
      if (igroup >= 0) {
        uint32_t c_base = s_tmem_base + istage_tile * kTileM;
        tCt.data() = make_tmem_ptr<float>(c_base);
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

        uint32_t sf_base = s_tmem_base + kStageTile * kTileM;

        wait_barrier(tmem_writable[istage_tile], phase_tile);

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(utccp_done[istage_k], phase_utccp_done);
          wait_barrier(tma_readable[istage_k], phase);
          wait_barrier(cp_async_readable[istage_k], phase_cpasync);

          uint32_t sf_slot = sf_base + istage_k * kScaleColsPerTile;

#pragma unroll
          for (uint32_t ik = 0; ik < size<2>(tBr); ik++) {
            uint32_t sfb_addr_ki = (sf_slot & 0x3FFFFFFFu) | (ik << 30);
            uint32_t sfa_addr_ki = ((sf_slot + 4) & 0x3FFFFFFFu) | (ik << 30);
            auto sfb_ki = make_tensor(make_tmem_ptr<cutlass::float_ue8m0_t>(sfb_addr_ki),
                                      make_layout(make_shape(1)));
            auto sfa_ki = make_tensor(make_tmem_ptr<cutlass::float_ue8m0_t>(sfa_addr_ki),
                                      make_layout(make_shape(1)));
            cute::gemm(tiled_mma.with(tiled_mma.accumulate_, sfb_ki, sfa_ki),
                       tBr(_, _, ik, istage_k), tAr(_, _, ik, istage_k), tCt);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }

          cutlass::arch::umma_arrive(&cp_async_writable[istage_k]);

          istage_k++;
          if (istage_k == kStage) {
            phase ^= 1;
            phase_cpasync ^= 1;
            phase_utccp_done ^= 1;
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

      wait_barrier(task_readable[istage_task], phase_task);
      igroup = task_shm[istage_task][0];
      itile_m = task_shm[istage_task][1];
      itile_n = task_shm[istage_task][2];
      arrive_barrier(task_writable[istage_task]);

      if (igroup < 0) {
        break;
      }

      istage_task++;
      if (istage_task == kStageTask) {
        istage_task = 0;
        phase_task ^= 1;
      }
    }
  } else if (iwarp == 3 && elected) {
    // W3: FindTask
    int phase_task_write = 1;
    int istage_task = 0;

    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;
    int iblock = blockIdx.x;

    while (true) {
      wait_barrier(task_writable[istage_task], phase_task_write);
      iblock += gridDim.x;
      get_next_tile_horizon_mxfp8<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                          sum_tile_m, flat_divider);
      task_shm[istage_task][0] = igroup;
      task_shm[istage_task][1] = itile_m;
      task_shm[istage_task][2] = itile_n;
      arrive_barrier(task_readable[istage_task]);

      if (igroup < 0) {
        break;
      }

      istage_task++;
      if (istage_task == kStageTask) {
        istage_task = 0;
        phase_task_write ^= 1;
      }
    }
  } else if (idx >= 128 && idx < 256) {
    // W4-W7: Epilogue (TMEM → STSM → TMA store)
    int epi_idx = idx - 128;
    bool is_leader = elected && (iwarp == 4);

    auto epi_tiler = make_tile(Int<kTileN>{}, Int<kEpiTileM>{});
    auto tCt_epi = zipped_divide(tCt, make_tile(epi_tiler));
    auto sY_epi = zipped_divide(sY, epi_tiler);
    auto sYT_epi = zipped_divide(sYT, epi_tiler);

    using TmemLoadAtom = typename GemmConfig::TmemLoadAtom;
    auto tiled_copy_t2r = make_tmem_copy(TmemLoadAtom{}, tCt_epi(_, _0{}));
    auto thr_copy_t2r = tiled_copy_t2r.get_slice(epi_idx);
    auto tCt4r = thr_copy_t2r.partition_S(tCt_epi);
    auto tCr4t = make_tensor_like<float>(thr_copy_t2r.partition_D(sY_epi(_, 0)));

    using StsmTAtom = typename GemmConfig::StsmTAtom;
    auto tiled_copy_r2s = make_tiled_copy_D(Copy_Atom<StsmTAtom, Tout>{}, tiled_copy_t2r);
    auto thr_copy_r2s = tiled_copy_r2s.get_slice(epi_idx);
    auto tCr4s = make_tensor_like<Tout>(thr_copy_r2s.partition_S(sY_epi(_, 0)));
    auto tCs4r = thr_copy_r2s.partition_D(sYT_epi);

    auto nepi_tile = size<2>(tCt4r);

    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    int phase_tile = 0;
    int phase_task = 0;

    int istage_tile = 0;
    int istage_task = 0;
    int istage_tma = 0;

    auto tCt4r_base_ptr = tCt4r.data();
    int iblock = blockIdx.x;
    get_next_tile_horizon_mxfp8<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                        sum_tile_m, flat_divider);

    while (true) {
      if (igroup >= 0) {
        tCt4r.data() = tCt4r_base_ptr + istage_tile * kTileM;
        auto *td_y_g = td_ay + igroup * 2 + 1;
        prefetch_tma_descriptor(td_y_g);
        wait_barrier(tmem_readable[istage_tile], phase_tile);
#pragma unroll
        for (int iepi = 0; iepi < nepi_tile; iepi++) {
          copy(tiled_copy_t2r, tCt4r(_, _, iepi), tCr4t);
          cutlass::arch::fence_view_async_tmem_load();

          if (iepi == nepi_tile - 1) {
            if (is_leader) {
              arrive_barrier(tmem_writable[istage_tile]);
            }
          }

          auto tCr_fp2 = recast<float2>(tCr4t);
          auto tCr_bf162 = recast<__nv_bfloat162>(tCr4s);
#pragma unroll
          for (int i = 0; i < cute::size(tCr_bf162); i++) {
            tCr_bf162(i) = __float22bfloat162_rn(tCr_fp2(i));
          }

          tma_store_wait<kStageTMA - 1>();
          cutlass::arch::NamedBarrier::sync(128, 0);

          copy(tiled_copy_r2s, tCr4s, tCs4r(_, _, istage_tma));

          tma_store_fence();
          cutlass::arch::NamedBarrier::sync(128, 0);

          if (is_leader) {
            auto gYT_tma = tma_y.get_tma_tensor(make_shape(n, m));
            auto btma_y = tma_y.get_slice(0);
            auto tDs = btma_y.partition_S(sYT);
            auto tDg = btma_y.partition_D(gYT_tma);

            cute::copy(tma_y.with(td_y_g), tDs(_, 0, 0, istage_tma),
                       tDg(_, itile_n, itile_m * nepi_tile + iepi));
            tma_store_arrive();
          }

          istage_tma = (istage_tma + 1) % kStageTMA;
        }

        istage_tile++;
        if (istage_tile == kStageTile) {
          phase_tile ^= 1;
          istage_tile = 0;
        }
      }

      wait_barrier(task_readable[istage_task], phase_task);
      igroup = task_shm[istage_task][0];
      itile_m = task_shm[istage_task][1];
      itile_n = task_shm[istage_task][2];
      arrive_barrier(task_writable[istage_task]);

      if (igroup < 0) {
        break;
      }

      istage_task++;
      if (istage_task == kStageTask) {
        istage_task = 0;
        phase_task ^= 1;
      }
    }
  }

  __syncthreads();
  if (iwarp == 1) {
    tmem_allocator.free(s_tmem_base, kTmemCols);
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels
}  // namespace group_gemm
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM100_MXFP8_GROUP_GEMM_CP_ASYNC_MXFP8_CUH_
