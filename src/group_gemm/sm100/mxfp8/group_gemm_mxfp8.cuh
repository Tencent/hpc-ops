// Copyright 2026 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM100_MXFP8_GROUP_GEMM_MXFP8_CUH_
#define SRC_GROUP_GEMM_SM100_MXFP8_GROUP_GEMM_MXFP8_CUH_

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
#include "src/group_gemm/sm100/mxfp8/utils.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace group_gemm {

using namespace cute;  // NOLINT

namespace kernels {

template <int kMmaSM>
__device__ __forceinline__ void get_next_tile_horizon_mxfp8(const int *tiles_ptr, int iblock,
                                                            int num_group, int &igroup,
                                                            int &itile_n, int &itile_m,
                                                            int &sum_tile_m,
                                                            cutlass::FastDivmod flat_divider) {
  int num_tile_m, itile_m_total;
  flat_divider(itile_m_total, itile_n, iblock);
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

template <typename GemmConfig, typename TmaA, typename TmaB, typename TmaY, typename TmaSFA,
          typename TmaSFB, bool kUsePDL = false>
__global__ __launch_bounds__(256, 1) void group_gemm_1sm_mxfp8_kernel(
    const __grid_constant__ TmaB tma_b, const __grid_constant__ TmaSFB tma_sfb,
    const __grid_constant__ TmaSFA tma_sfa, cute::TmaDescriptor *td_ay, int *seqlens_ptr,
    int *tiles_ptr, int *cu_tiles_ptr, int num_group, int m, int n, int k,
    cutlass::FastDivmod flat_divider) {
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
  constexpr int kABPerSF = GemmConfig::kABPerSF;

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
  TmaA tma_a;
  TmaY tma_y;
  auto gA = tma_a.get_tma_tensor(make_shape(m, k));
  auto gB = tma_b.get_tma_tensor(make_shape(n, k, num_group));

  // sfx_max_tiles = m / kTileM + num_group   (matches launcher's SFB make_tensor; worst-case)
  int sfx_max_tiles = m / kTileM + num_group;
  int k_sf_tiles = (k / kSfVec + 3) / 4;
  auto gSFA =
      tma_sfa.get_tma_tensor(make_shape(Int<kSfxRows>{}, Int<16>{}, sfx_max_tiles, k_sf_tiles));
  auto gSFB =
      tma_sfb.get_tma_tensor(make_shape(Int<32>{}, Int<16>{}, num_group * (n / 128), k_sf_tiles));
  auto gY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));

  // TMA partition
  auto btma_a = tma_a.get_slice(0);
  auto btma_b = tma_b.get_slice(0);
  auto btma_sfa = tma_sfa.get_slice(0);
  auto btma_sfb = tma_sfb.get_slice(0);
  auto tAg = btma_a.partition_S(gA);
  auto tAs = btma_a.partition_D(sA);
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
  __shared__ uint64_t ab_readable[kStage];  // AB SMEM data ready
  __shared__ uint64_t ab_writable[kStage];  // AB SMEM reusable
  __shared__ uint64_t sf_readable[kStage];  // SF SMEM ready
  __shared__ uint64_t sf_writable[kStage];  // SF SMEM+TMEM reusable
  __shared__ uint64_t utccp_done[kStage];   // SF TMEM ready
  __shared__ uint64_t tmem_readable[kStageTile];
  __shared__ uint64_t tmem_writable[kStageTile];

  __shared__ uint64_t task_readable[kStageTask];
  __shared__ uint64_t task_writable[kStageTask];
  __shared__ int task_shm[kStageTask][4];

  if (iwarp == 0 && elected) {
#pragma unroll
    for (int i = 0; i < kStage; i++) {
      initialize_barrier(ab_readable[i], 1);
      initialize_barrier(ab_writable[i], 1);
      initialize_barrier(sf_readable[i], 1);
      initialize_barrier(sf_writable[i], 1);
      initialize_barrier(utccp_done[i], 1);
    }
#pragma unroll
    for (int i = 0; i < kStageTile; i++) {
      initialize_barrier(tmem_readable[i], 1);
      initialize_barrier(tmem_writable[i], 1);
    }

    constexpr int kMmaThreads = 32;    // W5
    constexpr int kEpiThreads = 128;   // W0-W3
    constexpr int kTmaThreads = 1;     // W4 (elected)
    constexpr int kUtccpThreads = 32;  // W7
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
  __syncthreads();

  tCt.data() = make_tmem_ptr<float>(s_tmem_base);

  constexpr uint32_t kExpectedBytesAB =
      (cosize(SLayoutB{}) / kStage + cosize(SLayoutA{}) / kStage) * sizeof(Tin);
  constexpr uint32_t kExpectedBytesSF = 32 * 16 + kSfxRows * 16;

  if (iwarp == 0 && elected) {
    // TMA warp: loads AB (per kTileK tile) and SF (per 128-K packed tile).
    int phase_ab = 1;  // ab_writable phase
    int phase_sf = 1;  // sf_writable phase
    int istage_ab = 0;
    int istage_sf = 0;

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
        auto *td_a = td_ay + igroup * 2;
        prefetch_tma_descriptor(td_a);

        int ntile_k_ab = ntile_k;
        for (int itile_k = 0; itile_k < ntile_k_ab; ++itile_k) {
          int iab_in_sf = itile_k % kABPerSF;
          int itile_sf = itile_k / kABPerSF;  // SF tile index into gmem

          // Wait for AB SMEM slot to be reusable
          wait_barrier(ab_writable[istage_ab], phase_ab);

          if (iab_in_sf == 0) {
            // First AB in this SF batch: also load SF
            wait_barrier(sf_writable[istage_sf], phase_sf);

            int sfb_flat_n = igroup * (n / 128) + itile_n;
            int sfa_tile_global = shm_cu_tiles[igroup] + itile_m;
            copy(tma_sfb.with(sf_readable[istage_sf], 0, TMA::CacheHintSm90::EVICT_FIRST),
                 tSFBg(_, 0, 0, sfb_flat_n, itile_sf), tSFBs(_, 0, 0, istage_sf));
            copy(tma_sfa.with(sf_readable[istage_sf], 0, TMA::CacheHintSm90::EVICT_FIRST),
                 tSFAg(_, 0, 0, sfa_tile_global, itile_sf), tSFAs(_, 0, 0, istage_sf));
            set_barrier_transaction_bytes(sf_readable[istage_sf], kExpectedBytesSF);
          }

          // load A/B
          copy(tma_b.with(ab_readable[istage_ab], 0, TMA::CacheHintSm90::EVICT_FIRST),
               tBg(_, itile_n, itile_k, igroup), tBs(_, 0, 0, istage_ab));
          copy(tma_a.with(td_a, ab_readable[istage_ab], 0, TMA::CacheHintSm90::EVICT_FIRST),
               tAg(_, itile_m, itile_k), tAs(_, 0, 0, istage_ab));
          set_barrier_transaction_bytes(ab_readable[istage_ab], kExpectedBytesAB);

          istage_ab++;
          if (istage_ab == kStage) {
            phase_ab ^= 1;
            istage_ab = 0;
          }
          if (iab_in_sf == kABPerSF - 1 || itile_k == ntile_k - 1) {
            istage_sf++;
            if (istage_sf == kStage) {
              phase_sf ^= 1;
              istage_sf = 0;
            }
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
    // UTCCP warp: SF SMEM -> SF TMEM. Iterates over SF tiles (128-K granularity).
    int phase_sf_r = 0;  // sf_readable phase
    int phase_sf_w = 1;  // sf_writable phase
    int istage_sf = 0;

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
        int ntile_k_sf = (ntile_k + kABPerSF - 1) / kABPerSF;
        for (int isf = 0; isf < ntile_k_sf; ++isf) {
          wait_barrier(sf_writable[istage_sf], phase_sf_w);
          wait_barrier(sf_readable[istage_sf], phase_sf_r);

          uint32_t sf_slot = sf_base + istage_sf * kScaleColsPerTile;

          if (elected) {
            SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(sfb_desc[istage_sf], sf_slot);
            SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(sfa_low_desc[istage_sf],
                                                                    sf_slot + 4);
            if constexpr (!kSmallTM) {
              SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(sfa_high_desc[istage_sf],
                                                                      sf_slot + 4 + 4);
            }
          }

          cutlass::arch::umma_arrive(&utccp_done[istage_sf]);

          istage_sf++;
          if (istage_sf == kStage) {
            phase_sf_r ^= 1;
            phase_sf_w ^= 1;
            istage_sf = 0;
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
    // MMA warp: decoupled AB/SF pipeline. SF TMEM persists for kABPerSF AB iterations.
    int phase_ab = 0;          // ab_readable phase
    int phase_utccp_done = 0;  // utccp_done phase (SF TMEM ready)
    int istage_ab = 0;
    int istage_sf = 0;
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
        // TMEM for C
        uint32_t c_base = s_tmem_base + istage_tile * kTileM;
        tCt.data() = make_tmem_ptr<float>(c_base);
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

        uint32_t sf_base = s_tmem_base + kStageTile * kTileM;

        wait_barrier(tmem_writable[istage_tile], phase_tile);

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          int iab_in_sf = itile_k % kABPerSF;

          // Wait for SF TMEM (once per SF tile, i.e. when iab_in_sf==0)
          if (iab_in_sf == 0) {
            wait_barrier(utccp_done[istage_sf], phase_utccp_done);
          }
          // Wait for AB SMEM data.
          wait_barrier(ab_readable[istage_ab], phase_ab);

          uint32_t sf_slot = sf_base + istage_sf * kScaleColsPerTile;

#pragma unroll
          for (uint32_t ik = 0; ik < size<2>(tBr); ik++) {
            // SF sub-index: offset by iab_in_sf*(kTileK/32) so that kTileK=64
            // uses ik=0,1 on first AB tile, ik=2,3 on second.
            uint32_t sf_ik = iab_in_sf * (kTileK / kSfVec) + ik;
            uint32_t sfb_addr_ki = (sf_slot & 0x3FFFFFFFu) | (sf_ik << 30);
            uint32_t sfa_addr_ki = ((sf_slot + 4) & 0x3FFFFFFFu) | (sf_ik << 30);
            auto sfb_ki = make_tensor(make_tmem_ptr<cutlass::float_ue8m0_t>(sfb_addr_ki),
                                      make_layout(make_shape(1)));
            auto sfa_ki = make_tensor(make_tmem_ptr<cutlass::float_ue8m0_t>(sfa_addr_ki),
                                      make_layout(make_shape(1)));
            cute::gemm(tiled_mma.with(tiled_mma.accumulate_, sfb_ki, sfa_ki),
                       tBr(_, _, ik, istage_ab), tAr(_, _, ik, istage_ab), tCt);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }

          // Release AB SMEM
          cutlass::arch::umma_arrive(&ab_writable[istage_ab]);

          // Release SF SMEM+TMEM after the last AB in this SF batch
          // (or on the final k-tile if ntile_k is not a multiple of kABPerSF)
          if (iab_in_sf == kABPerSF - 1 || itile_k == ntile_k - 1) {
            cutlass::arch::umma_arrive(&sf_writable[istage_sf]);
          }

          istage_ab++;
          if (istage_ab == kStage) {
            phase_ab ^= 1;
            istage_ab = 0;
          }
          if (iab_in_sf == kABPerSF - 1 || itile_k == ntile_k - 1) {
            istage_sf++;
            if (istage_sf == kStage) {
              phase_utccp_done ^= 1;
              istage_sf = 0;
            }
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
    // Warp 6: Find task
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
  } else if (idx >= 128) {
    // Warps 0-3: Epilogue (TMEM → stmatrix.trans → smem (col-major) → TMA store)
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

    // epi warpgroup
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
        auto *td_y = td_ay + igroup * 2 + 1;
        prefetch_tma_descriptor(td_y);
        wait_barrier(tmem_readable[istage_tile], phase_tile);
#pragma unroll
        for (int iepi = 0; iepi < nepi_tile; iepi++) {
          // TMEM -> RMEM
          copy(tiled_copy_t2r, tCt4r(_, _, iepi), tCr4t);
          cutlass::arch::fence_view_async_tmem_load();

          if (iepi == nepi_tile - 1) {
            if (is_leader) {
              arrive_barrier(tmem_writable[istage_tile]);
            }
          }

          // cast f32 -> bf16
          auto tCr_fp2 = recast<float2>(tCr4t);
          auto tCr_bf162 = recast<__nv_bfloat162>(tCr4s);
#pragma unroll
          for (int i = 0; i < cute::size(tCr_bf162); i++) {
            tCr_bf162(i) = __float22bfloat162_rn(tCr_fp2(i));
          }

          tma_store_wait<kStageTMA - 1>();
          cutlass::arch::NamedBarrier::sync(128, 0);

          // RMEM -> SMEM
          copy(tiled_copy_r2s, tCr4s, tCs4r(_, _, istage_tma));

          // SMEM -> GMEM
          tma_store_fence();
          cutlass::arch::NamedBarrier::sync(128, 0);

          if (is_leader) {
            auto gYT_tma = tma_y.get_tma_tensor(make_shape(n, m));
            auto btma_y = tma_y.get_slice(0);
            auto tDs = btma_y.partition_S(sYT);
            auto tDg = btma_y.partition_D(gYT_tma);

            cute::copy(tma_y.with(td_y), tDs(_, 0, 0, istage_tma),
                       tDg(_, itile_n, itile_m * nepi_tile + iepi));
            tma_store_arrive();
          }

          istage_tma = (istage_tma + 1) % kStageTMA;
        }

        // if (is_leader) {
        //   arrive_barrier(tmem_writable[istage_tile]);
        // }

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

template <typename GemmConfig, typename TmaA, typename TmaB, typename TmaY, typename TmaSFA,
          typename TmaSFB, bool kUsePDL = false>
__global__ __launch_bounds__(256, 1) void group_gemm_2sm_mxfp8_kernel(
    const __grid_constant__ TmaB tma_b, const __grid_constant__ TmaSFB tma_sfb,
    const __grid_constant__ TmaSFA tma_sfa, cute::TmaDescriptor *td_ay, int *seqlens_ptr,
    int *tiles_ptr, int *cu_tiles_ptr, int num_group, int m, int n, int k,
    cutlass::FastDivmod flat_divider) {
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

  constexpr int kTileM = GemmConfig::kTileM;
  constexpr int kTileN = GemmConfig::kTileN;
  constexpr int kTileK = GemmConfig::kTileK;
  constexpr int kCtaTileN = GemmConfig::kCtaTileN;
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
  constexpr int kABPerSF = GemmConfig::kABPerSF;

  // TMEM budget enforcement: handled by static_assert in Config (config.h).

  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int elected = cute::elect_one_sync();

  int block_rank = cute::block_rank_in_cluster();
  int sm_idx = block_rank;
  bool elected_cta = (sm_idx == 0);
  int cluster_id = blockIdx.x / kMmaSM;
  int num_clusters = gridDim.x / kMmaSM;
  int iblock = cluster_id;
  uint16_t mcast_mask = 0x3;  // 2-CTA cluster

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
  TmaA tma_a;
  TmaY tma_y;
  auto gA = tma_a.get_tma_tensor(make_shape(m, k));  // X (m, k), per-group desc (td_ay[g*2+0])
  auto gB = tma_b.get_tma_tensor(make_shape(n, k, num_group));
  // SFB static desc; the 3rd dim is "global GEMM m-tile index" (flat across groups).
  // sfx_max_tiles = m / kTileM + num_group   (worst-case)
  int sfx_max_tiles = m / kTileM + num_group;
  // Ksf_tiles = ceil(k_sf / 4) where k_sf = k / kSfVec. When k % 128 != 0 the
  // trailing SF tile is partial: prepack zero-pads it, AB TMA OOB-loads zeros,
  // so the inner_k MMA iterations covering the OOB SF lanes contribute 0.
  int k_sf_tiles = (k / kSfVec + 3) / 4;
  auto gSFA =
      tma_sfa.get_tma_tensor(make_shape(Int<kSfxRows>{}, Int<16>{}, sfx_max_tiles, k_sf_tiles));
  // SFB n-tile dim uses n/256 (kTileN granularity); the prepack ROWS_PER_TILE=256
  // on the 2SM SFW path keeps shape (64, 16, num_group * (n/256), Ksf_tiles).
  auto gSFB = tma_sfb.get_tma_tensor(
      make_shape(Int<64>{}, Int<16>{}, num_group * (n / kTileN), k_sf_tiles));
  auto gY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));

  // TMA partition (sm_idx-aware for cooperative A/B/SFB; SFA via slice(0) for multicast)
  auto btma_ga = tma_a.get_slice(sm_idx);
  auto btma_sa = tma_a.get_slice(0);
  auto btma_gb = tma_b.get_slice(sm_idx);
  auto btma_sb = tma_b.get_slice(0);
  auto btma_gsfb = tma_sfb.get_slice(sm_idx);
  auto btma_ssfb = tma_sfb.get_slice(0);
  auto btma_sfa = tma_sfa.get_slice(0);

  auto tAg = btma_ga.partition_S(gA);
  auto tAs = btma_sa.partition_D(sA);
  auto tBg = btma_gb.partition_S(gB);
  auto tBs = btma_sb.partition_D(sB);
  auto tSFAg = btma_sfa.partition_S(gSFA);
  auto tSFAs = btma_sfa.partition_D(sSFA);
  auto tSFBg = btma_gsfb.partition_S(gSFB);
  auto tSFBs = btma_ssfb.partition_D(sSFB);

  int ntile_k = size<2>(tBg);

  // TiledMma partition (SwapAB: math B → MMA-A, math A → MMA-B; sm_idx-aware)
  TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(sm_idx);
  auto tBr = cta_mma.make_fragment_A(cta_mma.partition_A(sB));
  auto tAr = cta_mma.make_fragment_B(cta_mma.partition_B(sA));
  auto tCt = cta_mma.make_fragment_C(cta_mma.partition_C(gY));

  // SF UTCCP descriptors (pre-built for all SF stages; one per packed 128-K SF tile)
  uint64_t sfb_desc[kStage];
  uint64_t sfa_low_desc[kStage];
  uint64_t sfa_high_desc[kStage];  // only used when !kSmallTM
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

  // TMEM allocation (2SM)
  using TmemAllocator = TMEM::Allocator2Sm;
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
  __shared__ uint64_t ab_writable[kStage];
  __shared__ uint64_t tma_sf_readable[kStage];
  __shared__ uint64_t sf_writable[kStage];
  __shared__ uint64_t utccp_done[kStage];
  __shared__ uint64_t tmem_readable[kStageTile];
  __shared__ uint64_t tmem_writable[kStageTile];
  __shared__ int task_shm[kStageTask][4];
  __shared__ uint64_t task_readable[kStageTask];
  __shared__ uint64_t task_writable[kStageTask];

  if (iwarp == 0 && elected) {
#pragma unroll
    for (int i = 0; i < kStage; i++) {
      initialize_barrier(tma_readable[i], 2);
      initialize_barrier(ab_writable[i], 1);
      initialize_barrier(tma_sf_readable[i], 3);
      initialize_barrier(sf_writable[i], 1);
      initialize_barrier(utccp_done[i], 1);
    }
#pragma unroll
    for (int i = 0; i < kStageTile; i++) {
      initialize_barrier(tmem_readable[i], 1);
      initialize_barrier(tmem_writable[i], 2);
    }
    constexpr int kMmaThreads = 32;    // W2
    constexpr int kEpiThreads = 128;   // W4-W7
    constexpr int kTmaThreads = 1;     // W0 (elected)
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

  // Load shm_tiles + shm_cu_tiles (per-group caches)
  for (int i = idx; i < num_group; i += blockDim.x) {
    shm_tiles[i] = tiles_ptr[i];
  }
  for (int i = idx; i <= num_group; i += blockDim.x) {
    shm_cu_tiles[i] = cu_tiles_ptr[i];
  }
  // __syncthreads();
  cluster_relaxed_sync();

  tCt.data() = make_tmem_ptr<float>(s_tmem_base);

  // Per-CTA byte counts (after sm_idx slice).
  constexpr uint32_t kBytesAB =
      (cosize(SLayoutA{}) / kStage + cosize(SLayoutB{}) / kStage) * sizeof(Tin);
  constexpr uint32_t kExpectedBytesAB = kMmaSM * kBytesAB;
  constexpr uint32_t kExpectedBytesSFB = kMmaSM * (32 * 16);  // 2 CTAs × 32×16
  constexpr uint32_t kExpectedBytesSFA = kSfxRows * 16;

  if (iwarp == 0 && elected) {
    // TMA warp (both CTAs)
    int phase_ab = 1;
    int phase_sf = 1;
    int istage_ab = 0;
    int istage_sf = 0;
    int phase_task = 0;
    int istage_task = 0;
    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    get_next_tile_horizon_mxfp8<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                        sum_tile_m, flat_divider);

    while (true) {
      if (igroup >= 0) {
        auto *td_a = td_ay + igroup * 2 + 0;
        prefetch_tma_descriptor(td_a);

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          int iab_in_sf = itile_k % kABPerSF;
          int itile_sf = itile_k / kABPerSF;  // SF K-tile index into gmem

          // Wait for AB SMEM slot
          wait_barrier(ab_writable[istage_ab], phase_ab);

          // SF TMA: load only on first AB of each SF batch
          if (iab_in_sf == 0) {
            wait_barrier(sf_writable[istage_sf], phase_sf);

            int sfb_flat_n = igroup * (n / kTileN) + itile_n;
            copy(tma_sfb.with(tma_sf_readable[istage_sf], 0, TMA::CacheHintSm100::EVICT_FIRST),
                 tSFBg(_, 0, 0, sfb_flat_n, itile_sf), tSFBs(_, 0, 0, istage_sf));
            if (elected_cta) {
              set_barrier_transaction_bytes(tma_sf_readable[istage_sf], kExpectedBytesSFB);
            } else {
              arrive_cluster_barrier(tma_sf_readable[istage_sf], 0);
            }

            int sfa_tile_global = shm_cu_tiles[igroup] + itile_m;
            if constexpr (kSmallTM) {
              if (elected_cta) {
                copy(tma_sfa.with(tma_sf_readable[istage_sf], mcast_mask),
                     tSFAg(_, 0, 0, sfa_tile_global, itile_sf), tSFAs(_, 0, 0, istage_sf));
              }
              set_barrier_transaction_bytes(tma_sf_readable[istage_sf], kExpectedBytesSFA);
            } else {
              copy(tma_sfa.with(tma_sf_readable[istage_sf], mcast_mask),
                   tSFAg(_, sm_idx, 0, sfa_tile_global, itile_sf), tSFAs(_, sm_idx, 0, istage_sf));
              set_barrier_transaction_bytes(tma_sf_readable[istage_sf], kExpectedBytesSFA);
            }
          }

          // A/B cooperative loads
          copy(tma_b.with(tma_readable[istage_ab], 0, TMA::CacheHintSm100::EVICT_FIRST),
               tBg(_, itile_n, itile_k, igroup), tBs(_, 0, 0, istage_ab));
          copy(tma_a.with(td_a, tma_readable[istage_ab], 0, TMA::CacheHintSm100::EVICT_FIRST),
               tAg(_, itile_m, itile_k), tAs(_, 0, 0, istage_ab));
          if (elected_cta) {
            set_barrier_transaction_bytes(tma_readable[istage_ab], kExpectedBytesAB);
          } else {
            arrive_cluster_barrier(tma_readable[istage_ab], 0);
          }

          istage_ab++;
          if (istage_ab == kStage) {
            phase_ab ^= 1;
            istage_ab = 0;
          }
          if (iab_in_sf == kABPerSF - 1 || itile_k == ntile_k - 1) {
            istage_sf++;
            if (istage_sf == kStage) {
              phase_sf ^= 1;
              istage_sf = 0;
            }
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
    // UTCCP warp: SF SMEM -> SF TMEM (CTA0 issues, CTA1 keeps task pipeline alive).
    int phase_sf_r = 0;
    int phase_sf_w = 1;
    int istage_sf = 0;
    int phase_task = 0;
    int istage_task = 0;
    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    uint32_t sf_base = s_tmem_base + kStageTile * kTileM;

    get_next_tile_horizon_mxfp8<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                        sum_tile_m, flat_divider);

    while (true) {
      if (igroup >= 0 && elected_cta) {
        int ntile_k_sf = (ntile_k + kABPerSF - 1) / kABPerSF;
        for (int isf = 0; isf < ntile_k_sf; ++isf) {
          wait_barrier(sf_writable[istage_sf], phase_sf_w);
          wait_barrier(tma_sf_readable[istage_sf], phase_sf_r);

          uint32_t sf_slot = sf_base + istage_sf * kScaleColsPerTile;

          if (elected) {
            SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_2cta::copy(sfb_desc[istage_sf], sf_slot);
            SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_2cta::copy(sfa_low_desc[istage_sf],
                                                                    sf_slot + 4);
            if constexpr (!kSmallTM) {
              SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_2cta::copy(sfa_high_desc[istage_sf],
                                                                      sf_slot + 4 + 4);
            }
          }

          cutlass::arch::umma_arrive_2x1SM(&utccp_done[istage_sf]);

          istage_sf++;
          if (istage_sf == kStage) {
            phase_sf_r ^= 1;
            phase_sf_w ^= 1;
            istage_sf = 0;
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
    // MMA warp
    int phase_ab = 0;
    int phase_utccp_done = 0;
    int istage_ab = 0;
    int istage_sf = 0;
    int istage_tile = 0;
    int phase_tile = 1;
    int phase_task = 0;
    int istage_task = 0;
    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    get_next_tile_horizon_mxfp8<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                        sum_tile_m, flat_divider);

    while (true) {
      if (igroup >= 0 && elected_cta) {
        uint32_t c_base = s_tmem_base + istage_tile * kTileM;
        tCt.data() = make_tmem_ptr<float>(c_base);
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

        uint32_t sf_base = s_tmem_base + kStageTile * kTileM;

        wait_barrier(tmem_writable[istage_tile], phase_tile);

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          int iab_in_sf = itile_k % kABPerSF;

          // Wait for SF TMEM (once per SF tile)
          if (iab_in_sf == 0) {
            wait_barrier(utccp_done[istage_sf], phase_utccp_done);
          }
          // Wait for AB SMEM
          wait_barrier(tma_readable[istage_ab], phase_ab);

          uint32_t sf_slot = sf_base + istage_sf * kScaleColsPerTile;

#pragma unroll
          for (uint32_t ik = 0; ik < size<2>(tBr); ik++) {
            // SF sub-index: offset by iab_in_sf*(kTileK/32) so kTileK=64
            // uses ik=0,1 on first AB, ik=2,3 on second.
            uint32_t sf_ik = iab_in_sf * (kTileK / kSfVec) + ik;
            uint32_t sfb_addr_ki = (sf_slot & 0x3FFFFFFFu) | (sf_ik << 30);
            uint32_t sfa_addr_ki = ((sf_slot + 4) & 0x3FFFFFFFu) | (sf_ik << 30);
            auto sfb_ki = make_tensor(make_tmem_ptr<cutlass::float_ue8m0_t>(sfb_addr_ki),
                                      make_layout(make_shape(1)));
            auto sfa_ki = make_tensor(make_tmem_ptr<cutlass::float_ue8m0_t>(sfa_addr_ki),
                                      make_layout(make_shape(1)));
            cute::gemm(tiled_mma.with(tiled_mma.accumulate_, sfb_ki, sfa_ki),
                       tBr(_, _, ik, istage_ab), tAr(_, _, ik, istage_ab), tCt);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }

          // Release AB SMEM
          cutlass::arch::umma_arrive_multicast_2x1SM(&ab_writable[istage_ab], mcast_mask);

          // Release SF SMEM+TMEM after the last AB in this SF batch (or last k-tile)
          if (iab_in_sf == kABPerSF - 1 || itile_k == ntile_k - 1) {
            cutlass::arch::umma_arrive_multicast_2x1SM(&sf_writable[istage_sf], mcast_mask);
          }

          istage_ab++;
          if (istage_ab == kStage) {
            phase_ab ^= 1;
            istage_ab = 0;
          }
          if (iab_in_sf == kABPerSF - 1 || itile_k == ntile_k - 1) {
            istage_sf++;
            if (istage_sf == kStage) {
              phase_utccp_done ^= 1;
              istage_sf = 0;
            }
          }
        }

        cutlass::arch::umma_arrive_multicast_2x1SM(&tmem_readable[istage_tile], mcast_mask);

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
    // Find task (both CTAs; each maintains its own task_shm).
    int phase_task_write = 1;
    int istage_task = 0;
    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    while (true) {
      wait_barrier(task_writable[istage_task], phase_task_write);
      iblock += num_clusters;
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
  } else if (idx >= 128) {
    // Epilogue warps W4-W7 (both CTAs; each stores its own kCtaTileN slab).
    int epi_idx = idx - 128;
    int tmem_rd_phase = 0;
    int istage_tile = 0;
    int istage_tma = 0;
    bool is_leader = elected && (iwarp == 4);

    auto epi_tiler = make_tile(Int<kCtaTileN>{}, Int<kEpiTileM>{});
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

    int phase_task = 0;
    int istage_task = 0;
    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    auto tCt4r_base_ptr = tCt4r.data();

    get_next_tile_horizon_mxfp8<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                        sum_tile_m, flat_divider);

    auto gYT_tma = tma_y.get_tma_tensor(make_shape(n, m));
    auto btma_y = tma_y.get_slice(0);
    auto tDs = btma_y.partition_S(sYT);
    auto tDg = btma_y.partition_D(gYT_tma);

    while (true) {
      if (igroup >= 0) {
        tCt4r.data() = tCt4r_base_ptr + istage_tile * kTileM;
        auto *td_y = td_ay + igroup * 2 + 1;
        if (is_leader) {
          prefetch_tma_descriptor(td_y);
        }
        wait_barrier(tmem_readable[istage_tile], tmem_rd_phase);

        for (int iepi = 0; iepi < nepi_tile; iepi++) {
          // TMEM -> RMEM
          copy(tiled_copy_t2r, tCt4r(_, _, iepi), tCr4t);
          cutlass::arch::fence_view_async_tmem_load();

          if (iepi == nepi_tile - 1) {
            if (is_leader) {
              arrive_cluster_barrier(tmem_writable[istage_tile], 0);
            }
          }

          // cast f32 -> bf16
          {
            auto tCr_fp2 = recast<float2>(tCr4t);
            auto tCr_bf162 = recast<__nv_bfloat162>(tCr4s);
#pragma unroll
            for (int i = 0; i < cute::size(tCr_bf162); i++) {
              tCr_bf162(i) = __float22bfloat162_rn(tCr_fp2(i));
            }
          }

          tma_store_wait<kStageTMA - 1>();
          cutlass::arch::NamedBarrier::sync(128, 0);

          // RMEM -> SMEM
          copy(tiled_copy_r2s, tCr4s, tCs4r(_, _, istage_tma));

          // SMEM -> GMEM
          tma_store_fence();
          cutlass::arch::NamedBarrier::sync(128, 0);

          if (is_leader) {
            int m_tile_idx = itile_m * (kTileM / kEpiTileM) + iepi;
            int n_tile_idx = itile_n * kMmaSM + sm_idx;
            cute::copy(tma_y.with(td_y), tDs(_, 0, 0, istage_tma), tDg(_, n_tile_idx, m_tile_idx));
            tma_store_arrive();
          }

          istage_tma = (istage_tma + 1) % kStageTMA;
        }

        // Arrive tmem_writable AFTER all iepi sub-tiles are done (fp8 2SM style).
        // if (is_leader) {
        //   arrive_cluster_barrier(tmem_writable[istage_tile], 0);
        // }

        istage_tile++;
        if (istage_tile == kStageTile) {
          tmem_rd_phase ^= 1;
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

  // cute::cluster_sync();
  cluster_relaxed_sync();
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

#endif  // SRC_GROUP_GEMM_SM100_MXFP8_GROUP_GEMM_MXFP8_CUH_
