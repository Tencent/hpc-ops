// Copyright 2026 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM100_MXFP8_GROUP_GEMM_GATHER4_MXFP8_CUH_
#define SRC_GROUP_GEMM_SM100_MXFP8_GROUP_GEMM_GATHER4_MXFP8_CUH_

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

// cp.async 4B with zero-fill: copies src_size bytes, zero-fills (4 - src_size) bytes.
// When src_size=0, fills 4 zeros without accessing gmem_src.
__device__ __forceinline__ void cp_async_4b_g4(void *smem_dst, const void *gmem_src, int src_size) {
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem_addr), "l"(gmem_src),
               "r"(src_size));
}

// TMA gather4 inline PTX (1SM, shared::cta): loads 4 rows x boxDim[0] bytes into own SMEM.
// The mbarrier is auto-arrived with complete_tx::bytes.
__device__ __forceinline__ void tma_gather4_load(void *smem_dst, const CUtensorMap *tmap, int col,
                                                 int row0, int row1, int row2, int row3,
                                                 uint64_t &mbar) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  uint64_t tmap_addr = reinterpret_cast<uint64_t>(tmap);
  uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));

  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
      ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
      :
      : "r"(smem_addr), "l"(tmap_addr), "r"(col), "r"(row0), "r"(row1), "r"(row2), "r"(row3),
        "r"(mbar_addr)
      : "memory");
}

// TMA gather4 inline PTX (2SM, cta_group::2 shared::cluster): cooperative load across 2-CTA
// cluster. Each CTA loads its own rows; bytes route to the cluster-mapped mbarrier
// on the elected CTA (cta_id=0). The `cluster_mbar_addr` must be obtained via mapa.
__device__ __forceinline__ void tma_gather4_load_2sm(void *smem_dst, const CUtensorMap *tmap,
                                                     uint32_t cluster_mbar_addr, int col, int row0,
                                                     int row1, int row2, int row3) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  uint64_t tmap_addr = reinterpret_cast<uint64_t>(tmap);

  asm volatile(
      "cp.async.bulk.tensor.2d.cta_group::2.tile::gather4.shared::cluster.global"
      ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
      :
      : "r"(smem_addr), "l"(tmap_addr), "r"(col), "r"(row0), "r"(row1), "r"(row2), "r"(row3),
        "r"(cluster_mbar_addr)
      : "memory");
}

// Helper: get cluster-mapped mbarrier address (maps local smem mbar to target CTA's view)
__device__ __forceinline__ uint32_t get_cluster_mbar_addr(uint64_t &local_mbar,
                                                          uint32_t target_cta = 0) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&local_mbar));
  uint32_t result;
  asm volatile(
      "{\n\t"
      "mapa.shared::cluster.u32 %0, %1, %2;\n\t"
      "}"
      : "=r"(result)
      : "r"(smem_addr), "r"(target_cta));
  return result;
}

// ============================================================================
// 1SM gather4 kernel
// ============================================================================
template <typename GemmConfig, typename TmaB, typename TmaY, typename TmaSFB>
__global__ __launch_bounds__(256, 1) void group_gemm_1sm_gather4_mxfp8_kernel(
    const __grid_constant__ CUtensorMap tma_a_gather4, const __grid_constant__ TmaB tma_b,
    const __grid_constant__ TmaSFB tma_sfb, cute::TmaDescriptor *td_ay,
    const uint8_t *__restrict__ sfx_ptr, const int *__restrict__ x_row_map_ptr, int x_num_rows,
    int *seqlens_ptr, int *cu_seqlens_ptr, int *tiles_ptr, int *cu_tiles_ptr, int num_group, int m,
    int n, int k, cutlass::FastDivmod flat_divider) {
  cudaGridDependencySynchronize();
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

  static_assert(kMmaSM == 1, "1SM gather4 kernel requires kMmaSM == 1");

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
  int *shm_seqlens = shm_cu_tiles + (num_group + 1);
  int *shm_cu_seqlens = shm_seqlens + (num_group + 1);

  Tensor sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
  Tensor sSFA = make_tensor(make_smem_ptr(shm_sfa), SLayoutSFA{});
  Tensor sSFB = make_tensor(make_smem_ptr(shm_sfb), SLayoutSFB{});
  Tensor sY = make_tensor(make_smem_ptr(shm_yt), SLayoutY{});
  Tensor sYT = make_tensor(make_smem_ptr(shm_yt), SLayoutYT{});

  // Global tensors
  TmaY tma_y;
  auto gB = tma_b.get_tma_tensor(make_shape(n, k, num_group));
  int Ksf_tiles_runtime = (k / kSfVec + 3) / 4;
  auto gSFB = tma_sfb.get_tma_tensor(
      make_shape(Int<32>{}, Int<16>{}, num_group * (n / 128), Ksf_tiles_runtime));
  auto gY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));

  // SFA row-major params
  const int K_sf = k / kSfVec;

  // TMA partition (B, SFB only — A via gather4, SFA via cp.async)
  auto btma_b = tma_b.get_slice(0);
  auto btma_sfb = tma_sfb.get_slice(0);
  auto tBg = btma_b.partition_S(gB);
  auto tBs = btma_b.partition_D(sB);
  auto tSFBg = btma_sfb.partition_S(gSFB);
  auto tSFBs = btma_sfb.partition_D(sSFB);

  int ntile_k = size<2>(tBg);

  // TiledMma partition
  TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(0);
  auto tBr = cta_mma.make_fragment_A(cta_mma.partition_A(sB));
  auto tAr = cta_mma.make_fragment_B(cta_mma.partition_B(sA));
  auto tCt = cta_mma.make_fragment_C(cta_mma.partition_C(gY));

  // SF UTCCP descriptors
  __shared__ uint64_t sfb_desc[kStage];
  __shared__ uint64_t sfa_low_desc[kStage];
  __shared__ uint64_t sfa_high_desc[kStage];
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
      next_power2(std::integral_constant<int, kStageTile * kTileM + kScaleColsPerTile>{});

  __shared__ uint32_t s_tmem_base;
  __shared__ uint64_t ab_readable[kStage];
  __shared__ uint64_t ab_writable[kStage];
  __shared__ uint64_t tmem_readable[kStageTile];
  __shared__ uint64_t tmem_writable[kStageTile];

  __shared__ uint64_t task_readable[kStageTask];
  __shared__ uint64_t task_writable[kStageTask];
  __shared__ int task_shm[kStageTask][4];

  constexpr int kSfaCpAsyncThreads = 32;  // W1 (cp.async SFA)

  if (iwarp == 0 && elected) {
#pragma unroll
    for (int i = 0; i < kStage; i++) {
      // ab_readable: 32 cp.async noinc (SFA from W1) + 1 TMA expect_tx (B+SFB+A gather4) = 33
      initialize_barrier(ab_readable[i], kSfaCpAsyncThreads + 1);
      initialize_barrier(ab_writable[i], 1);
    }
#pragma unroll
    for (int i = 0; i < kStageTile; i++) {
      initialize_barrier(tmem_readable[i], 1);
      initialize_barrier(tmem_writable[i], 1);
    }

    constexpr int kTmaThreads = 1;    // W0 (elected)
    constexpr int kMmaThreads = 32;   // W2
    constexpr int kEpiThreads = 128;  // W4-W7
#pragma unroll
    for (int i = 0; i < kStageTask; i++) {
      initialize_barrier(task_readable[i], 1);
      initialize_barrier(task_writable[i],
                         kSfaCpAsyncThreads + kMmaThreads + kTmaThreads + kEpiThreads);
    }
    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 2) {
    tmem_allocator.allocate(kTmemCols, &s_tmem_base);
    tmem_allocator.release_allocation_lock();
  }

  // Load shm_tiles, shm_seqlens, shm_cu_seqlens
  for (int i = idx; i < num_group; i += blockDim.x) {
    shm_tiles[i] = tiles_ptr[i];
    shm_seqlens[i] = seqlens_ptr[i];
  }
  for (int i = idx; i <= num_group; i += blockDim.x) {
    shm_cu_tiles[i] = cu_tiles_ptr[i];
    shm_cu_seqlens[i] = cu_seqlens_ptr[i];
  }
  // if constexpr (GemmConfig::kNeedPreZeroB) {
  //   for (int i = idx; i < static_cast<int>(cosize(SLayoutB{})); i += blockDim.x) {
  //     reinterpret_cast<uint8_t *>(shm_b)[i] = 0;
  //   }
  // }
  __syncthreads();

  tCt.data() = make_tmem_ptr<float>(s_tmem_base);

  using TinB = typename GemmConfig::TinB;
  constexpr uint32_t kBytesA = (cosize(SLayoutA{}) / kStage) * sizeof(Tin);
  constexpr uint32_t kBytesB = (cosize(SLayoutB{}) / kStage) * cute::sizeof_bits_v<TinB> / 8;
  constexpr uint32_t kExpectedBytesAB = kBytesA + kBytesB + 32 * 16;  // A + B + SFB

  if (iwarp == 0 && elected) {
    // W0: TMA warp — TMA B + TMA SFB only (gather4 moved to W1)
    int phase_ab = 1;
    int istage_ab = 0;

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
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(ab_writable[istage_ab], phase_ab);

          // TMA B
          copy(tma_b.with(ab_readable[istage_ab], 0, TMA::CacheHintSm90::EVICT_FIRST),
               tBg(_, itile_n, itile_k, igroup), tBs(_, 0, 0, istage_ab));
          // TMA SFB
          int sfb_flat_n = igroup * (n / 128) + itile_n;
          copy(tma_sfb.with(ab_readable[istage_ab], 0, TMA::CacheHintSm90::EVICT_FIRST),
               tSFBg(_, 0, 0, sfb_flat_n, itile_k), tSFBs(_, 0, 0, istage_ab));

          set_barrier_transaction_bytes(ab_readable[istage_ab], kExpectedBytesAB);

          istage_ab++;
          if (istage_ab == kStage) {
            istage_ab = 0;
            phase_ab ^= 1;
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
    // W1 (32 threads): gather4 A (elected) + cp.async SFA (all threads)
    int local_idx = idx - 32;

    int phase_ab = 1;
    int istage_ab = 0;

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
        int cu_seq = shm_cu_seqlens[igroup];
        int tile_base_row = cu_seq + itile_m * kTileM;
        int rows_in_group = shm_seqlens[igroup];
        int remaining_m = rows_in_group - itile_m * kTileM;

        constexpr int kSfaRowsPerThread = (kTileM + kSfaCpAsyncThreads - 1) / kSfaCpAsyncThreads;
        const uint8_t *sfa_src_base[kSfaRowsPerThread];
        int sfa_smem_off[kSfaRowsPerThread];
        int sfa_valid_size[kSfaRowsPerThread];
        if (local_idx < kTileM) {
#pragma unroll
          for (int i = 0; i < kSfaRowsPerThread; i++) {
            int r = local_idx + i * kSfaCpAsyncThreads;
            bool valid = (r < remaining_m);
            int abs_row = tile_base_row + r;
            int src_row = valid ? x_row_map_ptr[abs_row] : 0;
            sfa_src_base[i] = sfx_ptr + src_row * K_sf;
            int block_32x16 = r / 128;
            int block_32x4 = (r / 32) % 4;
            int row_in_block = r % 32;
            sfa_smem_off[i] = block_32x16 * (32 * 16) + row_in_block * 16 + block_32x4 * 4;
            sfa_valid_size[i] = valid ? 4 : 0;
          }
        }

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(ab_writable[istage_ab], phase_ab);

          // Gather4 A: elected thread issues TMA gather4 for kTileM rows
          if (elected) {
            int col = itile_k * kTileK;
            auto *a_smem_stage =
                reinterpret_cast<uint8_t *>(shm_a) + istage_ab * kTileM * kTileK * sizeof(Tin);
            for (int r = 0; r < kTileM; r += 4) {
              int base_row = cu_seq + itile_m * kTileM + r;
              int r0 = (r + 0 < remaining_m) ? x_row_map_ptr[base_row + 0] : 0;
              int r1 = (r + 1 < remaining_m) ? x_row_map_ptr[base_row + 1] : 0;
              int r2 = (r + 2 < remaining_m) ? x_row_map_ptr[base_row + 2] : 0;
              int r3 = (r + 3 < remaining_m) ? x_row_map_ptr[base_row + 3] : 0;
              void *dst = a_smem_stage + r * kTileK * sizeof(Tin);
              tma_gather4_load(dst, &tma_a_gather4, col, r0, r1, r2, r3, ab_readable[istage_ab]);
            }
          }

          // cp.async SFA: all 32 threads
          if (local_idx < kTileM) {
            auto *sfa_stage = reinterpret_cast<uint8_t *>(shm_sfa) + istage_ab * kSfxRows * 16;
#pragma unroll
            for (int i = 0; i < kSfaRowsPerThread; i++) {
              const void *gmem_src = sfa_src_base[i] + itile_k * 4;
              void *smem_dst = sfa_stage + sfa_smem_off[i];
              cp_async_4b_g4(smem_dst, gmem_src, sfa_valid_size[i]);
            }
          }

          cpasync_barrier_arrive_noinc(reinterpret_cast<uint64_t *>(&ab_readable[istage_ab]));

          istage_ab++;
          if (istage_ab == kStage) {
            istage_ab = 0;
            phase_ab ^= 1;
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
    // W2: MMA warp (inlines UTCCP)
    int phase_ab = 0;
    int istage_ab = 0;

    int phase_tile = 1;
    int istage_tile = 0;

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
        uint32_t c_base = s_tmem_base + istage_tile * kTileM;
        tCt.data() = make_tmem_ptr<float>(c_base);
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

        wait_barrier(tmem_writable[istage_tile], phase_tile);
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(ab_readable[istage_ab], phase_ab);

          if (elected) {
            SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(sfb_desc[istage_ab], sf_base);
            SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(sfa_low_desc[istage_ab],
                                                                    sf_base + 4);
            if constexpr (!kSmallTM) {
              SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(sfa_high_desc[istage_ab],
                                                                      sf_base + 4 + 4);
            }
          }

#pragma unroll
          for (uint32_t ik = 0; ik < size<2>(tBr); ik++) {
            uint32_t sfb_addr_ki = (sf_base & 0x3FFFFFFFu) | (ik << 30);
            uint32_t sfa_addr_ki = ((sf_base + 4) & 0x3FFFFFFFu) | (ik << 30);
            auto sfb_ki = make_tensor(make_tmem_ptr<cutlass::float_ue8m0_t>(sfb_addr_ki),
                                      make_layout(make_shape(1)));
            auto sfa_ki = make_tensor(make_tmem_ptr<cutlass::float_ue8m0_t>(sfa_addr_ki),
                                      make_layout(make_shape(1)));
            cute::gemm(tiled_mma.with(tiled_mma.accumulate_, sfb_ki, sfa_ki),
                       tBr(_, _, ik, istage_ab), tAr(_, _, ik, istage_ab), tCt);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }

          cutlass::arch::umma_arrive(&ab_writable[istage_ab]);

          istage_ab++;
          if (istage_ab == kStage) {
            phase_ab ^= 1;
            istage_ab = 0;
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
    // W4-W7: Epilogue (TMEM -> Regs -> Cast -> SMEM -> TMA Store)
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
  if (iwarp == 2) {
    tmem_allocator.free(s_tmem_base, kTmemCols);
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

// ============================================================================
// 2SM gather4 kernel
// ============================================================================
template <typename GemmConfig, typename TmaB, typename TmaY, typename TmaSFB>
__global__ __launch_bounds__(256, 1) void group_gemm_2sm_gather4_mxfp8_kernel(
    const __grid_constant__ CUtensorMap tma_a_gather4, const __grid_constant__ TmaB tma_b,
    const __grid_constant__ TmaSFB tma_sfb, cute::TmaDescriptor *td_ay,
    const uint8_t *__restrict__ sfx_ptr, const int *__restrict__ x_row_map_ptr, int x_num_rows,
    int *seqlens_ptr, int *cu_seqlens_ptr, int *tiles_ptr, int *cu_tiles_ptr, int num_group, int m,
    int n, int k, cutlass::FastDivmod flat_divider) {
  cudaGridDependencySynchronize();
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
  constexpr int kCtaTileM = GemmConfig::kCtaTileM;
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

  static_assert(kMmaSM == 2, "2SM gather4 kernel requires kMmaSM == 2");

  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int elected = cute::elect_one_sync();

  // 2SM cluster setup
  int block_rank = cute::block_rank_in_cluster();
  int sm_idx = block_rank;
  bool elected_cta = (sm_idx == 0);
  int cluster_id = blockIdx.x / kMmaSM;
  int num_clusters = gridDim.x / kMmaSM;
  int iblock = cluster_id;
  uint16_t mcast_mask = 0x3;

  // Smem
  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_a = reinterpret_cast<Tin *>(shm_data);
  auto *shm_b = shm_a + cosize(SLayoutA{});
  auto *shm_sfa = reinterpret_cast<Tsf *>(shm_b + cosize(SLayoutB{}));
  auto *shm_sfb = shm_sfa + cosize(SLayoutSFA{});
  auto *shm_yt = reinterpret_cast<Tout *>(shm_sfb + cosize(SLayoutSFB{}));
  int *shm_tiles = reinterpret_cast<int *>(shm_yt + cosize(SLayoutYT{}));
  int *shm_cu_tiles = shm_tiles + (num_group + 1);
  int *shm_seqlens = shm_cu_tiles + (num_group + 1);
  int *shm_cu_seqlens = shm_seqlens + (num_group + 1);

  Tensor sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
  Tensor sSFA = make_tensor(make_smem_ptr(shm_sfa), SLayoutSFA{});
  Tensor sSFB = make_tensor(make_smem_ptr(shm_sfb), SLayoutSFB{});
  Tensor sY = make_tensor(make_smem_ptr(shm_yt), SLayoutY{});
  Tensor sYT = make_tensor(make_smem_ptr(shm_yt), SLayoutYT{});

  // Global tensors
  TmaY tma_y;
  auto gB = tma_b.get_tma_tensor(make_shape(n, k, num_group));
  int Ksf_tiles_runtime = (k / kSfVec + 3) / 4;
  auto gSFB = tma_sfb.get_tma_tensor(
      make_shape(Int<64>{}, Int<16>{}, num_group * (n / kTileN), Ksf_tiles_runtime));
  auto gY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));

  // SFA row-major params
  const int K_sf = k / kSfVec;

  // TMA partition (B/SFB: cooperative via sm_idx)
  auto btma_gb = tma_b.get_slice(sm_idx);
  auto btma_sb = tma_b.get_slice(0);
  auto btma_gsfb = tma_sfb.get_slice(sm_idx);
  auto btma_ssfb = tma_sfb.get_slice(0);
  auto tBg = btma_gb.partition_S(gB);
  auto tBs = btma_sb.partition_D(sB);
  auto tSFBg = btma_gsfb.partition_S(gSFB);
  auto tSFBs = btma_ssfb.partition_D(sSFB);

  int ntile_k = size<2>(tBg);

  // TiledMma partition (2SM)
  TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(sm_idx);
  auto tBr = cta_mma.make_fragment_A(cta_mma.partition_A(sB));
  auto tAr = cta_mma.make_fragment_B(cta_mma.partition_B(sA));
  auto tCt = cta_mma.make_fragment_C(cta_mma.partition_C(gY));

  // SF UTCCP descriptors
  uint64_t sfb_desc[kStage];
  uint64_t sfa_low_desc[kStage];
  uint64_t sfa_high_desc[kStage];
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
      next_power2(std::integral_constant<int, kStageTile * kTileM + kScaleColsPerTile>{});

  __shared__ uint32_t s_tmem_base;
  __shared__ uint64_t ab_readable[kStage];
  __shared__ uint64_t ab_writable[kStage];
  __shared__ uint64_t tmem_readable[kStageTile];
  __shared__ uint64_t tmem_writable[kStageTile];

  __shared__ uint64_t task_readable[kStageTask];
  __shared__ uint64_t task_writable[kStageTask];
  __shared__ int task_shm[kStageTask][4];

  constexpr int kSfaCpAsyncThreads = 32;  // W1 (cp.async SFA)

  if (iwarp == 0 && elected) {
#pragma unroll
    for (int i = 0; i < kStage; i++) {
      // ab_readable: 32 cp.async noinc (SFA from W1) + 2 TMA tx (B+SFB cooperative + A gather4)
      initialize_barrier(ab_readable[i], kSfaCpAsyncThreads + 2);
      initialize_barrier(ab_writable[i], 1);
    }
#pragma unroll
    for (int i = 0; i < kStageTile; i++) {
      initialize_barrier(tmem_readable[i], 1);
      initialize_barrier(tmem_writable[i], 2);
    }
    constexpr int kMmaThreads = 32;
    constexpr int kEpiThreads = 128;
    constexpr int kTmaThreads = 1;
#pragma unroll
    for (int i = 0; i < kStageTask; i++) {
      initialize_barrier(task_readable[i], 1);
      initialize_barrier(task_writable[i],
                         kSfaCpAsyncThreads + kMmaThreads + kTmaThreads + kEpiThreads);
    }
    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 2) {
    tmem_allocator.allocate(kTmemCols, &s_tmem_base);
    tmem_allocator.release_allocation_lock();
  }

  // Load shm_tiles, shm_seqlens, shm_cu_seqlens
  for (int i = idx; i < num_group; i += blockDim.x) {
    shm_tiles[i] = tiles_ptr[i];
    shm_seqlens[i] = seqlens_ptr[i];
  }
  for (int i = idx; i <= num_group; i += blockDim.x) {
    shm_cu_tiles[i] = cu_tiles_ptr[i];
    shm_cu_seqlens[i] = cu_seqlens_ptr[i];
  }
  if constexpr (GemmConfig::kNeedPreZeroB) {
    for (int i = idx; i < static_cast<int>(cosize(SLayoutB{})); i += blockDim.x) {
      reinterpret_cast<uint8_t *>(shm_b)[i] = 0;
    }
  }

  uint32_t cluster_mbar[kStage];
#pragma unroll
  for (int i = 0; i < kStage; i++) {
    cluster_mbar[i] = get_cluster_mbar_addr(ab_readable[i], 0);
  }

  cluster_relaxed_sync();

  tCt.data() = make_tmem_ptr<float>(s_tmem_base);

  using TinB = typename GemmConfig::TinB;
  constexpr uint32_t kBytesA = (cosize(SLayoutA{}) / kStage) * sizeof(Tin);
  constexpr uint32_t kBytesB = (cosize(SLayoutB{}) / kStage) * cute::sizeof_bits_v<TinB> / 8;
  // Transaction bytes: cluster-aggregate of A (gather4) + B (2SM TMA) + SFB (2SM TMA).
  // Both CTAs' gather4 bytes route to elected CTA's cluster mbar via cta_group::2.
  constexpr uint32_t kExpectedBytesAB = kMmaSM * kBytesA + kMmaSM * kBytesB + kMmaSM * (32 * 16);

  if (iwarp == 0 && elected) {
    // W0: TMA warp — cooperative TMA B + SFB
    int phase_ab = 1;
    int istage_ab = 0;

    int phase_task = 0;
    int istage_task = 0;

    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    get_next_tile_horizon_mxfp8<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                        sum_tile_m, flat_divider);

    while (true) {
      if (igroup >= 0) {
        int cu_seq = shm_cu_seqlens[igroup];
        int rows_in_group = shm_seqlens[igroup];
        int remaining_m = rows_in_group - itile_m * kTileM;

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(ab_writable[istage_ab], phase_ab);

          // Cooperative TMA B
          copy(tma_b.with(ab_readable[istage_ab], 0, TMA::CacheHintSm100::EVICT_FIRST),
               tBg(_, itile_n, itile_k, igroup), tBs(_, 0, 0, istage_ab));
          // Cooperative TMA SFB
          int sfb_flat_n = igroup * (n / kTileN) + itile_n;
          copy(tma_sfb.with(ab_readable[istage_ab], 0, TMA::CacheHintSm100::EVICT_FIRST),
               tSFBg(_, 0, 0, sfb_flat_n, itile_k), tSFBs(_, 0, 0, istage_ab));

          // Gather4 A: each CTA loads its own kCtaTileM rows via cta_group::2 gather4.
          int col = itile_k * kTileK;
          auto *a_smem_stage =
              reinterpret_cast<uint8_t *>(shm_a) + istage_ab * kCtaTileM * kTileK * sizeof(Tin);
          int cta_remaining_m = remaining_m - sm_idx * kCtaTileM;
          // uint32_t cluster_mbar = get_cluster_mbar_addr(ab_readable[istage_ab], 0);
#pragma unroll
          for (int r = 0; r < kCtaTileM; r += 4) {
            int base_row = cu_seq + itile_m * kTileM + sm_idx * kCtaTileM + r;
            int r0, r1, r2, r3;
            r0 = (r + 0 < cta_remaining_m) ? x_row_map_ptr[base_row + 0] : 0;
            r1 = (r + 1 < cta_remaining_m) ? x_row_map_ptr[base_row + 1] : 0;
            r2 = (r + 2 < cta_remaining_m) ? x_row_map_ptr[base_row + 2] : 0;
            r3 = (r + 3 < cta_remaining_m) ? x_row_map_ptr[base_row + 3] : 0;
            void *dst = a_smem_stage + r * kTileK * sizeof(Tin);
            tma_gather4_load_2sm(dst, &tma_a_gather4, cluster_mbar[istage_ab], col, r0, r1, r2, r3);
          }

          if (elected_cta) {
            set_barrier_transaction_bytes(ab_readable[istage_ab], kExpectedBytesAB);
          } else {
            arrive_cluster_barrier(ab_readable[istage_ab], 0);
          }

          istage_ab++;
          if (istage_ab == kStage) {
            istage_ab = 0;
            phase_ab ^= 1;
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
    // W1 (32 threads): cp.async SFA inline prepack
    int local_idx = idx - 32;

    int phase_ab = 1;
    int istage_ab = 0;

    int phase_task = 0;
    int istage_task = 0;

    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    get_next_tile_horizon_mxfp8<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                        sum_tile_m, flat_divider);

    while (true) {
      if (igroup >= 0) {
        int cu_seq = shm_cu_seqlens[igroup];
        int tile_base_row = cu_seq + itile_m * kTileM;
        int rows_in_group = shm_seqlens[igroup];
        int remaining_m = rows_in_group - itile_m * kTileM;

        constexpr int kSfaRowsPerThread = (kTileM + kSfaCpAsyncThreads - 1) / kSfaCpAsyncThreads;
        const uint8_t *sfa_src_base[kSfaRowsPerThread];
        int sfa_smem_off[kSfaRowsPerThread];
        int sfa_valid_size[kSfaRowsPerThread];
        if (local_idx < kTileM) {
#pragma unroll
          for (int i = 0; i < kSfaRowsPerThread; i++) {
            int r = local_idx + i * kSfaCpAsyncThreads;
            bool valid = (r < remaining_m);
            int abs_row = tile_base_row + r;
            int src_row;
            src_row = valid ? x_row_map_ptr[abs_row] : 0;
            sfa_src_base[i] = sfx_ptr + src_row * K_sf;
            int block_32x16 = r / 128;
            int block_32x4 = (r / 32) % 4;
            int row_in_block = r % 32;
            sfa_smem_off[i] = block_32x16 * (32 * 16) + row_in_block * 16 + block_32x4 * 4;
            sfa_valid_size[i] = valid ? 4 : 0;
          }
        }

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(ab_writable[istage_ab], phase_ab);
          // cp.async SFA: all 32 threads
          if (local_idx < kTileM) {
            auto *sfa_stage = reinterpret_cast<uint8_t *>(shm_sfa) + istage_ab * kSfxRows * 16;
#pragma unroll
            for (int i = 0; i < kSfaRowsPerThread; i++) {
              const void *gmem_src = sfa_src_base[i] + itile_k * 4;
              void *smem_dst = sfa_stage + sfa_smem_off[i];
              cp_async_4b_g4(smem_dst, gmem_src, sfa_valid_size[i]);
            }
          }

          cpasync_barrier_arrive_noinc(reinterpret_cast<uint64_t *>(&ab_readable[istage_ab]));

          istage_ab++;
          if (istage_ab == kStage) {
            istage_ab = 0;
            phase_ab ^= 1;
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
    // W2: MMA warp (elected_cta only computes, inlines UTCCP)
    int phase_ab = 0;
    int istage_ab = 0;

    int phase_tile = 1;
    int istage_tile = 0;

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
        uint32_t c_base = s_tmem_base + istage_tile * kTileM;
        tCt.data() = make_tmem_ptr<float>(c_base);
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

        wait_barrier(tmem_writable[istage_tile], phase_tile);
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(ab_readable[istage_ab], phase_ab);

          if (elected) {
            SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_2cta::copy(sfb_desc[istage_ab], sf_base);
            SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_2cta::copy(sfa_low_desc[istage_ab],
                                                                    sf_base + 4);
            if constexpr (!kSmallTM) {
              SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_2cta::copy(sfa_high_desc[istage_ab],
                                                                      sf_base + 4 + 4);
            }
          }

#pragma unroll
          for (uint32_t ik = 0; ik < size<2>(tBr); ik++) {
            uint32_t sfb_addr_ki = (sf_base & 0x3FFFFFFFu) | (ik << 30);
            uint32_t sfa_addr_ki = ((sf_base + 4) & 0x3FFFFFFFu) | (ik << 30);
            auto sfb_ki = make_tensor(make_tmem_ptr<cutlass::float_ue8m0_t>(sfb_addr_ki),
                                      make_layout(make_shape(1)));
            auto sfa_ki = make_tensor(make_tmem_ptr<cutlass::float_ue8m0_t>(sfa_addr_ki),
                                      make_layout(make_shape(1)));
            cute::gemm(tiled_mma.with(tiled_mma.accumulate_, sfb_ki, sfa_ki),
                       tBr(_, _, ik, istage_ab), tAr(_, _, ik, istage_ab), tCt);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }

          cutlass::arch::umma_arrive_multicast_2x1SM(&ab_writable[istage_ab], mcast_mask);

          istage_ab++;
          if (istage_ab == kStage) {
            phase_ab ^= 1;
            istage_ab = 0;
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
    // W3: FindTask (cluster-aware)
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
  } else if (idx >= 128 && idx < 256) {
    // W4-W7: Epilogue (both CTAs store their own N-half via TMA Store)
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
          copy(tiled_copy_t2r, tCt4r(_, _, iepi), tCr4t);
          cutlass::arch::fence_view_async_tmem_load();

          if (iepi == nepi_tile - 1) {
            if (is_leader) {
              arrive_cluster_barrier(tmem_writable[istage_tile], 0);
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
            int m_tile_idx = itile_m * (kTileM / kEpiTileM) + iepi;
            int n_tile_idx = itile_n * kMmaSM + sm_idx;
            cute::copy(tma_y.with(td_y), tDs(_, 0, 0, istage_tma), tDg(_, n_tile_idx, m_tile_idx));
            tma_store_arrive();
          }

          istage_tma = (istage_tma + 1) % kStageTMA;
        }

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

  cluster_relaxed_sync();
  if (iwarp == 2) {
    tmem_allocator.free(s_tmem_base, kTmemCols);
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace kernels
}  // namespace group_gemm
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM100_MXFP8_GROUP_GEMM_GATHER4_MXFP8_CUH_
