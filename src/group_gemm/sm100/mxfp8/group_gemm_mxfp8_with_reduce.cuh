// Copyright 2026 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM100_MXFP8_GROUP_GEMM_MXFP8_WITH_REDUCE_CUH_
#define SRC_GROUP_GEMM_SM100_MXFP8_GROUP_GEMM_MXFP8_WITH_REDUCE_CUH_

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
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/float8.h"
#include "cutlass/numeric_types.h"
#include "src/group_gemm/sm100/mxfp8/config.h"
#include "src/group_gemm/sm100/mxfp8/group_gemm_mxfp8.cuh"
#include "src/group_gemm/sm100/mxfp8/utils.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace group_gemm {

using namespace cute;  // NOLINT

namespace kernels {

// cp.async 4B with zero-fill: copies src_size bytes, zero-fills (4 - src_size) bytes.
// When src_size=0, fills 4 zeros without accessing gmem_src.
__device__ __forceinline__ void cp_async_4b(void *smem_dst, const void *gmem_src, int src_size) {
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem_addr), "l"(gmem_src),
               "r"(src_size));
}

__device__ __forceinline__ void atomic_reduce_bf16_mxfp8(const void *gmem_ptr, void *smem_ptr,
                                                         int bytes) {
  vec_t<uint32_t, 4> src = load<uint32_t, 4>(smem_ptr);

  asm volatile("red.global.v4.bf16x2.add.noftz [%0], {%1, %2, %3, %4};\n" ::"l"(gmem_ptr),
               "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]));
}

template <typename GemmConfig, typename TmaA, typename TmaB, typename TmaSFB, bool kUsePDL = false>
__global__ __launch_bounds__(256, 1) void group_gemm_1sm_mxfp8_with_reduce_kernel(
    const __grid_constant__ TmaB tma_b, const __grid_constant__ TmaSFB tma_sfb,
    cute::TmaDescriptor *td_ay, typename GemmConfig::Tout *y_ptr,
    const uint8_t *__restrict__ sfx_ptr, int *seqlens_ptr, int *cu_seqlens_ptr, int *tiles_ptr,
    int *cu_tiles_ptr, int *x_row_map_ptr, float *topk_scale_row_map_ptr, int num_group, int m,
    int n, int k, cutlass::FastDivmod flat_divider) {
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
  constexpr int kCtaTileN = GemmConfig::kCtaTileN;

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

  // Shared memory for topk_scale caching
  // Placed after the dynamic shm_tiles/shm_cu_tiles region
  // We use a static __shared__ declaration for it
  __shared__ float reduce_topk_scale[kTileM];

  // Global tensors
  TmaA tma_a;
  auto gA = tma_a.get_tma_tensor(make_shape(m, k));
  auto gB = tma_b.get_tma_tensor(make_shape(n, k, num_group));

  int k_sf_tiles = (k / kSfVec + 3) / 4;
  auto gSFB =
      tma_sfb.get_tma_tensor(make_shape(Int<32>{}, Int<16>{}, num_group * (n / 128), k_sf_tiles));
  auto gY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));

  // SFA row-major params (K_sf = number of scale values per row)
  const int K_sf = k / kSfVec;

  // TMA partition (A, B, SFB — SFA is loaded via cp.async inline prepack by W1)
  auto btma_a = tma_a.get_slice(0);
  auto btma_b = tma_b.get_slice(0);
  auto btma_sfb = tma_sfb.get_slice(0);
  auto tAg = btma_a.partition_S(gA);
  auto tAs = btma_a.partition_D(sA);
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
  // SF TMEM needs only ONE slot: UTCCP (tcgen05.cp) and MMA (tcgen05.mma) run on
  // the same tensor-core unit in the same warp, so they are strictly serialized.
  constexpr int kTmemCols =
      next_power2(std::integral_constant<int, kStageTile * kTileM + kScaleColsPerTile>{});

  __shared__ uint32_t s_tmem_base;
  __shared__ uint64_t ab_readable[kStage];  // A+B+SFA+SFB SMEM data ready
  __shared__ uint64_t ab_writable[kStage];  // A+B+SFA+SFB SMEM reusable
  __shared__ uint64_t tmem_readable[kStageTile];
  __shared__ uint64_t tmem_writable[kStageTile];

  __shared__ uint64_t task_readable[kStageTask];
  __shared__ uint64_t task_writable[kStageTask];
  __shared__ int task_shm[kStageTask][4];

  constexpr int kSfaCpAsyncThreads = 32;  // W1 (cp.async SFA prepack)

  if (iwarp == 0 && elected) {
#pragma unroll
    for (int i = 0; i < kStage; i++) {
      // ab_readable: 32 noinc arrives (SFA, W1) + 1 TMA expect_tx (A+B+SFB, W0) = 33.
      initialize_barrier(ab_readable[i], kSfaCpAsyncThreads + 1);
      initialize_barrier(ab_writable[i], 1);
    }
#pragma unroll
    for (int i = 0; i < kStageTile; i++) {
      initialize_barrier(tmem_readable[i], 1);
      initialize_barrier(tmem_writable[i], 1);
    }

    constexpr int kMmaThreads = 32;   // W2
    constexpr int kEpiThreads = 128;  // W4-W7
    constexpr int kTmaThreads = 1;    // W0 (elected)
#pragma unroll
    for (int i = 0; i < kStageTask; i++) {
      initialize_barrier(task_readable[i], 1);
      initialize_barrier(task_writable[i],
                         kSfaCpAsyncThreads + kMmaThreads + kTmaThreads + kEpiThreads);
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

  // fp4 weight: TMA scatter-unpack does not fill every byte of SMEM B, so pre-zero it.
  if constexpr (GemmConfig::kNeedPreZeroB) {
    for (int i = idx; i < static_cast<int>(cosize(SLayoutB{})); i += blockDim.x) {
      reinterpret_cast<uint8_t *>(shm_b)[i] = 0;
    }
  }
  __syncthreads();

  tCt.data() = make_tmem_ptr<float>(s_tmem_base);

  constexpr uint32_t kExpectedBytesAB =
      GemmConfig::kExpectedBytesA + GemmConfig::kExpectedBytesB + 32 * 16;  // A + B + SFB (no SFA)

  if (iwarp == 0 && elected) {
    // TMA warp: loads A, B, SFB (SFA is handled by W1 via cp.async)
    int phase_ab = 1;  // ab_writable phase
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
        auto *td_a = td_ay + igroup * 2;
        prefetch_tma_descriptor(td_a);

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(ab_writable[istage_ab], phase_ab);
          // load A + B + SFB (same barrier)
          copy(tma_b.with(ab_readable[istage_ab], 0, TMA::CacheHintSm90::EVICT_FIRST),
               tBg(_, itile_n, itile_k, igroup), tBs(_, 0, 0, istage_ab));
          copy(tma_a.with(td_a, ab_readable[istage_ab], 0, TMA::CacheHintSm90::EVICT_FIRST),
               tAg(_, itile_m, itile_k), tAs(_, 0, 0, istage_ab));
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
    // W1 (32 threads): cp.async SFA inline prepack
    int local_idx = idx - 32;

    int phase_ab = 1;  // ab_writable phase
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
        int cu_seq = cu_seqlens_ptr[igroup];
        int tile_base_row = cu_seq + itile_m * kTileM;
        int rows_in_group = seqlens_ptr[igroup];
        int remaining_m = rows_in_group - itile_m * kTileM;

        // Precompute SFA per-row info (invariant across k-tiles).
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
            int src_row = valid ? abs_row : 0;
            sfa_src_base[i] = sfx_ptr + src_row * K_sf;
            // Packed SMEM addressing: (32,16) block → (32,4) column → row within block
            int block_32x16 = r / 128;
            int block_32x4 = (r / 32) % 4;
            int row_in_block = r % 32;
            sfa_smem_off[i] = block_32x16 * (32 * 16) + row_in_block * 16 + block_32x4 * 4;
            sfa_valid_size[i] = valid ? 4 : 0;
          }
        }

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(ab_writable[istage_ab], phase_ab);

          // cp.async load SFA
          if (local_idx < kTileM) {
            auto *sfa_stage = reinterpret_cast<uint8_t *>(shm_sfa) + istage_ab * kSfxRows * 16;
#pragma unroll
            for (int i = 0; i < kSfaRowsPerThread; i++) {
              const void *gmem_src = sfa_src_base[i] + itile_k * 4;
              void *smem_dst = sfa_stage + sfa_smem_off[i];
              cp_async_4b(smem_dst, gmem_src, sfa_valid_size[i]);
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
    // MMA warp (inlines UTCCP: tcgen05.cp + tcgen05.mma on same warp → hardware serialized)
    int phase_ab = 0;  // ab_readable phase

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
        // TMEM for C
        uint32_t c_base = s_tmem_base + istage_tile * kTileM;
        tCt.data() = make_tmem_ptr<float>(c_base);
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

        wait_barrier(tmem_writable[istage_tile], phase_tile);
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          // Wait for A+B+SFA+SFB SMEM data
          wait_barrier(ab_readable[istage_ab], phase_ab);

          // Inline UTCCP: SF SMEM → TMEM (single slot, no per-stage offset)
          if (elected) {
            SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(sfb_desc[istage_ab], sf_base);
            SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(sfa_low_desc[istage_ab],
                                                                    sf_base + 4);
            if constexpr (!kSmallTM) {
              SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(sfa_high_desc[istage_ab],
                                                                      sf_base + 4 + 4);
            }
          }

          // MMA (immediately follows UTCCP — hardware serialized on same tensor core)
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

          // Release A+B+SFA+SFB SMEM (single arrive frees the stage for all producers)
          cutlass::arch::umma_arrive(&ab_writable[istage_ab]);

          istage_ab++;
          if (istage_ab == kStage) {
            istage_ab = 0;
            phase_ab ^= 1;
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
    // Find task
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
    // Epilogue warps (W4-W7): TMEM -> RMEM -> scale(topk) -> cast -> SMEM -> atomic reduce -> GMEM
    int epi_idx = idx - 128;
    bool is_leader = elected && (iwarp == 4);

    auto gY_local = make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                                make_shape(Int<kCtaTileN>{}, Int<kEpiTileM>{}),
                                make_stride(Int<kEpiTileM>{}, Int<1>{}));

    auto IY = make_identity_tensor(gY.shape());
    auto ISY = make_identity_tensor(gY_local.shape());
    auto epi_tiler = make_tile(Int<kCtaTileN>{}, Int<kEpiTileM>{});
    auto tCt_epi = zipped_divide(tCt, make_tile(epi_tiler));
    auto rC_epi = zipped_divide(gY, epi_tiler);
    auto sC_epi = zipped_divide(sY, epi_tiler);
    auto sYT_epi = zipped_divide(sYT, epi_tiler);
    auto IC_epi = zipped_divide(IY, epi_tiler);

    // TiledCopy TMEM -> RMEM
    using TmemLoadAtom = typename GemmConfig::TmemLoadAtom;
    auto tiled_copy_t2r = make_tmem_copy(TmemLoadAtom{}, tCt_epi(_, _0{}));
    auto thr_copy_t2r = tiled_copy_t2r.get_slice(epi_idx);
    auto tCt4r = thr_copy_t2r.partition_S(tCt_epi);
    auto tCr4t = make_tensor_like<float>(thr_copy_t2r.partition_D(rC_epi(_, 0)));
    auto tICr4t = thr_copy_t2r.partition_D(IC_epi(_, 0));

    // TiledCopy RMEM -> SMEM
    using StsmTAtom = typename GemmConfig::StsmTAtom;
    auto tiled_copy_r2s = make_tiled_copy_D(Copy_Atom<StsmTAtom, Tout>{}, tiled_copy_t2r);
    auto thr_copy_r2s = tiled_copy_r2s.get_slice(epi_idx);
    auto tCr4s = make_tensor_like<Tout>(thr_copy_r2s.partition_S(rC_epi(_, 0)));
    auto tCs4r = thr_copy_r2s.partition_D(sYT_epi);

    // S2G copy for computing output addresses (atomic reduce path)
    auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(n, 8192),
                         make_stride(Int<1>{}, n));
    auto gYY = local_tile(Y, make_tile(Int<kCtaTileN>{}, Int<kEpiTileM>{}), make_coord(_, _));

    using s2g_copy_op = UniversalCopy<cute::uint128_t>;
    using s2g_copy_traits = Copy_Traits<s2g_copy_op>;
    using s2g_copy_atom = Copy_Atom<s2g_copy_traits, Tout>;
    using S2GCopy = decltype(make_tiled_copy(
        s2g_copy_atom{},
        make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<1>{}, Int<16>{})),
        make_layout(make_shape(Int<8>{}, Int<1>{}))));
    S2GCopy s2g_tiled_copy;
    auto s2g_thr_copy = s2g_tiled_copy.get_slice(epi_idx);
    auto tYg = s2g_thr_copy.partition_D(gYY);
    auto tYs = s2g_thr_copy.partition_S(sYT);
    auto tIY = s2g_thr_copy.partition_S(ISY);

    auto nepi_tile = size<2>(tCt4r);

    // epi warpgroup state
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

        constexpr int kCopyM = size<2>(tYs);
        int cu_seq = cu_seqlens_ptr[igroup];
        int tile_base_row = cu_seq + itile_m * kTileM;
        int remaining_m = seqlens_ptr[igroup] - itile_m * kTileM;

        // Cooperative load of topk_scale into shared memory
        float *topk_scale_row_map_group = topk_scale_row_map_ptr + tile_base_row;
        for (int i = epi_idx; i < kTileM; i += 128) {
          if (i < remaining_m) {
            reduce_topk_scale[i] = topk_scale_row_map_group[i];
          }
        }

        // Precompute row_offsets and row_valid
        int row_offsets[nepi_tile][kCopyM];
        bool row_valid[nepi_tile][kCopyM];

#pragma unroll
        for (int iepi = 0; iepi < nepi_tile; iepi++) {
#pragma unroll
          for (int ir = 0; ir < kCopyM; ir++) {
            int ir_in_epi = get<1>(tIY(0, 0, ir));
            int ir_in_tile = iepi * kEpiTileM + ir_in_epi;
            row_valid[iepi][ir] = (ir_in_tile < remaining_m);
            int abs_row = tile_base_row + ir_in_tile;
            int src_row = x_row_map_ptr[abs_row];
            row_offsets[iepi][ir] = (src_row - ir_in_epi) * n;
          }
        }

        wait_barrier(tmem_readable[istage_tile], phase_tile);
        cutlass::arch::NamedBarrier::sync(128, 0);  // sync after topk_scale load

#pragma unroll
        for (int iepi = 0; iepi < nepi_tile; iepi++) {
          // TMEM -> RMEM
          copy(tiled_copy_t2r, tCt4r(_, _, iepi), tCr4t);
          cutlass::arch::fence_view_async_tmem_load();

          auto tCr_fp2 = recast<float2>(tCr4t);
          auto tCr_bf162 = recast<__nv_bfloat162>(tCr4s);

          // cast (with per-row topk_scale)
#pragma unroll
          for (int i = 0; i < cute::size(tCr_bf162); i++) {
            int irow = iepi * kEpiTileM + get<1>(tICr4t(2 * i));
            float oscale0 = reduce_topk_scale[irow];
            float oscale1 = reduce_topk_scale[irow + 1];
            float2 v = tCr_fp2(i);
            v.x *= oscale0;
            v.y *= oscale1;
            tCr_bf162(i) = __float22bfloat162_rn(v);
          }

          // RMEM -> SMEM
          copy(tiled_copy_r2s, tCr4s, tCs4r(_, _, istage_tma));

          // SMEM -> GMEM (atomic reduce per row)
          cutlass::arch::NamedBarrier::sync(128, 0);

#pragma unroll
          for (int ir = 0; ir < kCopyM; ir++) {
            if (row_valid[iepi][ir]) {
              atomic_reduce_bf16_mxfp8(&tYg(0, 0, ir, itile_n, 0) + row_offsets[iepi][ir],
                                       &tYs(0, 0, ir, istage_tma), 16);
            }
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

template <typename GemmConfig, typename TmaA, typename TmaB, typename TmaSFB, bool kUsePDL = false>
__global__ __launch_bounds__(256, 1) void group_gemm_2sm_mxfp8_with_reduce_kernel(
    const __grid_constant__ TmaB tma_b, const __grid_constant__ TmaSFB tma_sfb,
    cute::TmaDescriptor *td_ay, typename GemmConfig::Tout *y_ptr,
    const uint8_t *__restrict__ sfx_ptr, int *seqlens_ptr, int *cu_seqlens_ptr, int *tiles_ptr,
    int *cu_tiles_ptr, int *x_row_map_ptr, float *topk_scale_row_map_ptr, int num_group, int m,
    int n, int k, cutlass::FastDivmod flat_divider) {
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

  // Shared memory for topk_scale caching
  __shared__ float reduce_topk_scale[kTileM];

  // Global tensors
  TmaA tma_a;
  auto gA = tma_a.get_tma_tensor(make_shape(m, k));
  auto gB = tma_b.get_tma_tensor(make_shape(n, k, num_group));
  int k_sf_tiles = (k / kSfVec + 3) / 4;
  auto gSFB = tma_sfb.get_tma_tensor(
      make_shape(Int<64>{}, Int<16>{}, num_group * (n / kTileN), k_sf_tiles));
  auto gY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));

  // SFA row-major params (K_sf = number of scale values per row)
  const int K_sf = k / kSfVec;

  // TMA partition (sm_idx-aware for cooperative A/B/SFB; SFA loaded via cp.async by W1)
  auto btma_ga = tma_a.get_slice(sm_idx);
  auto btma_sa = tma_a.get_slice(0);
  auto btma_gb = tma_b.get_slice(sm_idx);
  auto btma_sb = tma_b.get_slice(0);
  auto btma_gsfb = tma_sfb.get_slice(sm_idx);
  auto btma_ssfb = tma_sfb.get_slice(0);

  auto tAg = btma_ga.partition_S(gA);
  auto tAs = btma_sa.partition_D(sA);
  auto tBg = btma_gb.partition_S(gB);
  auto tBs = btma_sb.partition_D(sB);
  auto tSFBg = btma_gsfb.partition_S(gSFB);
  auto tSFBs = btma_ssfb.partition_D(sSFB);

  int ntile_k = size<2>(tBg);

  // TiledMma partition
  TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(sm_idx);
  auto tBr = cta_mma.make_fragment_A(cta_mma.partition_A(sB));
  auto tAr = cta_mma.make_fragment_B(cta_mma.partition_B(sA));
  auto tCt = cta_mma.make_fragment_C(cta_mma.partition_C(gY));

  // SF UTCCP descriptors
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
  // SF TMEM needs only ONE slot (same reasoning as 1SM).
  constexpr int kTmemCols =
      next_power2(std::integral_constant<int, kStageTile * kTileM + kScaleColsPerTile>{});

  __shared__ uint32_t s_tmem_base;
  __shared__ uint64_t ab_readable[kStage];  // A+B+SFA+SFB SMEM data ready
  __shared__ uint64_t ab_writable[kStage];  // A+B+SFA+SFB SMEM reusable
  __shared__ uint64_t tmem_readable[kStageTile];
  __shared__ uint64_t tmem_writable[kStageTile];

  __shared__ int task_shm[kStageTask][4];
  __shared__ uint64_t task_readable[kStageTask];
  __shared__ uint64_t task_writable[kStageTask];

  constexpr int kSfaCpAsyncThreads2SM = 32;  // W1 (cp.async SFA prepack)

  if (iwarp == 0 && elected) {
#pragma unroll
    for (int i = 0; i < kStage; i++) {
      // ab_readable arrives on the LEADER CTA's barrier (CTA1 never waits its
      // own ab_readable — the MMA warp runs only on the leader CTA):
      //   1  expect_tx (A+B+SFB bytes, leader CTA0 W0)
      // + 1  cluster arrive (A+B+SFB, non-leader CTA1 W0)
      // + 32 noinc arrives (leader CTA0 SFA cp.async, W1)
      // + 1  cluster arrive (non-leader CTA1 SFA cp.async ready, W1)
      // = 35.  CTA1's SFA must be resident before the 2-CTA UTCCP/MMA reads it,
      // so CTA1 signals completion to the leader via a cross-CTA arrive rather
      // than a CTA-local cp.async noinc (which the leader could not observe and
      // which would overflow CTA1's never-waited barrier).
      initialize_barrier(ab_readable[i], kSfaCpAsyncThreads2SM + 3);
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
                         kSfaCpAsyncThreads2SM + kMmaThreads + kTmaThreads + kEpiThreads);
    }
    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 1) {
    tmem_allocator.allocate(kTmemCols, &s_tmem_base);
    tmem_allocator.release_allocation_lock();
  }

  // Load shm_tiles + shm_cu_tiles
  for (int i = idx; i < num_group; i += blockDim.x) {
    shm_tiles[i] = tiles_ptr[i];
  }
  for (int i = idx; i <= num_group; i += blockDim.x) {
    shm_cu_tiles[i] = cu_tiles_ptr[i];
  }
  if constexpr (GemmConfig::kNeedPreZeroB) {
    for (int i = idx; i < static_cast<int>(cosize(SLayoutB{})); i += blockDim.x) {
      reinterpret_cast<uint8_t *>(shm_b)[i] = 0;
    }
  }
  cluster_relaxed_sync();

  tCt.data() = make_tmem_ptr<float>(s_tmem_base);

  using TinB = typename GemmConfig::TinB;
  constexpr uint32_t kBytesA = (cosize(SLayoutA{}) / kStage) * sizeof(Tin);
  constexpr uint32_t kBytesB = (cosize(SLayoutB{}) / kStage) * cute::sizeof_bits_v<TinB> / 8;
  constexpr uint32_t kExpectedBytesAB =
      kMmaSM * (kBytesA + kBytesB) + kMmaSM * (32 * 16);  // A + B + SFB (no SFA)

  if (iwarp == 0 && elected) {
    // TMA warp (both CTAs): loads A, B, SFB (SFA handled by W1 via cp.async)
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
        auto *td_a = td_ay + igroup * 2 + 0;
        prefetch_tma_descriptor(td_a);

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(ab_writable[istage_ab], phase_ab);
          // load A + B + SFB
          copy(tma_b.with(ab_readable[istage_ab], 0, TMA::CacheHintSm100::EVICT_FIRST),
               tBg(_, itile_n, itile_k, igroup), tBs(_, 0, 0, istage_ab));
          copy(tma_a.with(td_a, ab_readable[istage_ab], 0, TMA::CacheHintSm100::EVICT_FIRST),
               tAg(_, itile_m, itile_k), tAs(_, 0, 0, istage_ab));
          int sfb_flat_n = igroup * (n / kTileN) + itile_n;
          copy(tma_sfb.with(ab_readable[istage_ab], 0, TMA::CacheHintSm100::EVICT_FIRST),
               tSFBg(_, 0, 0, sfb_flat_n, itile_k), tSFBs(_, 0, 0, istage_ab));
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
    // W1 (32 threads): cp.async SFA inline prepack (2SM variant)
    int local_idx = idx - 32;

    int phase_ab = 1;  // ab_writable phase
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
        int cu_seq = cu_seqlens_ptr[igroup];
        int tile_base_row = cu_seq + itile_m * kTileM;
        int rows_in_group = seqlens_ptr[igroup];
        int remaining_m = rows_in_group - itile_m * kTileM;

        // Precompute SFA per-row info (invariant across k-tiles).
        constexpr int kSfaRowsPerThread =
            (kTileM + kSfaCpAsyncThreads2SM - 1) / kSfaCpAsyncThreads2SM;
        const uint8_t *sfa_src_base[kSfaRowsPerThread];
        int sfa_smem_off[kSfaRowsPerThread];
        int sfa_valid_size[kSfaRowsPerThread];
        if (local_idx < kTileM) {
#pragma unroll
          for (int i = 0; i < kSfaRowsPerThread; i++) {
            int r = local_idx + i * kSfaCpAsyncThreads2SM;
            bool valid = (r < remaining_m);
            int abs_row = tile_base_row + r;
            int src_row = valid ? abs_row : 0;
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

          // Both CTAs load the full SFA tile into their own smem: the 2-CTA
          // UTCCP in the leader's MMA warp sources each CTA's TMEM SF from that
          // CTA's smem, so CTA1's smem must hold valid SFA too.
          if (local_idx < kTileM) {
            auto *sfa_stage = reinterpret_cast<uint8_t *>(shm_sfa) + istage_ab * kSfxRows * 16;
#pragma unroll
            for (int i = 0; i < kSfaRowsPerThread; i++) {
              const void *gmem_src = sfa_src_base[i] + itile_k * 4;
              void *smem_dst = sfa_stage + sfa_smem_off[i];
              cp_async_4b(smem_dst, gmem_src, sfa_valid_size[i]);
            }
          }

          if (elected_cta) {
            // Leader CTA: bind the local cp.async copies to the leader's
            // ab_readable via noinc (one arrive per W1 thread = 32).
            cpasync_barrier_arrive_noinc(reinterpret_cast<uint64_t *>(&ab_readable[istage_ab]));
          } else {
            // Non-leader CTA: cp.async.mbarrier.arrive.noinc can only signal a
            // CTA-local barrier, and CTA1's own ab_readable is never waited on
            // (the MMA warp runs only on the leader) — issuing noinc here would
            // pile up on a barrier whose phase never advances and overflow it.
            // Instead, drain the local cp.async, publish the smem writes to the
            // async proxy (the UTCCP reads via the async proxy), then signal the
            // LEADER's ab_readable with a single cross-CTA arrive.
            cutlass::arch::cp_async_fence();
            cutlass::arch::cp_async_wait<0>();
            __syncwarp();
            cutlass::arch::fence_view_async_shared();
            if (elected) {
              arrive_cluster_barrier(ab_readable[istage_ab], 0);
            }
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
  } else if (iwarp == 2) {
    // MMA warp (inlines UTCCP: tcgen05.cp + tcgen05.mma on same warp → hardware serialized)
    int phase_ab = 0;

    int istage_ab = 0;

    int istage_tile = 0;
    int phase_tile = 1;

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
          // Wait for A+B+SFA+SFB SMEM data
          wait_barrier(ab_readable[istage_ab], phase_ab);

          // Inline UTCCP: SF SMEM → TMEM (single slot)
          if (elected) {
            SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_2cta::copy(sfb_desc[istage_ab], sf_base);
            SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_2cta::copy(sfa_low_desc[istage_ab],
                                                                    sf_base + 4);
            if constexpr (!kSmallTM) {
              SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_2cta::copy(sfa_high_desc[istage_ab],
                                                                      sf_base + 4 + 4);
            }
          }

          // MMA (immediately follows UTCCP — hardware serialized on same tensor core)
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

          // Release A+B+SFA+SFB SMEM (single multicast arrive)
          cutlass::arch::umma_arrive_multicast_2x1SM(&ab_writable[istage_ab], mcast_mask);

          istage_ab++;
          if (istage_ab == kStage) {
            istage_ab = 0;
            phase_ab ^= 1;
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
    // Find task
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
    // Epilogue warps (both CTAs): TMEM -> scale(topk) -> cast -> SMEM -> atomic reduce -> GMEM
    int epi_idx = idx - 128;
    int tmem_rd_phase = 0;
    int istage_tile = 0;
    int istage_tma = 0;
    bool is_leader = elected && (iwarp == 4);

    auto gY_local = make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                                make_shape(Int<kCtaTileN>{}, Int<kEpiTileM>{}),
                                make_stride(Int<kEpiTileM>{}, Int<1>{}));

    auto IY = make_identity_tensor(gY.shape());
    auto ISY = make_identity_tensor(gY_local.shape());
    auto epi_tiler = make_tile(Int<kCtaTileN>{}, Int<kEpiTileM>{});
    auto tCt_epi = zipped_divide(tCt, make_tile(epi_tiler));
    auto rC_epi = zipped_divide(gY, epi_tiler);
    auto sC_epi = zipped_divide(sY, epi_tiler);
    auto sYT_epi = zipped_divide(sYT, epi_tiler);
    auto IC_epi = zipped_divide(IY, epi_tiler);

    // TiledCopy TMEM -> RMEM
    using TmemLoadAtom = typename GemmConfig::TmemLoadAtom;
    auto tiled_copy_t2r = make_tmem_copy(TmemLoadAtom{}, tCt_epi(_, _0{}));
    auto thr_copy_t2r = tiled_copy_t2r.get_slice(epi_idx);
    auto tCt4r = thr_copy_t2r.partition_S(tCt_epi);
    auto tCr4t = make_tensor_like<float>(thr_copy_t2r.partition_D(rC_epi(_, 0)));
    auto tICr4t = thr_copy_t2r.partition_D(IC_epi(_, 0));

    // TiledCopy RMEM -> SMEM
    using StsmTAtom = typename GemmConfig::StsmTAtom;
    auto tiled_copy_r2s = make_tiled_copy_D(Copy_Atom<StsmTAtom, Tout>{}, tiled_copy_t2r);
    auto thr_copy_r2s = tiled_copy_r2s.get_slice(epi_idx);
    auto tCr4s = make_tensor_like<Tout>(thr_copy_r2s.partition_S(rC_epi(_, 0)));
    auto tCs4r = thr_copy_r2s.partition_D(sYT_epi);

    // S2G copy for computing output addresses (atomic reduce path)
    auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(n, 8192),
                         make_stride(Int<1>{}, n));
    auto gYY = local_tile(Y, make_tile(Int<kCtaTileN>{}, Int<kEpiTileM>{}), make_coord(_, _));

    using s2g_copy_op = UniversalCopy<cute::uint128_t>;
    using s2g_copy_traits = Copy_Traits<s2g_copy_op>;
    using s2g_copy_atom = Copy_Atom<s2g_copy_traits, Tout>;
    using S2GCopy = decltype(make_tiled_copy(
        s2g_copy_atom{},
        make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<1>{}, Int<16>{})),
        make_layout(make_shape(Int<8>{}, Int<1>{}))));
    S2GCopy s2g_tiled_copy;
    auto s2g_thr_copy = s2g_tiled_copy.get_slice(epi_idx);
    auto tYg = s2g_thr_copy.partition_D(gYY);
    auto tYs = s2g_thr_copy.partition_S(sYT);
    auto tIY = s2g_thr_copy.partition_S(ISY);

    auto nepi_tile = size<2>(tCt4r);

    int phase_task = 0;
    int istage_task = 0;
    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;

    auto tCt4r_base_ptr = tCt4r.data();

    get_next_tile_horizon_mxfp8<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_n, itile_m,
                                        sum_tile_m, flat_divider);

    while (true) {
      if (igroup >= 0) {
        tCt4r.data() = tCt4r_base_ptr + istage_tile * kTileM;

        constexpr int kCopyM = size<2>(tYs);
        int cu_seq = cu_seqlens_ptr[igroup];
        int tile_base_row = cu_seq + itile_m * kTileM;
        int remaining_m = seqlens_ptr[igroup] - itile_m * kTileM;

        // Cooperative load of topk_scale into shared memory
        float *topk_scale_row_map_group = topk_scale_row_map_ptr + tile_base_row;
        for (int i = epi_idx; i < kTileM; i += 128) {
          if (i < remaining_m) {
            reduce_topk_scale[i] = topk_scale_row_map_group[i];
          }
        }

        // Precompute row_offsets and row_valid
        int row_offsets[nepi_tile][kCopyM];
        bool row_valid[nepi_tile][kCopyM];

#pragma unroll
        for (int iepi = 0; iepi < nepi_tile; iepi++) {
#pragma unroll
          for (int ir = 0; ir < kCopyM; ir++) {
            int ir_in_epi = get<1>(tIY(0, 0, ir));
            int ir_in_tile = iepi * kEpiTileM + ir_in_epi;
            row_valid[iepi][ir] = (ir_in_tile < remaining_m);
            int abs_row = tile_base_row + ir_in_tile;
            int src_row = x_row_map_ptr[abs_row];
            row_offsets[iepi][ir] = (src_row - ir_in_epi) * n;
          }
        }

        wait_barrier(tmem_readable[istage_tile], tmem_rd_phase);
        cutlass::arch::NamedBarrier::sync(128, 0);  // sync after topk_scale load

#pragma unroll
        for (int iepi = 0; iepi < nepi_tile; iepi++) {
          // TMEM -> RMEM
          copy(tiled_copy_t2r, tCt4r(_, _, iepi), tCr4t);
          cutlass::arch::fence_view_async_tmem_load();

          if (iepi == nepi_tile - 1) {
            if (is_leader) {
              arrive_cluster_barrier(tmem_writable[istage_tile], 0);
            }
          }

          auto tCr_fp2 = recast<float2>(tCr4t);
          auto tCr_bf162 = recast<__nv_bfloat162>(tCr4s);

          // cast (with per-row topk_scale)
#pragma unroll
          for (int i = 0; i < cute::size(tCr_bf162); i++) {
            int irow = iepi * kEpiTileM + get<1>(tICr4t(2 * i));
            float oscale0 = reduce_topk_scale[irow];
            float oscale1 = reduce_topk_scale[irow + 1];
            float2 v = tCr_fp2(i);
            v.x *= oscale0;
            v.y *= oscale1;
            tCr_bf162(i) = __float22bfloat162_rn(v);
          }

          // RMEM -> SMEM
          copy(tiled_copy_r2s, tCr4s, tCs4r(_, _, istage_tma));

          // SMEM -> GMEM (atomic reduce per row)
          cutlass::arch::NamedBarrier::sync(128, 0);

#pragma unroll
          for (int ir = 0; ir < kCopyM; ir++) {
            if (row_valid[iepi][ir]) {
              atomic_reduce_bf16_mxfp8(
                  &tYg(0, 0, ir, itile_n * kMmaSM + sm_idx, 0) + row_offsets[iepi][ir],
                  &tYs(0, 0, ir, istage_tma), 16);
            }
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

#endif  // SRC_GROUP_GEMM_SM100_MXFP8_GROUP_GEMM_MXFP8_WITH_REDUCE_CUH_
