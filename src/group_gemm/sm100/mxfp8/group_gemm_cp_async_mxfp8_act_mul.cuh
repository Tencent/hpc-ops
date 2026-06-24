// Copyright 2026 hpc-ops authors
//
// Fused GateUp GEMM + SiLU(gate)*up + MXFP8 quantization kernel.
//
// Weight layout (offline interleaved): N-dim every 16 elements:
//   [gate_0..15, up_0..15, gate_16..31, up_16..31, ...]
// SFB (weight scale) interleaved identically.
//
// Output: fp8_e4m3 activations (via STSM + TMA store) + UE8M0 scale (global store).

#ifndef SRC_GROUP_GEMM_SM100_MXFP8_GROUP_GEMM_CP_ASYNC_MXFP8_ACT_MUL_CUH_
#define SRC_GROUP_GEMM_SM100_MXFP8_GROUP_GEMM_CP_ASYNC_MXFP8_ACT_MUL_CUH_

#include <cuda.h>
#include <cuda_fp8.h>

#include <algorithm>

#include "src/group_gemm/sm100/mxfp8/group_gemm_cp_async_mxfp8.cuh"

namespace hpc {
namespace group_gemm {

using namespace cute;  // NOLINT

namespace kernels {

__device__ __forceinline__ uint8_t fp32_absmax_to_ue8m0_act(float absmax) {
  if (absmax == 0.f) {
    return 0;
  }
  uint32_t bits = __float_as_uint(absmax);
  int exp_biased = (bits >> 23) & 0xFF;
  uint32_t mant = bits & 0x7FFFFF;
  int sf_bits = exp_biased - 8 + (mant > 0x600000u ? 1 : 0);
  if (sf_bits < 0) {
    sf_bits = 0;
  }
  if (sf_bits > 255) {
    sf_bits = 255;
  }
  return static_cast<uint8_t>(sf_bits);
}

template <typename GemmConfig, typename TmaB, typename TmaSFB, bool kUsePDL = false>
__global__ __launch_bounds__(384, 1) void group_gemm_1sm_cp_async_mxfp8_act_mul_kernel(
    const __grid_constant__ TmaB tma_b, const __grid_constant__ TmaSFB tma_sfb,
    const typename GemmConfig::Tin *__restrict__ x_ptr, const uint8_t *__restrict__ sfx_ptr,
    const int *__restrict__ x_row_map_ptr, int x_num_rows,
    typename GemmConfig::Tout *__restrict__ y_ptr, __nv_fp8_e4m3 *y_fp8_ptr,
    uint8_t *__restrict__ out_scale_ptr, int *seqlens_ptr, int *cu_seqlens_ptr, int *tiles_ptr,
    int *cu_tiles_ptr, int num_group, int m, int n, int k, cutlass::FastDivmod flat_divider) {
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
  using TiledMma = typename GemmConfig::TiledMma;
  using G2SCopy = typename GemmConfig::G2SCopy;

  constexpr int kTileM = GemmConfig::kTileM;
  constexpr int kTileN = GemmConfig::kTileN;  // 128 (interleaved gate+up)
  constexpr int kTileK = GemmConfig::kTileK;
  constexpr int kEpiTileM = GemmConfig::kEpiTileM;
  constexpr int kStage = GemmConfig::kStage;
  constexpr int kStageTile = GemmConfig::kStageTile;
  constexpr int kStageTask = 5;
  constexpr int kSfVec = GemmConfig::kSfVec;
  constexpr bool kSmallTM = GemmConfig::kSmallTM;
  constexpr int kSfxRows = GemmConfig::kSfxRows;
  constexpr int kMmaSM = GemmConfig::kMmaSM;
  constexpr int kScaleColsPerTile = GemmConfig::kScaleColsPerTile;

  static_assert(kMmaSM == 1, "act_mul mxfp8 kernel is 1SM only");

  // SMEM layout for bf16 intermediate (TMEM → STSM landing): (kTileN=128, kEpiTileM, kStageTMA)
  // Uses SLayoutYT (SW128 swizzle) — same as non-fused kernel's STSM target.
  constexpr int kOutTileN = kTileN / 2;
  (void)kOutTileN;
  constexpr int kStageTMA = GemmConfig::kStageTMA;
  using SLayoutYT = typename GemmConfig::SLayoutYT;

  const int n_half = n / 2;
  (void)n_half;

  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int elected = cute::elect_one_sync();

  // SMEM layout:
  //   [A (activation)] [B (weight)] [SFA] [SFB] [YT (bf16 intermediate)] [tiles/cu_tiles]
  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_a = reinterpret_cast<Tin *>(shm_data);
  auto *shm_b = shm_a + cosize(SLayoutA{});
  auto *shm_sfa = reinterpret_cast<Tsf *>(shm_b + cosize(SLayoutB{}));
  auto *shm_sfb = shm_sfa + cosize(SLayoutSFA{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_sfb + cosize(SLayoutSFB{}));
  int *shm_tiles = reinterpret_cast<int *>(shm_y + cosize(SLayoutYT{}));
  int *shm_cu_tiles = shm_tiles + (num_group + 1);

  Tensor sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
  Tensor sSFA = make_tensor(make_smem_ptr(shm_sfa), SLayoutSFA{});
  Tensor sSFB = make_tensor(make_smem_ptr(shm_sfb), SLayoutSFB{});

  // Global tensors
  auto gB = tma_b.get_tma_tensor(make_shape(n, k, num_group));
  int Ksf_tiles_runtime = (k / kSfVec + 3) / 4;
  auto gSFB = tma_sfb.get_tma_tensor(
      make_shape(Int<32>{}, Int<16>{}, num_group * (n / 128), Ksf_tiles_runtime));

  // TMEM output: (kTileN=128, kTileM) with gate/up interleaved
  auto gY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));

  const int K_sf = k / kSfVec;

  auto gA_full = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                             make_stride(k, Int<1>{}));
  auto gA_tiled = local_tile(gA_full, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(0, _));
  auto IA = make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileK>{}));

  // TMA partition
  auto btma_b = tma_b.get_slice(0);
  auto btma_sfb = tma_sfb.get_slice(0);
  auto tBg = btma_b.partition_S(gB);
  auto tBs = btma_b.partition_D(sB);
  auto tSFBg = btma_sfb.partition_S(gSFB);
  auto tSFBs = btma_sfb.partition_D(sSFB);

  int ntile_k = size<2>(tBg);

  // TiledMma
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
    auto *sfb_ptr_s = shm_sfb + s * 32 * 16;
    auto *sfa_ptr_s = shm_sfa + s * kSfxRows * 16;
    Tensor t_sfb = make_tensor(
        make_smem_ptr(sfb_ptr_s),
        make_layout(make_shape(Int<32>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));
    Tensor t_sfa_lo = make_tensor(
        make_smem_ptr(sfa_ptr_s),
        make_layout(make_shape(Int<32>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));
    sfb_desc[s] = static_cast<uint64_t>(UMMA::make_umma_desc<UMMA::Major::K>(t_sfb));
    sfa_low_desc[s] = static_cast<uint64_t>(UMMA::make_umma_desc<UMMA::Major::K>(t_sfa_lo));
    if constexpr (!kSmallTM) {
      Tensor t_sfa_hi = make_tensor(
          make_smem_ptr(sfa_ptr_s + 32 * 16),
          make_layout(make_shape(Int<32>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));
      sfa_high_desc[s] = static_cast<uint64_t>(UMMA::make_umma_desc<UMMA::Major::K>(t_sfa_hi));
    }
  }

  // TMEM
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

  constexpr int kCpAsyncThreads = 128;
  constexpr uint32_t kExpectedBytesB = sizeof(typename GemmConfig::TinB) * kTileN * kTileK;
  constexpr uint32_t kExpectedBytesSFB = 32 * 16;

  if (iwarp == 0 && elected) {
#pragma unroll
    for (int i = 0; i < kStage; i++) {
      initialize_barrier(ab_readable[i], kCpAsyncThreads + 1);
      initialize_barrier(ab_writable[i], 1);
    }
#pragma unroll
    for (int i = 0; i < kStageTile; i++) {
      initialize_barrier(tmem_readable[i], 1);
      initialize_barrier(tmem_writable[i], 1);
    }
#pragma unroll
    for (int i = 0; i < kStageTask; i++) {
      initialize_barrier(task_readable[i], 1);
      initialize_barrier(task_writable[i], 32 + 1 + 128 + kCpAsyncThreads);
    }
    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 1) {
    tmem_allocator.allocate(kTmemCols, &s_tmem_base);
    tmem_allocator.release_allocation_lock();
  }

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
  __syncthreads();

  tCt.data() = make_tmem_ptr<float>(s_tmem_base);

  if (idx >= 256) {
    // W8-W11 (128 threads): cp.async load A + SFA inline prepack
    int local_idx = idx - 256;

    int phase_a = 1;  // ab_writable phase (whole-stage reusable signal)
    int istage_a = 0;

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
        // Precompute per-row A fix-up offsets for this (igroup, itile_m) tile.
        int cu_seq = cu_seqlens_ptr[igroup];
        int tile_base_row = cu_seq + itile_m * kTileM;
        int remaining_m = seqlens_ptr[igroup] - itile_m * kTileM;
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

        // Precompute SFA per-row info (invariant across k-tiles).
        // Only threads with local_idx < kTileM participate in SFA loads.
        // kTileM <= 256, kCpAsyncThreads = 128, so at most 2 rows per thread.
        constexpr int kSfaRowsPerThread = (kTileM + kCpAsyncThreads - 1) / kCpAsyncThreads;
        const uint8_t *sfa_src_base[kSfaRowsPerThread];
        int sfa_smem_off[kSfaRowsPerThread];
        int sfa_valid_size[kSfaRowsPerThread];
        if (local_idx < kTileM) {
#pragma unroll
          for (int i = 0; i < kSfaRowsPerThread; i++) {
            int r = local_idx + i * kCpAsyncThreads;
            bool valid = (r < remaining_m);
            int abs_row = tile_base_row + r;
            int src_row;
            if (kUseRowMap) {
              src_row = valid ? x_row_map_ptr[abs_row] : 0;
            } else {
              src_row = valid ? abs_row : 0;
            }
            sfa_src_base[i] = reinterpret_cast<const uint8_t *>(sfx_ptr + src_row * K_sf);
            int block_32x16 = r / 128;
            int block_32x4 = (r / 32) % 4;
            int row_in_block = r % 32;
            sfa_smem_off[i] = block_32x16 * (32 * 16) + row_in_block * 16 + block_32x4 * 4;
            sfa_valid_size[i] = valid ? 4 : 0;
          }
        }
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          // cp.async load A — wait stage reusable, then issue per-row copies.
          wait_barrier(ab_writable[istage_a], phase_a);
#pragma unroll
          for (int ir = 0; ir < kCopyM; ir++) {
            auto tAg_src = make_tensor(tAg(_, ir, _, itile_k).data() + a_row_offsets[ir],
                                       tAg(_, ir, _, itile_k).layout());
            cute::copy(g2s_tiled_copy, tAg_src, tAs(_, ir, _, istage_a));
          }
          if (local_idx < kTileM) {
            auto *sfa_stage = reinterpret_cast<uint8_t *>(shm_sfa) + istage_a * kSfxRows * 16;
#pragma unroll
            for (int i = 0; i < kSfaRowsPerThread; i++) {
              const void *gmem_src = sfa_src_base[i] + itile_k * 4;
              void *smem_dst = sfa_stage + sfa_smem_off[i];
              cp_async_4b(smem_dst, gmem_src, sfa_valid_size[i]);
            }
          }

          // cp.async commit + noinc arrive covers both A and SFA async copies.
          cpasync_barrier_arrive_noinc(reinterpret_cast<uint64_t *>(&ab_readable[istage_a]));

          istage_a++;
          if (istage_a == kStage) {
            phase_a ^= 1;
            istage_a = 0;
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
  } else if (iwarp == 0 && elected) {
    // W0: TMA load B + SFB (SFA is handled by W8-W11 inline prepack)
    int phase = 1;
    int istage_k = 0;

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
          // load B + SFB into the shared ab_readable barrier.
          wait_barrier(ab_writable[istage_k], phase);
          int sfb_flat_n = igroup * (n / 128) + itile_n;
          copy(tma_b.with(ab_readable[istage_k], 0, TMA::CacheHintSm90::EVICT_FIRST),
               tBg(_, itile_n, itile_k, igroup), tBs(_, 0, 0, istage_k));
          copy(tma_sfb.with(ab_readable[istage_k], 0, TMA::CacheHintSm90::EVICT_FIRST),
               tSFBg(_, 0, 0, sfb_flat_n, itile_k), tSFBs(_, 0, 0, istage_k));
          set_barrier_transaction_bytes(ab_readable[istage_k], kExpectedBytesB + kExpectedBytesSFB);

          istage_k++;
          if (istage_k == kStage) {
            istage_k = 0;
            phase ^= 1;
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
    // W1: MMA warp (also performs the inlined SMEM->TMEM UTCCP for SF).
    int phase = 0;  // A/B/SFA/SFB ready phase
    int istage_k = 0;

    int phase_tile = 1;
    int istage_tile = 0;

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
          wait_barrier(ab_readable[istage_k], phase);

          if (elected) {
            SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(sfb_desc[istage_k], sf_base);
            SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(sfa_low_desc[istage_k],
                                                                    sf_base + 4);
            if constexpr (!kSmallTM) {
              SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta::copy(sfa_high_desc[istage_k],
                                                                      sf_base + 4 + 4);
            }
          }
#pragma unroll
          for (uint32_t ik = 0; ik < size<2>(tBr); ik++) {
            uint32_t sfb_addr = (sf_base & 0x3FFFFFFFu) | (ik << 30);
            uint32_t sfa_addr = ((sf_base + 4) & 0x3FFFFFFFu) | (ik << 30);
            auto sfb_ki = make_tensor(make_tmem_ptr<cutlass::float_ue8m0_t>(sfb_addr),
                                      make_layout(make_shape(1)));
            auto sfa_ki = make_tensor(make_tmem_ptr<cutlass::float_ue8m0_t>(sfa_addr),
                                      make_layout(make_shape(1)));
            cute::gemm(tiled_mma.with(tiled_mma.accumulate_, sfb_ki, sfa_ki),
                       tBr(_, _, ik, istage_k), tAr(_, _, ik, istage_k), tCt);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }

          // Release A+B+SF SMEM (single arrive frees the stage for all producers)
          cutlass::arch::umma_arrive(&ab_writable[istage_k]);

          istage_k++;
          if (istage_k == kStage) {
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
  } else if (iwarp == 2 && elected) {
    // W2: FindTask
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
    // W4-W7: Epilogue
    //   Phase A: TMEM → reg (fp32) → bf16 → STSM → SMEM  (identical to non-fused)
    //   Phase B: SMEM read → act_mul (SiLU(gate)*up) + mxfp8 quant → global store
    int epi_idx = idx - 128;
    bool is_leader = elected && (iwarp == 4);

    // ---- Phase A setup: TMEM → STSM (same as group_gemm_1sm_cp_async_mxfp8_kernel) ----
    using SLayoutY = typename GemmConfig::SLayoutY;
    using SLayoutYT = typename GemmConfig::SLayoutYT;
    Tensor sY = make_tensor(make_smem_ptr(reinterpret_cast<Tout *>(shm_y)), SLayoutY{});
    Tensor sYT = make_tensor(make_smem_ptr(reinterpret_cast<Tout *>(shm_y)), SLayoutYT{});

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
        wait_barrier(tmem_readable[istage_tile], phase_tile);
#pragma unroll
        for (int iepi = 0; iepi < nepi_tile; iepi++) {
          // ===== Phase A: TMEM → reg → bf16 → STSM → SMEM =====
          copy(tiled_copy_t2r, tCt4r(_, _, iepi), tCr4t);
          cutlass::arch::fence_view_async_tmem_load();

          if (iepi == nepi_tile - 1) {
            if (is_leader) {
              arrive_barrier(tmem_writable[istage_tile]);
            }
          }

          // fp32 → bf16 conversion in registers
          auto tCr_fp2 = recast<float2>(tCr4t);
          auto tCr_bf162 = recast<__nv_bfloat162>(tCr4s);
#pragma unroll
          for (int i = 0; i < cute::size(tCr_bf162); i++) {
            tCr_bf162(i) = __float22bfloat162_rn(tCr_fp2(i));
          }

          // STSM: reg → SMEM (all 128 threads participate)
          cutlass::arch::NamedBarrier::sync(128, 0);
          copy(tiled_copy_r2s, tCr4s, tCs4r(_, _, istage_tma));
          cutlass::arch::NamedBarrier::sync(128, 0);

          // act_mul and quant
          {
            constexpr int kActMulVec = 16;
            constexpr int kThreadsPerCol = kTileN / (2 * kActMulVec);
            constexpr int kTotalWork = kThreadsPerCol * kEpiTileM;
            constexpr int kItersPerThread = (kTotalWork + 127) / 128;
            constexpr int kSfVecOut = 32;

            int cu_seq = cu_seqlens_ptr[igroup];
            int global_m_base = cu_seq + itile_m * kTileM + iepi * kEpiTileM;
            int global_n_base = itile_n * kOutTileN;

#pragma unroll
            for (int iter = 0; iter < kItersPerThread; iter++) {
              int work_id = epi_idx + iter * 128;
              if (work_id >= kTotalWork) {
                break;
              }

              int n_row = work_id % kThreadsPerCol;
              int m_col = work_id / kThreadsPerCol;
              int n_base = n_row * (2 * kActMulVec);

              vec_t<float, kActMulVec> v;
              float local_max = 0.f;
              {
#pragma unroll
                for (int i = 0; i < 2; i++) {
                  auto g =
                      to<float>(load<__nv_bfloat16, 8>(&sYT(n_base + i * 8, m_col, istage_tma)));
                  auto u = to<float>(
                      load<__nv_bfloat16, 8>(&sYT(n_base + kActMulVec + i * 8, m_col, istage_tma)));
#pragma unroll
                  for (int j = 0; j < 8; j++) {
                    v[i * 8 + j] = silu(g[j]) * u[j];
                    local_max = fmaxf(local_max, fabsf(v[i * 8 + j]));
                  }
                }
              }

              float partner_max = __shfl_xor_sync(0xFFFFFFFF, local_max, 1);
              float absmax = fmaxf(local_max, partner_max);

              uint8_t sf_bits = fp32_absmax_to_ue8m0_act(absmax);
              float inv_sf;
              if (sf_bits == 0) {
                inv_sf = 0.f;
              } else {
                inv_sf = 1.f / exp2f(static_cast<float>(sf_bits) - 127.f);
              }

              int out_n_base = global_n_base + n_row * kActMulVec;
              int global_m = global_m_base + m_col;
              if (global_m < m) {
                vec_t<__nv_fp8_e4m3, kActMulVec> v_fp8;
#pragma unroll
                for (int i = 0; i < kActMulVec; i++) {
                  v_fp8[i] = static_cast<__nv_fp8_e4m3>(v[i] * inv_sf);
                }
                store<__nv_fp8_e4m3, kActMulVec>(
                    &y_fp8_ptr[static_cast<int64_t>(global_m) * n_half + out_n_base], v_fp8);

                if ((n_row & 1) == 0) {
                  int K_sf_out = n_half / kSfVecOut;
                  int sf_n_idx = (global_n_base / kSfVecOut) + (n_row / 2);
                  out_scale_ptr[static_cast<int64_t>(global_m) * K_sf_out + sf_n_idx] = sf_bits;
                }
              }
            }
          }  //  act_mul and quant end
          cutlass::arch::NamedBarrier::sync(128, 0);

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

#endif  // SRC_GROUP_GEMM_SM100_MXFP8_GROUP_GEMM_CP_ASYNC_MXFP8_ACT_MUL_CUH_
