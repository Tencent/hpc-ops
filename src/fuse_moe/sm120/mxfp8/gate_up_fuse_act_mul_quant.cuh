// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_SM120_MXFP8_GATE_UP_FUSE_ACT_MUL_QUANT_CUH_
#define SRC_FUSE_MOE_SM120_MXFP8_GATE_UP_FUSE_ACT_MUL_QUANT_CUH_

#include <cuda.h>
#include <cuda_fp8.h>

#include "cute/atom/mma_traits_sm120.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/fast_math.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace fuse_moe {
namespace kernels {
namespace mxfp8 {

// Called by all 32 lanes of a warp.
__device__ __forceinline__ void get_next_tile_horizon(const int *tiles_ptr, int iblock,
                                                      int num_group, int &igroup, int &itile_n,
                                                      int &itile_m, int &itile_mg, int &sum_tile_m,
                                                      cutlass::FastDivmod flat_divider) {
  int ilane = threadIdx.x % 32;

  flat_divider(itile_mg, itile_n, iblock);
  for (int base = igroup; base < num_group; base += 32) {
    int num_tile_m = base + ilane < num_group ? tiles_ptr[base + ilane] : 0;
    int incl = num_tile_m;
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
      int up = __shfl_up_sync(0xFFFFFFFF, incl, d);
      incl += ilane >= d ? up : 0;
    }
    int total = __shfl_sync(0xFFFFFFFF, incl, 31);
    if (itile_mg < sum_tile_m + total) {
      int src = __ffs(__ballot_sync(0xFFFFFFFF, sum_tile_m + incl > itile_mg)) - 1;
      igroup = base + src;
      sum_tile_m += __shfl_sync(0xFFFFFFFF, incl - num_tile_m, src);
      itile_m = itile_mg - sum_tile_m;
      return;
    }
    sum_tile_m += total;
  }
  igroup = -1;
}

// cp_async_16b with an L2 cache policy (the TMA CacheHintSm90 encodings).
__device__ __forceinline__ void cp_async_16b_hint(uint32_t smem_addr, const void *gmem_src,
                                                  uint64_t cache_hint) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(smem_addr),
               "l"(gmem_src), "l"(cache_hint));
}

template <typename Config, typename TmaB, typename TmaY, typename TmaSFB>
__global__ void __launch_bounds__(Config::kNumThreads, 1) gate_up_fuse_act_mul_quant_kernel(
    const __grid_constant__ TmaB tma_b, const __grid_constant__ TmaY tma_y,
    const __grid_constant__ TmaSFB tma_sfb, typename Config::TSF *y_scale_ptr,
    const typename Config::TA *x_ptr, const typename Config::TSF *x_scale_ptr,
    const int *seqlens_ptr, const int *cu_seqlens_ptr, const int *tiles_ptr,
    const int *permuted_idx_to_raw_idx_ptr, int num_group, int m_pad, int n, int k, float limit,
    cutlass::FastDivmod tile_n_divider, cutlass::FastDivmod topk_divider,
    cute::TMA::CacheHintSm90 w_cache_hint, cute::TMA::CacheHintSm90 a_cache_hint) {
  using namespace cute;  // NOLINT

  using TA = typename Config::TA;
  using TB = typename Config::TB;
  using TC = typename Config::TC;
  using TSF = typename Config::TSF;

  using SLayoutA = typename Config::SLayoutA;
  using SLayoutB = typename Config::SLayoutB;
  using SLayoutC = typename Config::SLayoutC;
  using SLayoutSFA = typename Config::SLayoutSFA;
  using SLayoutSFB = typename Config::SLayoutSFB;
  using SLayoutSFARead = typename Config::SLayoutSFARead;
  using SLayoutSFBRead = typename Config::SLayoutSFBRead;
  using TiledMma = typename Config::TiledMma;
  using TiledCopyA = typename Config::TiledCopyA;
  using TiledCopyB = typename Config::TiledCopyB;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStageAB = Config::kStageAB;
  constexpr int kStageTask = Config::kStageTask;
  constexpr int kMathThreads = Config::kMathThreads;
  constexpr int kSFBlockM = Config::kSFBlockM;
  constexpr int kSFBlockBytes = Config::kSFBlockBytes;
  constexpr int kSFPerTile = Config::kSFPerTile;

  // idx
  int idx = threadIdx.x;
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  int elected = cute::elect_one_sync();
  bool is_leader_in_block = (iwarp == 0) && elected;
  int iblock = blockIdx.x;

  // barrier
  __shared__ uint64_t shmab_readable[kStageAB];
  __shared__ uint64_t shmab_writable[kStageAB];

  __shared__ uint64_t task_readable[kStageTask];
  __shared__ uint64_t task_writable[kStageTask];
  __shared__ int task_shm[kStageTask][4];

  __shared__ float max_shm[4 * kTileM];

  // The token of each of this task's valid A rows.
  __shared__ int a_token_shm[kTileM];

  // shm
  // 1024B aligned for the position-independent swizzle views.
  extern __shared__ __align__(1024) uint8_t shm_data[];
  auto *shm_a = reinterpret_cast<TA *>(shm_data);
  auto *shm_b = reinterpret_cast<TB *>(shm_a + cosize(SLayoutA{}));
  auto *shm_y = reinterpret_cast<TC *>(shm_b + cosize(SLayoutB{}));
  auto *shm_sfa = reinterpret_cast<TSF *>(shm_y + cosize(SLayoutC{}));
  auto *shm_sfb = reinterpret_cast<TSF *>(shm_sfa + cosize(SLayoutSFA{}));

  auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutC{});
  auto sSFA = make_tensor(make_smem_ptr(shm_sfa), SLayoutSFA{});
  auto sSFB = make_tensor(make_smem_ptr(shm_sfb), SLayoutSFB{});

  // gmem
  auto gB = tma_b.get_tma_tensor(make_shape(n, k, num_group));
  auto gSFB = tma_sfb.get_tma_tensor(
      make_shape(Int<4>{}, Int<128>{}, n / kSFBlockM, k / kTileK, num_group));

  // tma
  auto btma_b = tma_b.get_slice(0);
  auto btma_sfb = tma_sfb.get_slice(0);

  auto tBg = btma_b.partition_S(gB);  // (V, N, K, G)
  auto tBs = btma_b.partition_D(sB);  // (V, N, K, kStageAB)

  auto tSFBg = btma_sfb.partition_S(gSFB);  // (V, 1, 1, N / 128, K, G)
  auto tSFBs = btma_sfb.partition_D(sSFB);  // (V, 1, 1, kNumSFB, kStageAB)

  int num_tile_k = k / kTileK;

  constexpr int kTransactionBytes =
      (sizeof_bits_v<typename Config::TW> * cosize(SLayoutB{}(_, _, 0)) +  // NOLINT
       sizeof_bits_v<TSF> * cosize(SLayoutSFB{}(_, _, _, 0))) /            // NOLINT
      8;

  constexpr int kNumMmaWarps = Config::kNumMmaWarps;
  constexpr int kNumALoadThreads = 64;
  constexpr int kChunksPerRow = kTileK * sizeof(TA) / 16;
  constexpr int kRowStride = kNumALoadThreads / kChunksPerRow;
  constexpr int kRowsPerThread = kTileM / kRowStride;
  static_assert(Config::kNumThreads - kMathThreads == 64 + kNumALoadThreads,
                "warps 8-9 schedule and issue TMA, warps 10-11 load A");

  // barrier
  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStageAB; ++i) {
      initialize_barrier(shmab_readable[i], 1 + 32 + kNumALoadThreads);
      initialize_barrier(shmab_writable[i], kNumMmaWarps * 32);
    }
#pragma unroll
    for (int i = 0; i < kStageTask; ++i) {
      initialize_barrier(task_readable[i], 1);
      initialize_barrier(task_writable[i], kNumMmaWarps * 32 + 1 + kNumALoadThreads / 32);
    }

    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  // PDL
  cudaGridDependencySynchronize();

  if (idx >= kMathThreads) {
    cutlass::arch::warpgroup_reg_dealloc<Config::kLoadRegisters>();
  }

  // main
  if (iwarp == 8) {
    // scheduler warp
    int istage_task = 0;
    int phase_task = 1;

    int igroup = 0;
    int itile_m = 0;
    int itile_n = 0;
    int itile_mg = 0;
    int sum_tile_m = 0;

    while (true) {
      iblock += gridDim.x;
      get_next_tile_horizon(tiles_ptr, iblock, num_group, igroup, itile_n, itile_m, itile_mg,
                            sum_tile_m, tile_n_divider);

      wait_barrier(task_writable[istage_task], phase_task);
      if (elected) {
        task_shm[istage_task][0] = igroup;
        task_shm[istage_task][1] = itile_m;
        task_shm[istage_task][2] = itile_n;
        task_shm[istage_task][3] = itile_mg;
        arrive_barrier(task_readable[istage_task]);
      }
      if (igroup < 0) {
        break;
      }
      if (++istage_task == kStageTask) {
        istage_task = 0;
        phase_task ^= 1;
      }
    }  // while
  } else if (iwarp == 9) {
    // tma warp
    int ilane = idx % 32;
    int ismem_write = 0;
    int phase = 1;
    int istage_task = 0;
    int phase_task = 0;
    int num_sf_k = k / Config::kSFVecSize;

    // this warp's lanes split SFA's kTileM rows, kSFARowsPerLane each
    constexpr int kSFARowsPerLane = Config::kSFARowsPerLane;
    bool issuing_sfa = ilane < kTileM / kSFARowsPerLane;

    int igroup = 0;
    int itile_m = 0;
    int itile_n = 0;
    int itile_mg = 0;
    int sum_tile_m = 0;
    get_next_tile_horizon(tiles_ptr, iblock, num_group, igroup, itile_n, itile_m, itile_mg,
                          sum_tile_m, tile_n_divider);

    while (igroup >= 0) {
      int seqlen = seqlens_ptr[igroup];
      int cu_seqlen = cu_seqlens_ptr[igroup];
      int itoken[kSFARowsPerLane];
#pragma unroll
      for (int i = 0; i < kSFARowsPerLane; i++) {
        int irow = itile_m * kTileM + ilane * kSFARowsPerLane + i;
        int route =
            (issuing_sfa && irow < seqlen) ? permuted_idx_to_raw_idx_ptr[cu_seqlen + irow] : -1;
        int token = -1;
        if (route >= 0) {
          int slot;
          topk_divider(token, slot, route);
        }
        itoken[i] = token;
      }
      // Valid rows come first, so a lane starting on padding holds only padding and is skipped.
      bool valid_sfa = issuing_sfa && itoken[0] >= 0;

#pragma unroll 1
      for (int itile_k = 0; itile_k < num_tile_k; itile_k++) {
        wait_barrier(shmab_writable[ismem_write], phase);

        // load B + SFB
        if (elected) {
          set_barrier_transaction_bytes(shmab_readable[ismem_write], kTransactionBytes);
          cute::copy(tma_b.with(shmab_readable[ismem_write], 0, w_cache_hint),
                     tBg(_, itile_n, itile_k, igroup), tBs(_, 0, 0, ismem_write));
#pragma unroll
          for (int b = 0; b < Config::kNumSFB; b++) {
            cute::copy(tma_sfb.with(shmab_readable[ismem_write]),
                       tSFBg(_, 0, 0, itile_n * Config::kNumSFB + b, itile_k, igroup),
                       tSFBs(_, 0, 0, b, ismem_write));
          }
        }

        // load SFA
        if (valid_sfa) {
#pragma unroll
          for (int i = 0; i < kSFARowsPerLane; i++) {
            const auto *src = x_scale_ptr +
                              static_cast<int64_t>(itoken[i] >= 0 ? itoken[i] : 0) * num_sf_k +
                              itile_k * kSFPerTile;
            cp_async_4b(&sSFA(ilane * kSFARowsPerLane + i, 0, ismem_write), src,
                        itoken[i] >= 0 ? kSFPerTile : 0);
          }
        }

        cutlass::arch::cpasync_barrier_arrive_noinc(&shmab_readable[ismem_write]);

        ++ismem_write;
        if (ismem_write == kStageAB) {
          ismem_write = 0;
          phase ^= 1;
        }
      }  // for itile_k

      // fetch the next task
      wait_barrier(task_readable[istage_task], phase_task);
      igroup = task_shm[istage_task][0];
      itile_m = task_shm[istage_task][1];
      itile_n = task_shm[istage_task][2];
      itile_mg = task_shm[istage_task][3];
      if (elected) {
        arrive_barrier(task_writable[istage_task]);
      }
      if (++istage_task == kStageTask) {
        istage_task = 0;
        phase_task ^= 1;
      }
    }  // while
  } else if (iwarp >= 10) {
    // A load warps
    int iload = idx - kMathThreads - 64;
    int ichunk = iload % kChunksPerRow;
    int irow0 = iload / kChunksPerRow;
    // A thread's rows are kRowStride rows (1024B) apart, so they share one swizzle phase and sit
    // at constant offsets from its first row.
    constexpr int kRowBytesA = kTileK * sizeof(TA);
    constexpr int kStageBytesA = kTileM * kRowBytesA;
    static_assert(kRowStride * kRowBytesA == 1024, "one swizzle period per row step");
    uint32_t dst0 = cute::cast_smem_ptr_to_uint(&sA(irow0, ichunk * 16, 0));
    int ismem_write = 0;
    int phase = 1;
    int istage_task = 0;
    int phase_task = 0;

    int igroup = 0;
    int itile_m = 0;
    int itile_n = 0;
    int itile_mg = 0;
    int sum_tile_m = 0;
    get_next_tile_horizon(tiles_ptr, iblock, num_group, igroup, itile_n, itile_m, itile_mg,
                          sum_tile_m, tile_n_divider);

    while (igroup >= 0) {
      int seqlen = seqlens_ptr[igroup];
      int cu_seqlen = cu_seqlens_ptr[igroup];
      int num_valid_rows = min(seqlen - itile_m * kTileM, kTileM);
      // The first sync holds back the other warp's reads of the last task's tokens.
      cutlass::arch::NamedBarrier::sync(kNumALoadThreads, 2);
      for (int irow = iload; irow < num_valid_rows; irow += kNumALoadThreads) {
        int route = permuted_idx_to_raw_idx_ptr[cu_seqlen + itile_m * kTileM + irow];
        int token, slot;
        topk_divider(token, slot, route);
        a_token_shm[irow] = token;
      }
      cutlass::arch::NamedBarrier::sync(kNumALoadThreads, 2);

#pragma unroll 1
      for (int itile_k = 0; itile_k < num_tile_k; itile_k++) {
        wait_barrier(shmab_writable[ismem_write], phase);

        // Padding rows are left as they are: the MMA warps skip their 16-row blocks.
        const auto *src = x_ptr + itile_k * kTileK + ichunk * 16;
        uint32_t dst = dst0 + ismem_write * kStageBytesA;
#pragma unroll
        for (int i = 0; i < kRowsPerThread; i++) {
          int irow = irow0 + i * kRowStride;
          if (irow < num_valid_rows) {
            cp_async_16b_hint(dst + i * kRowStride * kRowBytesA,
                              src + static_cast<int64_t>(a_token_shm[irow]) * k,
                              static_cast<uint64_t>(a_cache_hint));
          }
        }
        cutlass::arch::cpasync_barrier_arrive_noinc(&shmab_readable[ismem_write]);

        ++ismem_write;
        if (ismem_write == kStageAB) {
          ismem_write = 0;
          phase ^= 1;
        }
      }  // for itile_k

      // fetch the next task
      wait_barrier(task_readable[istage_task], phase_task);
      igroup = task_shm[istage_task][0];
      itile_m = task_shm[istage_task][1];
      itile_n = task_shm[istage_task][2];
      itile_mg = task_shm[istage_task][3];
      if (elected) {
        arrive_barrier(task_writable[istage_task]);
      }
      if (++istage_task == kStageTask) {
        istage_task = 0;
        phase_task ^= 1;
      }
    }  // while
  } else if (iwarp < 8) {
    // MMA warp
    cutlass::arch::warpgroup_reg_alloc<Config::kMathRegisters>();

    int ismem_read = 0;
    int phase_shmab = 0;
    int ilane = idx % 32;
    int warp_n = idx / 32 % 4;

    // mma
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
    auto acc = partition_fragment_C(tiled_mma, Shape<Int<kTileM>, Int<kTileN>>{});
    auto tIC = thr_mma.partition_C(make_identity_tensor(Shape<Int<kTileM>, Int<kTileN>>{}));
    CUTE_STATIC_ASSERT_V(size(acc) == Int<kTileM * kTileN / (kNumMmaWarps * 32)>{});

    TiledCopyA tiled_copy_a;
    TiledCopyB tiled_copy_b;
    auto thr_copy_a = tiled_copy_a.get_slice(idx);
    auto thr_copy_b = tiled_copy_b.get_slice(idx);
    auto psA = as_position_independent_swizzle_tensor(sA);
    auto psB = as_position_independent_swizzle_tensor(sB);
    auto psY = as_position_independent_swizzle_tensor(sY);
    auto tAsA = thr_copy_a.partition_S(psA);
    auto tBsB = thr_copy_b.partition_S(psB);
    auto tArA = thr_copy_a.retile_D(tCrA);
    auto tBrB = thr_copy_b.retile_D(tCrB);

    constexpr int kNumM = size<1>(decltype(tCrA){});
    constexpr int kNumN = size<1>(decltype(tCrB){});
    int warp_m = idx / 32 / 4;
    auto sSFAr = make_tensor(make_smem_ptr(shm_sfa), SLayoutSFARead{});
    auto sSFBr = make_tensor(make_smem_ptr(shm_sfb), SLayoutSFBRead{});

    int sfa_row[kNumM];
    int sfb_row[kNumN];
#pragma unroll
    for (int im = 0; im < kNumM; im++) {
      sfa_row[im] = im * 32 + warp_m * 16 + ilane / 4 + (ilane % 2) * 8;
    }
#pragma unroll
    for (int in = 0; in < kNumN; in++) {
      sfb_row[in] = warp_n * (kTileN / 4) + in * 8 + ilane / 4;
    }

    // y's scales, (row, scale col) in layout_128x4: row r's c-th scale sits in block
    // (r / 128, c / 4) at (r % 32) * 16 + (r % 128 / 32) * 4 + c % 4. y carries n / 2 / 32 per row.
    int num_sf_blk_y = n / 2 / Config::kSFVecSize / kSFPerTile;
    auto gYSF = make_tensor(
        make_gmem_ptr(y_scale_ptr),
        make_layout(make_shape(make_shape(_32{}, _4{}, m_pad / kSFBlockM),
                               make_shape(Int<kSFPerTile>{}, num_sf_blk_y)),
                    make_stride(make_stride(_16{}, _4{},
                                            static_cast<int64_t>(num_sf_blk_y) * kSFBlockBytes),
                                make_stride(_1{}, Int<kSFBlockBytes>{}))));

    int istage_task = 0;
    int phase_task = 0;

    int igroup = 0;
    int itile_m = 0;
    int itile_n = 0;
    int itile_mg = 0;
    int sum_tile_m = 0;
    get_next_tile_horizon(tiles_ptr, iblock, num_group, igroup, itile_n, itile_m, itile_mg,
                          sum_tile_m, tile_n_divider);

    while (igroup >= 0) {
      clear(acc);
      int seqlen = seqlens_ptr[igroup];

      // Skip this warp's 16-row blocks past the valid rows.
      int num_valid_m = (seqlen - itile_m * kTileM - warp_m * 16 + 31) / 32;
      dispatch_num_valid_m<kNumM>(num_valid_m, [&](auto valid_m) {
        constexpr int kNumValidM = decltype(valid_m)::value;
        for (int itile_k = 0; itile_k < num_tile_k; itile_k++) {
          wait_barrier(shmab_readable[ismem_read], phase_shmab);

          uint32_t sfa4[kNumM];
          uint32_t sfb4[kNumN];

          // load sfa
#pragma unroll
          for (int im = 0; im < kNumValidM; im++) {
            sfa4[im] = *reinterpret_cast<const uint32_t *>(&sSFAr(sfa_row[im], 0, ismem_read));
          }
          // load sfb
#pragma unroll
          for (int in = 0; in < kNumN; in++) {
            sfb4[in] = *reinterpret_cast<const uint32_t *>(
                &sSFBr(sfb_row[in] % 32, sfb_row[in] % kSFBlockM / 32 * 4, sfb_row[in] / kSFBlockM,
                       ismem_read));
          }

#pragma unroll
          for (int ik = 0; ik < (kNumValidM > 0 ? kTileK / 32 : 0); ik++) {
#pragma unroll
            for (int im = 0; im < kNumValidM; im++) {
              cute::copy(tiled_copy_a, tAsA(_, im, ik, ismem_read), tArA(_, im, ik));
            }
            cute::copy(tiled_copy_b, tBsB(_, _, ik, ismem_read), tBrB(_, _, ik));
            if constexpr (Config::kIsFp4W) {
              // ldmatrix .b4x16_p64 puts each fp4 in a byte's low 4 bits, mxf8f6f4 wants it in
              // the middle 4: shift left by 2 (same as cute's fp4_shift_B).
              cute::transform(recast<uint32_t>(tCrB(_, _, ik)), [](uint32_t &v) { return v << 2; });
            }
            // Every (im, in) has its own scales, so they are set per MMA.
#pragma unroll
            for (int im = 0; im < kNumValidM; im++) {
#pragma unroll
              for (int in = 0; in < kNumN; in++) {
                cute::gemm(tiled_mma.with(sfa4[im], sfb4[in], ik), tCrA(_, im, ik), tCrB(_, in, ik),
                           acc(_, im, in));
              }
            }
          }

          // tell TMA this stage may be overwritten
          cutlass::arch::fence_view_async_shared();
          arrive_barrier(shmab_writable[ismem_read]);

          ++ismem_read;
          if (ismem_read == kStageAB) {
            ismem_read = 0;
            phase_shmab ^= 1;
          }
        }  // for itile_k
      });

      // epilogue: act = silu(gate) * up, where in % 4 = 0,1 are gate columns and in % 4 = 2,3 the
      // matching up columns (+16). acc's v = 0,1 sit on row q and v = 2,3 on row q + 8.
      constexpr int kNumJ = kNumN / 2;
      auto gate_in = [](int j) { return j / 2 * 4 + j % 2; };
      float act[4][kNumM][kNumJ];
#pragma unroll
      for (int im = 0; im < kNumM; im++) {
#pragma unroll
        for (int j = 0; j < kNumJ; j++) {
#pragma unroll
          for (int v = 0; v < 4; v++) {
            float g = fminf(limit, acc(v, im, gate_in(j)));
            float u = fminf(fmaxf(acc(v, im, gate_in(j) + 2), -limit), limit);
            act[v][im][j] = silu(g) * u;
          }
        }
      }

      // One quantization group is 32 act values per row: they reduce within the quad first, then
      // (kTileN = 128) merge with the neighbouring warp (warp_n ^ 1).
      constexpr bool kMergeWarps = kNumJ * 8 < Config::kSFVecSize;
      constexpr int kGroupsPerTile = Config::kTileNC / Config::kSFVecSize;
      float amax[kNumM][2];
#pragma unroll
      for (int im = 0; im < kNumM; im++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          float v = 0.f;
#pragma unroll
          for (int j = 0; j < kNumJ; j++) {
            v = fmaxf(v, fmaxf(fabsf(act[2 * h][im][j]), fabsf(act[2 * h + 1][im][j])));
          }
          v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 1));
          v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 2));
          amax[im][h] = v;
          if (kMergeWarps && ilane % 4 == 0) {
            max_shm[warp_n * kTileM + get<0>(tIC(2 * h, im, 0))] = v;
          }
        }
      }
      if constexpr (kMergeWarps) {
        cutlass::arch::NamedBarrier::sync(kNumMmaWarps * 32, 1);
      }

#pragma unroll
      for (int im = 0; im < kNumM; im++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          int row = get<0>(tIC(2 * h, im, 0));
          float row_amax =
              kMergeWarps ? fmaxf(amax[im][h], max_shm[(warp_n ^ 1) * kTileM + row]) : amax[im][h];

          constexpr float kE4m3MaxInv = 1.f / 448.f;
          __nv_fp8_e8m0 sf;
          sf.__x = __nv_cvt_float_to_e8m0(row_amax * kE4m3MaxInv, __NV_SATFINITE, cudaRoundPosInf);
          // From the exponent: rcp of an all-zero group's 2^-127 scale flushes to inf.
          float inv = __uint_as_float(static_cast<uint32_t>(254 - sf.__x) << 23);

          // Each quantization group's scale is written by the even-warp_n warps when merged.
          if ((!kMergeWarps || warp_n % 2 == 0) && ilane % 4 == 0) {
            store(&gYSF(itile_mg * kTileM + row,
                        itile_n * kGroupsPerTile + (kMergeWarps ? warp_n / 2 : warp_n)),
                  sf);
          }

#pragma unroll
          for (int j = 0; j < kNumJ; j++) {
            // The gate column c's position within act: the first 16 of every 32 columns.
            int c = get<1>(tIC(2 * h, im, gate_in(j)));
            int icol = c / 32 * 16 + c % 16;
            __nv_fp8x2_storage_t q = __nv_cvt_float2_to_fp8x2(
                make_float2(act[2 * h][im][j] * inv, act[2 * h + 1][im][j] * inv), __NV_SATFINITE,
                __NV_E4M3);
            store(&psY(row, icol), q);
          }
        }
      }

      tma_store_fence();
      cutlass::arch::NamedBarrier::sync(kNumMmaWarps * 32, 1);
      if (idx == 0) {
        auto gY = tma_y.get_tma_tensor(make_shape(m_pad, n / 2));
        auto btma_y = tma_y.get_slice(0);
        auto tYg = btma_y.partition_S(gY);
        auto tYs = btma_y.partition_D(sY);
        cute::copy(tma_y, tYs(_, 0, 0), tYg(_, itile_mg, itile_n));
        tma_store_arrive();
        tma_store_wait<0>();
      }
      // Without this some warps race ahead into the next task.
      cutlass::arch::NamedBarrier::sync(kNumMmaWarps * 32, 1);

      // fetch the next task
      wait_barrier(task_readable[istage_task], phase_task);
      igroup = task_shm[istage_task][0];
      itile_m = task_shm[istage_task][1];
      itile_n = task_shm[istage_task][2];
      itile_mg = task_shm[istage_task][3];
      arrive_barrier(task_writable[istage_task]);
      if (++istage_task == kStageTask) {
        istage_task = 0;
        phase_task ^= 1;
      }
    }
  }  // mma warp

  // Must stay after cudaGridDependencySynchronize: down reads the routing outputs before its own.
  cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace mxfp8
}  // namespace kernels
}  // namespace fuse_moe
}  // namespace hpc

#endif  // SRC_FUSE_MOE_SM120_MXFP8_GATE_UP_FUSE_ACT_MUL_QUANT_CUH_
