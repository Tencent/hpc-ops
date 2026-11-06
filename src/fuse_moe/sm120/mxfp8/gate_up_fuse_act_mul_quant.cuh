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

__device__ __forceinline__ void get_next_tile_horizon(const int *tiles_ptr, int iblock,
                                                      int num_group, int &igroup, int &itile_n,
                                                      int &itile_m, int &itile_mg, int &sum_tile_m,
                                                      cutlass::FastDivmod flat_divider) {
  int num_tile_m;

  flat_divider(itile_mg, itile_n, iblock);
  for (int i = igroup; i < num_group; i++) {
    num_tile_m = tiles_ptr[i];
    sum_tile_m += num_tile_m;
    if (itile_mg < sum_tile_m) {
      igroup = i;
      sum_tile_m = sum_tile_m - num_tile_m;
      itile_m = itile_mg - sum_tile_m;
      return;
    }
  }
  igroup = -1;
}

template <typename Config, int kNumTopk, typename TmaB, typename TmaY, typename TmaSFB>
__global__ void __launch_bounds__(Config::kNumThreads, 1) gate_up_fuse_act_mul_quant_kernel(
    const __grid_constant__ TmaB tma_b, const __grid_constant__ TmaY tma_y,
    const __grid_constant__ TmaSFB tma_sfb, typename Config::TSF *y_scale_ptr,
    const typename Config::TA *x_ptr, const typename Config::TSF *x_scale_ptr,
    const int *seqlens_ptr, const int *cu_seqlens_ptr, const int *tiles_ptr,
    const int *permuted_idx_to_raw_idx_ptr, int num_group, int m_pad, int n, int k, float limit,
    cutlass::FastDivmod tile_n_divider) {
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
  using SLayoutSFAMK = typename Config::SLayoutSFAMK;
  using SLayoutSFBMK = typename Config::SLayoutSFBMK;
  using TiledMma = typename Config::TiledMma;
  using TiledSF = typename Config::TiledSF;
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

  // The token of each of this task's A rows, -1 meaning zero-fill.
  __shared__ int a_token_shm[kTileM];

  // shm
  extern __shared__ uint8_t shm_data[] alignas(1024);
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
  auto tSFBs = btma_sfb.partition_D(sSFB);  // (V, 1, 1, kStageAB)

  int num_tile_k = k / kTileK;

  constexpr int kTransactionBytes =
      (sizeof_bits_v<typename Config::TW> * cosize(SLayoutB{}(_, _, 0)) +  // NOLINT
       sizeof_bits_v<TSF> * cosize(SLayoutSFB{}(_, _, 0))) /               // NOLINT
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
      initialize_barrier(task_writable[i], kNumMmaWarps * 32 + 1 + kNumALoadThreads);
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
    if (elected) {
      int istage_task = 0;
      int phase_task = 1;

      int igroup = 0;
      int itile_m = 0;
      int itile_n = 0;
      int itile_mg = 0;
      int sum_tile_m = 0;

      while (true) {
        wait_barrier(task_writable[istage_task], phase_task);

        iblock += gridDim.x;
        get_next_tile_horizon(tiles_ptr, iblock, num_group, igroup, itile_n, itile_m, itile_mg,
                              sum_tile_m, tile_n_divider);
        task_shm[istage_task][0] = igroup;
        task_shm[istage_task][1] = itile_m;
        task_shm[istage_task][2] = itile_n;
        task_shm[istage_task][3] = itile_mg;

        arrive_barrier(task_readable[istage_task]);
        if (igroup < 0) {
          break;
        }
        if (++istage_task == kStageTask) {
          istage_task = 0;
          phase_task ^= 1;
        }
      }  // while
    }
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
        itoken[i] = route >= 0 ? route / kNumTopk : -1;
      }

#pragma unroll 1
      for (int itile_k = 0; itile_k < num_tile_k; itile_k++) {
        wait_barrier(shmab_writable[ismem_write], phase);

        // load B + SFB
        if (elected) {
          set_barrier_transaction_bytes(shmab_readable[ismem_write], kTransactionBytes);
          cute::copy(tma_b.with(shmab_readable[ismem_write], 0, TMA::CacheHintSm90::EVICT_FIRST),
                     tBg(_, itile_n, itile_k, igroup), tBs(_, 0, 0, ismem_write));
          cute::copy(tma_sfb.with(shmab_readable[ismem_write], 0, TMA::CacheHintSm90::EVICT_FIRST),
                     tSFBg(_, 0, 0, itile_n, itile_k, igroup), tSFBs(_, 0, 0, ismem_write));
        }

        // load SFA
        if (issuing_sfa) {
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
      // The first sync holds back the other warp's reads of the last task's tokens.
      cutlass::arch::NamedBarrier::sync(kNumALoadThreads, 2);
      for (int irow = iload; irow < kTileM; irow += kNumALoadThreads) {
        int jrow = itile_m * kTileM + irow;
        int route = jrow < seqlen ? permuted_idx_to_raw_idx_ptr[cu_seqlen + jrow] : -1;
        a_token_shm[irow] = route >= 0 ? route / kNumTopk : -1;
      }
      cutlass::arch::NamedBarrier::sync(kNumALoadThreads, 2);

#pragma unroll 1
      for (int itile_k = 0; itile_k < num_tile_k; itile_k++) {
        wait_barrier(shmab_writable[ismem_write], phase);

#pragma unroll
        for (int i = 0; i < kRowsPerThread; i++) {
          // A row past this group's length copies 0 bytes, which zero-fills it.
          int irow = irow0 + i * kRowStride;
          int itoken = a_token_shm[irow];
          const auto *src = x_ptr + static_cast<int64_t>(itoken >= 0 ? itoken : 0) * k +
                            itile_k * kTileK + ichunk * 16;
          cp_async_16b(&sA(irow, ichunk * 16, ismem_write), src, itoken >= 0 ? 16 : 0);
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
      arrive_barrier(task_writable[istage_task]);
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

    TiledSF tiled_sf;
    auto thr_sf = tiled_sf.get_slice(idx);
    auto tCsSFA = thr_sf.partition_A(make_tensor(make_smem_ptr(shm_sfa), SLayoutSFAMK{}));
    auto tCsSFB = thr_sf.partition_B(make_tensor(make_smem_ptr(shm_sfb), SLayoutSFBMK{}));
    auto tCrSFA = make_fragment_like(tCsSFA(_, _, _, 0));
    auto tCrSFB = make_fragment_like(tCsSFB(_, _, _, 0));

    TiledCopyA tiled_copy_a;
    TiledCopyB tiled_copy_b;
    auto thr_copy_a = tiled_copy_a.get_slice(idx);
    auto thr_copy_b = tiled_copy_b.get_slice(idx);
    auto tAsA = thr_copy_a.partition_S(sA);
    auto tBsB = thr_copy_b.partition_S(sB);
    auto tArA = thr_copy_a.retile_D(tCrA);
    auto tBrB = thr_copy_b.retile_D(tCrB);

    constexpr int kNumM = size<1>(decltype(tCrA){});

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

      for (int itile_k = 0; itile_k < num_tile_k; itile_k++) {
        wait_barrier(shmab_readable[ismem_read], phase_shmab);

        // cp sfa/sfb from smem to reg
        cute::copy(tCsSFA(_, _, _, ismem_read), tCrSFA);
        cute::copy(tCsSFB(_, _, _, ismem_read), tCrSFB);

#pragma unroll
        for (int ik = 0; ik < size<2>(tCrA); ++ik) {
          cute::copy(tiled_copy_a, tAsA(_, _, ik, ismem_read), tArA(_, _, ik));
          cute::copy(tiled_copy_b, tBsB(_, _, ik, ismem_read), tBrB(_, _, ik));
          // fp4: ldmatrix leaves each value in a byte's low 4 bits, the MMA reads bits [5:2]
          fp4_shift_B(typename TiledMma::MMA_Op{}, tCrB(_, _, ik));
          cute::gemm(tiled_mma, make_zip_tensor(tCrA(_, _, ik), tCrSFA(_, _, ik)),
                     make_zip_tensor(tCrB(_, _, ik), tCrSFB(_, _, ik)), acc);
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

      // epilogue: act = silu(gate) * up, where in = 0,1 are gate columns and in = 2,3 the matching
      // up columns (+16). acc's v = 0,1 sit on row q and v = 2,3 on row q + 8.
      float act[4][kNumM][2];
#pragma unroll
      for (int im = 0; im < kNumM; im++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
#pragma unroll
          for (int v = 0; v < 4; v++) {
            float g = fminf(limit, acc(v, im, j));
            float u = fminf(fmaxf(acc(v, im, j + 2), -limit), limit);
            act[v][im][j] = silu(g) * u;
          }
        }
      }

      // One quantization group is 32 act values per row: this warp's 16 columns reduce within the
      // quad first, then merge with the neighbouring warp (warp_n ^ 1).
      float amax[kNumM][2];
#pragma unroll
      for (int im = 0; im < kNumM; im++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          float v = fmaxf(fmaxf(fabsf(act[2 * h][im][0]), fabsf(act[2 * h + 1][im][0])),
                          fmaxf(fabsf(act[2 * h][im][1]), fabsf(act[2 * h + 1][im][1])));
          v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 1));
          v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, 2));
          amax[im][h] = v;
          if (ilane % 4 == 0) {
            max_shm[warp_n * kTileM + get<0>(tIC(2 * h, im, 0))] = v;
          }
        }
      }
      cutlass::arch::NamedBarrier::sync(kNumMmaWarps * 32, 1);

#pragma unroll
      for (int im = 0; im < kNumM; im++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          int row = get<0>(tIC(2 * h, im, 0));
          float row_amax = fmaxf(amax[im][h], max_shm[(warp_n ^ 1) * kTileM + row]);

          constexpr float kE4m3MaxInv = 1.f / 448.f;
          __nv_fp8_e8m0 sf;
          sf.__x = __nv_cvt_float_to_e8m0(row_amax * kE4m3MaxInv, __NV_SATFINITE, cudaRoundPosInf);
          float sfv = static_cast<float>(sf);
          float inv = (sfv != 0.f) ? rcpf_ftz(sfv) : 0.f;

          // Each quantization group's scale is written by the even-warp_n warps.
          if (warp_n % 2 == 0 && ilane % 4 == 0) {
            store(&gYSF(itile_mg * kTileM + row, itile_n * 2 + warp_n / 2), sf);
          }

#pragma unroll
          for (int j = 0; j < 2; j++) {
            // The gate column c's position within act: the first 16 of every 32 columns.
            int c = get<1>(tIC(2 * h, im, j));
            int icol = c / 32 * 16 + c % 16;
            __nv_fp8x2_storage_t q = __nv_cvt_float2_to_fp8x2(
                make_float2(act[2 * h][im][j] * inv, act[2 * h + 1][im][j] * inv), __NV_SATFINITE,
                __NV_E4M3);
            store(&sY(row, icol), q);
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

  cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace mxfp8
}  // namespace kernels
}  // namespace fuse_moe
}  // namespace hpc

#endif  // SRC_FUSE_MOE_SM120_MXFP8_GATE_UP_FUSE_ACT_MUL_QUANT_CUH_
