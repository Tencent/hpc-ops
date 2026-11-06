// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_SM120_MXFP8_DOWN_FUSE_REDUCE_CUH_
#define SRC_FUSE_MOE_SM120_MXFP8_DOWN_FUSE_REDUCE_CUH_

#include <cuda.h>
#include <cuda_bf16.h>

#include "cute/atom/mma_traits_sm120.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
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

template <typename Config, int kNumTopk, typename TmaA, typename TmaB, typename TmaSFA,
          typename TmaSFB>
__global__ void __launch_bounds__(Config::kNumThreads, 1)
    down_fuse_reduce_kernel(const __grid_constant__ TmaA tma_a, const __grid_constant__ TmaB tma_b,
                            const __grid_constant__ TmaSFA tma_sfa,
                            const __grid_constant__ TmaSFB tma_sfb, typename Config::TC *y_ptr,
                            const float *topk_scale_ptr, const int *seqlens_ptr,
                            const int *cu_seqlens_ptr, const int *tiles_ptr,
                            const int *permuted_idx_to_raw_idx_ptr, int num_group, int m_pad, int n,
                            int k, cutlass::FastDivmod tile_n_divider) {
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
  using TiledCopyC = typename Config::TiledCopyC;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStageAB = Config::kStageAB;
  constexpr int kStageTask = Config::kStageTask;
  constexpr int kMathThreads = Config::kMathThreads;
  constexpr int kSFBlockM = Config::kSFBlockM;
  constexpr int kTilePerSFBlock = Config::kTilePerSFBlock;

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

  __shared__ int row_token_shm[kTileM];

  // shm
  extern __shared__ uint8_t shm_data[] alignas(1024);
  auto *shm_a = reinterpret_cast<TA *>(shm_data);
  auto *shm_b = reinterpret_cast<TB *>(shm_a + cosize(SLayoutA{}));
  auto *shm_c = reinterpret_cast<TC *>(shm_b + cosize(SLayoutB{}));
  auto *shm_sfa = reinterpret_cast<TSF *>(shm_c + cosize(SLayoutC{}));
  auto *shm_sfb = reinterpret_cast<TSF *>(shm_sfa + cosize(SLayoutSFA{}));

  auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
  auto sC = make_tensor(make_smem_ptr(shm_c), SLayoutC{});
  auto sSFA = make_tensor(make_smem_ptr(shm_sfa), SLayoutSFA{});
  auto sSFB = make_tensor(make_smem_ptr(shm_sfb), SLayoutSFB{});

  // gmem
  auto gA = tma_a.get_tma_tensor(make_shape(m_pad, k));
  auto gB = tma_b.get_tma_tensor(make_shape(n, k, num_group));
  auto gSFA = tma_sfa.get_tma_tensor(
      make_shape(Int<4>{}, Int<128>{}, (m_pad + kSFBlockM - 1) / kSFBlockM, k / kTileK));
  auto gSFB = tma_sfb.get_tma_tensor(
      make_shape(Int<4>{}, Int<128>{}, n / kSFBlockM, k / kTileK, num_group));

  // tma
  auto btma_a = tma_a.get_slice(0);
  auto btma_b = tma_b.get_slice(0);
  auto btma_sfa = tma_sfa.get_slice(0);
  auto btma_sfb = tma_sfb.get_slice(0);

  auto tAg = btma_a.partition_S(gA);  // (V, M, K)
  auto tAs = btma_a.partition_D(sA);  // (V, M, K, kStageAB)

  auto tBg = btma_b.partition_S(gB);  // (V, N, K, G)
  auto tBs = btma_b.partition_D(sB);  // (V, N, K, kStageAB)

  auto tSFAg = btma_sfa.partition_S(gSFA);  // (V, 1, 1, M / 128, K)
  auto tSFAs = btma_sfa.partition_D(sSFA);  // (V, 1, 1, kStageAB)

  auto tSFBg = btma_sfb.partition_S(gSFB);  // (V, 1, 1, N / 128, K, G)
  auto tSFBs = btma_sfb.partition_D(sSFB);  // (V, 1, 1, kStageAB)

  int num_tile_k = k / kTileK;

  // A + B + SFA + SFB
  constexpr int kTransactionBytes =
      (sizeof_bits_v<TA> * cosize(SLayoutA{}(_, _, 0)) +                   // NOLINT
       sizeof_bits_v<typename Config::TW> * cosize(SLayoutB{}(_, _, 0)) +  // NOLINT
       sizeof_bits_v<TSF> * cosize(SLayoutSFA{}(_, _, 0)) +                // NOLINT
       sizeof_bits_v<TSF> * cosize(SLayoutSFB{}(_, _, 0))) /               // NOLINT
      8;

  constexpr int kNumMmaWarps = Config::kNumMmaWarps;

  // barrier
  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStageAB; ++i) {
      initialize_barrier(shmab_readable[i], 1);
      initialize_barrier(shmab_writable[i], kNumMmaWarps * 32);
    }
#pragma unroll
    for (int i = 0; i < kStageTask; ++i) {
      initialize_barrier(task_readable[i], 1);
      initialize_barrier(task_writable[i], kNumMmaWarps * 32 + 1);
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
    if (elected) {
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
#pragma unroll 1
        for (int itile_k = 0; itile_k < num_tile_k; itile_k++) {
          wait_barrier(shmab_writable[ismem_write], phase);

          cute::copy(tma_a.with(shmab_readable[ismem_write]), tAg(_, itile_mg, itile_k),
                     tAs(_, 0, 0, ismem_write));
          cute::copy(tma_b.with(shmab_readable[ismem_write], 0, TMA::CacheHintSm90::EVICT_FIRST),
                     tBg(_, itile_n, itile_k, igroup), tBs(_, 0, 0, ismem_write));
          cute::copy(tma_sfa.with(shmab_readable[ismem_write]),
                     tSFAg(_, 0, 0, itile_mg / kTilePerSFBlock, itile_k),
                     tSFAs(_, 0, 0, ismem_write));
          cute::copy(tma_sfb.with(shmab_readable[ismem_write], 0, TMA::CacheHintSm90::EVICT_FIRST),
                     tSFBg(_, 0, 0, itile_n, itile_k, igroup), tSFBs(_, 0, 0, ismem_write));
          set_barrier_transaction_bytes(shmab_readable[ismem_write], kTransactionBytes);

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
    }
  } else if (iwarp < 8) {
    // MMA warp
    cutlass::arch::warpgroup_reg_alloc<Config::kMathRegisters>();

    int ismem_read = 0;
    int phase_shmab = 0;

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
    auto tCrSFA = make_fragment_like(tCsSFA(_, _, _, 0, 0));
    auto tCrSFB = make_fragment_like(tCsSFB(_, _, _, 0));

    TiledCopyA tiled_copy_a;
    TiledCopyB tiled_copy_b;
    TiledCopyC tiled_copy_c;
    auto thr_copy_a = tiled_copy_a.get_slice(idx);
    auto thr_copy_b = tiled_copy_b.get_slice(idx);
    auto thr_copy_c = tiled_copy_c.get_slice(idx);
    auto tAsA = thr_copy_a.partition_S(sA);
    auto tBsB = thr_copy_b.partition_S(sB);
    auto tArA = thr_copy_a.retile_D(tCrA);
    auto tBrB = thr_copy_b.retile_D(tCrB);
    auto tCsC = thr_copy_c.partition_D(sC);

    constexpr int kNumM = size<1>(decltype(tCrA){});
    constexpr int kNumN = size<1>(decltype(tCrB){});

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

      int isf_tile = itile_mg % kTilePerSFBlock;

      for (int itile_k = 0; itile_k < num_tile_k; itile_k++) {
        wait_barrier(shmab_readable[ismem_read], phase_shmab);

        // cp sfa/sfb from smem to reg
        cute::copy(tCsSFA(_, _, _, ismem_read, isf_tile), tCrSFA);
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

      // epilogue
      int seqlen = seqlens_ptr[igroup];
      int cu_seqlen = cu_seqlens_ptr[igroup];
      float row_scale[kNumM][2];
#pragma unroll
      for (int im = 0; im < kNumM; im++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          int irow = itile_m * kTileM + get<0>(tIC(2 * h, im, 0));
          row_scale[im][h] =
              (irow < seqlen) ? topk_scale_ptr[permuted_idx_to_raw_idx_ptr[cu_seqlen + irow]] : 0.f;
        }
      }

      // The token each row accumulates into, -1 meaning do not write.
      if (idx < kTileM) {
        int irow = itile_m * kTileM + idx;
        row_token_shm[idx] =
            (irow < seqlen) ? permuted_idx_to_raw_idx_ptr[cu_seqlen + irow] / kNumTopk : -1;
      }

      auto rC = make_tensor_like<TC>(acc);
#pragma unroll
      for (int im = 0; im < kNumM; im++) {
#pragma unroll
        for (int in = 0; in < kNumN; in++) {
#pragma unroll
          for (int v = 0; v < 4; v++) {
            rC(v, im, in) = static_cast<TC>(acc(v, im, in) * row_scale[im][v / 2]);
          }
        }
      }
      auto tCrC = thr_copy_c.retile_S(rC);
      cute::copy(tiled_copy_c, tCrC, tCsC);

      // Read 16B rows (eight bf16) back out of shared -- adjacent threads cover adjacent columns
      // of one row -- and atomically add them to y by token.
      cutlass::arch::NamedBarrier::sync(kNumMmaWarps * 32, 1);
      constexpr int kVecPerRow = kTileN / 8;
#pragma unroll
      for (int i = idx; i < kTileM * kVecPerRow; i += kNumMmaWarps * 32) {
        int row = i / kVecPerRow;
        int col = i % kVecPerRow * 8;
        int token = row_token_shm[row];
        if (token >= 0) {
          red_add_v4_bf16x2(y_ptr + static_cast<int64_t>(token) * n + itile_n * kTileN + col,
                            *reinterpret_cast<const uint4 *>(&sC(row, col)));
        }
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

#endif  // SRC_FUSE_MOE_SM120_MXFP8_DOWN_FUSE_REDUCE_CUH_
