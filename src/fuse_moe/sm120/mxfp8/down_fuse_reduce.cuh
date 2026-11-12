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

// 16B unit c of row r in a warp's 8x32 bf16 block, swizzled against bank conflicts.
__device__ __forceinline__ int epi_unit(int r, int c) { return r * 4 + (c ^ (r / 2)); }

// mma accumulator pairs of the 8x32 block in, columns lane % 4 * 8 .. + 8 of row lane / 4 out.
__device__ __forceinline__ uint4 epi_transpose(const uint32_t (&pairs)[4], uint4 *buf) {
  int ilane = threadIdx.x % 32;
  uint32_t addr = cute::cast_smem_ptr_to_uint(buf + epi_unit(ilane % 8, ilane / 8));
  asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(addr),
               "r"(pairs[0]), "r"(pairs[1]), "r"(pairs[2]), "r"(pairs[3])
               : "memory");
  __syncwarp();
  uint4 v = buf[epi_unit(ilane / 4, ilane % 4)];
  __syncwarp();
  return v;
}

template <typename Config, typename TmaA, typename TmaB, typename TmaSFA, typename TmaSFB>
__global__ void __launch_bounds__(Config::kNumThreads, 1)
    down_fuse_reduce_kernel(const __grid_constant__ TmaA tma_a, const __grid_constant__ TmaB tma_b,
                            const __grid_constant__ TmaSFA tma_sfa,
                            const __grid_constant__ TmaSFB tma_sfb, typename Config::TC *y_ptr,
                            const float *permuted_topk_scale_ptr, const int *seqlens_ptr,
                            const int *cu_seqlens_ptr, const int *tiles_ptr,
                            const int *permuted_idx_to_raw_idx_ptr, int num_group, int m_pad, int n,
                            int k, cutlass::FastDivmod tile_n_divider,
                            cutlass::FastDivmod topk_divider, cute::TMA::CacheHintSm90 w_cache_hint,
                            cute::TMA::CacheHintSm90 a_cache_hint) {
  using namespace cute;  // NOLINT

  using TA = typename Config::TA;
  using TB = typename Config::TB;
  using TC = typename Config::TC;
  using TSF = typename Config::TSF;

  using SLayoutA = typename Config::SLayoutA;
  using SLayoutB = typename Config::SLayoutB;
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
  // One more stage: the MMA warps hold a task through the epilogue.
  constexpr int kStageTask = Config::kStageTask + 1;
  constexpr int kMathThreads = Config::kMathThreads;
  constexpr int kSFBlockM = Config::kSFBlockM;
  constexpr int kTilePerSFBlock = kSFBlockM / kTileM;

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
  // {igroup, itile_m, itile_n, itile_mg, seqlen}
  __shared__ int task_shm[kStageTask][5];
  // {token, topk scale bits} per row, token -1 for padding.
  __shared__ int2 task_rows_shm[kStageTask][kTileM];
  __shared__ uint4 epi_shm[Config::kNumMmaWarps][32];

  // shm
  // 1024B aligned for the position-independent swizzle views.
  extern __shared__ __align__(1024) uint8_t shm_data[];
  auto *shm_a = reinterpret_cast<TA *>(shm_data);
  auto *shm_b = reinterpret_cast<TB *>(shm_a + cosize(SLayoutA{}));
  auto *shm_sfa = reinterpret_cast<TSF *>(shm_b + cosize(SLayoutB{}));
  auto *shm_sfb = reinterpret_cast<TSF *>(shm_sfa + cosize(SLayoutSFA{}));

  auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
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
  auto tSFBs = btma_sfb.partition_D(sSFB);  // (V, 1, 1, 1, kStageAB)

  int num_tile_k = k / kTileK;

  // A + B + SFA + SFB
  constexpr int kTransactionBytes =
      (sizeof_bits_v<TA> * cosize(SLayoutA{}(_, _, 0)) +                   // NOLINT
       sizeof_bits_v<typename Config::TW> * cosize(SLayoutB{}(_, _, 0)) +  // NOLINT
       sizeof_bits_v<TSF> * cosize(SLayoutSFA{}(_, _, 0)) +                // NOLINT
       sizeof_bits_v<TSF> * cosize(SLayoutSFB{}(_, _, _, 0))) /            // NOLINT
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

  // PDL: only A and SFA come from gate_up; the routing outputs were complete before gate_up got
  // past its own cudaGridDependencySynchronize, which precedes its launch trigger. So only the
  // TMA thread waits, after issuing the weights of its first task.

  if (idx >= kMathThreads) {
    cutlass::arch::warpgroup_reg_dealloc<Config::kLoadRegisters>();
  }

  // main
  if (iwarp == 8) {
    // scheduler warp
    int istage_task = 0;
    int phase_task = 1;

    int ilane = idx % 32;
    int igroup = 0;
    int itile_m = 0;
    int itile_n = 0;
    int itile_mg = 0;
    int sum_tile_m = 0;

    for (;; iblock += gridDim.x) {
      get_next_tile_horizon(tiles_ptr, iblock, num_group, igroup, itile_n, itile_m, itile_mg,
                            sum_tile_m, tile_n_divider);
      int seqlen = igroup >= 0 ? seqlens_ptr[igroup] : 0;
      int cu_seqlen = igroup >= 0 ? cu_seqlens_ptr[igroup] : 0;
      int2 rows[kTileM / 32];
#pragma unroll
      for (int j = 0; j < kTileM / 32; j++) {
        int irow = itile_m * kTileM + ilane + 32 * j;
        rows[j] = make_int2(-1, 0);
        if (irow < seqlen) {
          int route = permuted_idx_to_raw_idx_ptr[cu_seqlen + irow];
          int token;
          int slot;
          topk_divider(token, slot, route);
          rows[j] = make_int2(token, __float_as_int(permuted_topk_scale_ptr[cu_seqlen + irow]));
        }
      }

      wait_barrier(task_writable[istage_task], phase_task);
#pragma unroll
      for (int j = 0; j < kTileM / 32; j++) {
        task_rows_shm[istage_task][ilane + 32 * j] = rows[j];
      }
      __syncwarp();
      if (elected) {
        task_shm[istage_task][0] = igroup;
        task_shm[istage_task][1] = itile_m;
        task_shm[istage_task][2] = itile_n;
        task_shm[istage_task][3] = itile_mg;
        task_shm[istage_task][4] = seqlen;
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
    if (elected) {
      int ismem_write = 0;
      int phase = 1;
      int istage_task = 0;
      int phase_task = 0;
      bool dep_ready = false;

      while (true) {
        wait_barrier(task_readable[istage_task], phase_task);
        int igroup = task_shm[istage_task][0];
        int itile_n = task_shm[istage_task][2];
        int itile_mg = task_shm[istage_task][3];
        arrive_barrier(task_writable[istage_task]);
        if (++istage_task == kStageTask) {
          istage_task = 0;
          phase_task ^= 1;
        }
        if (igroup < 0) {
          break;
        }

        int itile_k = 0;
        if (!dep_ready) {
          // The first stages are still free: fill their B/SFB now, A/SFA once gate_up is done.
          int num_pre = num_tile_k < kStageAB ? num_tile_k : kStageAB;
#pragma unroll 1
          for (; itile_k < num_pre; itile_k++) {
            set_barrier_transaction_bytes(shmab_readable[itile_k], kTransactionBytes);
            cute::copy(tma_b.with(shmab_readable[itile_k], 0, w_cache_hint),
                       tBg(_, itile_n, itile_k, igroup), tBs(_, 0, 0, itile_k));
            cute::copy(tma_sfb.with(shmab_readable[itile_k]),
                       tSFBg(_, 0, 0, itile_n, itile_k, igroup), tSFBs(_, 0, 0, 0, itile_k));
          }
          cudaGridDependencySynchronize();
          dep_ready = true;
#pragma unroll 1
          for (int i = 0; i < num_pre; i++) {
            cute::copy(tma_a.with(shmab_readable[i], 0, a_cache_hint), tAg(_, itile_mg, i),
                       tAs(_, 0, 0, i));
            cute::copy(tma_sfa.with(shmab_readable[i], 0, a_cache_hint),
                       tSFAg(_, 0, 0, itile_mg / kTilePerSFBlock, i), tSFAs(_, 0, 0, i));
          }
          ismem_write = num_pre % kStageAB;
          if (num_pre == kStageAB) {
            phase ^= 1;
          }
        }

#pragma unroll 1
        for (; itile_k < num_tile_k; itile_k++) {
          wait_barrier(shmab_writable[ismem_write], phase);

          cute::copy(tma_a.with(shmab_readable[ismem_write], 0, a_cache_hint),
                     tAg(_, itile_mg, itile_k), tAs(_, 0, 0, ismem_write));
          cute::copy(tma_b.with(shmab_readable[ismem_write], 0, w_cache_hint),
                     tBg(_, itile_n, itile_k, igroup), tBs(_, 0, 0, ismem_write));
          cute::copy(tma_sfa.with(shmab_readable[ismem_write], 0, a_cache_hint),
                     tSFAg(_, 0, 0, itile_mg / kTilePerSFBlock, itile_k),
                     tSFAs(_, 0, 0, ismem_write));
          cute::copy(tma_sfb.with(shmab_readable[ismem_write]),
                     tSFBg(_, 0, 0, itile_n, itile_k, igroup), tSFBs(_, 0, 0, 0, ismem_write));
          set_barrier_transaction_bytes(shmab_readable[ismem_write], kTransactionBytes);

          ++ismem_write;
          if (ismem_write == kStageAB) {
            ismem_write = 0;
            phase ^= 1;
          }
        }  // for itile_k
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

    TiledCopyA tiled_copy_a;
    TiledCopyB tiled_copy_b;
    auto thr_copy_a = tiled_copy_a.get_slice(idx);
    auto thr_copy_b = tiled_copy_b.get_slice(idx);
    auto psA = as_position_independent_swizzle_tensor(sA);
    auto psB = as_position_independent_swizzle_tensor(sB);
    auto tAsA = thr_copy_a.partition_S(psA);
    auto tBsB = thr_copy_b.partition_S(psB);
    auto tArA = thr_copy_a.retile_D(tCrA);
    auto tBrB = thr_copy_b.retile_D(tCrB);

    constexpr int kNumM = size<1>(decltype(tCrA){});
    constexpr int kNumN = size<1>(decltype(tCrB){});
    constexpr int kWarpCols = kTileN / 4;
    static_assert(kNumN == 4 && kWarpCols == 32, "the epilogue transposes four 8-column blocks");
    int ilane = idx % 32;
    int warp_m = idx / 32 / 4;
    int warp_n = idx / 32 % 4;
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
      sfb_row[in] = warp_n * 32 + in * 8 + ilane / 4;
    }

    int istage_task = 0;
    int phase_task = 0;

    while (true) {
      wait_barrier(task_readable[istage_task], phase_task);
      int igroup = task_shm[istage_task][0];
      int itile_m = task_shm[istage_task][1];
      int itile_n = task_shm[istage_task][2];
      int itile_mg = task_shm[istage_task][3];
      int seqlen = task_shm[istage_task][4];
      if (igroup < 0) {
        break;
      }
      const int2 *rows = task_rows_shm[istage_task];
      clear(acc);

      int sfa_col = (itile_mg % kTilePerSFBlock) * (kTileM / 32) * 4;

      // Skip this warp's 16-row blocks past the valid rows.
      int num_valid_m = (seqlen - itile_m * kTileM - warp_m * 16 + 31) / 32;
      dispatch_num_valid_m<kNumM>(num_valid_m, [&](auto valid_m) {
        constexpr int kNumValidM = decltype(valid_m)::value;
        for (int itile_k = 0; itile_k < num_tile_k; itile_k++) {
          wait_barrier(shmab_readable[ismem_read], phase_shmab);

          // load sfa
          uint32_t sfa4[kNumM];
          uint32_t sfb4[kNumN];
#pragma unroll
          for (int im = 0; im < kNumValidM; im++) {
            sfa4[im] = *reinterpret_cast<const uint32_t *>(
                &sSFAr(sfa_row[im] % 32, sfa_row[im] / 32 * 4 + sfa_col, ismem_read));
          }

          // load sfb
#pragma unroll
          for (int in = 0; in < kNumN; in++) {
            sfb4[in] = *reinterpret_cast<const uint32_t *>(
                &sSFBr(sfb_row[in] % 32, sfb_row[in] / 32 * 4, 0, ismem_read));
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

      // epilogue: transpose 8 rows at a time, then each lane adds 16B of one row into y.
#pragma unroll
      for (int im = 0; im < kNumM; im++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          int2 row = rows[get<0>(tIC(2 * h, im, 0))];
          float row_scale = __int_as_float(row.y);
          uint32_t pairs[kNumN];
#pragma unroll
          for (int in = 0; in < kNumN; in++) {
            __nv_bfloat162 v = __floats2bfloat162_rn(acc(2 * h, im, in) * row_scale,
                                                     acc(2 * h + 1, im, in) * row_scale);
            pairs[in] = *reinterpret_cast<uint32_t *>(&v);
          }
          uint4 block = epi_transpose(pairs, epi_shm[idx / 32]);
          if (row.x >= 0) {
            int col = warp_n * kWarpCols + ilane % 4 * 8;
            red_add_v4_bf16x2(y_ptr + static_cast<int64_t>(row.x) * n + itile_n * kTileN + col,
                              block);
          }
        }
      }

      // done with the task's rows
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
