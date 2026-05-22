// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "src/gemm/gemm.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace gemm {
namespace kernels {

template <int kElementsPerThread, int TM, int TN>
__global__ void pad_and_transpose(float *new_scale_ptr, const float *scale_ptr, int m, int n,
                                  int m_pad, int num_tile_n) {
  int iblock = blockIdx.x;
  int idx = threadIdx.x;

  int start_row = (iblock / num_tile_n) * TM;
  int start_col = (iblock % num_tile_n) * TN;

  constexpr int kVecsPerRow = TN / kElementsPerThread;
  __shared__ float shm_tile[TM * TN];

  int local_row = idx / kVecsPerRow;
  int local_col = idx % kVecsPerRow * kElementsPerThread;

  int src_global_row = start_row + local_row;
  int src_global_col = start_col + local_col;

  // g -> s
  if (src_global_row < m && src_global_col < n) {
    auto r = load<float, kElementsPerThread>(scale_ptr + src_global_row * n + src_global_col);
    store(shm_tile + local_row * TN + local_col, r);
  }

  __syncthreads();

  // s -> g
  int v_idx = idx * 4;
  local_row = v_idx % TM;
  local_col = v_idx / TM;

  int dst_global_row = start_col + local_col;
  int dst_global_col = start_row + local_row;

  if (dst_global_row < n && dst_global_col < m) {
    // printf("thread: %d, dst_global_row: %d, dst_global_col: %d\n", idx, dst_global_row,
    // dst_global_col);
    vec_t<float, 4> r;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      r[i] = shm_tile[(local_row + i) * TN + local_col];
    }
    store(new_scale_ptr + dst_global_row * m_pad + dst_global_col, r);
  }
}

template <int kBlockSwizzle, int kSplitK>
__device__ __forceinline__ auto get_next_tile(int iblock, int num_tile_m, int num_tile_n,
                                              cutlass::FastDivmod swizzle_divider,
                                              cutlass::FastDivmod flat_divider) {
  int itile_m, itile_n;
  int num_tile_bxn = kBlockSwizzle * num_tile_n * kSplitK;
  int total_sizzle_blocks = num_tile_m / kBlockSwizzle * num_tile_bxn;

  if (iblock >= total_sizzle_blocks) {
    flat_divider(itile_m, itile_n, iblock);
  } else {
    int i_bxn, i_bxn_res;
    swizzle_divider(i_bxn, i_bxn_res, iblock);

    itile_m = i_bxn * kBlockSwizzle + i_bxn_res % kBlockSwizzle;
    itile_n = i_bxn_res / kBlockSwizzle;
  }

  int ichunk = itile_n % kSplitK;
  itile_n = itile_n / kSplitK;

  return cute::make_tuple(itile_m, itile_n, ichunk);
}

template <typename Tout, int kTileM, int kTileN, int kSplitK, int kWarpCount, bool HasBias>
__device__ __forceinline__ void splitk_reduce(Tout *y_ptr, float *splitk_y_ptr,
                                              const float *bias_ptr, int m, int n, int itile_m,
                                              int itile_n) {
  int iwarp = threadIdx.x / 32;
  int ilane = threadIdx.x % 32;

  constexpr int kN = 4;
  int row = itile_m * kTileM;
  int col = itile_n * kTileN;

  auto *y_tile = y_ptr + row * n + col;
  auto *splitk_y_tile = splitk_y_ptr + row * n + col;

#pragma unroll
  for (int irow = iwarp; irow < kTileM; irow += kWarpCount) {
    if (row + irow >= m) {
      return;
    }

    int icol = ilane * kN;
    if (col + icol >= n) {
      return;
    }

    auto y = load<float, kN>(splitk_y_tile + irow * n + icol);
    if constexpr (HasBias) {
      auto bias = load<float, kN>(bias_ptr + col + icol);
      for (int i = 0; i < kN; i++) {
        y[i] += bias[i];
      }
    }

#pragma unroll
    for (int ichunk = 1; ichunk < kSplitK; ++ichunk) {
      auto split_y = load<float, kN>(splitk_y_tile + ichunk * m * n + irow * n + icol);
#pragma unroll
      for (int i = 0; i < kN; i++) {
        y[i] += split_y[i];
      }
    }
    store(y_tile + irow * n + icol, to<Tout>(y));
  }
}

template <typename TiledMma, typename TmaA, typename TmaB, typename TmaAS, typename TmaBS,
          typename TmaBias, typename TmaD, int kTileM, int kTileN, int kStage, typename Tin,
          typename TY, typename Tout, typename SLayoutA, typename SLayoutB, typename SLayoutAS,
          typename SLayoutBS, typename SLayoutBIAS, typename SLayoutCT, int kBlockSwizzle,
          int kSplitK, bool HasBias>
__global__ void __launch_bounds__(384, 1)
    gemm_blockwise_fp8(const __grid_constant__ TmaA tma_a, const __grid_constant__ TmaB tma_b,
                       const __grid_constant__ TmaAS tma_as, const __grid_constant__ TmaBS tma_bs,
                       const __grid_constant__ TmaBias tma_bias, const __grid_constant__ TmaD tma_d,
                       Tout *y_ptr, float *splitk_y_ptr, int *split_flag_ptr, const float *bias_ptr,
                       int m, int n, int k, int m_pad, int num_block_n, int num_block_k,
                       cutlass::FastDivmod swizzle_divider, cutlass::FastDivmod flat_divider,
                       cutlass::FastDivmod reduce_flat_divider) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;

  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  int elected = cute::elect_one_sync();
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  __shared__ uint64_t writable[kStage];
  __shared__ uint64_t readable[kStage];
  __shared__ uint64_t bias_writable;
  __shared__ uint64_t bias_readable;

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_a = (Tin *)shm_data;
  auto *shm_b = (Tin *)(shm_a + cosize(SLayoutA{}));
  auto *shm_c = (TY *)(shm_b + cosize(SLayoutB{}));
  auto *shm_as = (float *)(shm_c + cosize(SLayoutCT{}));
  auto *shm_bs = (float *)(shm_as + cosize(SLayoutAS{}));
  auto *shm_bias = (float *)(shm_bs + cosize(SLayoutBS{}));

  auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
  auto sAS = make_tensor(make_smem_ptr(shm_as), SLayoutAS{});
  auto sBS = make_tensor(make_smem_ptr(shm_bs), SLayoutBS{});
  auto sBIAS = make_tensor(make_smem_ptr(shm_bias), SLayoutBIAS{});

  auto gA = tma_a.get_tma_tensor(make_shape(m, k));
  auto gB = tma_b.get_tma_tensor(make_shape(n, k));
  auto gC = make_tensor(make_gmem_ptr((Tout *)(nullptr)), make_shape(Int<kTileN>{}, Int<kTileM>{}),
                        make_stride(Int<kTileM>{}, Int<1>{}));
  auto gAS = tma_as.get_tma_tensor(make_shape(num_block_k, m_pad));
  auto gBS = tma_bs.get_tma_tensor(make_shape(num_block_n, num_block_k));
  auto gBIAS = tma_bias.get_tma_tensor(make_shape(1, n));

  auto btma_a = tma_a.get_slice(0);
  auto btma_b = tma_b.get_slice(0);
  auto btma_as = tma_as.get_slice(0);
  auto btma_bs = tma_bs.get_slice(0);
  auto btma_bias = tma_bias.get_slice(0);

  auto tAg = btma_a.partition_S(gA);  // (cpbox_x, num_tile_m, num_tile_k)
  auto tAs = btma_a.partition_D(sA);  // (cpbox_x, _1, _1, kStage)

  auto tBg = btma_b.partition_S(gB);  // (cpbox_w, num_tile_n, num_tile_k)
  auto tBs = btma_b.partition_D(sB);  // (cpbox_w, _1, _1, kStage)

  auto tASg = btma_as.partition_S(gAS);  // (cpbox_xs, num_tile_k, num_tile_m)
  auto tASs = btma_as.partition_D(sAS);  // (cpbox_xs, kStage, _1)

  auto tBSg = btma_bs.partition_S(gBS);  // (cpbox_ws, num_tile_n, num_tile_k / 4)
  auto tBSs = btma_bs.partition_D(sBS);  // (cpbox_ws, kStage, 32 / 4)

  auto tBIASg = btma_bias.partition_S(gBIAS);  // (TMA, num_tile_n)
  auto tBIASs = btma_bias.partition_D(sBIAS);  // (TMA, _1)

  int num_tile_m = size<1>(tAg);
  int num_tile_n = size<1>(tBg);

  if (is_leader_in_block) {
    if constexpr (HasBias && kSplitK == 1) {
      initialize_barrier(bias_readable, 1);
      initialize_barrier(bias_writable, size(TiledMma{}) / 128);
    }
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable[i], 1);
      initialize_barrier(writable[i], size(TiledMma{}) / 128);
    }
  }
  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  constexpr int kNumThreads = size(TiledMma{});
  // load warpgroup
  if (idx >= kNumThreads) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= kNumThreads;
    constexpr int kTransactionBytes = sizeof(Tin) * cosize(SLayoutA{}(_, _, 0)) +
                                      sizeof(Tin) * cosize(SLayoutB{}(_, _, 0)) +
                                      sizeof(float) * cosize(SLayoutAS{}(0, _)) + sizeof(float) * 4;
    constexpr int kBiasTransactionBytes = sizeof(float) * cosize(SLayoutBIAS{});

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;       // start with ok
      int phase_bias = 1;  // start with ok
      // int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);
      int ismem_write = 0;
      int iblock = blockIdx.x;
      int num_tile_k = size<2>(tAg);

      while (true) {
        auto [itile_m, itile_n, ichunk] = get_next_tile<kBlockSwizzle, kSplitK>(
            iblock, num_tile_m, num_tile_n, swizzle_divider, flat_divider);

        if (itile_m >= num_tile_m) {
          break;
        }
        iblock += gridDim.x;

        // load bias
        if constexpr (HasBias && kSplitK == 1) {
          wait_barrier(bias_writable, phase_bias);
          set_barrier_transaction_bytes(bias_readable, kBiasTransactionBytes);
          cute::copy(tma_bias.with(bias_readable), tBIASg(_, 0, itile_n), tBIASs(_, 0, 0));
          phase_bias ^= 1;
        }

#pragma unroll 1
        for (int itile_k = ichunk; itile_k < num_tile_k; itile_k += kSplitK) {
          // load a, b
          wait_barrier(writable[ismem_write], phase);
          set_barrier_transaction_bytes(readable[ismem_write], kTransactionBytes);
          cute::copy(tma_a.with(readable[ismem_write]), tAg(_, itile_m, itile_k),
                     tAs(_, 0, 0, ismem_write));
          cute::copy(tma_b.with(readable[ismem_write]), tBg(_, itile_n, itile_k),
                     tBs(_, 0, 0, ismem_write));
          cute::copy(tma_as.with(readable[ismem_write]), tASg(_, itile_k, itile_m),
                     tASs(_, ismem_write, 0));
          cute::copy(tma_bs.with(readable[ismem_write]), tBSg(_, itile_n, itile_k / 4),
                     tBSs(_, ismem_write, 0));
          ++ismem_write;
          if (ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }  // ntile_todo
      }  // while
    }  // if idx == 0

  } else {
    // math warpgroup
    cutlass::arch::warpgroup_reg_alloc<168>();

    int idx_in_warpgroup = idx % 128;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    TiledMma tiled_mma;

    auto thr_mma = tiled_mma.get_slice(idx);
    auto tBs4r = thr_mma.partition_A(sB);
    auto tAs4r = thr_mma.partition_B(sA);

    auto tBr = thr_mma.make_fragment_A(tBs4r);  // (MMA, MMA_N, MMA_K, kStage)
    auto tAr = thr_mma.make_fragment_B(tAs4r);  // (MMA, MMA_M, MMA_K, kStage)

    auto tCr = thr_mma.partition_fragment_C(gC);  // (MMA, MMA_N, MMA_M)

    auto tCr_mn = retile_fragment(tCr);
    constexpr int kM = size<0>(tCr_mn);
    constexpr int kN = size<1>(tCr_mn);

    auto gI = make_identity_tensor(gC.shape());
    auto tI = thr_mma.partition_C(gI);
    auto tI_mn = retile_fragment(tI);

    int ismem_read = 0;
    int phase = 0;
    int phase_bias = 0;

    int iblock = blockIdx.x;
    int last_tile_m = -1;
    int last_tile_n = -1;
    while (true) {
      auto [itile_m, itile_n, ichunk] = get_next_tile<kBlockSwizzle, kSplitK>(
          iblock, num_tile_m, num_tile_n, swizzle_divider, flat_divider);

      if (itile_m >= num_tile_m) {
        break;
      }
      iblock += gridDim.x;

      auto tDr = make_tensor_like(tCr);
      clear(tDr);

      int num_tile_k = size<2>(tAg);
#pragma unroll 1
      for (int itile_k = ichunk; itile_k < num_tile_k; itile_k += kSplitK) {
        wait_barrier(readable[ismem_read], phase);

        float tCS[kN];
        float wscale = sBS(ismem_read, itile_k % 4);
#pragma unroll
        for (int in = 0; in < kN; in++) {
          float xscale = sAS(ismem_read, get<1>(tI_mn(0, in)));
          tCS[in] = xscale * wscale;
        }

        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        // mma
        warpgroup_fence_operand(tCr);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tAr); ++ik) {
          cute::gemm(tiled_mma, tBr(_, _, ik, ismem_read), tAr(_, _, ik, ismem_read), tCr(_, _, _));
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCr);

        auto tDr_mn = retile_fragment(tDr);
#pragma unroll
        for (int in = 0; in < kN; in++) {
          float yscale = tCS[in];
#pragma unroll
          for (int im = 0; im < kM; im++) {
            tDr_mn(im, in) += tCr_mn(im, in) * yscale;
          }
        }

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable[ismem_read]);
        }

        ++ismem_read;
        if (ismem_read == kStage) {
          phase ^= 1;
          ismem_read = 0;
        }
      }  // for end
      // add bias
      if constexpr (HasBias && kSplitK == 1) {
        wait_barrier(bias_readable, phase_bias);
        auto tDr_mn = retile_fragment(tDr);
#pragma unroll
        for (int im = 0; im < kM; im++) {
          auto bias = sBIAS(get<0>(tI_mn(im, 0)));
#pragma unroll
          for (int in = 0; in < kN; in++) {
            tDr_mn(im, in) += bias;
          }
        }
      }

      // float32 -> bfloat16
      auto tCrh = make_tensor_like<TY>(tCr);

#pragma unroll
      for (int i = 0; i < size(tCr); ++i) {
        tCrh(i) = (TY)(tDr(i));
      }

      // Epilogue
      auto sCT = make_tensor(make_smem_ptr((TY *)shm_c), SLayoutCT{});  // (TN, TM):(1, TN)
      using STSM_ATOM =
          std::conditional_t<(kTileM >= 16), cute::SM90_U16x8_STSM_T, cute::SM90_U16x4_STSM_T>;
      using STS_ATOM =
          std::conditional_t<std::is_same_v<TY, float>, UniversalCopy<uint32_t>, STSM_ATOM>;
      using R2SCopyAtomY = Copy_Atom<STS_ATOM, TY>;
      auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomY{}, tiled_mma);
      auto thr_copy_c = tiled_copy_c.get_slice(idx);

      auto tCr4s = thr_copy_c.retile_S(tCrh);
      auto tCs4r = thr_copy_c.partition_D(sCT);

      cute::tma_store_wait<0>();
      syncwarpgroup(iwarpgroup);

      // bias arrive must wait all thread in warp have handled bias
      if constexpr (HasBias && kSplitK == 1) {
        if (elected_idx_in_warpgroup) {
          arrive_barrier(bias_writable);
        }
        phase_bias ^= 1;
      }

      if constexpr (kSplitK > 1) {
        if (is_leader_in_warpgroup) {
          if (last_tile_m != -1 && last_tile_n != -1) {
            auto *split_flag = split_flag_ptr + last_tile_m * num_tile_n + last_tile_n;
            atomicAdd(split_flag, 1);
          }
          last_tile_m = itile_m;
          last_tile_n = itile_n;
        }
      }

      cute::copy(tiled_copy_c, tCr4s, tCs4r);
      syncwarpgroup(iwarpgroup);
      cute::tma_store_fence();

      if (is_leader_in_warpgroup) {
        auto gD = tma_d.get_tma_tensor(make_shape(n, m, kSplitK));
        auto btma_d = tma_d.get_slice(0);

        auto tDs = btma_d.partition_S(sCT);  // (TMA, _2, _1)
        auto tDg = btma_d.partition_D(gD);   // (TMA, TMA_N, TMA_M, kSplitK)

        cute::copy(tma_d, tDs(_, iwarpgroup, Int<0>{}),
                   tDg(_, itile_n * 2 + iwarpgroup, itile_m, ichunk));
        tma_store_arrive();
      }
    }  // while
    if constexpr (kSplitK > 1) {
      cute::tma_store_wait<0>();

      fence_async_global();
      __threadfence();
      syncwarpgroup(iwarpgroup);

      if (is_leader_in_warpgroup) {
        if (last_tile_m != -1 && last_tile_n != -1) {
          auto *split_flag = split_flag_ptr + last_tile_m * num_tile_n + last_tile_n;
          atomicAdd(split_flag, 1);
        }
      }

      bar_sync<128 * 2>(2);

      iblock = blockIdx.x;
      __threadfence();
      while (true) {
        int itile_m, itile_n;
        reduce_flat_divider(itile_m, itile_n, iblock);

        if (itile_m >= num_tile_m) {
          break;
        }
        iblock += gridDim.x;
        auto *split_flag = split_flag_ptr + itile_m * num_tile_n + itile_n;
        while (load_global_volatile(split_flag) != kSplitK * 2) {
        }
        using NVTout = __nv_bfloat16;
        splitk_reduce<NVTout, kTileM, kTileN, kSplitK, 256 / 32, HasBias>(
            reinterpret_cast<NVTout *>(y_ptr), splitk_y_ptr, bias_ptr, m, n, itile_m, itile_n);
      }
    }  // if constexpr (kSplitK > 1)
  }  // else
}

}  // namespace kernels

template <int kTileM>
static constexpr auto mma_selector() {
  if constexpr (kTileM == 8) {
    return cute::SM90_64x8x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 16) {
    return cute::SM90_64x16x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 32) {
    return cute::SM90_64x32x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 48) {
    return cute::SM90_64x48x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 64) {
    return cute::SM90_64x64x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 96) {
    return cute::SM90_64x96x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 128) {
    return cute::SM90_64x128x32_F32E4M3E4M3_SS_TN<>{};
  } else {
    return cute::SM90_64x64x32_F32E4M3E4M3_SS_TN<>{};
  }
}

template <int kTileM, int kTileN, int kTileK, int kStage, int kBlockSwizzle, int kSplitK,
          bool HasBias>
void launch_gemm_blockwise_fp8(void *y_ptr, void *split_y_ptr, void *split_flag_ptr,
                               const void *x_ptr, const void *w_ptr, const void *x_scale_ptr,
                               const void *weight_scale_ptr, const void *bias_ptr, int m, int n,
                               int k, int m_pad, int num_block_k, int num_block_n,
                               cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;
  using TS = float;
  using TBIAS = float;
  using TY = std::conditional_t<(kSplitK > 1), float, Tout>;

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto W = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)), make_shape(n, k),
                       make_stride(k, Int<1>{}));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<TY *>(kSplitK > 1 ? split_y_ptr : y_ptr)),
                       make_shape(n, m, kSplitK), make_stride(Int<1>{}, n, n * m));
  auto XS = make_tensor(make_gmem_ptr(reinterpret_cast<const TS *>(x_scale_ptr)),
                        make_shape(num_block_k, m_pad), make_stride(m_pad, Int<1>{}));
  auto WS = make_tensor(make_gmem_ptr(reinterpret_cast<const TS *>(weight_scale_ptr)),
                        make_shape(num_block_n, num_block_k), make_stride(num_block_k, Int<1>{}));
  auto BIAS = make_tensor(make_gmem_ptr(reinterpret_cast<const TBIAS *>(bias_ptr)),
                          make_shape(1, n), make_stride(n, Int<1>{}));

  auto slayout_x = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_w = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{}));
  static constexpr int kMDim = (kTileM + 31) / 32 * 32;
  auto slayout_xs = make_layout(
      make_shape(Int<kStage>{}, Int<kMDim>{}),  // shared memory address must be 128 byte aligned
      make_stride(Int<kMDim>{}, Int<1>{}));
  auto slayout_ws = make_layout(
      make_shape(Int<kStage>{}, Int<32>{}),
      make_stride(Int<32>{}, Int<1>{}));  // shared memory address must be 128 byte aligned
  auto slayout_bias =
      make_layout(make_shape(Int<1>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto slayout_yt =
      tile_to_shape(GMMA::Layout_MN_SW64_Atom<TY>{}, make_shape(Int<kTileN>{}, Int<kTileM>{}));

  auto cpbox_xs =
      make_layout(make_shape(Int<1>{}, Int<kMDim>{}), make_stride(Int<kMDim>{}, Int<1>{}));
  auto cpbox_ws = make_layout(
      make_shape(Int<1>{}, Int<4>{}),
      make_stride(Int<4>{}, Int<1>{}));  // size of transfer must be a multiple of 16 bytes.
  auto cpbox_bias =
      make_layout(make_shape(Int<1>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto cpbox_yt =
      tile_to_shape(GMMA::Layout_MN_SW64_Atom<TY>{}, make_shape(Int<kTileN / 2>{}, Int<kTileM>{}));

  static constexpr int shm_xw = sizeof(Tin) * (cosize(slayout_x) + cosize(slayout_w));
  static constexpr int shm_y = sizeof(TY) * cosize(slayout_yt);
  static constexpr int shm_xws = (cosize(slayout_xs) + cosize(slayout_ws)) * sizeof(TS);
  static constexpr int shm_bias = HasBias ? cosize(slayout_bias) * sizeof(TBIAS) : 0;
  static constexpr int shm_size = shm_xw + shm_y + shm_xws + shm_bias;

  auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, X, slayout_x(_, _, 0));
  auto tma_w = make_tma_copy(SM90_TMA_LOAD{}, W, slayout_w(_, _, 0));
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, cpbox_yt);
  auto tma_xs = make_tma_copy(SM90_TMA_LOAD{}, XS, cpbox_xs);
  auto tma_ws = make_tma_copy(SM90_TMA_LOAD{}, WS, cpbox_ws);
  auto tma_bias = make_tma_copy(SM90_TMA_LOAD{}, BIAS, cpbox_bias);

  auto warpgroup_layout = make_layout(make_shape(Int<2>{}, Int<1>{}, Int<1>{}));
  auto tiled_mma = make_tiled_mma(mma_selector<kTileM>(), warpgroup_layout);

  auto kernel = kernels::gemm_blockwise_fp8<
      decltype(tiled_mma), decltype(tma_x), decltype(tma_w), decltype(tma_xs), decltype(tma_ws),
      decltype(tma_bias), decltype(tma_y), kTileM, kTileN, kStage, Tin, TY, Tout,
      decltype(slayout_x), decltype(slayout_w), decltype(slayout_xs), decltype(slayout_ws),
      decltype(slayout_bias), decltype(slayout_yt), kBlockSwizzle, kSplitK, HasBias>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  int num_tile_m = (m + kTileM - 1) / kTileM;
  int num_tile_n = (n + kTileN - 1) / kTileN * kSplitK;
  int num_tile = num_tile_m * num_tile_n;
  int num_tile_bxn = kBlockSwizzle * num_tile_n;
  cutlass::FastDivmod swizzle_divider(num_tile_bxn);
  cutlass::FastDivmod flat_divider(num_tile_n);
  cutlass::FastDivmod reduce_flat_divider(num_tile_n / kSplitK);

  dim3 block(size(tiled_mma) + 128);
  dim3 grid(std::min(get_sm_count(), num_tile));

  kernel<<<grid, block, shm_size, stream>>>(
      tma_x, tma_w, tma_xs, tma_ws, tma_bias, tma_y, reinterpret_cast<Tout *>(y_ptr),
      reinterpret_cast<float *>(split_y_ptr), reinterpret_cast<int *>(split_flag_ptr),
      reinterpret_cast<const float *>(bias_ptr), m, n, k, m_pad, num_block_n, num_block_k,
      swizzle_divider, flat_divider, reduce_flat_divider);
}

#define LAUNCH_GEMM_INTERNAL(TILE_M_VAL, SPLIT_K_VAL)                                           \
  do {                                                                                          \
    constexpr int kTileM = TILE_M_VAL;                                                          \
    constexpr int kSplitK = SPLIT_K_VAL;                                                        \
    if (bias_ptr) {                                                                             \
      launch_gemm_blockwise_fp8<kTileM, kTileN, kTileK, kStage, kBlockSwizzle, kSplitK, true>(  \
          y_ptr, split_y_ptr, split_flag_ptr, x_ptr, w_ptr, x_scale_ptr, weight_scale_ptr,      \
          bias_ptr, m, n, k, m_pad, num_block_k, num_block_n, stream);                          \
    } else {                                                                                    \
      launch_gemm_blockwise_fp8<kTileM, kTileN, kTileK, kStage, kBlockSwizzle, kSplitK, false>( \
          y_ptr, split_y_ptr, split_flag_ptr, x_ptr, w_ptr, x_scale_ptr, weight_scale_ptr,      \
          bias_ptr, m, n, k, m_pad, num_block_k, num_block_n, stream);                          \
    }                                                                                           \
  } while (0)

#define DISPATCH_SPLIT_K(TM)     \
  if (splitk == 8) {             \
    LAUNCH_GEMM_INTERNAL(TM, 8); \
  } else if (splitk == 4) {      \
    LAUNCH_GEMM_INTERNAL(TM, 4); \
  } else if (splitk == 2) {      \
    LAUNCH_GEMM_INTERNAL(TM, 2); \
  } else {                       \
    LAUNCH_GEMM_INTERNAL(TM, 1); \
  }

bool gemm_blockwise_fp8_async(void *y_ptr, void *split_y_ptr, void *split_flag_ptr,
                              const void *x_ptr, const void *w_ptr, const void *x_scale_ptr,
                              const void *weight_scale_ptr, const void *bias_ptr, int m, int n,
                              int k, int m_pad, int num_block_k, int num_block_n, int splitk,
                              cudaStream_t stream) {
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;
  constexpr int kStage = 6;
  constexpr int kBlockSwizzle = 4;

  if (m <= 8) {
    DISPATCH_SPLIT_K(8);
  } else if (m <= 16) {
    DISPATCH_SPLIT_K(16);
  } else if (m <= 32) {
    DISPATCH_SPLIT_K(32);
  } else if (m <= 48) {
    DISPATCH_SPLIT_K(48);
  } else {
    DISPATCH_SPLIT_K(64);
  }

  return true;
}

bool pad_and_transpose_async(void *new_scale_ptr, const void *scale_ptr, int m, int n, int m_pad,
                             cudaStream_t stream) {
  constexpr int TM = 32;
  constexpr int TN = 32;
  int num_tile_m = (m + 31) / 32;
  int num_tile_n = (n + 31) / 32;
  int num_block = num_tile_m * num_tile_n;
  dim3 grid(num_block);

  if (n % 4 == 0) {
    dim3 block(256);
    constexpr int kElementsPerThread = 4;
    kernels::pad_and_transpose<kElementsPerThread, TM, TN><<<grid, block, 0, stream>>>(
        reinterpret_cast<float *>(new_scale_ptr), reinterpret_cast<const float *>(scale_ptr), m, n,
        m_pad, num_tile_n);
  } else if (n % 2 == 0) {
    dim3 block(512);
    constexpr int kElementsPerThread = 2;
    kernels::pad_and_transpose<kElementsPerThread, TM, TN><<<grid, block, 0, stream>>>(
        reinterpret_cast<float *>(new_scale_ptr), reinterpret_cast<const float *>(scale_ptr), m, n,
        m_pad, num_tile_n);
  } else {
    dim3 block(1024);
    constexpr int kElementsPerThread = 1;
    kernels::pad_and_transpose<kElementsPerThread, TM, TN><<<grid, block, 0, stream>>>(
        reinterpret_cast<float *>(new_scale_ptr), reinterpret_cast<const float *>(scale_ptr), m, n,
        m_pad, num_tile_n);
  }
  return true;
}

#undef LAUNCH_GEMM_INTERNAL
#undef DISPATCH_SPLIT_K
}  // namespace gemm
}  // namespace hpc
