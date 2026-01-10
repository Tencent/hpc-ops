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

template <int kNumWarpsPerCTA, int kElementsPerThread, int TM, int TN>
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
    auto r = load<float, 4>(scale_ptr + src_global_row * n + src_global_col);
    store(shm_tile + local_row * TN + local_col, r);
  }

  __syncthreads();

  // s -> g
  local_row = idx / TN * kElementsPerThread;
  local_col = idx % TN;

  int dst_global_row = start_col + local_col;
  int dst_global_col = start_row + local_row;

  if (dst_global_row < n && dst_global_col < m) {
    vec_t<float, kElementsPerThread> r;
#pragma unroll
    for (int i = 0; i < kElementsPerThread; ++i) {
      r[i] = shm_tile[(local_row + i) * TN + local_col];
    }
    store(new_scale_ptr + dst_global_row * m_pad + dst_global_col, r);
  }
}

template <int kBlockSwizzle>
__device__ __forceinline__ auto get_next_tile(int iblock, int num_tile_m, int num_tile_n,
                                              cutlass::FastDivmod swizzle_divider,
                                              cutlass::FastDivmod flat_divider) {
  int itile_m, itile_n;
  int num_tile_bxn = kBlockSwizzle * num_tile_n;
  int total_sizzle_blocks = num_tile_m / kBlockSwizzle * num_tile_bxn;

  if (iblock >= total_sizzle_blocks) {
    flat_divider(itile_m, itile_n, iblock);
  } else {
    int i_bxn, i_bxn_res;
    swizzle_divider(i_bxn, i_bxn_res, iblock);

    itile_m = i_bxn * kBlockSwizzle + i_bxn_res % kBlockSwizzle;
    itile_n = i_bxn_res / kBlockSwizzle;
  }

  return cute::make_tuple(itile_m, itile_n);
}

template <typename TiledMma, typename TmaA, typename TmaB, typename TmaAS, typename TmaBS,
          typename TmaBIAS, typename TmaD, int kTileM, int kTileN, int kStage, int kTileS,
          typename Tin, typename Tout, typename SLayoutA, typename SLayoutB, typename SLayoutAS,
          typename SLayoutBS, typename SLayoutBIAS, typename SLayoutC, int kBlockSwizzle>
__global__ void gemm_blockwise(
    const __grid_constant__ TmaA tma_a, const __grid_constant__ TmaB tma_b,
    const __grid_constant__ TmaAS tma_as, const __grid_constant__ TmaBS tma_bs,
    const __grid_constant__ TmaBIAS tma_bias, const __grid_constant__ TmaD tma_d, int m, int n,
    int k, int m_pad, int num_block_n, int num_block_k, cutlass::FastDivmod swizzle_divider,
    cutlass::FastDivmod flat_divider) {
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
  auto *shm_c = (Tout *)(shm_b + cosize(SLayoutB{}));
  auto *shm_as = (float *)(shm_c + cosize(SLayoutC{}));
  auto *shm_bs = (float *)(shm_as + cosize(SLayoutAS{}));
  auto *shm_bias = (float *)(shm_bs + cosize(SLayoutBS{}));

  auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
  auto sAS = make_tensor(make_smem_ptr(shm_as), SLayoutAS{});
  auto sBS = make_tensor(make_smem_ptr(shm_bs), SLayoutBS{});
  auto sBIAS = make_tensor(make_smem_ptr(shm_bias), SLayoutBIAS{});

  auto gA = tma_a.get_tma_tensor(make_shape(m, k));
  auto gB = tma_b.get_tma_tensor(make_shape(n, k));
  auto gC =
      make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gAS = tma_as.get_tma_tensor(make_shape(num_block_k, m_pad));
  auto gBS = tma_bs.get_tma_tensor(make_shape(num_block_n, num_block_k));
  auto gBIAS = tma_bias.get_tma_tensor(make_shape(1, n));

  auto btma_a = tma_a.get_slice(0);
  auto btma_b = tma_b.get_slice(0);
  auto btma_as = tma_as.get_slice(0);
  auto btma_bs = tma_bs.get_slice(0);
  auto btma_bias = tma_bias.get_slice(0);

  auto tAg = btma_a.partition_S(gA);  // (TMA, TMA_M, TMA_K)
  auto tAs = btma_a.partition_D(sA);  // (TMA, _1, _1, kStage)

  auto tBg = btma_b.partition_S(gB);  // (TMA, TMA_N, TMA_K)
  auto tBs = btma_b.partition_D(sB);  // (TMA, -1, -1, stage)

  auto tASg = btma_as.partition_S(gAS);  // (TMA, TMA_k, TMA_M)
  auto tASs = btma_as.partition_D(sAS);  // (TMA, kStage, _1)

  auto tBSg = btma_bs.partition_S(gBS);  // (TMA, TMA_N, TMA_K)
  auto tBSs = btma_bs.partition_D(sBS);  // (TMA, _1, TMA_K)

  auto tBIASg = btma_bias.partition_S(gBIAS);  // (TMA, TMA_N, TMA_K)
  auto tBIASs = btma_bias.partition_D(sBIAS);  // (TMA, _1, )

  int num_tile_m = size<1>(tAg);
  int num_tile_n = size<1>(tBg);

  if (is_leader_in_block) {
    initialize_barrier(bias_readable, 1);
    initialize_barrier(bias_writable, size(TiledMma{}) / 128);
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable[i], 1);
      initialize_barrier(writable[i], size(TiledMma{}) / 128);
    }
  }
  __syncthreads();

  // load warpgroup
  if (idx >= 256) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    idx -= 256;
    constexpr int kTransactionBytes =
        sizeof(Tin) * cosize(SLayoutA{}(_, _, 0)) + sizeof(Tin) * cosize(SLayoutB{}(_, _, 0)) +
        sizeof(float) * cosize(SLayoutAS{}(0, _)) + sizeof(float) * kTileS;
    constexpr int kBiasTransactionBytes = sizeof(float) * cosize(SLayoutBIAS{});

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;  // start with ok
      int phase_bias = 1;
      int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);
      int iblock = blockIdx.x;
      int ntile_k = size<2>(tAg);

      while (true) {
        auto [itile_m, itile_n] = get_next_tile<kBlockSwizzle>(iblock, num_tile_m, num_tile_n,
                                                               swizzle_divider, flat_divider);

        if (itile_m >= num_tile_m) {
          break;
        }
        iblock += gridDim.x;

        // load bias
        wait_barrier(bias_writable, phase_bias);
        cute::copy(tma_bias.with(bias_readable), tBIASg(_, 0, itile_n), tBIASs(_, 0, 0));
        set_barrier_transaction_bytes(bias_readable, kBiasTransactionBytes);
        phase_bias ^= 1;

#pragma unroll 1
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          // load a, b
          wait_barrier(writable[ismem_write], phase);

          cute::copy(tma_a.with(readable[ismem_write]), tAg(_, itile_m, itile_k),
                     tAs(_, 0, 0, ismem_write));
          cute::copy(tma_b.with(readable[ismem_write]), tBg(_, itile_n, itile_k),
                     tBs(_, 0, 0, ismem_write));
          cute::copy(tma_as.with(readable[ismem_write]), tASg(_, itile_k, itile_m),
                     tASs(_, ismem_write, 0));
          cute::copy(tma_bs.with(readable[ismem_write]), tBSg(_, itile_n, itile_k / 4),
                     tBSs(_, ismem_write, 0));
          set_barrier_transaction_bytes(readable[ismem_write], kTransactionBytes);

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
    auto tAs4r = thr_mma.partition_A(sA);
    auto tBs4r = thr_mma.partition_B(sB);

    auto tAr = thr_mma.make_fragment_A(tAs4r);  // (MMA, MMA_M, MMA_K, kStage)
    auto tBr = thr_mma.make_fragment_B(tBs4r);  // (MMA, MMA_N, MMA_K, kStage)

    auto tCr = thr_mma.partition_fragment_C(gC);  // (MMA, MMA_N, MMA_M) ((_2, _2, _16), _1, _1)
    auto tCr_mn = retile_fragment(tCr);           // (_2, (_2, _16))

    constexpr int kM = size<0>(tCr_mn);
    constexpr int kN = size<1>(tCr_mn);

    auto gI = make_identity_tensor(gC.shape());
    auto tI = thr_mma.partition_C(gI);
    auto tI_mn = retile_fragment(tI);  // (_2, (_2, _16))

    int ismem_read = 0;
    int phase = 0;
    int phase_bias = 0;

    int iblock = blockIdx.x;
    while (true) {
      auto [itile_m, itile_n] = get_next_tile<kBlockSwizzle>(iblock, num_tile_m, num_tile_n,
                                                             swizzle_divider, flat_divider);
      if (itile_m >= num_tile_m) {
        break;
      }

      iblock += gridDim.x;

      auto tDr = make_tensor_like(tCr);
      clear(tDr);

      int ntile_k = size<2>(tAg);  // tAg: (TMA, TMA_M, TMA_K)
#pragma unroll 1
      for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
        wait_barrier(readable[ismem_read], phase);

        float tCS[kM];
        float wscale = sBS(ismem_read, itile_k % 4);
#pragma unroll
        for (int im = 0; im < kM; im++) {
          tCS[im] = sAS(ismem_read, get<0>(tI_mn(im, 0))) * wscale;
        }

        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        // mma
        warpgroup_fence_operand(tCr);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tAr); ++ik) {
          cute::gemm(tiled_mma, tAr(_, _, ik, ismem_read), tBr(_, _, ik, ismem_read), tCr(_, _, _));
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCr);

        auto tDr_mn = retile_fragment(tDr);
#pragma unroll
        for (int im = 0; im < kM; im++) {
          float yscale = tCS[im];
#pragma unroll
          for (int in = 0; in < kN; in++) {
            tDr_mn(im, in) = tCr_mn(im, in) * yscale + tDr_mn(im, in);
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
      wait_barrier(bias_readable, phase_bias);
      auto tDr_mn = retile_fragment(tDr);
#pragma unroll
      for (int in = 0; in < kN; in++) {
        auto bias = sBIAS(get<1>(tI_mn(0, in)));
#pragma unroll
        for (int im = 0; im < kM; im++) {
          tDr_mn(im, in) += bias;
        }
      }

      // float32 -> bfloat16
      auto tCrh = make_tensor_like<cute::bfloat16_t>(tCr);
#pragma unroll
      for (int i = 0; i < size(tCr); ++i) {
        tCrh(i) = (Tout)(tDr(i));
      }

      // Epilogue
      auto sC = make_tensor(make_smem_ptr((Tout *)shm_c), SLayoutC{});
      using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x1_STSM_N, Tout>;
      auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);

      auto thr_copy_c = tiled_copy_c.get_slice(idx);

      auto tCr4s = thr_copy_c.retile_S(tCrh);
      auto tCs4r = thr_copy_c.partition_D(sC);

      cute::tma_store_wait<0>();
      syncwarpgroup(iwarpgroup);

      // bias arrive must wait all thread in warp have handled bias
      if (elected_idx_in_warpgroup) {
        arrive_barrier(bias_writable);
      }
      phase_bias ^= 1;

      cute::copy(tiled_copy_c, tCr4s, tCs4r);
      syncwarpgroup(iwarpgroup);
      cute::tma_store_fence();

      if (is_leader_in_warpgroup) {
        auto gD = tma_d.get_tma_tensor(make_shape(m, n));
        auto btma_d = tma_d.get_slice(0);

        auto tDs = btma_d.partition_S(sC);  // (TMA, _2, _1)
        auto tDg = btma_d.partition_D(gD);  // (TMA, TMA_M, TMA_N)

        cute::copy(tma_d, tDs(_, iwarpgroup, Int<0>{}), tDg(_, itile_m * 2 + iwarpgroup, itile_n));
        tma_store_arrive();
      }
    }  // while
  }  // else
}

}  // namespace kernels

void pad_and_transpose_async(void *new_scale_ptr, const void *scale_ptr, int m, int n, int m_pad,
                             cudaStream_t stream) {
  constexpr int kNumWarpsPerCTA = 8;
  constexpr int kElementsPerThread = 4;
  constexpr int TM = 32;
  constexpr int TN = 32;
  int num_tile_m = (m + 31) / 32;
  int num_tile_n = (n + 31) / 32;
  int num_block = num_tile_m * num_tile_n;

  dim3 block(256);
  dim3 grid(num_block);
  kernels::pad_and_transpose<kNumWarpsPerCTA, kElementsPerThread, TM, TN>
      <<<grid, block, 0, stream>>>(reinterpret_cast<float *>(new_scale_ptr),
                                   reinterpret_cast<const float *>(scale_ptr), m, n, m_pad,
                                   num_tile_n);
}

void gemm_blockwise_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                          const void *x_scale_ptr, const void *weight_scale_ptr,
                          const void *bias_ptr, int m, int n, int k, int m_pad,
                          cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;
  using TS = float;
  using TBIAS = float;

  constexpr int kTileM = 128;
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;
  constexpr int kTileS = 4;
  constexpr int kStage = 4;
  constexpr int kBlockSwizzle = 4;

  int num_block_k = k / kTileK;
  int num_block_n = n / kTileN;

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto W = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)), make_shape(n, k),
                       make_stride(k, Int<1>{}));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(m, n),
                       make_stride(n, Int<1>{}));
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
  auto slayout_y =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tout>{}, make_shape(Int<kTileM>{}, Int<kTileN>{}));
  auto slayout_xs =
      make_layout(make_shape(Int<kStage>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));
  auto slayout_ws =
      make_layout(make_shape(Int<kStage>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
  auto slayout_bias =
      make_layout(make_shape(Int<1>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));

  auto cpbox_xs =
      make_layout(make_shape(Int<1>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));
  auto cpbox_ws =
      make_layout(make_shape(Int<1>{}, Int<kTileS>{}), make_stride(Int<kTileS>{}, Int<1>{}));
  auto cpbox_bias =
      make_layout(make_shape(Int<1>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto cpbox_y = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tout>{},
                               make_shape(Int<kTileM / 2>{}, Int<kTileN>{}));

  static constexpr int shm_xw = (cosize(slayout_x) + cosize(slayout_w)) * sizeof(Tin);
  static constexpr int shm_y = cosize(slayout_y) * sizeof(Tout);
  static constexpr int shm_xws = (cosize(slayout_xs) + cosize(slayout_ws)) * sizeof(TS);
  static constexpr int shm_bias = cosize(slayout_bias) * sizeof(TBIAS);
  static constexpr int shm_size = shm_xw + shm_y + shm_xws + shm_bias;

  auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, X, slayout_x(_, _, 0));
  auto tma_w = make_tma_copy(SM90_TMA_LOAD{}, W, slayout_w(_, _, 0));
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, cpbox_y);
  auto tma_xs = make_tma_copy(SM90_TMA_LOAD{}, XS, cpbox_xs);
  auto tma_ws = make_tma_copy(SM90_TMA_LOAD{}, WS, cpbox_ws);
  auto tma_bias = make_tma_copy(SM90_TMA_LOAD{}, BIAS, cpbox_bias);

  auto warpgroup_layout = make_layout(make_shape(Int<2>{}, Int<1>{}, Int<1>{}));
  auto tiled_mma = make_tiled_mma(SM90_64x128x32_F32E4M3E4M3_SS_TN<>{}, warpgroup_layout);

  auto kernel = kernels::gemm_blockwise<
      decltype(tiled_mma), decltype(tma_x), decltype(tma_w), decltype(tma_xs), decltype(tma_ws),
      decltype(tma_bias), decltype(tma_y), kTileM, kTileN, kStage, kTileS, Tin, Tout,
      decltype(slayout_x), decltype(slayout_w), decltype(slayout_xs), decltype(slayout_ws),
      decltype(slayout_bias), decltype(slayout_y), kBlockSwizzle>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("cudaFuncSetAttribute error: %s - %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
  }

  int num_tile_m = (m + kTileM - 1) / kTileM;
  int num_tile_n = (n + kTileN - 1) / kTileN;
  int num_tile = num_tile_m * num_tile_n;
  int num_tile_bxn = kBlockSwizzle * num_tile_n;
  cutlass::FastDivmod swizzle_divider(num_tile_bxn);
  cutlass::FastDivmod flat_divider(num_tile_n);

  dim3 block(size(tiled_mma) + 128);
  dim3 grid(std::min(get_sm_count(), num_tile));

  kernel<<<grid, block, shm_size, stream>>>(tma_x, tma_w, tma_xs, tma_ws, tma_bias, tma_y, m, n, k,
                                            m_pad, num_block_n, num_block_k, swizzle_divider,
                                            flat_divider);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("kernel launch error: %s - %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
  }
}

}  // namespace gemm
}  // namespace hpc
