// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "src/attention/attention.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

__device__ __forceinline__ auto get_next_tile(int iblock, int num_tile_m, int num_head_q,
                                              int num_batch, int total_blocks,
                                              cutlass::FastDivmod head_q_divmod,
                                              cutlass::FastDivmod tile_m_divmod) {
  int itile_m, ihead_q, ibatch;
  int itile_m_res;

  iblock = total_blocks - iblock - 1;

  tile_m_divmod(itile_m, itile_m_res, iblock);
  head_q_divmod(ibatch, ihead_q, itile_m_res);

  return cute::make_tuple(itile_m, ihead_q, ibatch);
}

template <typename YTensor, typename STensor>
__device__ __forceinline__ void final_online_softmax(YTensor &&tYr_mn, STensor &&gSum, int kM) {
#pragma unroll
  for (int im = 0; im < kM; ++im) {
    gSum(im) = warp_4lane_reduce_sum_xor(gSum(im));

    float one_over_gsum = rcpf_ftz(gSum(im));
#pragma unroll
    for (int in = 0; in < cute::size<1>(tYr_mn); ++in) {
      tYr_mn(im, in) = tYr_mn(im, in) * one_over_gsum;
    }
  }
}

template <typename ATensor, typename MTensor, typename STensor, typename YTensor>
__device__ __forceinline__ void online_softmax(ATensor &&tAttr_mn, MTensor &&gMax, STensor &&gSum,
                                               YTensor &&tYr_mn, int kM, int kN,
                                               float one_over_dk_log2e) {
#pragma unroll
  for (int im = 0; im < kM; ++im) {
    float row_max = tAttr_mn(im, 0);
    float row_sum = 0.f;

#pragma unroll
    for (int in = 1; in < kN; ++in) {
      row_max = fmaxf(row_max, tAttr_mn(im, in));
    }

    row_max = warp_4lane_reduce_max_xor(row_max) * one_over_dk_log2e;
    float last_max = gMax(im);
    gMax(im) = fmaxf(last_max, row_max);

#pragma unroll
    for (int in = 0; in < kN; ++in) {
      tAttr_mn(im, in) = exp2f_ftz(tAttr_mn(im, in) * one_over_dk_log2e - gMax(im));
      row_sum += tAttr_mn(im, in);
    }

    float scale = exp2f_ftz(last_max - gMax(im));
    gSum(im) = gSum(im) * scale + row_sum;

#pragma unroll
    for (int in = 0; in < cute::size<1>(tYr_mn); ++in) {
      tYr_mn(im, in) = tYr_mn(im, in) * scale;
    }
  }
}

template <typename Tout, typename Tin, int kTileM, int kTileN, int kTileK, int kTileV, int kStage,
          typename TiledMmaQK, typename TiledMmaPV, typename TmaQ, typename TmaK, typename TmaV,
          typename TmaY, typename SLayoutQ, typename SLayoutK, typename SLayoutV, typename SLayoutY>
__global__ void __launch_bounds__(384, 1) attention_prefill_bf16_kernel(
    const __grid_constant__ TmaQ tma_q, const __grid_constant__ TmaK tma_k,
    const __grid_constant__ TmaV tma_v, const __grid_constant__ TmaY tma_y, int num_batch,
    int num_seq_q, int num_seq_kv, int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv,
    float one_over_dk_log2e, cutlass::FastDivmod head_kv_divmod, cutlass::FastDivmod head_q_divmod,
    cutlass::FastDivmod tile_m_divmod) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  __shared__ uint64_t bar_q;
  __shared__ uint64_t writable_q;
  __shared__ uint64_t bar_k[kStage];
  __shared__ uint64_t writable_k[kStage];
  __shared__ uint64_t bar_v[kStage];
  __shared__ uint64_t writable_v[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = (Tin *)shm_data;
  auto *shm_k = ((Tin *)shm_q) + cosize(SLayoutQ{});
  auto *shm_v = ((Tin *)shm_k) + cosize(SLayoutK{});
  auto *shm_y = ((Tout *)shm_v) + cosize(SLayoutV{});  // Reuse All

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(num_seq_q, num_dim_qk, num_head_q, num_batch));
  auto gK = tma_k.get_tma_tensor(make_shape(num_seq_kv, num_dim_qk, num_head_kv, num_batch));
  auto gV = tma_v.get_tma_tensor(make_shape(num_dim_v, num_seq_kv, num_head_kv, num_batch));
  auto gY = tma_y.get_tma_tensor(make_shape(num_seq_q, num_dim_v, num_head_q, num_batch));

  auto gAtt = make_tensor(make_gmem_ptr((float *)nullptr), make_shape(Int<kTileM>{}, Int<kTileN>{}),
                          make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY = make_tensor(make_gmem_ptr((Tout *)nullptr), make_shape(Int<kTileM>{}, Int<kTileV>{}),
                         make_stride(Int<kTileV>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);
  auto btma_y = tma_y.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, head, batch)
  auto tKg = btma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, head, batch)
  auto tVg = btma_v.partition_S(gV);  // (TMA, TMA_V, TMA_N, head, batch)

  auto tQs = btma_q.partition_D(sQ);  // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);  // (TMA, _1, _1, kStage)
  auto tVs = btma_v.partition_D(sV);  // (TMA, _1, _1, kStage)

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;

  auto thr_mma_qk = tiled_mma_qk.get_slice(idx);
  auto thr_mma_pv = tiled_mma_pv.get_slice(idx);

  auto tQs4r = thr_mma_qk.partition_A(sQ);
  auto tKs4r = thr_mma_qk.partition_B(sK);
  auto tVs4r = thr_mma_pv.partition_B(sV);

  auto tQr = thr_mma_qk.make_fragment_A(tQs4r);  // (MMA, MMA_M, MMA_K)
  auto tKr = thr_mma_qk.make_fragment_B(tKs4r);  // (MMA, MMA_N, MMA_K)
  auto tVr = thr_mma_pv.make_fragment_B(tVs4r);  // (MMA, MMA_V, MMA_N)

  auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
  auto tYr = thr_mma_pv.partition_fragment_C(gYY);

  // init k/v barrier
  if (is_leader_in_block) {
    initialize_barrier(bar_q, 1);
    initialize_barrier(writable_q, 2);
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(bar_k[i], 1);
      initialize_barrier(writable_k[i], 2);
      initialize_barrier(bar_v[i], 1);
      initialize_barrier(writable_v[i], 2);
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  int num_tile_m = (num_seq_q + kTileM - 1) / kTileM;

  int total_blocks = num_head_q * num_batch * num_tile_m;
  if (idx >= 256) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= 256;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;    // start with ok
      int phase_q = 1;  // start with ok
      int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);

      int iter = 0;
      int iblock_true = iblock;
      while (true) {
        if (iblock >= total_blocks) {
          break;
        }

        auto [itile_m, ihead_q, ibatch] = get_next_tile(iblock, num_tile_m, num_head_q, num_batch,
                                                        total_blocks, head_q_divmod, tile_m_divmod);

        int iblock_res = 78 - iblock % 78;
        iblock = iblock + 2 * iblock_res - 1;

        if (itile_m * kTileM >= num_seq_q) {
          continue;
        }

        int ihead_kv, res;
        head_kv_divmod(ihead_kv, res, ihead_q);

        int num_tile_kv = min(((itile_m + 1) * kTileM + kTileN - 1) / kTileN, size<1>(tKg));

        // Load Q
        wait_barrier(writable_q, phase_q);
        cute::copy(tma_q.with(bar_q), tQg(_, itile_m, _, ihead_q, ibatch), tQs(_, 0, _));
        set_barrier_transaction_bytes(bar_q, sizeof(Tin) * cosize(SLayoutQ{}));
        phase_q ^= 1;

        // load KV
#pragma unroll 1
        for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
          // k
          wait_barrier(writable_k[ismem_write], phase);
          cute::copy(tma_k.with(bar_k[ismem_write]), tKg(_, itile_seq_kv, _, ihead_kv, ibatch),
                     tKs(_, 0, _, ismem_write));
          set_barrier_transaction_bytes(bar_k[ismem_write],
                                        sizeof(Tin) * cosize(SLayoutK{}(_, _, 0)));

          // v
          wait_barrier(writable_v[ismem_write], phase);
          cute::copy(tma_v.with(bar_v[ismem_write]), tVg(_, _, itile_seq_kv, ihead_kv, ibatch),
                     tVs(_, _, 0, ismem_write));
          set_barrier_transaction_bytes(bar_v[ismem_write],
                                        sizeof(Tin) * cosize(SLayoutV{}(_, _, 0)));

          ++ismem_write;
          if (ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<168>();

    int idx_in_warpgroup = idx % 128;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    auto layout_asC = thr_mma_qk.partition_C(gAtt).layout();
    auto layout_asA = thr_mma_pv.partition_A(gAtt).layout();
    auto tAttA = make_tensor(tAttr.data(), left_inverse(layout_asC).compose(layout_asA));

    auto gI = make_identity_tensor(gAtt.shape());
    auto tI = thr_mma_qk.partition_C(gI);

    auto tAttr_mn = retile_fragment(tAttr);
    constexpr int kM = size<0>(tAttr_mn);
    constexpr int kN = size<1>(tAttr_mn);
    Tensor gMax = make_tensor<float>(Int<kM>{});
    Tensor gSum = make_tensor<float>(Int<kM>{});

    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

    int ismem_read = 0;
    int phase = 0;
    int phase_q = 0;

    int iter = 0;
    int iblock_true = iblock;
    while (true) {
      if (iblock >= total_blocks) {
        break;
      }

      auto [itile_m, ihead_q, ibatch] = get_next_tile(iblock, num_tile_m, num_head_q, num_batch,
                                                      total_blocks, head_q_divmod, tile_m_divmod);

      int iblock_res = 78 - iblock % 78;
      iblock = iblock + 2 * iblock_res - 1;

      if (itile_m * kTileM >= num_seq_q) {
        continue;
      }

      int ihead_kv, res;
      head_kv_divmod(ihead_kv, res, ihead_q);
      int num_tile_kv = min(((itile_m + 1) * kTileM + kTileN - 1) / kTileN, size<1>(tKg));
      int num_tile_full = min(itile_m * kTileM, num_seq_kv) / kTileN;

      clear(tYr);
      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());

      wait_barrier(bar_q, phase_q);
#pragma unroll 1
      for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        wait_barrier(bar_k[ismem_read], phase);

        // P = QK
        tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;

        warpgroup_fence_operand(tAttr);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tQr); ++ik) {
          cute::gemm(tiled_mma_qk, tQr(_, _, ik), tKr(_, _, ik, ismem_read), tAttr(_, _, _));
          tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tAttr);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_k[ismem_read]);
        }

        if (itile_seq_kv == (num_tile_kv - 1)) {
          if (elected_idx_in_warpgroup) {
            arrive_barrier(writable_q);
          }
          phase_q ^= 1;
        }

        // do causal mask
        auto tI_mn = retile_fragment(tI);

        if (itile_seq_kv >= num_tile_full) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              int irow = itile_m * kTileM + get<0>(tI_mn(im, in));
              int icol = itile_seq_kv * kTileN + get<1>(tI_mn(im, in));

              if ((icol > irow) || (icol >= num_seq_kv)) {
                tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
              }
            }
          }
        }

        auto tYr_mn = retile_fragment(tYr);
        // online softmax
        online_softmax(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);

        // Y = PV
        auto tAttAbf16 = make_tensor_like<cute::bfloat16_t>(tAttA);
#pragma unroll
        for (int i = 0; i < size(tAttA); ++i) {
          tAttAbf16(i) = (cute::bfloat16_t)(tAttA(i));
        }

        wait_barrier(bar_v[ismem_read], phase);

        warpgroup_fence_operand(tYr);
        warpgroup_arrive();
        cute::gemm(tiled_mma_pv, tAttAbf16, tVr(_, _, _, ismem_read), tYr);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_v[ismem_read]);
        }

        ++ismem_read;
        if (ismem_read == kStage) {
          phase ^= 1;
          ismem_read = 0;
        }
      }

      auto tYr_mn = retile_fragment(tYr);
      // final online softmax
      final_online_softmax(tYr_mn, gSum, kM);

      // to bfloat16
      auto tYr_bf16 = make_tensor_like<Tout>(tYr);

#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        Tout v{tYr(i)};
        tYr_bf16(i) = v;
      }

      // Epilogue: write register-C to global memory
      using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>;
      auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma_pv);
      auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx);

      auto tYr4s = r2s_thr_copy.retile_S(tYr_bf16);
      auto tYs4r = r2s_thr_copy.partition_D(sY);

      cute::copy(r2s_tiled_copy, tYr4s, tYs4r);
      asm volatile("barrier.sync %0, 128;\n" ::"r"(iwarpgroup) : "memory");
      tma_store_fence();

      // using TMA to store
      if (is_leader_in_warpgroup) {
        auto cY = tma_y.get_tma_tensor(make_shape(num_seq_q, num_dim_v, num_head_q, num_batch));
        auto btma_y = tma_y.get_slice(0);

        auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
        auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

        cute::copy(tma_y, tYss(_, iwarpgroup, 0),
                   tYgg(_, itile_m * 2 + iwarpgroup, 0, ihead_q, ibatch));
        tma_store_arrive();
      }
    }
  }
}

}  // namespace kernels

void attention_prefill_bf16_async(void *y_ptr, const void *q_ptr, const void *k_ptr,
                                  const void *v_ptr, int num_batch, int num_seq_q, int num_seq_kv,
                                  int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv,
                                  int ldY, int ldQ, int ldK, int ldV, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::bfloat16_t;
  using Tout = cute::bfloat16_t;

  constexpr int kTileM = 128;
  constexpr int kTileN = 64;
  constexpr int kTileK = 128;
  constexpr int kTileV = 128;
  constexpr int kStage = 3;

  auto Q = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(q_ptr)),
                       make_shape(num_seq_q, num_dim_qk, num_head_q, num_batch),
                       make_stride(ldQ, Int<1>{}, num_dim_qk, num_seq_q * ldQ));

  auto K = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(k_ptr)),
                       make_shape(num_seq_kv, num_dim_qk, num_head_kv, num_batch),
                       make_stride(ldK, Int<1>{}, num_dim_qk, num_seq_kv * ldK));
  auto V = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(v_ptr)),
                       make_shape(num_dim_v, num_seq_kv, num_head_kv, num_batch),
                       make_stride(Int<1>{}, ldV, num_dim_v, num_seq_kv * ldV));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<const Tout *>(y_ptr)),
                       make_shape(num_seq_q, num_dim_v, num_head_q, num_batch),
                       make_stride(ldY, Int<1>{}, num_dim_v, num_seq_q * ldY));

  auto slayout_q =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto slayout_k = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_v = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileV>{}, Int<kTileN>{}, Int<kStage>{}));
  auto slayout_y =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tout>{}, make_shape(Int<kTileM>{}, Int<kTileV>{}));

  auto cpbox_y = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tout>{},
                               make_shape(Int<kTileM / 2>{}, Int<kTileV>{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, slayout_q);
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K, slayout_k(_, _, 0));
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, V, slayout_v(_, _, 0));
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, cpbox_y);

  auto warpgroup_layout = make_layout(make_shape(Int<2>{}, Int<1>{}, Int<1>{}));
  using TiledMmaQK = decltype(make_tiled_mma(
      SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}, warpgroup_layout));
  using TiledMmaPV = decltype(make_tiled_mma(
      SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}, warpgroup_layout));

  static_assert(kTileM >= get<0>(TiledMmaQK::Shape_MNK{}));
  static_assert(kTileM % get<0>(TiledMmaQK::Shape_MNK{}) == 0);
  static_assert(kTileM >= get<0>(TiledMmaPV::Shape_MNK{}));
  static_assert(kTileM % get<0>(TiledMmaPV::Shape_MNK{}) == 0);

  static_assert(kTileN >= get<1>(TiledMmaQK::Shape_MNK{}));
  static_assert(kTileN % get<1>(TiledMmaQK::Shape_MNK{}) == 0);
  static_assert(kTileN >= get<2>(TiledMmaPV::Shape_MNK{}));
  static_assert(kTileN % get<2>(TiledMmaPV::Shape_MNK{}) == 0);

  static_assert(kTileK >= get<2>(TiledMmaQK::Shape_MNK{}));
  static_assert(kTileK % get<2>(TiledMmaQK::Shape_MNK{}) == 0);

  static_assert(kTileV >= get<1>(TiledMmaPV::Shape_MNK{}));
  static_assert(kTileV % get<1>(TiledMmaPV::Shape_MNK{}) == 0);

  dim3 block(384);
  dim3 grid(78);

  int shm_qkv = (cosize(slayout_q) + cosize(slayout_k) + cosize(slayout_v)) * sizeof(Tin);
  int shm_y = cosize(slayout_y) * sizeof(Tout);
  int shm_size = shm_qkv + shm_y;

  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;

  int kv_group = num_head_q / num_head_kv;
  cutlass::FastDivmod head_kv_divmod(kv_group);
  cutlass::FastDivmod head_q_divmod(num_head_q);
  cutlass::FastDivmod tile_m_divmod(num_batch * num_head_q);

  auto kernel = kernels::attention_prefill_bf16_kernel<
      Tout, Tin, kTileM, kTileN, kTileK, kTileV, kStage, TiledMmaQK, TiledMmaPV, decltype(tma_q),
      decltype(tma_k), decltype(tma_v), decltype(tma_y), decltype(slayout_q), decltype(slayout_k),
      decltype(slayout_v), decltype(slayout_y)>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
  kernel<<<grid, block, shm_size, stream>>>(
      tma_q, tma_k, tma_v, tma_y, num_batch, num_seq_q, num_seq_kv, num_dim_qk, num_dim_v,
      num_head_q, num_head_kv, one_over_dk_log2e, head_kv_divmod, head_q_divmod, tile_m_divmod);
}
}  // namespace attention
}  // namespace hpc
