// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/attention/attention.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

template <typename TensorY, typename TensorS>
__device__ __forceinline__ void final_online_softmax(TensorY &tYr_mn, TensorS &gSum, int kM) {
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

template <typename TensorA, typename TensorM, typename TensorS, typename TensorY>
__device__ __forceinline__ void online_softmax(TensorA &tAttr_mn, TensorM &gMax, TensorS &gSum,
                                               TensorY &tYr_mn, int kM, int kN, float one_over_dk) {
#pragma unroll
  for (int im = 0; im < kM; ++im) {
    float row_max = tAttr_mn(im, 0);
    float row_sum = 0.f;

#pragma unroll
    for (int in = 1; in < kN; ++in) {
      row_max = fmaxf(row_max, tAttr_mn(im, in));
    }

    row_max = warp_4lane_reduce_max_xor(row_max) * one_over_dk;
    float last_max = gMax(im);
    gMax(im) = fmaxf(last_max, row_max);

#pragma unroll
    for (int in = 0; in < kN; ++in) {
      tAttr_mn(im, in) = expf_ftz(tAttr_mn(im, in) * one_over_dk - gMax(im));
      row_sum += tAttr_mn(im, in);
    }

    float scale = expf_ftz(last_max - gMax(im));
    gSum(im) = gSum(im) * scale + row_sum;

#pragma unroll
    for (int in = 0; in < cute::size<1>(tYr_mn); ++in) {
      tYr_mn(im, in) = tYr_mn(im, in) * scale;
    }
  }
}

template <typename Tout, typename Tin, int kTileM, int kTileN, int kTileK, int kTileV,
          typename TiledMmaQK, typename TiledMmaPV, typename TmaQ, typename TmaK, typename TmaV,
          typename TmaY, typename SLayoutQ, typename SLayoutK, typename SLayoutV, typename SLayoutY>
__global__ void attention_prefill_bf16_kernel(const __grid_constant__ TmaQ tma_q,
                                              const __grid_constant__ TmaK tma_k,
                                              const __grid_constant__ TmaV tma_v,
                                              const __grid_constant__ TmaY tma_y, int num_batch,
                                              int num_seq_q, int num_seq_kv, int num_dim_qk,
                                              int num_dim_v, int num_head_q, int num_head_kv,
                                              float one_over_dk, cutlass::FastDivmod HeadKV) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int itile_m = blockIdx.x;
  int ihead_q = blockIdx.y;
  int ibatch = blockIdx.z;

  int ihead_kv, res;
  HeadKV(ihead_kv, res, ihead_q);

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  __shared__ uint64_t bar_q;
  __shared__ uint64_t bar_k;
  __shared__ uint64_t bar_v;
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = (Tin *)shm_data;
  auto *shm_k = ((Tin *)shm_q) + cosize(SLayoutQ{});
  auto *shm_v = ((Tin *)shm_k) + cosize(SLayoutK{});
  auto *shm_y = ((Tout *)shm_data);  // Reuse All

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
  auto tKs = btma_k.partition_D(sK);  // (TMA, _1, _1)
  auto tVs = btma_v.partition_D(sV);  // (TMA, _1, _1)

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

  auto gI = make_identity_tensor(gAtt.shape());
  auto tI = thr_mma_qk.partition_C(gI);

  auto tAttr_mn = retile_fragment(tAttr);
  constexpr int kM = size<0>(tAttr_mn);
  constexpr int kN = size<1>(tAttr_mn);
  Tensor gMax = make_tensor<float>(Int<kM>{});
  Tensor gSum = make_tensor<float>(Int<kM>{});

  clear(gSum);
  fill(gMax, -std::numeric_limits<float>::infinity());

  // Load Q
  if (is_leader_in_block) {
    initialize_barrier(bar_q, 1);
    cute::copy(tma_q.with(bar_q), tQg(_, itile_m, _, ihead_q, ibatch), tQs(_, 0, _));
    set_barrier_transaction_bytes(bar_q, sizeof(Tin) * cosize(SLayoutQ{}));
  }

  // init k/v barrier
  if (is_leader_in_block) {
    initialize_barrier(bar_k, 1);
    initialize_barrier(bar_v, 1);
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();
  wait_barrier(bar_q, 0);

  auto layout_asC = thr_mma_qk.partition_C(gAtt).layout();
  auto layout_asA = thr_mma_pv.partition_A(gAtt).layout();
  auto tAttA = make_tensor(tAttr.data(), left_inverse(layout_asC).compose(layout_asA));

  clear(tYr);

  tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

  int num_tile_kv = min(((itile_m + 1) * kTileM + kTileN - 1) / kTileN, size<1>(tKg));
  int num_tile_full = min(itile_m * kTileM, num_seq_kv) / kTileN;

  int phase = 0;
#pragma unroll 1
  for (int itile_seq_kv = num_tile_full; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
    // load k/scale/v
    if (is_leader_in_block) {
      // k
      cute::copy(tma_k.with(bar_k), tKg(_, itile_seq_kv, _, ihead_kv, ibatch), tKs(_, 0, _));
      set_barrier_transaction_bytes(bar_k, sizeof(Tin) * cosize(SLayoutK{}));

      // v
      cute::copy(tma_v.with(bar_v), tVg(_, _, itile_seq_kv, ihead_kv, ibatch), tVs(_, _, 0));
      set_barrier_transaction_bytes(bar_v, sizeof(Tin) * cosize(SLayoutV{}));
    }
    wait_barrier(bar_k, phase);

    // P = QK
    tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;

    warpgroup_fence_operand(tAttr);
    warpgroup_arrive();
    for (int ik = 0; ik < size<2>(tQr); ++ik) {
      cute::gemm(tiled_mma_qk, tQr(_, _, ik), tKr(_, _, ik), tAttr(_, _, _));
      tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
    }

    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tAttr);

    // do causal mask
    auto tAttr_mn = retile_fragment(tAttr);
    auto tI_mn = retile_fragment(tI);
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

    auto tYr_mn = retile_fragment(tYr);
    // online softmax
    online_softmax(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk);

    // Y = PV
    auto tAttAbf16 = make_tensor_like<cute::bfloat16_t>(tAttA);
#pragma unroll
    for (int i = 0; i < size(tAttA); ++i) {
      tAttAbf16(i) = (cute::bfloat16_t)(tAttA(i));
    }

    wait_barrier(bar_v, phase);
    phase ^= 1;

    warpgroup_fence_operand(tYr);
    warpgroup_arrive();
    cute::gemm(tiled_mma_pv, tAttAbf16, tVr, tYr);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    warpgroup_fence_operand(tYr);
    __syncthreads();
  }

#pragma unroll 1
  for (int itile_seq_kv = 0; itile_seq_kv < num_tile_full; ++itile_seq_kv) {
    // load k/scale/v
    if (is_leader_in_block) {
      // k
      cute::copy(tma_k.with(bar_k), tKg(_, itile_seq_kv, _, ihead_kv, ibatch), tKs(_, 0, _));
      set_barrier_transaction_bytes(bar_k, sizeof(Tin) * cosize(SLayoutK{}));

      // v
      cute::copy(tma_v.with(bar_v), tVg(_, _, itile_seq_kv, ihead_kv, ibatch), tVs(_, _, 0));
      set_barrier_transaction_bytes(bar_v, sizeof(Tin) * cosize(SLayoutV{}));
    }
    wait_barrier(bar_k, phase);

    // P = QK
    tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;

    warpgroup_fence_operand(tAttr);
    warpgroup_arrive();
    for (int ik = 0; ik < size<2>(tQr); ++ik) {
      cute::gemm(tiled_mma_qk, tQr(_, _, ik), tKr(_, _, ik), tAttr(_, _, _));
      tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
    }

    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tAttr);

    auto tYr_mn = retile_fragment(tYr);
    // online softmax
    online_softmax(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk);

    // Y = PV
    auto tAttAbf16 = make_tensor_like<cute::bfloat16_t>(tAttA);
#pragma unroll
    for (int i = 0; i < size(tAttA); ++i) {
      tAttAbf16(i) = (cute::bfloat16_t)(tAttA(i));
    }

    wait_barrier(bar_v, phase);
    phase ^= 1;

    warpgroup_fence_operand(tYr);
    warpgroup_arrive();
    cute::gemm(tiled_mma_pv, tAttAbf16, tVr, tYr);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    warpgroup_fence_operand(tYr);
    __syncthreads();
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
  __syncthreads();
  tma_store_fence();

  // using TMA to store
  if (is_leader_in_block) {
    auto cY = tma_y.get_tma_tensor(make_shape(num_seq_q, num_dim_v, num_head_q, num_batch));
    auto btma_y = tma_y.get_slice(0);

    auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
    auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

    cute::copy(tma_y, tYss(_, 0, 0), tYgg(_, itile_m, 0, ihead_q, ibatch));
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

  constexpr int kTileM = 64;
  constexpr int kTileN = 64;
  constexpr int kTileK = 80;
  constexpr int kTileV = 80;

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
      tile_to_shape(GMMA::Layout_K_SW32_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto slayout_k =
      tile_to_shape(GMMA::Layout_K_SW32_Atom<Tin>{}, make_shape(Int<kTileN>{}, Int<kTileK>{}));
  auto slayout_v =
      tile_to_shape(GMMA::Layout_MN_SW32_Atom<Tin>{}, make_shape(Int<kTileV>{}, Int<kTileN>{}));
  auto slayout_y =
      tile_to_shape(GMMA::Layout_K_SW32_Atom<Tout>{}, make_shape(Int<kTileM>{}, Int<kTileV>{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, slayout_q);
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K, slayout_k);
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, V, slayout_v);
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, slayout_y);

  using TiledMmaQK =
      decltype(make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
  using TiledMmaPV =
      decltype(make_tiled_mma(SM90_64x80x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));

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

  dim3 block(size(TiledMmaQK{}));
  dim3 grid((num_seq_q + kTileM - 1) / kTileM, num_head_q, num_batch);

  int shm_qkv = (cosize(slayout_q) + cosize(slayout_k) + cosize(slayout_v)) * sizeof(Tin);
  int shm_y = cosize(slayout_y) * sizeof(Tout);
  int shm_size = std::max(shm_qkv, shm_y);

  float one_over_dk = 1.f / sqrtf(float(num_dim_qk));
  int kv_group = num_head_q / num_head_kv;
  cutlass::FastDivmod HeadKV(kv_group);

  auto kernel = kernels::attention_prefill_bf16_kernel<
      Tout, Tin, kTileM, kTileN, kTileK, kTileV, TiledMmaQK, TiledMmaPV, decltype(tma_q),
      decltype(tma_k), decltype(tma_v), decltype(tma_y), decltype(slayout_q), decltype(slayout_k),
      decltype(slayout_v), decltype(slayout_y)>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
  kernel<<<grid, block, shm_size, stream>>>(tma_q, tma_k, tma_v, tma_y, num_batch, num_seq_q,
                                            num_seq_kv, num_dim_qk, num_dim_v, num_head_q,
                                            num_head_kv, one_over_dk, HeadKV);
}
}  // namespace attention
}  // namespace hpc
