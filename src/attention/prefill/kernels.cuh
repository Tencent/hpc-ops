// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_PREFILL_KERNELS_CUH_
#define SRC_ATTENTION_PREFILL_KERNELS_CUH_

#include <cuda.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/attention/prefill/fp8_v_to_half_dequant.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

constexpr float kFp8PrefillPScale = 256.0f;
constexpr float kFp8PrefillPScaleInv = 1.0f / kFp8PrefillPScale;

constexpr int kKCvtBarrierId = 2;
constexpr int kKCvtThreads = 96;
constexpr int kVCvtBarrierId = 3;
constexpr int kVCvtThreads = 128;

template <int kTileM>
__device__ __forceinline__ auto get_next_tile(const int *seqlens_q_ptr, int &iblock, int num_head_q,
                                              int num_batch, int max_total_blocks,
                                              int max_num_tile_m, cutlass::FastDivmod head_q_divmod,
                                              cutlass::FastDivmod tile_m_divmod) {
  int itile_m, ihead_q, ibatch;
  int itile_m_res;
  int num_seq_q, num_tile_m, num_tile_m_res;

  int iblock_res = max_total_blocks - iblock - 1;

  tile_m_divmod(itile_m, itile_m_res, iblock_res);
  head_q_divmod(ibatch, ihead_q, itile_m_res);

  num_seq_q = seqlens_q_ptr[ibatch];
  num_tile_m = (num_seq_q + kTileM - 1) / kTileM;
  num_tile_m_res = max_num_tile_m - num_tile_m;
  itile_m = itile_m - num_tile_m_res;

  iblock = iblock + 2 * (gridDim.x - iblock % gridDim.x) - 1;

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

template <typename T, typename TmaQ, typename TmaK, typename TmaV, typename TmaY>
__global__ void update_batched_tma(const vec_t<cute::TmaDescriptor, 4> td_qkvy,
                                   cute::TmaDescriptor *tma_qkvy, const T *q_ptr, const T *k_ptr,
                                   const T *v_ptr, const T *y_ptr, const int *seqlens_q_ptr,
                                   const int *cu_seqlens_q_ptr, int num_batch, int max_seq_q,
                                   int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv,
                                   int ldQ, int ldK, int ldV, int ldY) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int ibatch = blockIdx.x;

  __shared__ cute::TmaDescriptor smem_tma_desc[4];

  int num_seq = seqlens_q_ptr[ibatch];
  int cu_seqlen_q = cu_seqlens_q_ptr[ibatch];
  auto *q_ibatch_ptr = q_ptr + cu_seqlen_q * ldQ;
  auto *k_ibatch_ptr = k_ptr + cu_seqlen_q * ldK;
  auto *v_ibatch_ptr = v_ptr + cu_seqlen_q * ldV;
  auto *y_ibatch_ptr = y_ptr + cu_seqlen_q * ldY;

  if (idx < 4) {
    smem_tma_desc[idx] = td_qkvy[idx];
  }
  __syncwarp();

  // Q
  if (idx == 0) {
    auto gQ = make_tensor(make_gmem_ptr(q_ibatch_ptr), make_shape(num_seq, num_dim_qk, num_head_q),
                          make_stride(ldQ, Int<1>{}, num_dim_qk));
    update_tma_gtensor<TmaQ>(smem_tma_desc[idx], gQ);
  }

  // K
  if (idx == 1) {
    auto gK = make_tensor(make_gmem_ptr(k_ibatch_ptr), make_shape(num_seq, num_dim_qk, num_head_kv),
                          make_stride(ldK, Int<1>{}, num_dim_qk));
    update_tma_gtensor<TmaK>(smem_tma_desc[idx], gK);
  }

  // V
  if (idx == 2) {
    auto gV = make_tensor(make_gmem_ptr(v_ibatch_ptr), make_shape(num_dim_v, num_seq, num_head_kv),
                          make_stride(Int<1>{}, ldV, num_dim_v));
    update_tma_gtensor<TmaV>(smem_tma_desc[idx], gV);
  }

  // Y
  if (idx == 3) {
    auto gY = make_tensor(make_gmem_ptr(y_ibatch_ptr), make_shape(num_seq, num_dim_v, num_head_q),
                          make_stride(ldY, Int<1>{}, num_dim_v));
    update_tma_gtensor<TmaY>(smem_tma_desc[idx], gY);
  }

#pragma unroll
  for (int i = 0; i < 4; i++) {
    __syncwarp();
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    tma_descriptor_cp_fence_release(tma_qkvy + ibatch * 4 + i, smem_tma_desc[i]);
  }
}

template <typename T1, typename T2, typename TmaQ, typename TmaY>
__global__ void update_batched_tma_with_kvcache(const vec_t<cute::TmaDescriptor, 2> td_qy,
                                                cute::TmaDescriptor *tma_qy, const T1 *q_ptr,
                                                const T2 *y_ptr, const int *cu_seqlens_q_ptr,
                                                int num_batch, int max_seq_q, int num_dim_qk,
                                                int num_dim_v, int num_head_q, int ldQ, int ldY) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int ibatch = blockIdx.x;

  __shared__ cute::TmaDescriptor smem_tma_desc[2];

  int num_seq = cu_seqlens_q_ptr[ibatch + 1] - cu_seqlens_q_ptr[ibatch];
  int cu_seqlen_q = cu_seqlens_q_ptr[ibatch];
  auto *q_ibatch_ptr = q_ptr + cu_seqlen_q * ldQ;
  auto *y_ibatch_ptr = y_ptr + cu_seqlen_q * ldY;

  if (idx < 2) {
    smem_tma_desc[idx] = td_qy[idx];
  }
  __syncwarp();

  // Q
  if (idx == 0) {
    auto gQ = make_tensor(make_gmem_ptr(q_ibatch_ptr), make_shape(num_seq, num_dim_qk, num_head_q),
                          make_stride(ldQ, Int<1>{}, num_dim_qk));
    update_tma_gtensor<TmaQ>(smem_tma_desc[idx], gQ);
  }

  // Y
  if (idx == 1) {
    auto gY = make_tensor(make_gmem_ptr(y_ibatch_ptr), make_shape(num_seq, num_dim_v, num_head_q),
                          make_stride(ldY, Int<1>{}, num_dim_v));
    update_tma_gtensor<TmaY>(smem_tma_desc[idx], gY);
  }

#pragma unroll
  for (int i = 0; i < 2; i++) {
    __syncwarp();
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    tma_descriptor_cp_fence_release(tma_qy + ibatch * 2 + i, smem_tma_desc[i]);
  }
}

// TMA update kernel for FP8 varlen Q/K/V.
template <typename Tin, typename Tout, typename TmaQ, typename TmaK, typename TmaV, typename TmaY>
__global__ void update_batched_tma_fp8(const vec_t<cute::TmaDescriptor, 4> td_qkvy,
                                       cute::TmaDescriptor *tma_qkvy, const Tin *q_ptr,
                                       const Tin *k_ptr, const Tin *v_ptr, const Tout *y_ptr,
                                       const int *cu_seqlens_q_ptr, const int *cu_seqlens_kv_ptr,
                                       int num_batch, int max_seq_q, int max_seq_kv, int num_dim_qk,
                                       int num_dim_v, int num_head_q, int num_head_kv, int ldQ,
                                       int ldK, int ldV, int ldY) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int ibatch = blockIdx.x;

  __shared__ cute::TmaDescriptor smem_tma_desc[4];

  int num_seq_q = cu_seqlens_q_ptr[ibatch + 1] - cu_seqlens_q_ptr[ibatch];
  int num_seq_kv = cu_seqlens_kv_ptr[ibatch + 1] - cu_seqlens_kv_ptr[ibatch];
  int cu_seqlen_q = cu_seqlens_q_ptr[ibatch];
  int cu_seqlen_kv = cu_seqlens_kv_ptr[ibatch];

  auto *q_ibatch_ptr = q_ptr + cu_seqlen_q * ldQ;
  auto *k_ibatch_ptr = k_ptr + cu_seqlen_kv * ldK;
  auto *v_ibatch_ptr = v_ptr + cu_seqlen_kv * ldV;
  auto *y_ibatch_ptr = y_ptr + cu_seqlen_q * ldY;

  if (idx < 4) {
    smem_tma_desc[idx] = td_qkvy[idx];
  }
  __syncwarp();

  // Q
  if (idx == 0) {
    auto gQ =
        make_tensor(make_gmem_ptr(q_ibatch_ptr), make_shape(num_seq_q, num_dim_qk, num_head_q),
                    make_stride(ldQ, Int<1>{}, num_dim_qk));
    update_tma_gtensor<TmaQ>(smem_tma_desc[idx], gQ);
  }

  // K
  if (idx == 1) {
    auto gK =
        make_tensor(make_gmem_ptr(k_ibatch_ptr), make_shape(num_seq_kv, num_dim_qk, num_head_kv),
                    make_stride(ldK, Int<1>{}, num_dim_qk));
    update_tma_gtensor<TmaK>(smem_tma_desc[idx], gK);
  }

  // V
  if (idx == 2) {
    auto gV =
        make_tensor(make_gmem_ptr(v_ibatch_ptr), make_shape(num_dim_v, num_seq_kv, num_head_kv),
                    make_stride(Int<1>{}, ldV, num_dim_v));
    update_tma_gtensor<TmaV>(smem_tma_desc[idx], gV);
  }

  // Y
  if (idx == 3) {
    auto gY = make_tensor(make_gmem_ptr(y_ibatch_ptr), make_shape(num_seq_q, num_dim_v, num_head_q),
                          make_stride(ldY, Int<1>{}, num_dim_v));
    update_tma_gtensor<TmaY>(smem_tma_desc[idx], gY);
  }

#pragma unroll
  for (int i = 0; i < 4; i++) {
    __syncwarp();
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    tma_descriptor_cp_fence_release(tma_qkvy + ibatch * 4 + i, smem_tma_desc[i]);
  }
}

template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY>
__global__ void __launch_bounds__(384, 1) attention_prefill_bf16_warp_specialization_kernel(
    cute::TmaDescriptor *td_qkvy, int *seqlens_q_ptr, int num_batch, int max_seq_q, int num_dim_qk,
    int num_dim_v, int num_head_q, int num_head_kv, float one_over_dk_log2e,
    cutlass::FastDivmod head_kv_divmod, cutlass::FastDivmod head_q_divmod,
    cutlass::FastDivmod tile_m_divmod) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;
  using SLayoutQ = typename Config::SLayoutQ;
  using SLayoutK = typename Config::SLayoutK;
  using SLayoutV = typename Config::SLayoutV;
  using SLayoutY = typename Config::SLayoutY;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  __shared__ uint64_t readable_q;
  __shared__ uint64_t writable_q;
  __shared__ uint64_t readable_k[kStage];
  __shared__ uint64_t writable_k[kStage];
  __shared__ uint64_t readable_v[kStage];
  __shared__ uint64_t writable_v[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_v = shm_k + cosize(SLayoutK{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_v + cosize(SLayoutV{}));

  TmaQ tma_q;
  TmaK tma_k;
  TmaV tma_v;
  TmaY tma_y;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gK = tma_k.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_kv));
  auto gV = tma_v.get_tma_tensor(make_shape(num_dim_v, max_seq_q, num_head_kv));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileV>{}), make_stride(Int<kTileV>{}, Int<1>{}));

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
    initialize_barrier(readable_q, 1);
    initialize_barrier(writable_q, 2);
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable_k[i], 1);
      initialize_barrier(writable_k[i], 2);
      initialize_barrier(readable_v[i], 1);
      initialize_barrier(writable_v[i], 2);
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  int max_num_tile_m = (max_seq_q + kTileM - 1) / kTileM;
  int max_total_blocks = num_head_q * num_batch * max_num_tile_m;

  if (idx >= 256) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    idx -= 256;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;    // start with ok
      int phase_q = 1;  // start with ok
      int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);

      while (true) {
        if (iblock >= max_total_blocks) {
          break;
        }

        auto [itile_m, ihead_q, ibatch] =
            get_next_tile<kTileM>(seqlens_q_ptr, iblock, num_head_q, num_batch, max_total_blocks,
                                  max_num_tile_m, head_q_divmod, tile_m_divmod);

        if (itile_m < 0) {
          continue;
        }

        int ihead_kv, res;
        head_kv_divmod(ihead_kv, res, ihead_q);

        auto *td_q = td_qkvy + ibatch * 4;
        auto *td_k = td_qkvy + ibatch * 4 + 1;
        auto *td_v = td_qkvy + ibatch * 4 + 2;

        // Load Q
        wait_barrier(writable_q, phase_q);
        cute::copy(tma_q.with(td_q, readable_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
        set_barrier_transaction_bytes(readable_q, sizeof(Tin) * cosize(SLayoutQ{}));
        phase_q ^= 1;

        int num_tile_kv = ((itile_m + 1) * kTileM + kTileN - 1) / kTileN;
        constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;
        constexpr int kTransactionBytesV = sizeof(Tin) * kTileV * kTileN;
        // load KV
#pragma unroll 1
        for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
          // k
          wait_barrier(writable_k[ismem_write], phase);
          cute::copy(tma_k.with(td_k, readable_k[ismem_write]), tKg(_, itile_seq_kv, _, ihead_kv),
                     tKs(_, 0, _, ismem_write));
          set_barrier_transaction_bytes(readable_k[ismem_write], kTransactionBytesK);

          // v
          wait_barrier(writable_v[ismem_write], phase);
          cute::copy(tma_v.with(td_v, readable_v[ismem_write]), tVg(_, _, itile_seq_kv, ihead_kv),
                     tVs(_, _, 0, ismem_write));
          set_barrier_transaction_bytes(readable_v[ismem_write], kTransactionBytesV);

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

    while (true) {
      if (iblock >= max_total_blocks) {
        break;
      }

      auto [itile_m, ihead_q, ibatch] =
          get_next_tile<kTileM>(seqlens_q_ptr, iblock, num_head_q, num_batch, max_total_blocks,
                                max_num_tile_m, head_q_divmod, tile_m_divmod);

      int num_seq_kv = seqlens_q_ptr[ibatch];
      if (itile_m < 0) {
        continue;
      }

      int ihead_kv, res;
      head_kv_divmod(ihead_kv, res, ihead_q);
      int num_tile_kv = ((itile_m + 1) * kTileM + kTileN - 1) / kTileN;
      int num_tile_full = itile_m * kTileM / kTileN;

      clear(tYr);
      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());

      wait_barrier(readable_q, phase_q);
#pragma unroll 1
      for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        wait_barrier(readable_k[ismem_read], phase);

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
          tAttAbf16(i) = static_cast<cute::bfloat16_t>(tAttA(i));
        }

        wait_barrier(readable_v[ismem_read], phase);

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
        auto cY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
        auto btma_y = tma_y.get_slice(0);

        auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
        auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

        auto *td_y = td_qkvy + ibatch * 4 + 3;
        cute::copy(tma_y.with(td_y), tYss(_, iwarpgroup, 0),
                   tYgg(_, itile_m * 2 + iwarpgroup, 0, ihead_q));
        tma_store_arrive();
      }
    }
  }
}

template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY>
__global__ void __launch_bounds__(384, 1)
    attention_with_kvcache_prefill_bf16_warp_specialization_kernel(
        cute::TmaDescriptor *td_qy, const __grid_constant__ TmaK tma_k,
        const __grid_constant__ TmaV tma_v, const int *cu_seqlens_q_ptr,
        const int *seqlens_kvcache_ptr, const int *block_ids_ptr, int num_batch, int max_seq_q,
        int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
        int block_size, int num_seq_max_blocks, float one_over_dk_log2e,
        cutlass::FastDivmod head_kv_divmod, cutlass::FastDivmod head_q_divmod,
        cutlass::FastDivmod tile_m_divmod) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;
  using SLayoutQ = typename Config::SLayoutQ;
  using SLayoutK = typename Config::SLayoutK;
  using SLayoutV = typename Config::SLayoutV;
  using SLayoutY = typename Config::SLayoutY;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;
  constexpr int kBlockSize = Config::kBlockSize;

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  __shared__ uint64_t readable_q;
  __shared__ uint64_t writable_q;
  __shared__ uint64_t readable_k[kStage];
  __shared__ uint64_t writable_k[kStage];
  __shared__ uint64_t readable_v[kStage];
  __shared__ uint64_t writable_v[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_v = shm_k + cosize(SLayoutK{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_v + cosize(SLayoutV{}));
  auto *shm_seqlens_q = reinterpret_cast<int *>(shm_y + cosize(SLayoutY{}));
  auto *shm_seqlens_kv = shm_seqlens_q + num_batch;
  auto *shm_seqlens_qstart = shm_seqlens_kv + num_batch;

  TmaQ tma_q;
  TmaY tma_y;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks));
  auto gV =
      tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileV>{}), make_stride(Int<kTileV>{}, Int<1>{}));

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
    initialize_barrier(readable_q, 1);
    initialize_barrier(writable_q, 2);
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable_k[i], 1);
      initialize_barrier(writable_k[i], 2);
      initialize_barrier(readable_v[i], 1);
      initialize_barrier(writable_v[i], 2);
    }
  }

  for (int i = idx; i < num_batch; i += blockDim.x) {
    int num_seq_ibatch = cu_seqlens_q_ptr[i + 1] - cu_seqlens_q_ptr[i];
    shm_seqlens_q[i] = num_seq_ibatch;
    shm_seqlens_kv[i] = seqlens_kvcache_ptr[i];
    shm_seqlens_qstart[i] = seqlens_kvcache_ptr[i] - num_seq_ibatch;
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  int max_num_tile_m = (max_seq_q + kTileM - 1) / kTileM;
  int max_total_blocks = num_head_q * num_batch * max_num_tile_m;

  constexpr int kNumBlockPerTileN = kTileN / kBlockSize;

  if (idx >= 256) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= 256;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;    // start with ok
      int phase_q = 1;  // start with ok
      int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);

      while (true) {
        if (iblock >= max_total_blocks) {
          break;
        }

        auto [itile_m, ihead_q, ibatch] =
            get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                  max_num_tile_m, head_q_divmod, tile_m_divmod);

        int num_seq_kv = seqlens_kvcache_ptr[ibatch];
        int num_seq_q = shm_seqlens_q[ibatch];
        if (itile_m < 0) {
          continue;
        }

        int ihead_kv, res;
        head_kv_divmod(ihead_kv, res, ihead_q);

        auto *td_q = td_qy + ibatch * 2;

        int start_seq_q = shm_seqlens_qstart[ibatch];
        auto *block_ids_ibatch_ptr = block_ids_ptr + ibatch * num_seq_max_blocks;

        // Load Q
        wait_barrier(writable_q, phase_q);
        cute::copy(tma_q.with(td_q, readable_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
        set_barrier_transaction_bytes(readable_q, sizeof(Tin) * cosize(SLayoutQ{}));
        phase_q ^= 1;

        int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
        constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;
        constexpr int kTransactionBytesV = sizeof(Tin) * kTileV * kTileN;
        // load KV
#pragma unroll 1
        for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
          // k
          wait_barrier(writable_k[ismem_write], phase);

          int iblock_ids[kNumBlockPerTileN];
#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            iblock_ids[iblock_kv] = 0;
            int iblock_id = itile_seq_kv * kNumBlockPerTileN + iblock_kv;
            if (iblock_id < num_seq_max_blocks) {
              iblock_ids[iblock_kv] = block_ids_ibatch_ptr[iblock_id];
            }
          }

#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            int iblock_true = iblock_ids[iblock_kv];
            cute::copy(tma_k.with(readable_k[ismem_write]), tKg(_, 0, _, ihead_kv, iblock_true),
                       tKs(_, iblock_kv, _, ismem_write));
          }
          set_barrier_transaction_bytes(readable_k[ismem_write], kTransactionBytesK);

          // v
          wait_barrier(writable_v[ismem_write], phase);
#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            int iblock_true = iblock_ids[iblock_kv];
            cute::copy(tma_v.with(readable_v[ismem_write]), tVg(_, _, 0, ihead_kv, iblock_true),
                       tVs(_, _, iblock_kv, ismem_write));
          }
          set_barrier_transaction_bytes(readable_v[ismem_write], kTransactionBytesV);

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

    while (true) {
      if (iblock >= max_total_blocks) {
        break;
      }

      auto [itile_m, ihead_q, ibatch] =
          get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                max_num_tile_m, head_q_divmod, tile_m_divmod);

      if (itile_m < 0) {
        continue;
      }

      int num_seq_kv = shm_seqlens_kv[ibatch];
      int start_seq_q = shm_seqlens_qstart[ibatch];

      int ihead_kv, res;
      head_kv_divmod(ihead_kv, res, ihead_q);
      int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
      int num_tile_full = (start_seq_q + itile_m * kTileM) / kTileN;

      clear(tYr);
      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());

      wait_barrier(readable_q, phase_q);
#pragma unroll 1
      for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        wait_barrier(readable_k[ismem_read], phase);

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
              int irow = start_seq_q + itile_m * kTileM + get<0>(tI_mn(im, in));
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
          tAttAbf16(i) = static_cast<cute::bfloat16_t>(tAttA(i));
        }

        wait_barrier(readable_v[ismem_read], phase);

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
        auto cY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
        auto btma_y = tma_y.get_slice(0);

        auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
        auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

        auto *td_y = td_qy + ibatch * 2 + 1;
        cute::copy(tma_y.with(td_y), tYss(_, iwarpgroup, 0),
                   tYgg(_, itile_m * 2 + iwarpgroup, 0, ihead_q));
        tma_store_arrive();
      }
    }
  }
}

template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY,
          typename TmaQS>
__global__ void __launch_bounds__(384, 1)
    attention_with_kvcache_qpertoken_perhead_kvpertensor_prefill_fp8_warp_specialization_kernel(
        cute::TmaDescriptor *td_qy, const __grid_constant__ TmaK tma_k,
        const __grid_constant__ TmaV tma_v, const __grid_constant__ TmaQS tma_qs,
        const float *qscale_ptr, const float *kscale_ptr, const float *vscale_ptr,
        const int *cu_seqlens_q_ptr, const int *seqlens_kvcache_ptr, const int *block_ids_ptr,
        int num_batch, int max_seq_q, int max_seq_q_pad, int num_dim_qk, int num_dim_v,
        int num_head_q, int num_head_kv, int num_kvcache_blocks, int block_size,
        int num_seq_max_blocks, float one_over_dk_log2e, cutlass::FastDivmod head_kv_divmod,
        cutlass::FastDivmod head_q_divmod, cutlass::FastDivmod tile_m_divmod) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;
  using SLayoutQ = typename Config::SLayoutQ;
  using SLayoutK = typename Config::SLayoutK;
  using SLayoutV = typename Config::SLayoutV;
  using SLayoutVT = typename Config::SLayoutVT;
  using SLayoutY = typename Config::SLayoutY;
  using SLayoutQS = typename Config::SLayoutQS;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;
  constexpr int kBlockSize = Config::kBlockSize;

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  __shared__ uint64_t readable_q;
  __shared__ uint64_t writable_q;
  __shared__ uint64_t readable_k[kStage];
  __shared__ uint64_t writable_k[kStage];
  __shared__ uint64_t readable_v[kStage];
  __shared__ uint64_t writable_v[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_v = shm_k + cosize(SLayoutK{});
  auto *shm_vt = shm_v + cosize(SLayoutV{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_vt + cosize(SLayoutVT{}));
  auto *shm_qs = reinterpret_cast<float *>(shm_y + cosize(SLayoutY{}));
  auto *shm_seqlens_q = reinterpret_cast<int *>(shm_qs + cosize(SLayoutQS{}));
  auto *shm_seqlens_kv = shm_seqlens_q + num_batch;
  auto *shm_seqlens_qstart = shm_seqlens_kv + num_batch;

  TmaQ tma_q;
  TmaY tma_y;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks));
  auto gV =
      tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
  auto gQS = tma_qs.get_tma_tensor(make_shape(max_seq_q_pad, num_head_q, num_batch));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gAtt_fp8 =
      make_tensor(make_gmem_ptr(static_cast<Tin *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileV>{}), make_stride(Int<kTileV>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sVT = make_tensor(make_smem_ptr(shm_vt), SLayoutVT{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});
  auto sQS = make_tensor(make_smem_ptr(shm_qs), SLayoutQS{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);
  auto btma_y = tma_y.get_slice(0);
  auto btma_qs = tma_qs.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);     // (TMA, TMA_M, TMA_K, head, batch)
  auto tKg = btma_k.partition_S(gK);     // (TMA, TMA_N, TMA_K, head, batch)
  auto tVg = btma_v.partition_S(gV);     // (TMA, TMA_V, TMA_N, head, batch)
  auto tQSg = btma_qs.partition_S(gQS);  // (TMA, TMA_M, head, batch)

  auto tQs = btma_q.partition_D(sQ);     // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);     // (TMA, _1, _1, kStage)
  auto tVs = btma_v.partition_D(sV);     // (TMA, _1, _1, kStage)
  auto tQSs = btma_qs.partition_D(sQS);  // (TMA, _1)

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;

  auto thr_mma_qk = tiled_mma_qk.get_slice(idx);
  auto thr_mma_pv = tiled_mma_pv.get_slice(idx);

  auto tQs4r = thr_mma_qk.partition_A(sQ);
  auto tKs4r = thr_mma_qk.partition_B(sK);
  auto tVTs4r = thr_mma_pv.partition_B(sVT);

  auto tQr = thr_mma_qk.make_fragment_A(tQs4r);    // (MMA, MMA_M, MMA_K)
  auto tKr = thr_mma_qk.make_fragment_B(tKs4r);    // (MMA, MMA_N, MMA_K)
  auto tVTr = thr_mma_pv.make_fragment_B(tVTs4r);  // (MMA, MMA_V, MMA_N, kStage)

  auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
  auto tYr = thr_mma_pv.partition_fragment_C(gYY);

  // init k/v barrier
  if (is_leader_in_block) {
    initialize_barrier(readable_q, 1);
    initialize_barrier(writable_q, 2);
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable_k[i], 1);
      initialize_barrier(writable_k[i], 2);
      initialize_barrier(readable_v[i], 1);
      initialize_barrier(writable_v[i], 2);
    }
  }

  for (int i = idx; i < num_batch; i += blockDim.x) {
    int num_seq_ibatch = cu_seqlens_q_ptr[i + 1] - cu_seqlens_q_ptr[i];
    shm_seqlens_q[i] = num_seq_ibatch;
    shm_seqlens_kv[i] = seqlens_kvcache_ptr[i];
    shm_seqlens_qstart[i] = seqlens_kvcache_ptr[i] - num_seq_ibatch;
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  int max_num_tile_m = (max_seq_q + kTileM - 1) / kTileM;
  int max_total_blocks = num_head_q * num_batch * max_num_tile_m;

  constexpr int kNumBlockPerTileN = kTileN / kBlockSize;

  if (idx >= 256) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= 256;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;    // start with ok
      int phase_q = 1;  // start with ok
      int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);

      while (true) {
        if (iblock >= max_total_blocks) {
          break;
        }

        auto [itile_m, ihead_q, ibatch] =
            get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                  max_num_tile_m, head_q_divmod, tile_m_divmod);

        int num_seq_kv = seqlens_kvcache_ptr[ibatch];
        int num_seq_q = shm_seqlens_q[ibatch];
        if (itile_m < 0) {
          continue;
        }

        int ihead_kv, res;
        head_kv_divmod(ihead_kv, res, ihead_q);

        auto *td_q = td_qy + ibatch * 2;

        int start_seq_q = shm_seqlens_qstart[ibatch];
        auto *block_ids_ibatch_ptr = block_ids_ptr + ibatch * num_seq_max_blocks;

        // Load Q
        wait_barrier(writable_q, phase_q);
        cute::copy(tma_q.with(td_q, readable_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
        cute::copy(tma_qs.with(readable_q), tQSg(_, itile_m, ihead_q, ibatch), tQSs(_, 0));
        set_barrier_transaction_bytes(
            readable_q, sizeof(Tin) * cosize(SLayoutQ{}) + sizeof(float) * cosize(SLayoutQS{}));
        phase_q ^= 1;

        int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
        int num_blocks = (num_seq_kv + kBlockSize - 1) / kBlockSize;
        constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;
        constexpr int kTransactionBytesV = sizeof(Tin) * kTileV * kTileN;
        // load KV
#pragma unroll 1
        for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
          // k
          wait_barrier(writable_k[ismem_write], phase);

          int iblock_ids[kNumBlockPerTileN];
#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            iblock_ids[iblock_kv] = 0;
            int iblock_id = itile_seq_kv * kNumBlockPerTileN + iblock_kv;
            if (iblock_id < num_blocks) {
              iblock_ids[iblock_kv] = block_ids_ibatch_ptr[iblock_id];
            } else {
              iblock_ids[iblock_kv] = -1;
            }
          }

#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            int iblock_true = iblock_ids[iblock_kv];
            cute::copy(tma_k.with(readable_k[ismem_write]), tKg(_, 0, _, ihead_kv, iblock_true),
                       tKs(_, iblock_kv, _, ismem_write));
          }
          set_barrier_transaction_bytes(readable_k[ismem_write], kTransactionBytesK);

          // v
          wait_barrier(writable_v[ismem_write], phase);
#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            int iblock_true = iblock_ids[iblock_kv];
            cute::copy(tma_v.with(readable_v[ismem_write]), tVg(_, _, 0, ihead_kv, iblock_true),
                       tVs(_, _, iblock_kv, ismem_write));
          }
          set_barrier_transaction_bytes(readable_v[ismem_write], kTransactionBytesV);

          ++ismem_write;
          if (ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<192>();

    int idx_in_warpgroup = idx % 128;
    int idx_in_warp = idx % 32;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    auto tAttr_fp8 = make_tensor_like<Tin>(tAttr);
    auto layout_asC = tAttr.layout();
    auto layout_asA = thr_mma_pv.partition_fragment_A(gAtt_fp8).layout();
    // auto tAttA_fp8 = make_tensor(tAttr_fp8.data(), left_inverse(layout_asC).compose(layout_asA));
    auto tAttA_fp8 = make_tensor(tAttr_fp8.data(), layout_asA);

    auto gI = make_identity_tensor(gAtt.shape());
    auto tI = thr_mma_qk.partition_C(gI);

    auto tAttr_mn = retile_fragment(tAttr);
    constexpr int kM = size<0>(tAttr_mn);
    constexpr int kN = size<1>(tAttr_mn);
    Tensor gMax = make_tensor<float>(Int<kM>{});
    Tensor gSum = make_tensor<float>(Int<kM>{});

    constexpr int kTileTransV = 32;
    constexpr int kTileTransN = 16;
    auto sV_tile =
        local_tile(sV, make_tile(Int<kTileTransV>{}, Int<kTileTransN>{}), make_coord(_, _, _));
    auto sVT_tile =
        local_tile(sVT, make_tile(Int<kTileTransV>{}, Int<kTileTransN>{}), make_coord(_, _, _));

    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

    int ismem_read = 0;
    int phase = 0;
    int phase_q = 0;

    float tQS[kM];
    float kscale = kscale_ptr[0];
    float vscale = vscale_ptr[0];

    while (true) {
      if (iblock >= max_total_blocks) {
        break;
      }

      auto [itile_m, ihead_q, ibatch] =
          get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                max_num_tile_m, head_q_divmod, tile_m_divmod);

      if (itile_m < 0) {
        continue;
      }

      int num_seq_kv = shm_seqlens_kv[ibatch];
      int start_seq_q = shm_seqlens_qstart[ibatch];

      int ihead_kv, res;
      head_kv_divmod(ihead_kv, res, ihead_q);
      int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
      int num_tile_full = (start_seq_q + itile_m * kTileM) / kTileN;

      clear(tYr);
      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());

      wait_barrier(readable_q, phase_q);

      auto tI_mn = retile_fragment(tI);
#pragma unroll
      for (int im = 0; im < kM; im++) {
        tQS[im] = sQS(get<0>(tI_mn(im, 0))) * kscale;
      }

#pragma unroll 1
      for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        wait_barrier(readable_k[ismem_read], phase);

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

        if (itile_seq_kv >= num_tile_full) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              int irow = start_seq_q + itile_m * kTileM + get<0>(tI_mn(im, in));
              int icol = itile_seq_kv * kTileN + get<1>(tI_mn(im, in));

              if ((icol > irow) || (icol >= num_seq_kv)) {
                tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
              } else {
                tAttr_mn(im, in) *= tQS[im];
              }
            }
          }
        } else {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              tAttr_mn(im, in) *= tQS[im];
            }
          }
        }

        auto tYr_mn = retile_fragment(tYr);
        // online softmax
        // online_softmax_with_scale(tAttr_mn, gMax, gSum, tYr_mn, tQS, kM, kN, one_over_dk_log2e);
        online_softmax(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);

        // Scale P before FP8 quantization; epilogue applies the reciprocal.
#pragma unroll
        for (int i = 0; i < size(tAttr); ++i) {
          tAttr(i) *= kFp8PrefillPScale;
        }

        // convert P to fp8 and permute for pv gemm
        auto tAttr_float32x4 = recast<float4>(tAttr);
        auto tAttr_fp8x4 = recast<__nv_fp8x4_e4m3>(tAttr_fp8);

#pragma unroll
        for (int i = 0; i < size(tAttr_float32x4); i++) {
          tAttr_fp8x4(i) = __nv_fp8x4_e4m3(tAttr_float32x4(i));
        }

        auto tAttr_fp8_u64 = recast<uint2>(tAttr_fp8);
#pragma unroll
        for (int i = 0; i < size(tAttr_fp8_u64); i++) {
          uint32_t upper = tAttr_fp8_u64(i).x;
          uint32_t lower = tAttr_fp8_u64(i).y;
          tAttr_fp8_u64(i).x = __byte_perm(upper, lower, 0x5410);
          tAttr_fp8_u64(i).y = __byte_perm(upper, lower, 0x7632);
        }

        // Y = PV
        wait_barrier(readable_v[ismem_read], phase);

        auto sV_tile_iwarp = sV_tile(_, _, _, iwarp_in_warpgroup, ismem_read);
        auto sVT_tile_iwarp = sVT_tile(_, _, _, iwarp_in_warpgroup, ismem_read);

        constexpr int kNumTilePerWarp = 32 / kTileTransN;

        auto tOr = make_fragment_like(tYr);
        warpgroup_fence_operand(tOr);
        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;
#pragma unroll
        for (int in = 0; in < size<2>(tVTr); in++) {
#pragma unroll
          for (int iin = 0; iin < kNumTilePerWarp; iin++) {
            smem_trans_and_interleave0189_nm_nm(
                sV_tile(_, _, iwarp_in_warpgroup, in * kNumTilePerWarp + iin, ismem_read),
                sVT_tile(_, _, iwarp_in_warpgroup, in * kNumTilePerWarp + iin,
                         iwarpgroup * kStage + ismem_read),
                idx_in_warp);
          }
          cutlass::arch::fence_view_async_shared();
          // syncwarpgroup(iwarpgroup);

          warpgroup_arrive();
          cute::gemm(tiled_mma_pv, tAttA_fp8(_, _, in),
                     tVTr(_, _, in, iwarpgroup * kStage + ismem_read), tOr);
          warpgroup_commit_batch();
          tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_wait<0>();
        warpgroup_fence_operand(tOr);
#pragma unroll
        for (int i = 0; i < size(tYr); ++i) {
          tYr(i) += tOr(i);
        }

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

      float vscale_eff = vscale * kFp8PrefillPScaleInv;
#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        tYr_bf16(i) = Tout(tYr(i) * vscale_eff);
      }

      // Epilogue: write register-C to global memory
      using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>;
      auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma_pv);
      auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx);

      auto tYr4s = r2s_thr_copy.retile_S(tYr_bf16);
      auto tYs4r = r2s_thr_copy.partition_D(sY);

      cute::copy(r2s_tiled_copy, tYr4s, tYs4r);
      syncwarpgroup(iwarpgroup);
      tma_store_fence();

      // using TMA to store
      if (is_leader_in_warpgroup) {
        auto cY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
        auto btma_y = tma_y.get_slice(0);

        auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
        auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

        auto *td_y = td_qy + ibatch * 2 + 1;
        cute::copy(tma_y.with(td_y), tYss(_, iwarpgroup, 0),
                   tYgg(_, itile_m * 2 + iwarpgroup, 0, ihead_q));
        tma_store_arrive();
      }
    }
  }
}

template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY,
          typename TmaQS, typename TmaKS>
__global__ void __launch_bounds__(384, 1)
    attention_with_kvcache_qkpertoken_perhead_vperhead_prefill_fp8_warp_specialization_kernel(
        cute::TmaDescriptor *td_qy, const __grid_constant__ TmaK tma_k,
        const __grid_constant__ TmaV tma_v, const __grid_constant__ TmaQS tma_qs,
        const __grid_constant__ TmaKS tma_ks, const float *qscale_ptr, const float *kscale_ptr,
        const float *vscale_ptr, const int *cu_seqlens_q_ptr, const int *seqlens_kvcache_ptr,
        const int *block_ids_ptr, int num_batch, int max_seq_q, int max_seq_q_pad, int num_dim_qk,
        int num_dim_v, int num_dim_scale, int num_head_q, int num_head_kv, int num_kvcache_blocks,
        int num_scale_blocks, int block_size, int num_seq_max_blocks, float one_over_dk_log2e,
        cutlass::FastDivmod head_kv_divmod, cutlass::FastDivmod head_q_divmod,
        cutlass::FastDivmod tile_m_divmod) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;
  using SLayoutQ = typename Config::SLayoutQ;
  using SLayoutK = typename Config::SLayoutK;
  using SLayoutV = typename Config::SLayoutV;
  using SLayoutVT = typename Config::SLayoutVT;
  using SLayoutY = typename Config::SLayoutY;
  using SLayoutQS = typename Config::SLayoutQS;
  using SLayoutKS = typename Config::SLayoutKS;
  using SLayoutKSC = typename Config::SLayoutKSC;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;
  constexpr int kBlockSize = Config::kBlockSize;
  constexpr int kScaleBlockSize = Config::kScaleBlockSize;

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  __shared__ uint64_t readable_q;
  __shared__ uint64_t writable_q;
  __shared__ uint64_t readable_k[kStage];
  __shared__ uint64_t writable_k[kStage];
  __shared__ uint64_t readable_v[kStage];
  __shared__ uint64_t writable_v[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_v = shm_k + cosize(SLayoutK{});
  auto *shm_vt = shm_v + cosize(SLayoutV{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_vt + cosize(SLayoutVT{}));
  auto *shm_qs = reinterpret_cast<float *>(shm_y + cosize(SLayoutY{}));
  auto *shm_ks = reinterpret_cast<float *>(shm_qs + cosize(SLayoutQS{}));
  auto *shm_seqlens_q = reinterpret_cast<int *>(shm_ks + cosize(SLayoutKSC{}));
  auto *shm_seqlens_kv = shm_seqlens_q + num_batch;
  auto *shm_seqlens_qstart = shm_seqlens_kv + num_batch;

  TmaQ tma_q;
  TmaY tma_y;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks));
  auto gV =
      tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
  auto gQS = tma_qs.get_tma_tensor(make_shape(max_seq_q_pad, num_head_q, num_batch));
  auto gKS = tma_ks.get_tma_tensor(
      make_shape(kScaleBlockSize, num_dim_scale, num_head_kv, num_scale_blocks));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gAtt_fp8 =
      make_tensor(make_gmem_ptr(static_cast<Tin *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileV>{}), make_stride(Int<kTileV>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sVT = make_tensor(make_smem_ptr(shm_vt), SLayoutVT{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});
  auto sQS = make_tensor(make_smem_ptr(shm_qs), SLayoutQS{});
  auto sKS = make_tensor(make_smem_ptr(shm_ks), SLayoutKS{});
  auto sKSC = make_tensor(make_smem_ptr(shm_ks), SLayoutKSC{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);
  auto btma_y = tma_y.get_slice(0);
  auto btma_qs = tma_qs.get_slice(0);
  auto btma_ks = tma_ks.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);     // (TMA, TMA_M, TMA_K, head, batch)
  auto tKg = btma_k.partition_S(gK);     // (TMA, TMA_N, TMA_K, head, batch)
  auto tVg = btma_v.partition_S(gV);     // (TMA, TMA_V, TMA_N, head, batch)
  auto tQSg = btma_qs.partition_S(gQS);  // (TMA, TMA_M, head, batch)
  auto tKSg = btma_ks.partition_S(gKS);  // (TMA, TMA_M, head, batch)

  auto tQs = btma_q.partition_D(sQ);      // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);      // (TMA, _1, _1, kStage)
  auto tVs = btma_v.partition_D(sV);      // (TMA, _1, _1, kStage)
  auto tQSs = btma_qs.partition_D(sQS);   // (TMA, _1)
  auto tKSs = btma_ks.partition_D(sKSC);  // (TMA, _1)

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;

  auto thr_mma_qk = tiled_mma_qk.get_slice(idx);
  auto thr_mma_pv = tiled_mma_pv.get_slice(idx);

  auto tQs4r = thr_mma_qk.partition_A(sQ);
  auto tKs4r = thr_mma_qk.partition_B(sK);
  auto tVTs4r = thr_mma_pv.partition_B(sVT);

  auto tQr = thr_mma_qk.make_fragment_A(tQs4r);    // (MMA, MMA_M, MMA_K)
  auto tKr = thr_mma_qk.make_fragment_B(tKs4r);    // (MMA, MMA_N, MMA_K)
  auto tVTr = thr_mma_pv.make_fragment_B(tVTs4r);  // (MMA, MMA_V, MMA_N, kStage)

  auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
  auto tYr = thr_mma_pv.partition_fragment_C(gYY);

  // init k/v barrier
  if (is_leader_in_block) {
    initialize_barrier(readable_q, 1);
    initialize_barrier(writable_q, 2);
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable_k[i], 1);
      initialize_barrier(writable_k[i], 2);
      initialize_barrier(readable_v[i], 1);
      initialize_barrier(writable_v[i], 2);
    }
  }

  for (int i = idx; i < num_batch; i += blockDim.x) {
    int num_seq_ibatch = cu_seqlens_q_ptr[i + 1] - cu_seqlens_q_ptr[i];
    shm_seqlens_q[i] = num_seq_ibatch;
    shm_seqlens_kv[i] = seqlens_kvcache_ptr[i];
    shm_seqlens_qstart[i] = seqlens_kvcache_ptr[i] - num_seq_ibatch;
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  int max_num_tile_m = (max_seq_q + kTileM - 1) / kTileM;
  int max_total_blocks = num_head_q * num_batch * max_num_tile_m;

  constexpr int kNumBlockPerTileN = kTileN / kBlockSize;

  if (idx >= 256) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    idx -= 256;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;    // start with ok
      int phase_q = 1;  // start with ok
      int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);

      while (true) {
        if (iblock >= max_total_blocks) {
          break;
        }

        auto [itile_m, ihead_q, ibatch] =
            get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                  max_num_tile_m, head_q_divmod, tile_m_divmod);

        int num_seq_kv = seqlens_kvcache_ptr[ibatch];
        int num_seq_q = shm_seqlens_q[ibatch];
        if (itile_m < 0) {
          continue;
        }

        int ihead_kv, res;
        head_kv_divmod(ihead_kv, res, ihead_q);

        auto *td_q = td_qy + ibatch * 2;

        int start_seq_q = shm_seqlens_qstart[ibatch];
        auto *block_ids_ibatch_ptr = block_ids_ptr + ibatch * num_seq_max_blocks;

        // Load Q
        wait_barrier(writable_q, phase_q);
        cute::copy(tma_q.with(td_q, readable_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
        cute::copy(tma_qs.with(readable_q), tQSg(_, itile_m, ihead_q, ibatch), tQSs(_, 0));
        set_barrier_transaction_bytes(
            readable_q, sizeof(Tin) * cosize(SLayoutQ{}) + sizeof(float) * cosize(SLayoutQS{}));
        phase_q ^= 1;

        int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
        int num_blocks = (num_seq_kv + kBlockSize - 1) / kBlockSize;
        constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK + sizeof(float) * kTileN;
        // constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;
        constexpr int kTransactionBytesV = sizeof(Tin) * kTileV * kTileN;
        // load KV
#pragma unroll 1
        for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
          // k
          wait_barrier(writable_k[ismem_write], phase);

          int iblock_ids[kNumBlockPerTileN];
#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            iblock_ids[iblock_kv] = 0;
            int iblock_id = itile_seq_kv * kNumBlockPerTileN + iblock_kv;
            if (iblock_id < num_blocks) {
              iblock_ids[iblock_kv] = block_ids_ibatch_ptr[iblock_id];
            } else {
              iblock_ids[iblock_kv] = -1;
            }
          }

#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            int iblock_true = iblock_ids[iblock_kv];
            cute::copy(tma_k.with(readable_k[ismem_write]), tKg(_, 0, _, ihead_kv, iblock_true),
                       tKs(_, iblock_kv, _, ismem_write));
            cute::copy(tma_ks.with(readable_k[ismem_write]), tKSg(_, 0, _, ihead_kv, iblock_true),
                       tKSs(_, iblock_kv, _, ismem_write));
          }
          set_barrier_transaction_bytes(readable_k[ismem_write], kTransactionBytesK);

          // v
          wait_barrier(writable_v[ismem_write], phase);
#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            int iblock_true = iblock_ids[iblock_kv];
            cute::copy(tma_v.with(readable_v[ismem_write]), tVg(_, _, 0, ihead_kv, iblock_true),
                       tVs(_, _, iblock_kv, ismem_write));
          }
          set_barrier_transaction_bytes(readable_v[ismem_write], kTransactionBytesV);

          ++ismem_write;
          if (ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<224>();

    int idx_in_warpgroup = idx % 128;
    int idx_in_warp = idx % 32;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    auto tAttr_fp8 = make_tensor_like<Tin>(tAttr);
    auto layout_asC = tAttr.layout();
    auto layout_asA = thr_mma_pv.partition_fragment_A(gAtt_fp8).layout();
    // auto tAttA_fp8 = make_tensor(tAttr_fp8.data(), left_inverse(layout_asC).compose(layout_asA));
    auto tAttA_fp8 = make_tensor(tAttr_fp8.data(), layout_asA);

    auto gI = make_identity_tensor(gAtt.shape());
    auto tI = thr_mma_qk.partition_C(gI);

    auto tAttr_mn = retile_fragment(tAttr);
    constexpr int kM = size<0>(tAttr_mn);
    constexpr int kN = size<1>(tAttr_mn);
    Tensor gMax = make_tensor<float>(Int<kM>{});
    Tensor gSum = make_tensor<float>(Int<kM>{});

    constexpr int kTileTransV = 32;
    constexpr int kTileTransN = 16;
    auto sV_tile =
        local_tile(sV, make_tile(Int<kTileTransV>{}, Int<kTileTransN>{}), make_coord(_, _, _));
    auto sVT_tile =
        local_tile(sVT, make_tile(Int<kTileTransV>{}, Int<kTileTransN>{}), make_coord(_, _, _));

    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

    int ismem_read = 0;
    int phase = 0;
    int phase_q = 0;

    float tQS[kM];
    float tKS[kN];
    // float kscale = kscale_ptr[0];
    // float vscale = vscale_ptr[0];

    while (true) {
      if (iblock >= max_total_blocks) {
        break;
      }

      auto [itile_m, ihead_q, ibatch] =
          get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                max_num_tile_m, head_q_divmod, tile_m_divmod);

      if (itile_m < 0) {
        continue;
      }

      int num_seq_kv = shm_seqlens_kv[ibatch];
      int start_seq_q = shm_seqlens_qstart[ibatch];

      int ihead_kv, res;
      head_kv_divmod(ihead_kv, res, ihead_q);
      int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
      int num_tile_full = (start_seq_q + itile_m * kTileM) / kTileN;

      float vscale = vscale_ptr[ihead_kv];

      clear(tYr);
      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());

      wait_barrier(readable_q, phase_q);

      auto tI_mn = retile_fragment(tI);
#pragma unroll
      for (int im = 0; im < kM; im++) {
        tQS[im] = sQS(get<0>(tI_mn(im, 0)));
      }

#pragma unroll 1
      for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        wait_barrier(readable_k[ismem_read], phase);

#pragma unroll
        for (int ins = 0; ins < kN; ins++) {
          tKS[ins] = sKS(get<1>(tI_mn(0, ins)), ismem_read);
        }

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

        if (itile_seq_kv >= num_tile_full) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              int irow = start_seq_q + itile_m * kTileM + get<0>(tI_mn(im, in));
              int icol = itile_seq_kv * kTileN + get<1>(tI_mn(im, in));

              if ((icol > irow) || (icol >= num_seq_kv)) {
                tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
              } else {
                tAttr_mn(im, in) *= tQS[im] * tKS[in];
              }
            }
          }
        } else {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              tAttr_mn(im, in) *= tQS[im] * tKS[in];
            }
          }
        }

        auto tYr_mn = retile_fragment(tYr);
        // online softmax
        // online_softmax_with_scale(tAttr_mn, gMax, gSum, tYr_mn, tQS, tKS, kM, kN,
        // one_over_dk_log2e);
        online_softmax(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);

        // Scale P before FP8 quantization; epilogue applies the reciprocal.
#pragma unroll
        for (int i = 0; i < size(tAttr); ++i) {
          tAttr(i) *= kFp8PrefillPScale;
        }

        // convert P to fp8 and permute for pv gemm
        auto tAttr_float32x4 = recast<float4>(tAttr);
        auto tAttr_fp8x4 = recast<__nv_fp8x4_e4m3>(tAttr_fp8);

#pragma unroll
        for (int i = 0; i < size(tAttr_float32x4); i++) {
          tAttr_fp8x4(i) = __nv_fp8x4_e4m3(tAttr_float32x4(i));
        }

        auto tAttr_fp8_u64 = recast<uint2>(tAttr_fp8);
#pragma unroll
        for (int i = 0; i < size(tAttr_fp8_u64); i++) {
          uint32_t upper = tAttr_fp8_u64(i).x;
          uint32_t lower = tAttr_fp8_u64(i).y;
          tAttr_fp8_u64(i).x = __byte_perm(upper, lower, 0x5410);
          tAttr_fp8_u64(i).y = __byte_perm(upper, lower, 0x7632);
        }

        // Y = PV
        wait_barrier(readable_v[ismem_read], phase);

        auto sV_tile_iwarp = sV_tile(_, _, _, iwarp_in_warpgroup, ismem_read);
        auto sVT_tile_iwarp = sVT_tile(_, _, _, iwarp_in_warpgroup, ismem_read);

        constexpr int kNumTilePerWarp = 32 / kTileTransN;

        auto tOr = make_fragment_like(tYr);
        warpgroup_fence_operand(tOr);
        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;
#pragma unroll
        for (int in = 0; in < size<2>(tVTr); in++) {
#pragma unroll
          for (int iin = 0; iin < kNumTilePerWarp; iin++) {
            smem_trans_and_interleave0189_nm_nm(
                sV_tile(_, _, iwarp_in_warpgroup, in * kNumTilePerWarp + iin, ismem_read),
                sVT_tile(_, _, iwarp_in_warpgroup, in * kNumTilePerWarp + iin,
                         iwarpgroup * kStage + ismem_read),
                idx_in_warp);
          }
          cutlass::arch::fence_view_async_shared();
          // syncwarpgroup(iwarpgroup);

          warpgroup_arrive();
          cute::gemm(tiled_mma_pv, tAttA_fp8(_, _, in),
                     tVTr(_, _, in, iwarpgroup * kStage + ismem_read), tOr);
          warpgroup_commit_batch();
          tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_wait<0>();
        warpgroup_fence_operand(tOr);
#pragma unroll
        for (int i = 0; i < size(tYr); ++i) {
          tYr(i) += tOr(i);
        }

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

      float vscale_eff = vscale * kFp8PrefillPScaleInv;
#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        tYr_bf16(i) = Tout(tYr(i) * vscale_eff);
      }

      // Epilogue: write register-C to global memory
      using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>;
      auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma_pv);
      auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx);

      auto tYr4s = r2s_thr_copy.retile_S(tYr_bf16);
      auto tYs4r = r2s_thr_copy.partition_D(sY);

      cute::copy(r2s_tiled_copy, tYr4s, tYs4r);
      syncwarpgroup(iwarpgroup);
      tma_store_fence();

      // using TMA to store
      if (is_leader_in_warpgroup) {
        auto cY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
        auto btma_y = tma_y.get_slice(0);

        auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
        auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

        auto *td_y = td_qy + ibatch * 2 + 1;
        cute::copy(tma_y.with(td_y), tYss(_, iwarpgroup, 0),
                   tYgg(_, itile_m * 2 + iwarpgroup, 0, ihead_q));
        tma_store_arrive();
      }
    }
  }
}

// FP8 warp-specialization kernel with paged KV cache and Block-Sparse Attention support,
// dim_qk=128, dim_v=128.
template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY,
          typename TmaQS, bool kHasMask>
__global__ void __launch_bounds__(384, 1)
    attention_with_kvcache_blocksparse_qpertoken_perhead_kvpertensor_prefill_fp8_warp_specialization_kernel(  // NOLINT(whitespace/line_length)
        cute::TmaDescriptor *td_qy, const __grid_constant__ TmaK tma_k,
        const __grid_constant__ TmaV tma_v, const __grid_constant__ TmaQS tma_qs,
        const float *kscale_ptr, const float *vscale_ptr, const int *cu_seqlens_q_ptr,
        const int *seqlens_kvcache_ptr, const int *block_ids_ptr, int num_batch, int max_seq_q,
        int max_seq_q_pad, int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv,
        int num_kvcache_blocks, int num_seq_max_blocks, float one_over_dk_log2e,
        cutlass::FastDivmod head_kv_divmod, cutlass::FastDivmod head_q_divmod,
        cutlass::FastDivmod tile_m_divmod, const uint8_t *block_mask_ptr, int num_tile_kv_in_mask) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;
  using SLayoutQ = typename Config::SLayoutQ;
  using SLayoutK = typename Config::SLayoutK;
  using SLayoutV = typename Config::SLayoutV;
  using SLayoutVT = typename Config::SLayoutVT;
  using SLayoutY = typename Config::SLayoutY;
  using SLayoutQS = typename Config::SLayoutQS;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;
  constexpr int kBlockSize = Config::kBlockSize;

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  __shared__ uint64_t readable_q;
  __shared__ uint64_t writable_q;
  __shared__ uint64_t readable_list;
  __shared__ uint64_t writable_list;
  __shared__ uint64_t readable_k[kStage];
  __shared__ uint64_t writable_k[kStage];
  __shared__ uint64_t readable_v[kStage];
  __shared__ uint64_t writable_v[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_v = shm_k + cosize(SLayoutK{});
  auto *shm_vt = shm_v + cosize(SLayoutV{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_vt + cosize(SLayoutVT{}));
  auto *shm_qs = reinterpret_cast<float *>(shm_y + cosize(SLayoutY{}));
  auto *shm_seqlens_q = reinterpret_cast<int *>(shm_qs + cosize(SLayoutQS{}));
  auto *shm_seqlens_kv = shm_seqlens_q + num_batch;
  auto *shm_seqlens_qstart = shm_seqlens_kv + num_batch;
  // Active tile list: [num_tile_active (int)] [tile_indices (int[])]
  auto *shm_num_active = reinterpret_cast<int *>(shm_seqlens_qstart + num_batch);
  auto *shm_active_tiles = shm_num_active + 1;

  TmaQ tma_q;
  TmaY tma_y;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks));
  auto gV =
      tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
  auto gQS = tma_qs.get_tma_tensor(make_shape(max_seq_q_pad, num_head_q, num_batch));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gAtt_fp8 =
      make_tensor(make_gmem_ptr(static_cast<Tin *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileV>{}), make_stride(Int<kTileV>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sVT = make_tensor(make_smem_ptr(shm_vt), SLayoutVT{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});
  auto sQS = make_tensor(make_smem_ptr(shm_qs), SLayoutQS{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);
  auto btma_y = tma_y.get_slice(0);
  auto btma_qs = tma_qs.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);     // (TMA, TMA_M, TMA_K, head, batch)
  auto tKg = btma_k.partition_S(gK);     // (TMA, TMA_N, TMA_K, head, batch)
  auto tVg = btma_v.partition_S(gV);     // (TMA, TMA_V, TMA_N, head, batch)
  auto tQSg = btma_qs.partition_S(gQS);  // (TMA, TMA_M, head, batch)

  auto tQs = btma_q.partition_D(sQ);     // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);     // (TMA, _1, _1, kStage)
  auto tVs = btma_v.partition_D(sV);     // (TMA, _1, _1, kStage)
  auto tQSs = btma_qs.partition_D(sQS);  // (TMA, _1)

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;

  auto thr_mma_qk = tiled_mma_qk.get_slice(idx);
  auto thr_mma_pv = tiled_mma_pv.get_slice(idx);

  auto tQs4r = thr_mma_qk.partition_A(sQ);
  auto tKs4r = thr_mma_qk.partition_B(sK);
  auto tVTs4r = thr_mma_pv.partition_B(sVT);

  auto tQr = thr_mma_qk.make_fragment_A(tQs4r);    // (MMA, MMA_M, MMA_K)
  auto tKr = thr_mma_qk.make_fragment_B(tKs4r);    // (MMA, MMA_N, MMA_K)
  auto tVTr = thr_mma_pv.make_fragment_B(tVTs4r);  // (MMA, MMA_V, MMA_N, kStage)

  auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
  auto tYr = thr_mma_pv.partition_fragment_C(gYY);

  // init k/v barrier
  if (is_leader_in_block) {
    initialize_barrier(readable_q, 1);
    initialize_barrier(writable_q, 2);
    initialize_barrier(readable_list, 32);
    initialize_barrier(writable_list, 256);
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable_k[i], 1);
      initialize_barrier(writable_k[i], 2);
      initialize_barrier(readable_v[i], 1);
      initialize_barrier(writable_v[i], 2);
    }
  }

  for (int i = idx; i < num_batch; i += blockDim.x) {
    int num_seq_ibatch = cu_seqlens_q_ptr[i + 1] - cu_seqlens_q_ptr[i];
    shm_seqlens_q[i] = num_seq_ibatch;
    shm_seqlens_kv[i] = seqlens_kvcache_ptr[i];
    shm_seqlens_qstart[i] = seqlens_kvcache_ptr[i] - num_seq_ibatch;
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  int max_num_tile_m = (max_seq_q + kTileM - 1) / kTileM;
  int max_total_blocks = num_head_q * num_batch * max_num_tile_m;

  constexpr int kNumBlockPerTileN = kTileN / kBlockSize;

  // Producer Warpgroup
  if (idx >= 256) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= 256;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int idx_in_warp = idx % 32;

    // 32 lanes enter for warp-parallel mask scan; single-thread ops guarded by `if (elected)`.
    if (iwarp == 0) {
      int phase = 1;         // start with ok
      int phase_q = 1;       // start with ok
      int phase_list_w = 1;  // start with ok
      int ismem_write = 0;

      while (true) {
        if (iblock >= max_total_blocks) {
          break;
        }

        auto [itile_m, ihead_q, ibatch] =
            get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                  max_num_tile_m, head_q_divmod, tile_m_divmod);

        int num_seq_kv = seqlens_kvcache_ptr[ibatch];
        int num_seq_q = shm_seqlens_q[ibatch];
        if (itile_m < 0) {
          continue;
        }

        int ihead_kv, res;
        head_kv_divmod(ihead_kv, res, ihead_q);

        auto *td_q = td_qy + ibatch * 2;

        int start_seq_q = shm_seqlens_qstart[ibatch];
        auto *block_ids_ibatch_ptr = block_ids_ptr + ibatch * num_seq_max_blocks;

        // Load Q
        wait_barrier(writable_q, phase_q);
        if (elected) {
          cute::copy(tma_q.with(td_q, readable_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
          cute::copy(tma_qs.with(readable_q), tQSg(_, itile_m, ihead_q, ibatch), tQSs(_, 0));
        }

        int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
        int num_blocks = (num_seq_kv + kBlockSize - 1) / kBlockSize;
        constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;
        constexpr int kTransactionBytesV = sizeof(Tin) * kTileV * kTileN;

        // Build active tile list in SMEM (parallel with TMA Q in flight)
        int num_tile_active = num_tile_kv;
        if constexpr (kHasMask) {
          wait_barrier(writable_list, phase_list_w);
          phase_list_w ^= 1;

          int block_mask_offset = ibatch * (num_head_q * max_num_tile_m * num_tile_kv_in_mask) +
                                  ihead_q * (max_num_tile_m * num_tile_kv_in_mask) +
                                  itile_m * num_tile_kv_in_mask;
          int num_tile_with_mask = min(num_tile_kv, num_tile_kv_in_mask);

          // Warp-parallel stream compaction of active tiles into SMEM.
          num_tile_active = 0;
#pragma unroll 1
          for (int base = 0; base < num_tile_with_mask; base += 32) {
            int i = base + idx_in_warp;
            bool active = (i < num_tile_with_mask) && (block_mask_ptr[block_mask_offset + i] != 0);
            uint32_t ballot = __ballot_sync(0xFFFFFFFF, active);
            // rank within active lanes = popcount of earlier-lane bits in ballot
            if (active) {
              shm_active_tiles[num_tile_active + __popc(ballot & ((1u << idx_in_warp) - 1))] = i;
            }
            num_tile_active += __popc(ballot);
          }

          if (num_tile_with_mask < num_tile_kv) {
            if (elected) {
              shm_active_tiles[num_tile_active] = num_tile_with_mask;
            }
            num_tile_active += 1;
          }

          __syncwarp();

          if (elected) {
            *shm_num_active = num_tile_active;
          }
          arrive_barrier(readable_list);
        }

        if (elected) {
          set_barrier_transaction_bytes(
              readable_q, sizeof(Tin) * cosize(SLayoutQ{}) + sizeof(float) * cosize(SLayoutQS{}));
        }
        phase_q ^= 1;

#pragma unroll 1
        for (int i_active = 0; i_active < num_tile_active; ++i_active) {
          int itile_seq_kv;
          if constexpr (kHasMask) {
            itile_seq_kv = shm_active_tiles[i_active];
          } else {
            itile_seq_kv = i_active;
          }

          // k
          wait_barrier(writable_k[ismem_write], phase);

          int iblock_ids[kNumBlockPerTileN];
#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            iblock_ids[iblock_kv] = -1;
            int iblock_id = itile_seq_kv * kNumBlockPerTileN + iblock_kv;
            if (iblock_id < num_blocks) {
              iblock_ids[iblock_kv] = block_ids_ibatch_ptr[iblock_id];
            }
          }

          if (elected) {
#pragma unroll
            for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
              int iblock_true = iblock_ids[iblock_kv];
              cute::copy(tma_k.with(readable_k[ismem_write]), tKg(_, 0, _, ihead_kv, iblock_true),
                         tKs(_, iblock_kv, _, ismem_write));
            }
            set_barrier_transaction_bytes(readable_k[ismem_write], kTransactionBytesK);
          }

          // v
          wait_barrier(writable_v[ismem_write], phase);
          if (elected) {
#pragma unroll
            for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
              int iblock_true = iblock_ids[iblock_kv];
              cute::copy(tma_v.with(readable_v[ismem_write]), tVg(_, _, 0, ihead_kv, iblock_true),
                         tVs(_, _, iblock_kv, ismem_write));
            }
            set_barrier_transaction_bytes(readable_v[ismem_write], kTransactionBytesV);
          }

          ++ismem_write;
          if (ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }
        __syncwarp();
      }
    }
  } else {  // Consumer Warpgroup
    cutlass::arch::warpgroup_reg_alloc<192>();

    int idx_in_warpgroup = idx % 128;
    int idx_in_warp = idx % 32;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    auto tAttr_fp8 = make_tensor_like<Tin>(tAttr);
    auto layout_asA = thr_mma_pv.partition_fragment_A(gAtt_fp8).layout();
    auto tAttA_fp8 = make_tensor(tAttr_fp8.data(), layout_asA);

    auto gI = make_identity_tensor(gAtt.shape());
    auto tI = thr_mma_qk.partition_C(gI);

    auto tAttr_mn = retile_fragment(tAttr);
    constexpr int kM = size<0>(tAttr_mn);
    constexpr int kN = size<1>(tAttr_mn);
    Tensor gMax = make_tensor<float>(Int<kM>{});
    Tensor gSum = make_tensor<float>(Int<kM>{});

    constexpr int kTileTransV = 32;
    constexpr int kTileTransN = 16;
    auto sV_tile =
        local_tile(sV, make_tile(Int<kTileTransV>{}, Int<kTileTransN>{}), make_coord(_, _, _));
    auto sVT_tile =
        local_tile(sVT, make_tile(Int<kTileTransV>{}, Int<kTileTransN>{}), make_coord(_, _, _));

    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

    int ismem_read = 0;
    int phase = 0;
    int phase_q = 0;
    int phase_list = 0;

    float tQS[kM];
    float kscale = kscale_ptr[0];
    float vscale = vscale_ptr[0];

    while (true) {
      if (iblock >= max_total_blocks) {
        break;
      }

      auto [itile_m, ihead_q, ibatch] =
          get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                max_num_tile_m, head_q_divmod, tile_m_divmod);

      if (itile_m < 0) {
        continue;
      }

      int num_seq_kv = shm_seqlens_kv[ibatch];
      int start_seq_q = shm_seqlens_qstart[ibatch];

      int ihead_kv, res;
      head_kv_divmod(ihead_kv, res, ihead_q);
      int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
      int num_tile_full = (start_seq_q + itile_m * kTileM) / kTileN;

      clear(tYr);
      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());

      wait_barrier(readable_q, phase_q);

      int num_tile_active;
      if constexpr (kHasMask) {
        wait_barrier(readable_list, phase_list);
        phase_list ^= 1;
        num_tile_active = *shm_num_active;
      } else {
        num_tile_active = num_tile_kv;
      }

      auto tI_mn = retile_fragment(tI);
#pragma unroll
      for (int im = 0; im < kM; im++) {
        tQS[im] = sQS(get<0>(tI_mn(im, 0))) * kscale;
      }

#pragma unroll 1
      for (int i_active = 0; i_active < num_tile_active; ++i_active) {
        int itile_seq_kv;
        if constexpr (kHasMask) {
          itile_seq_kv = shm_active_tiles[i_active];
        } else {
          itile_seq_kv = i_active;
        }

        wait_barrier(readable_k[ismem_read], phase);

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

        if (i_active == (num_tile_active - 1)) {
          if (elected_idx_in_warpgroup) {
            arrive_barrier(writable_q);
          }
          phase_q ^= 1;
          if constexpr (kHasMask) {
            arrive_barrier(writable_list);
          }
        }

        if (itile_seq_kv >= num_tile_full) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              int irow = start_seq_q + itile_m * kTileM + get<0>(tI_mn(im, in));
              int icol = itile_seq_kv * kTileN + get<1>(tI_mn(im, in));

              if ((icol > irow) || (icol >= num_seq_kv)) {
                tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
              } else {
                tAttr_mn(im, in) *= tQS[im];
              }
            }
          }
        } else {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              tAttr_mn(im, in) *= tQS[im];
            }
          }
        }

        auto tYr_mn = retile_fragment(tYr);
        // online softmax
        // online_softmax_with_scale(tAttr_mn, gMax, gSum, tYr_mn, tQS, kM, kN, one_over_dk_log2e);
        online_softmax(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);

        // Scale P before FP8 quantization; epilogue applies the reciprocal.
#pragma unroll
        for (int i = 0; i < size(tAttr); ++i) {
          tAttr(i) *= kFp8PrefillPScale;
        }

        // convert P to fp8 and permute for pv gemm
        auto tAttr_float32x4 = recast<float4>(tAttr);
        auto tAttr_fp8x4 = recast<__nv_fp8x4_e4m3>(tAttr_fp8);

#pragma unroll
        for (int i = 0; i < size(tAttr_float32x4); i++) {
          tAttr_fp8x4(i) = __nv_fp8x4_e4m3(tAttr_float32x4(i));
        }

        auto tAttr_fp8_u64 = recast<uint2>(tAttr_fp8);
#pragma unroll
        for (int i = 0; i < size(tAttr_fp8_u64); i++) {
          uint32_t upper = tAttr_fp8_u64(i).x;
          uint32_t lower = tAttr_fp8_u64(i).y;
          tAttr_fp8_u64(i).x = __byte_perm(upper, lower, 0x5410);
          tAttr_fp8_u64(i).y = __byte_perm(upper, lower, 0x7632);
        }

        // Y = PV
        wait_barrier(readable_v[ismem_read], phase);

        constexpr int kNumTilePerWarp = 32 / kTileTransN;

        auto tOr = make_fragment_like(tYr);
        warpgroup_fence_operand(tOr);
        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;
#pragma unroll
        for (int in = 0; in < size<2>(tVTr); in++) {
#pragma unroll
          for (int iin = 0; iin < kNumTilePerWarp; iin++) {
            smem_trans_and_interleave0189_nm_nm(
                sV_tile(_, _, iwarp_in_warpgroup, in * kNumTilePerWarp + iin, ismem_read),
                sVT_tile(_, _, iwarp_in_warpgroup, in * kNumTilePerWarp + iin,
                         iwarpgroup * kStage + ismem_read),
                idx_in_warp);
          }
          cutlass::arch::fence_view_async_shared();

          warpgroup_arrive();
          cute::gemm(tiled_mma_pv, tAttA_fp8(_, _, in),
                     tVTr(_, _, in, iwarpgroup * kStage + ismem_read), tOr);
          warpgroup_commit_batch();
          tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_wait<0>();
        warpgroup_fence_operand(tOr);
#pragma unroll
        for (int i = 0; i < size(tYr); ++i) {
          tYr(i) += tOr(i);
        }

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_v[ismem_read]);
        }

        ++ismem_read;
        if (ismem_read == kStage) {
          phase ^= 1;
          ismem_read = 0;
        }
      }

      // Release Q barrier to prevent deadlock if violated.
      if (num_tile_active == 0) {
        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_q);
        }
        phase_q ^= 1;
      }

      auto tYr_mn = retile_fragment(tYr);
      // final online softmax
      final_online_softmax(tYr_mn, gSum, kM);

      // to bfloat16
      auto tYr_bf16 = make_tensor_like<Tout>(tYr);

      float vscale_eff = vscale * kFp8PrefillPScaleInv;
#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        tYr_bf16(i) = Tout(tYr(i) * vscale_eff);
      }

      // Epilogue: write register-C to global memory
      using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>;
      auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma_pv);
      auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx);

      auto tYr4s = r2s_thr_copy.retile_S(tYr_bf16);
      auto tYs4r = r2s_thr_copy.partition_D(sY);

      cute::copy(r2s_tiled_copy, tYr4s, tYs4r);
      syncwarpgroup(iwarpgroup);
      tma_store_fence();

      // using TMA to store
      if (is_leader_in_warpgroup) {
        auto cY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
        auto btma_y = tma_y.get_slice(0);

        auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
        auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

        auto *td_y = td_qy + ibatch * 2 + 1;
        cute::copy(tma_y.with(td_y), tYss(_, iwarpgroup, 0),
                   tYgg(_, itile_m * 2 + iwarpgroup, 0, ihead_q));
        tma_store_arrive();
      }
    }
  }
}

// FP8 warp-specialization kernel with paged KV cache and Block-Sparse Attention support,
// qkpertoken_perhead_vperhead scale layout, dim_qk=128, dim_v=128.
template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY,
          typename TmaQS, typename TmaKS, bool kHasMask>
__global__ void __launch_bounds__(384, 1)
    attention_with_kvcache_blocksparse_qkpertoken_perhead_vperhead_prefill_fp8_warp_specialization_kernel(  // NOLINT(whitespace/line_length)
        cute::TmaDescriptor *td_qy, const __grid_constant__ TmaK tma_k,
        const __grid_constant__ TmaV tma_v, const __grid_constant__ TmaQS tma_qs,
        const __grid_constant__ TmaKS tma_ks, const float *qscale_ptr, const float *kscale_ptr,
        const float *vscale_ptr, const int *cu_seqlens_q_ptr, const int *seqlens_kvcache_ptr,
        const int *block_ids_ptr, int num_batch, int max_seq_q, int max_seq_q_pad, int num_dim_qk,
        int num_dim_v, int num_dim_scale, int num_head_q, int num_head_kv, int num_kvcache_blocks,
        int num_scale_blocks, int num_seq_max_blocks, float one_over_dk_log2e,
        cutlass::FastDivmod head_kv_divmod, cutlass::FastDivmod head_q_divmod,
        cutlass::FastDivmod tile_m_divmod, const uint8_t *block_mask_ptr, int num_tile_kv_in_mask) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;
  using SLayoutQ = typename Config::SLayoutQ;
  using SLayoutK = typename Config::SLayoutK;
  using SLayoutV = typename Config::SLayoutV;
  using SLayoutVT = typename Config::SLayoutVT;
  using SLayoutY = typename Config::SLayoutY;
  using SLayoutQS = typename Config::SLayoutQS;
  using SLayoutKS = typename Config::SLayoutKS;
  using SLayoutKSC = typename Config::SLayoutKSC;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;
  constexpr int kBlockSize = Config::kBlockSize;
  constexpr int kScaleBlockSize = Config::kScaleBlockSize;

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  __shared__ uint64_t readable_q;
  __shared__ uint64_t writable_q;
  __shared__ uint64_t readable_list;
  __shared__ uint64_t writable_list;
  __shared__ uint64_t readable_k[kStage];
  __shared__ uint64_t writable_k[kStage];
  __shared__ uint64_t readable_v[kStage];
  __shared__ uint64_t writable_v[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_v = shm_k + cosize(SLayoutK{});
  auto *shm_vt = shm_v + cosize(SLayoutV{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_vt + cosize(SLayoutVT{}));
  auto *shm_qs = reinterpret_cast<float *>(shm_y + cosize(SLayoutY{}));
  auto *shm_ks = reinterpret_cast<float *>(shm_qs + cosize(SLayoutQS{}));
  auto *shm_seqlens_q = reinterpret_cast<int *>(shm_ks + cosize(SLayoutKSC{}));
  auto *shm_seqlens_kv = shm_seqlens_q + num_batch;
  auto *shm_seqlens_qstart = shm_seqlens_kv + num_batch;
  // Active tile list: [num_tile_active (int)] [tile_indices (int[])]
  auto *shm_num_active = reinterpret_cast<int *>(shm_seqlens_qstart + num_batch);
  auto *shm_active_tiles = shm_num_active + 1;

  TmaQ tma_q;
  TmaY tma_y;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks));
  auto gV =
      tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
  auto gQS = tma_qs.get_tma_tensor(make_shape(max_seq_q_pad, num_head_q, num_batch));
  auto gKS = tma_ks.get_tma_tensor(
      make_shape(kScaleBlockSize, num_dim_scale, num_head_kv, num_scale_blocks));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gAtt_fp8 =
      make_tensor(make_gmem_ptr(static_cast<Tin *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileV>{}), make_stride(Int<kTileV>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sVT = make_tensor(make_smem_ptr(shm_vt), SLayoutVT{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});
  auto sQS = make_tensor(make_smem_ptr(shm_qs), SLayoutQS{});
  auto sKS = make_tensor(make_smem_ptr(shm_ks), SLayoutKS{});
  auto sKSC = make_tensor(make_smem_ptr(shm_ks), SLayoutKSC{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);
  auto btma_y = tma_y.get_slice(0);
  auto btma_qs = tma_qs.get_slice(0);
  auto btma_ks = tma_ks.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);     // (TMA, TMA_M, TMA_K, head, batch)
  auto tKg = btma_k.partition_S(gK);     // (TMA, TMA_N, TMA_K, head, batch)
  auto tVg = btma_v.partition_S(gV);     // (TMA, TMA_V, TMA_N, head, batch)
  auto tQSg = btma_qs.partition_S(gQS);  // (TMA, TMA_M, head, batch)
  auto tKSg = btma_ks.partition_S(gKS);  // (TMA, TMA_M, head, batch)

  auto tQs = btma_q.partition_D(sQ);      // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);      // (TMA, _1, _1, kStage)
  auto tVs = btma_v.partition_D(sV);      // (TMA, _1, _1, kStage)
  auto tQSs = btma_qs.partition_D(sQS);   // (TMA, _1)
  auto tKSs = btma_ks.partition_D(sKSC);  // (TMA, _1)

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;

  auto thr_mma_qk = tiled_mma_qk.get_slice(idx);
  auto thr_mma_pv = tiled_mma_pv.get_slice(idx);

  auto tQs4r = thr_mma_qk.partition_A(sQ);
  auto tKs4r = thr_mma_qk.partition_B(sK);
  auto tVTs4r = thr_mma_pv.partition_B(sVT);

  auto tQr = thr_mma_qk.make_fragment_A(tQs4r);    // (MMA, MMA_M, MMA_K)
  auto tKr = thr_mma_qk.make_fragment_B(tKs4r);    // (MMA, MMA_N, MMA_K)
  auto tVTr = thr_mma_pv.make_fragment_B(tVTs4r);  // (MMA, MMA_V, MMA_N, kStage)

  auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
  auto tYr = thr_mma_pv.partition_fragment_C(gYY);

  // init k/v barrier
  if (is_leader_in_block) {
    initialize_barrier(readable_q, 1);
    initialize_barrier(writable_q, 2);
    initialize_barrier(readable_list, 32);
    initialize_barrier(writable_list, 256);
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable_k[i], 1);
      initialize_barrier(writable_k[i], 2);
      initialize_barrier(readable_v[i], 1);
      initialize_barrier(writable_v[i], 2);
    }
  }

  for (int i = idx; i < num_batch; i += blockDim.x) {
    int num_seq_ibatch = cu_seqlens_q_ptr[i + 1] - cu_seqlens_q_ptr[i];
    shm_seqlens_q[i] = num_seq_ibatch;
    shm_seqlens_kv[i] = seqlens_kvcache_ptr[i];
    shm_seqlens_qstart[i] = seqlens_kvcache_ptr[i] - num_seq_ibatch;
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  int max_num_tile_m = (max_seq_q + kTileM - 1) / kTileM;
  int max_total_blocks = num_head_q * num_batch * max_num_tile_m;

  constexpr int kNumBlockPerTileN = kTileN / kBlockSize;

  // Producer Warpgroup
  if (idx >= 256) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    idx -= 256;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int idx_in_warp = idx % 32;

    // 32 lanes enter for warp-parallel mask scan; single-thread ops guarded by `if (elected)`.
    if (iwarp == 0) {
      int phase = 1;         // start with ok
      int phase_q = 1;       // start with ok
      int phase_list_w = 1;  // start with ok
      int ismem_write = 0;

      while (true) {
        if (iblock >= max_total_blocks) {
          break;
        }

        auto [itile_m, ihead_q, ibatch] =
            get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                  max_num_tile_m, head_q_divmod, tile_m_divmod);

        int num_seq_kv = seqlens_kvcache_ptr[ibatch];
        int num_seq_q = shm_seqlens_q[ibatch];
        if (itile_m < 0) {
          continue;
        }

        int ihead_kv, res;
        head_kv_divmod(ihead_kv, res, ihead_q);

        auto *td_q = td_qy + ibatch * 2;

        int start_seq_q = shm_seqlens_qstart[ibatch];
        auto *block_ids_ibatch_ptr = block_ids_ptr + ibatch * num_seq_max_blocks;

        // Load Q
        wait_barrier(writable_q, phase_q);
        if (elected) {
          cute::copy(tma_q.with(td_q, readable_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
          cute::copy(tma_qs.with(readable_q), tQSg(_, itile_m, ihead_q, ibatch), tQSs(_, 0));
        }

        int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
        int num_blocks = (num_seq_kv + kBlockSize - 1) / kBlockSize;
        constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK + sizeof(float) * kTileN;
        constexpr int kTransactionBytesV = sizeof(Tin) * kTileV * kTileN;

        // Build active tile list in SMEM (parallel with TMA Q in flight)
        int num_tile_active = num_tile_kv;
        if constexpr (kHasMask) {
          wait_barrier(writable_list, phase_list_w);
          phase_list_w ^= 1;

          int block_mask_offset = ibatch * (num_head_q * max_num_tile_m * num_tile_kv_in_mask) +
                                  ihead_q * (max_num_tile_m * num_tile_kv_in_mask) +
                                  itile_m * num_tile_kv_in_mask;
          int num_tile_with_mask = min(num_tile_kv, num_tile_kv_in_mask);

          // Warp-parallel stream compaction of active tiles into SMEM.
          num_tile_active = 0;
#pragma unroll 1
          for (int base = 0; base < num_tile_with_mask; base += 32) {
            int i = base + idx_in_warp;
            bool active = (i < num_tile_with_mask) && (block_mask_ptr[block_mask_offset + i] != 0);
            uint32_t ballot = __ballot_sync(0xFFFFFFFF, active);
            // rank within active lanes = popcount of earlier-lane bits in ballot
            if (active) {
              shm_active_tiles[num_tile_active + __popc(ballot & ((1u << idx_in_warp) - 1))] = i;
            }
            num_tile_active += __popc(ballot);
          }

          if (num_tile_with_mask < num_tile_kv) {
            if (elected) {
              shm_active_tiles[num_tile_active] = num_tile_with_mask;
            }
            num_tile_active += 1;
          }

          __syncwarp();

          if (elected) {
            *shm_num_active = num_tile_active;
          }
          arrive_barrier(readable_list);
        }

        if (elected) {
          set_barrier_transaction_bytes(
              readable_q, sizeof(Tin) * cosize(SLayoutQ{}) + sizeof(float) * cosize(SLayoutQS{}));
        }
        phase_q ^= 1;

#pragma unroll 1
        for (int i_active = 0; i_active < num_tile_active; ++i_active) {
          int itile_seq_kv;
          if constexpr (kHasMask) {
            itile_seq_kv = shm_active_tiles[i_active];
          } else {
            itile_seq_kv = i_active;
          }

          // k
          wait_barrier(writable_k[ismem_write], phase);

          int iblock_ids[kNumBlockPerTileN];
#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            iblock_ids[iblock_kv] = -1;
            int iblock_id = itile_seq_kv * kNumBlockPerTileN + iblock_kv;
            if (iblock_id < num_blocks) {
              iblock_ids[iblock_kv] = block_ids_ibatch_ptr[iblock_id];
            }
          }

          if (elected) {
#pragma unroll
            for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
              int iblock_true = iblock_ids[iblock_kv];
              cute::copy(tma_k.with(readable_k[ismem_write]), tKg(_, 0, _, ihead_kv, iblock_true),
                         tKs(_, iblock_kv, _, ismem_write));
              cute::copy(tma_ks.with(readable_k[ismem_write]), tKSg(_, 0, _, ihead_kv, iblock_true),
                         tKSs(_, iblock_kv, _, ismem_write));
            }
            set_barrier_transaction_bytes(readable_k[ismem_write], kTransactionBytesK);
          }

          // v
          wait_barrier(writable_v[ismem_write], phase);
          if (elected) {
#pragma unroll
            for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
              int iblock_true = iblock_ids[iblock_kv];
              cute::copy(tma_v.with(readable_v[ismem_write]), tVg(_, _, 0, ihead_kv, iblock_true),
                         tVs(_, _, iblock_kv, ismem_write));
            }
            set_barrier_transaction_bytes(readable_v[ismem_write], kTransactionBytesV);
          }

          ++ismem_write;
          if (ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }
        __syncwarp();
      }
    }
  } else {  // Consumer Warpgroup
    cutlass::arch::warpgroup_reg_alloc<224>();

    int idx_in_warpgroup = idx % 128;
    int idx_in_warp = idx % 32;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    auto tAttr_fp8 = make_tensor_like<Tin>(tAttr);
    auto layout_asA = thr_mma_pv.partition_fragment_A(gAtt_fp8).layout();
    auto tAttA_fp8 = make_tensor(tAttr_fp8.data(), layout_asA);

    auto gI = make_identity_tensor(gAtt.shape());
    auto tI = thr_mma_qk.partition_C(gI);

    auto tAttr_mn = retile_fragment(tAttr);
    constexpr int kM = size<0>(tAttr_mn);
    constexpr int kN = size<1>(tAttr_mn);
    Tensor gMax = make_tensor<float>(Int<kM>{});
    Tensor gSum = make_tensor<float>(Int<kM>{});

    constexpr int kTileTransV = 32;
    constexpr int kTileTransN = 16;
    auto sV_tile =
        local_tile(sV, make_tile(Int<kTileTransV>{}, Int<kTileTransN>{}), make_coord(_, _, _));
    auto sVT_tile =
        local_tile(sVT, make_tile(Int<kTileTransV>{}, Int<kTileTransN>{}), make_coord(_, _, _));

    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

    int ismem_read = 0;
    int phase = 0;
    int phase_q = 0;
    int phase_list = 0;

    float tQS[kM];
    float tKS[kN];

    while (true) {
      if (iblock >= max_total_blocks) {
        break;
      }

      auto [itile_m, ihead_q, ibatch] =
          get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                max_num_tile_m, head_q_divmod, tile_m_divmod);

      if (itile_m < 0) {
        continue;
      }

      int num_seq_kv = shm_seqlens_kv[ibatch];
      int start_seq_q = shm_seqlens_qstart[ibatch];

      int ihead_kv, res;
      head_kv_divmod(ihead_kv, res, ihead_q);
      int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
      int num_tile_full = (start_seq_q + itile_m * kTileM) / kTileN;

      float vscale = vscale_ptr[ihead_kv];

      clear(tYr);
      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());

      wait_barrier(readable_q, phase_q);

      int num_tile_active;
      if constexpr (kHasMask) {
        wait_barrier(readable_list, phase_list);
        phase_list ^= 1;
        num_tile_active = *shm_num_active;
      } else {
        num_tile_active = num_tile_kv;
      }

      auto tI_mn = retile_fragment(tI);
#pragma unroll
      for (int im = 0; im < kM; im++) {
        tQS[im] = sQS(get<0>(tI_mn(im, 0)));
      }

#pragma unroll 1
      for (int i_active = 0; i_active < num_tile_active; ++i_active) {
        int itile_seq_kv;
        if constexpr (kHasMask) {
          itile_seq_kv = shm_active_tiles[i_active];
        } else {
          itile_seq_kv = i_active;
        }

        wait_barrier(readable_k[ismem_read], phase);

#pragma unroll
        for (int ins = 0; ins < kN; ins++) {
          tKS[ins] = sKS(get<1>(tI_mn(0, ins)), ismem_read);
        }

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

        if (i_active == (num_tile_active - 1)) {
          if (elected_idx_in_warpgroup) {
            arrive_barrier(writable_q);
          }
          phase_q ^= 1;
          if constexpr (kHasMask) {
            arrive_barrier(writable_list);
          }
        }

        if (itile_seq_kv >= num_tile_full) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              int irow = start_seq_q + itile_m * kTileM + get<0>(tI_mn(im, in));
              int icol = itile_seq_kv * kTileN + get<1>(tI_mn(im, in));

              if ((icol > irow) || (icol >= num_seq_kv)) {
                tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
              } else {
                tAttr_mn(im, in) *= tQS[im] * tKS[in];
              }
            }
          }
        } else {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              tAttr_mn(im, in) *= tQS[im] * tKS[in];
            }
          }
        }

        auto tYr_mn = retile_fragment(tYr);
        // online softmax
        online_softmax(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);

        // Scale P before FP8 quantization; epilogue applies the reciprocal.
#pragma unroll
        for (int i = 0; i < size(tAttr); ++i) {
          tAttr(i) *= kFp8PrefillPScale;
        }

        // convert P to fp8 and permute for pv gemm
        auto tAttr_float32x4 = recast<float4>(tAttr);
        auto tAttr_fp8x4 = recast<__nv_fp8x4_e4m3>(tAttr_fp8);

#pragma unroll
        for (int i = 0; i < size(tAttr_float32x4); i++) {
          tAttr_fp8x4(i) = __nv_fp8x4_e4m3(tAttr_float32x4(i));
        }

        auto tAttr_fp8_u64 = recast<uint2>(tAttr_fp8);
#pragma unroll
        for (int i = 0; i < size(tAttr_fp8_u64); i++) {
          uint32_t upper = tAttr_fp8_u64(i).x;
          uint32_t lower = tAttr_fp8_u64(i).y;
          tAttr_fp8_u64(i).x = __byte_perm(upper, lower, 0x5410);
          tAttr_fp8_u64(i).y = __byte_perm(upper, lower, 0x7632);
        }

        // Y = PV
        wait_barrier(readable_v[ismem_read], phase);

        constexpr int kNumTilePerWarp = 32 / kTileTransN;

        auto tOr = make_fragment_like(tYr);
        warpgroup_fence_operand(tOr);
        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;
#pragma unroll
        for (int in = 0; in < size<2>(tVTr); in++) {
#pragma unroll
          for (int iin = 0; iin < kNumTilePerWarp; iin++) {
            smem_trans_and_interleave0189_nm_nm(
                sV_tile(_, _, iwarp_in_warpgroup, in * kNumTilePerWarp + iin, ismem_read),
                sVT_tile(_, _, iwarp_in_warpgroup, in * kNumTilePerWarp + iin,
                         iwarpgroup * kStage + ismem_read),
                idx_in_warp);
          }
          cutlass::arch::fence_view_async_shared();

          warpgroup_arrive();
          cute::gemm(tiled_mma_pv, tAttA_fp8(_, _, in),
                     tVTr(_, _, in, iwarpgroup * kStage + ismem_read), tOr);
          warpgroup_commit_batch();
          tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_wait<0>();
        warpgroup_fence_operand(tOr);
#pragma unroll
        for (int i = 0; i < size(tYr); ++i) {
          tYr(i) += tOr(i);
        }

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_v[ismem_read]);
        }

        ++ismem_read;
        if (ismem_read == kStage) {
          phase ^= 1;
          ismem_read = 0;
        }
      }

      // Release Q barrier to prevent deadlock if violated.
      if (num_tile_active == 0) {
        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_q);
        }
        phase_q ^= 1;
      }

      auto tYr_mn = retile_fragment(tYr);
      // final online softmax
      final_online_softmax(tYr_mn, gSum, kM);

      // to bfloat16
      auto tYr_bf16 = make_tensor_like<Tout>(tYr);

      float vscale_eff = vscale * kFp8PrefillPScaleInv;
#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        tYr_bf16(i) = Tout(tYr(i) * vscale_eff);
      }

      // Epilogue: write register-C to global memory
      using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>;
      auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma_pv);
      auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx);

      auto tYr4s = r2s_thr_copy.retile_S(tYr_bf16);
      auto tYs4r = r2s_thr_copy.partition_D(sY);

      cute::copy(r2s_tiled_copy, tYr4s, tYs4r);
      syncwarpgroup(iwarpgroup);
      tma_store_fence();

      // using TMA to store
      if (is_leader_in_warpgroup) {
        auto cY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
        auto btma_y = tma_y.get_slice(0);

        auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
        auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

        auto *td_y = td_qy + ibatch * 2 + 1;
        cute::copy(tma_y.with(td_y), tYss(_, iwarpgroup, 0),
                   tYgg(_, itile_m * 2 + iwarpgroup, 0, ihead_q));
        tma_store_arrive();
      }
    }
  }
}

template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY>
__global__ void attention_prefill_bf16_multi_stage_kernel(cute::TmaDescriptor *td_qkvy,
                                                          int *seqlens_q_ptr, int num_batch,
                                                          int max_seq_q, int num_dim_qk,
                                                          int num_dim_v, int num_head_q,
                                                          int num_head_kv, float one_over_dk_log2e,
                                                          cutlass::FastDivmod HeadKV) {
  using namespace cute;  // NOLINT

  using Tout = typename Config::Tout;
  using Tin = typename Config::Tin;
  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;
  using SLayoutQ = typename Config::SLayoutQ;
  using SLayoutK = typename Config::SLayoutK;
  using SLayoutV = typename Config::SLayoutV;
  using SLayoutY = typename Config::SLayoutY;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;

  int idx = threadIdx.x;
  int itile_m = blockIdx.x;
  int ihead_q = blockIdx.y;
  int ibatch = blockIdx.z;

  int num_seq_q = seqlens_q_ptr[ibatch];
  int num_seq_kv = num_seq_q;
  if (itile_m * kTileM >= num_seq_q) {
    return;
  }

  int ihead_kv, res;
  HeadKV(ihead_kv, res, ihead_q);

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  __shared__ uint64_t bar_q;
  __shared__ uint64_t bar_k[kStage];
  __shared__ uint64_t bar_v[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_v = shm_k + cosize(SLayoutK{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_data);  // Reuse All

  TmaQ tma_q;
  TmaK tma_k;
  TmaV tma_v;
  TmaY tma_y;

  auto *td_q = td_qkvy + ibatch * 4;
  auto *td_k = td_qkvy + ibatch * 4 + 1;
  auto *td_v = td_qkvy + ibatch * 4 + 2;
  auto *td_y = td_qkvy + ibatch * 4 + 3;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gK = tma_k.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_kv));
  auto gV = tma_v.get_tma_tensor(make_shape(num_dim_v, max_seq_q, num_head_kv));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileV>{}), make_stride(Int<kTileV>{}, Int<1>{}));

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
    cute::copy(tma_q.with(td_q, bar_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
    set_barrier_transaction_bytes(bar_q, sizeof(Tin) * cosize(SLayoutQ{}));
  }

  // init k/v barrier
  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(bar_k[i], 1);
      initialize_barrier(bar_v[i], 1);
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();
  wait_barrier(bar_q, 0);

  constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;
  constexpr int kTransactionBytesV = sizeof(Tin) * kTileV * kTileN;
  int ismem_write = 0;
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    if (is_leader_in_block) {
      // k
      cute::copy(tma_k.with(td_k, bar_k[istage]), tKg(_, istage, _, ihead_kv),
                 tKs(_, 0, _, istage));
      set_barrier_transaction_bytes(bar_k[istage], kTransactionBytesK);

      // v
      cute::copy(tma_v.with(td_v, bar_v[istage]), tVg(_, _, istage, ihead_kv),
                 tVs(_, _, 0, istage));
      set_barrier_transaction_bytes(bar_v[istage], kTransactionBytesV);

      ++ismem_write;
    }
  }

  auto layout_asC = thr_mma_qk.partition_C(gAtt).layout();
  auto layout_asA = thr_mma_pv.partition_A(gAtt).layout();
  auto tAttA = make_tensor(tAttr.data(), left_inverse(layout_asC).compose(layout_asA));

  clear(tYr);

  tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

  int num_tile_kv = min(((itile_m + 1) * kTileM + kTileN - 1) / kTileN, size<1>(tKg));
  int num_tile_full = min(itile_m * kTileM, num_seq_kv) / kTileN;

  int phase = 0;
  int ismem_read = 0;
  int itile_write = ismem_write;
#pragma unroll 1
  for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
    // load k/scale/v
    if (itile_write < num_tile_kv) {
      if (is_leader_in_block) {
        // k
        cute::copy(tma_k.with(td_k, bar_k[ismem_write]), tKg(_, itile_write, _, ihead_kv),
                   tKs(_, 0, _, ismem_write));
        set_barrier_transaction_bytes(bar_k[ismem_write], kTransactionBytesK);

        // v
        cute::copy(tma_v.with(td_v, bar_v[ismem_write]), tVg(_, _, itile_write, ihead_kv),
                   tVs(_, _, 0, ismem_write));
        set_barrier_transaction_bytes(bar_v[ismem_write], kTransactionBytesV);

        ismem_write = (ismem_write + 1) % kStage;
      }
    }
    ++itile_write;

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

    // do causal mask
    auto tAttr_mn = retile_fragment(tAttr);
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
      tAttAbf16(i) = static_cast<cute::bfloat16_t>(tAttA(i));
    }

    wait_barrier(bar_v[ismem_read], phase);

    warpgroup_fence_operand(tYr);
    warpgroup_arrive();
    cute::gemm(tiled_mma_pv, tAttAbf16, tVr(_, _, _, ismem_read), tYr);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    ++ismem_read;
    if (ismem_read == kStage) {
      phase ^= 1;
      ismem_read = 0;
    }

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
    auto cY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
    auto btma_y = tma_y.get_slice(0);

    auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
    auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

    cute::copy(tma_y.with(td_y), tYss(_, 0, 0), tYgg(_, itile_m, 0, ihead_q));
  }
}

template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY>
__global__ void attention_with_kvcache_prefill_bf16_multi_stage_kernel(
    cute::TmaDescriptor *td_qy, const __grid_constant__ TmaK tma_k,
    const __grid_constant__ TmaV tma_v, const int *cu_seqlens_q_ptr, const int *seqlens_kvcache_ptr,
    const int *block_ids_ptr, int num_batch, int max_seq_q, int num_dim_qk, int num_dim_v,
    int num_head_q, int num_head_kv, int num_kvcache_blocks, int block_size, int num_seq_max_blocks,
    float one_over_dk_log2e, cutlass::FastDivmod HeadKV) {
  using namespace cute;  // NOLINT

  using Tout = typename Config::Tout;
  using Tin = typename Config::Tin;
  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;
  using SLayoutQ = typename Config::SLayoutQ;
  using SLayoutK = typename Config::SLayoutK;
  using SLayoutV = typename Config::SLayoutV;
  using SLayoutY = typename Config::SLayoutY;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;
  constexpr int kBlockSize = Config::kBlockSize;

  int idx = threadIdx.x;
  int itile_m = blockIdx.x;
  int ihead_q = blockIdx.y;
  int ibatch = blockIdx.z;

  int num_seq_q = cu_seqlens_q_ptr[ibatch + 1] - cu_seqlens_q_ptr[ibatch];
  int num_seq_kv = seqlens_kvcache_ptr[ibatch];
  if (itile_m * kTileM >= num_seq_q) {
    return;
  }

  int ihead_kv, res;
  HeadKV(ihead_kv, res, ihead_q);

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  __shared__ uint64_t bar_q;
  __shared__ uint64_t bar_k[kStage];
  __shared__ uint64_t bar_v[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_v = shm_k + cosize(SLayoutK{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_data);  // Reuse All

  TmaQ tma_q;
  TmaY tma_y;

  auto *td_q = td_qy + ibatch * 2;
  auto *td_y = td_qy + ibatch * 2 + 1;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks));
  auto gV =
      tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileV>{}), make_stride(Int<kTileV>{}, Int<1>{}));

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
    cute::copy(tma_q.with(td_q, bar_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
    set_barrier_transaction_bytes(bar_q, sizeof(Tin) * cosize(SLayoutQ{}));
  }

  // init k/v barrier
  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(bar_k[i], 1);
      initialize_barrier(bar_v[i], 1);
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();
  wait_barrier(bar_q, 0);

  constexpr int kNumBlockPerTileN = kTileN / kBlockSize;
  constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;
  constexpr int kTransactionBytesV = sizeof(Tin) * kTileV * kTileN;

  auto *block_ids_ibatch_ptr = block_ids_ptr + ibatch * num_seq_max_blocks;

  int ismem_write = 0;
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    if (is_leader_in_block) {
      // k

      int iblock_ids[kNumBlockPerTileN];
#pragma unroll
      for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
        iblock_ids[iblock_kv] = 0;
        int iblock_id = istage * kNumBlockPerTileN + iblock_kv;
        if (iblock_id < num_seq_max_blocks) {
          iblock_ids[iblock_kv] = block_ids_ibatch_ptr[iblock_id];
        }
      }
#pragma unroll
      for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
        int iblock_true = iblock_ids[iblock_kv];
        cute::copy(tma_k.with(bar_k[istage]), tKg(_, 0, _, ihead_kv, iblock_true),
                   tKs(_, iblock_kv, _, istage));
      }
      set_barrier_transaction_bytes(bar_k[istage], kTransactionBytesK);

      // v
#pragma unroll
      for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
        int iblock_true = iblock_ids[iblock_kv];
        cute::copy(tma_v.with(bar_v[istage]), tVg(_, _, 0, ihead_kv, iblock_true),
                   tVs(_, _, iblock_kv, istage));
      }
      set_barrier_transaction_bytes(bar_v[istage], kTransactionBytesV);

      ++ismem_write;
    }
  }

  auto layout_asC = thr_mma_qk.partition_C(gAtt).layout();
  auto layout_asA = thr_mma_pv.partition_A(gAtt).layout();
  auto tAttA = make_tensor(tAttr.data(), left_inverse(layout_asC).compose(layout_asA));

  clear(tYr);

  tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

  int start_seq_q = num_seq_kv - num_seq_q;
  int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
  int num_tile_full = (start_seq_q + itile_m * kTileM) / kTileN;

  int phase = 0;
  int ismem_read = 0;
  int itile_write = ismem_write;
#pragma unroll 1
  for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
    // load k/scale/v
    if (itile_write < num_tile_kv) {
      if (is_leader_in_block) {
        // k
        int iblock_ids[kNumBlockPerTileN];
#pragma unroll
        for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
          iblock_ids[iblock_kv] = 0;
          int iblock_id = itile_write * kNumBlockPerTileN + iblock_kv;
          if (iblock_id < num_seq_max_blocks) {
            iblock_ids[iblock_kv] = block_ids_ibatch_ptr[iblock_id];
          }
        }
#pragma unroll
        for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
          int iblock_true = iblock_ids[iblock_kv];
          cute::copy(tma_k.with(bar_k[ismem_write]), tKg(_, 0, _, ihead_kv, iblock_true),
                     tKs(_, iblock_kv, _, ismem_write));
        }
        set_barrier_transaction_bytes(bar_k[ismem_write], kTransactionBytesK);

        // v
#pragma unroll
        for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
          int iblock_true = iblock_ids[iblock_kv];
          cute::copy(tma_v.with(bar_v[ismem_write]), tVg(_, _, 0, ihead_kv, iblock_true),
                     tVs(_, _, iblock_kv, ismem_write));
        }
        set_barrier_transaction_bytes(bar_v[ismem_write], kTransactionBytesV);

        ismem_write = (ismem_write + 1) % kStage;
      }
    }
    ++itile_write;

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

    // do causal mask
    auto tAttr_mn = retile_fragment(tAttr);
    auto tI_mn = retile_fragment(tI);
    if (itile_seq_kv >= num_tile_full) {
#pragma unroll
      for (int im = 0; im < kM; ++im) {
#pragma unroll
        for (int in = 0; in < kN; ++in) {
          int irow = start_seq_q + itile_m * kTileM + get<0>(tI_mn(im, in));
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
      tAttAbf16(i) = static_cast<cute::bfloat16_t>(tAttA(i));
    }

    wait_barrier(bar_v[ismem_read], phase);

    warpgroup_fence_operand(tYr);
    warpgroup_arrive();
    cute::gemm(tiled_mma_pv, tAttAbf16, tVr(_, _, _, ismem_read), tYr);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    ++ismem_read;
    if (ismem_read == kStage) {
      phase ^= 1;
      ismem_read = 0;
    }

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
    auto cY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
    auto btma_y = tma_y.get_slice(0);

    auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
    auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

    cute::copy(tma_y.with(td_y), tYss(_, 0, 0), tYgg(_, itile_m, 0, ihead_q));
  }
}

// FP8 warp-specialization kernel with varlen Q/K/V and Block-Sparse Attention, dim_qk=192,
// dim_v=128
template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY,
          bool kHasMask>
__global__ void __launch_bounds__(384, 1)
    attention_blocksparse_prefill_fp8_dim192_warp_specialization_kernel(
        cute::TmaDescriptor *td_qkvy, const int *cu_seqlens_q_ptr, const int *cu_seqlens_kv_ptr,
        int num_batch, int max_seq_q, int max_seq_kv, int num_dim_qk, int num_dim_v, int num_head_q,
        int num_head_kv, float one_over_dk_log2e, float vscale, cutlass::FastDivmod head_kv_divmod,
        cutlass::FastDivmod head_q_divmod, cutlass::FastDivmod tile_m_divmod,
        const uint8_t *block_mask_ptr, int num_tile_kv_in_mask) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;
  using SLayoutQ = typename Config::SLayoutQ;
  using SLayoutK = typename Config::SLayoutK;
  using SLayoutV = typename Config::SLayoutV;
  using SLayoutVT = typename Config::SLayoutVT;
  using SLayoutY = typename Config::SLayoutY;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  __shared__ uint64_t readable_q;
  __shared__ uint64_t writable_q;
  __shared__ uint64_t readable_list;
  __shared__ uint64_t writable_list;
  __shared__ uint64_t readable_k[kStage];
  __shared__ uint64_t writable_k[kStage];
  __shared__ uint64_t readable_v[kStage];
  __shared__ uint64_t writable_v[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_v = shm_k + cosize(SLayoutK{});
  auto *shm_vt = shm_v + cosize(SLayoutV{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_vt + cosize(SLayoutVT{}));
  auto *shm_seqlens_q = reinterpret_cast<int *>(shm_y + cosize(SLayoutY{}));
  auto *shm_seqlens_kv = shm_seqlens_q + num_batch;
  auto *shm_seqlens_qstart = shm_seqlens_kv + num_batch;
  // Active tile list: [num_tile_active (int)] [tile_indices (int[])]
  auto *shm_num_active = reinterpret_cast<int *>(shm_seqlens_qstart + num_batch);
  auto *shm_active_tiles = shm_num_active + 1;

  TmaQ tma_q;
  TmaK tma_k;
  TmaV tma_v;
  TmaY tma_y;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gK = tma_k.get_tma_tensor(make_shape(max_seq_kv, num_dim_qk, num_head_kv));
  auto gV = tma_v.get_tma_tensor(make_shape(num_dim_v, max_seq_kv, num_head_kv));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gAtt_fp8 =
      make_tensor(make_gmem_ptr(static_cast<Tin *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileV>{}), make_stride(Int<kTileV>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sVT = make_tensor(make_smem_ptr(shm_vt), SLayoutVT{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);
  auto btma_y = tma_y.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, head)
  auto tKg = btma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, head)
  auto tVg = btma_v.partition_S(gV);  // (TMA, TMA_V, TMA_N, head)

  auto tQs = btma_q.partition_D(sQ);  // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);  // (TMA, _1, _1, kStage)
  auto tVs = btma_v.partition_D(sV);  // (TMA, _1, _1, kStage)

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;

  auto thr_mma_qk = tiled_mma_qk.get_slice(idx);
  auto thr_mma_pv = tiled_mma_pv.get_slice(idx);

  auto tQs4r = thr_mma_qk.partition_A(sQ);
  auto tKs4r = thr_mma_qk.partition_B(sK);
  auto tVTs4r = thr_mma_pv.partition_B(sVT);

  auto tQr = thr_mma_qk.make_fragment_A(tQs4r);
  auto tKr = thr_mma_qk.make_fragment_B(tKs4r);
  auto tVTr = thr_mma_pv.make_fragment_B(tVTs4r);

  auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
  auto tYr = thr_mma_pv.partition_fragment_C(gYY);

  // init k/v barrier
  if (is_leader_in_block) {
    initialize_barrier(readable_q, 1);
    initialize_barrier(writable_q, 2);
    initialize_barrier(readable_list, 32);
    initialize_barrier(writable_list, 256);
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable_k[i], 1);
      initialize_barrier(writable_k[i], 2);
      initialize_barrier(readable_v[i], 1);
      initialize_barrier(writable_v[i], 2);
    }
  }

  for (int i = idx; i < num_batch; i += blockDim.x) {
    int num_seq_ibatch = cu_seqlens_q_ptr[i + 1] - cu_seqlens_q_ptr[i];
    int num_seq_kv_ibatch = cu_seqlens_kv_ptr[i + 1] - cu_seqlens_kv_ptr[i];
    shm_seqlens_q[i] = num_seq_ibatch;
    shm_seqlens_kv[i] = num_seq_kv_ibatch;
    shm_seqlens_qstart[i] = num_seq_kv_ibatch - num_seq_ibatch;
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  int max_num_tile_m = (max_seq_q + kTileM - 1) / kTileM;
  int max_total_blocks = num_head_q * num_batch * max_num_tile_m;

  // Producer Warpgroup
  if (idx >= 256) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= 256;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int idx_in_warp = idx % 32;

    // 32 lanes enter for warp-parallel mask scan; single-thread ops guarded by `if (elected)`.
    if (iwarp == 0) {
      int phase = 1;         // start with ok
      int phase_q = 1;       // start with ok
      int phase_list_w = 1;  // start with ok
      int ismem_write = 0;

      while (true) {
        if (iblock >= max_total_blocks) {
          break;
        }

        auto [itile_m, ihead_q, ibatch] =
            get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                  max_num_tile_m, head_q_divmod, tile_m_divmod);

        if (itile_m < 0) {
          continue;
        }

        int ihead_kv, res;
        head_kv_divmod(ihead_kv, res, ihead_q);

        auto *td_q = td_qkvy + ibatch * 4;
        auto *td_k = td_qkvy + ibatch * 4 + 1;
        auto *td_v = td_qkvy + ibatch * 4 + 2;

        int start_seq_q = shm_seqlens_qstart[ibatch];

        // Load Q
        wait_barrier(writable_q, phase_q);
        if (elected) {
          cute::copy(tma_q.with(td_q, readable_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
        }

        int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
        constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;
        constexpr int kTransactionBytesV = sizeof(Tin) * kTileV * kTileN;

        // Build active tile list in SMEM (parallel with TMA Q in flight)
        int num_tile_active = num_tile_kv;
        if constexpr (kHasMask) {
          wait_barrier(writable_list, phase_list_w);
          phase_list_w ^= 1;

          int block_mask_offset = ibatch * (num_head_q * max_num_tile_m * num_tile_kv_in_mask) +
                                  ihead_q * (max_num_tile_m * num_tile_kv_in_mask) +
                                  itile_m * num_tile_kv_in_mask;
          int num_tile_with_mask = min(num_tile_kv, num_tile_kv_in_mask);

          // Warp-parallel stream compaction of active tiles into SMEM.
          num_tile_active = 0;
#pragma unroll 1
          for (int base = 0; base < num_tile_with_mask; base += 32) {
            int i = base + idx_in_warp;
            bool active = (i < num_tile_with_mask) && (block_mask_ptr[block_mask_offset + i] != 0);
            uint32_t ballot = __ballot_sync(0xFFFFFFFF, active);
            // rank within active lanes = popcount of earlier-lane bits in ballot
            if (active) {
              shm_active_tiles[num_tile_active + __popc(ballot & ((1u << idx_in_warp) - 1))] = i;
            }
            num_tile_active += __popc(ballot);
          }

          if (num_tile_with_mask < num_tile_kv) {
            if (elected) {
              shm_active_tiles[num_tile_active] = num_tile_with_mask;
            }
            num_tile_active += 1;
          }

          __syncwarp();

          if (elected) {
            *shm_num_active = num_tile_active;
          }
          arrive_barrier(readable_list);
        }

        if (elected) {
          set_barrier_transaction_bytes(readable_q, sizeof(Tin) * cosize(SLayoutQ{}));
        }
        phase_q ^= 1;

#pragma unroll 1
        for (int i_active = 0; i_active < num_tile_active; ++i_active) {
          int itile_seq_kv;
          if constexpr (kHasMask) {
            itile_seq_kv = shm_active_tiles[i_active];
          } else {
            itile_seq_kv = i_active;
          }

          // k
          wait_barrier(writable_k[ismem_write], phase);
          if (elected) {
            cute::copy(tma_k.with(td_k, readable_k[ismem_write]), tKg(_, itile_seq_kv, _, ihead_kv),
                       tKs(_, 0, _, ismem_write));
            set_barrier_transaction_bytes(readable_k[ismem_write], kTransactionBytesK);
          }

          // v
          wait_barrier(writable_v[ismem_write], phase);
          if (elected) {
            cute::copy(tma_v.with(td_v, readable_v[ismem_write]), tVg(_, _, itile_seq_kv, ihead_kv),
                       tVs(_, _, 0, ismem_write));
            set_barrier_transaction_bytes(readable_v[ismem_write], kTransactionBytesV);
          }

          ++ismem_write;
          if (ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }
        __syncwarp();
      }
    }
  } else {  // Consumer Warpgroup
    cutlass::arch::warpgroup_reg_alloc<192>();

    int idx_in_warpgroup = idx % 128;
    int idx_in_warp = idx % 32;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    auto tAttr_fp8 = make_tensor_like<Tin>(tAttr);
    auto layout_asA = thr_mma_pv.partition_fragment_A(gAtt_fp8).layout();
    auto tAttA_fp8 = make_tensor(tAttr_fp8.data(), layout_asA);

    auto gI = make_identity_tensor(gAtt.shape());
    auto tI = thr_mma_qk.partition_C(gI);

    auto tAttr_mn = retile_fragment(tAttr);
    constexpr int kM = size<0>(tAttr_mn);
    constexpr int kN = size<1>(tAttr_mn);
    Tensor gMax = make_tensor<float>(Int<kM>{});
    Tensor gSum = make_tensor<float>(Int<kM>{});

    constexpr int kTileTransV = 32;
    constexpr int kTileTransN = 16;
    auto sV_tile =
        local_tile(sV, make_tile(Int<kTileTransV>{}, Int<kTileTransN>{}), make_coord(_, _, _));
    auto sVT_tile =
        local_tile(sVT, make_tile(Int<kTileTransV>{}, Int<kTileTransN>{}), make_coord(_, _, _));

    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

    int ismem_read = 0;
    int phase = 0;
    int phase_q = 0;
    int phase_list = 0;

    while (true) {
      if (iblock >= max_total_blocks) {
        break;
      }

      auto [itile_m, ihead_q, ibatch] =
          get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                max_num_tile_m, head_q_divmod, tile_m_divmod);

      if (itile_m < 0) {
        continue;
      }

      int num_seq_kv = shm_seqlens_kv[ibatch];
      int start_seq_q = shm_seqlens_qstart[ibatch];

      int ihead_kv, res;
      head_kv_divmod(ihead_kv, res, ihead_q);
      int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
      int num_tile_full = (start_seq_q + itile_m * kTileM) / kTileN;

      clear(tYr);
      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());

      wait_barrier(readable_q, phase_q);

      int num_tile_active;
      if constexpr (kHasMask) {
        wait_barrier(readable_list, phase_list);
        phase_list ^= 1;
        num_tile_active = *shm_num_active;
      } else {
        num_tile_active = num_tile_kv;
      }

      auto tI_mn = retile_fragment(tI);

#pragma unroll 1
      for (int i_active = 0; i_active < num_tile_active; ++i_active) {
        int itile_seq_kv;
        if constexpr (kHasMask) {
          itile_seq_kv = shm_active_tiles[i_active];
        } else {
          itile_seq_kv = i_active;
        }

        wait_barrier(readable_k[ismem_read], phase);

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

        if (i_active == (num_tile_active - 1)) {
          if (elected_idx_in_warpgroup) {
            arrive_barrier(writable_q);
          }
          phase_q ^= 1;
          if constexpr (kHasMask) {
            arrive_barrier(writable_list);
          }
        }

        if (itile_seq_kv >= num_tile_full) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              int irow = start_seq_q + itile_m * kTileM + get<0>(tI_mn(im, in));
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

        // Scale P before FP8 quantization; epilogue applies the reciprocal.
#pragma unroll
        for (int i = 0; i < size(tAttr); ++i) {
          tAttr(i) *= kFp8PrefillPScale;
        }

        // convert P to fp8 and permute for pv gemm
        auto tAttr_float32x4 = recast<float4>(tAttr);
        auto tAttr_fp8x4 = recast<__nv_fp8x4_e4m3>(tAttr_fp8);

#pragma unroll
        for (int i = 0; i < size(tAttr_float32x4); i++) {
          tAttr_fp8x4(i) = __nv_fp8x4_e4m3(tAttr_float32x4(i));
        }

        auto tAttr_fp8_u64 = recast<uint2>(tAttr_fp8);
#pragma unroll
        for (int i = 0; i < size(tAttr_fp8_u64); i++) {
          uint32_t upper = tAttr_fp8_u64(i).x;
          uint32_t lower = tAttr_fp8_u64(i).y;
          tAttr_fp8_u64(i).x = __byte_perm(upper, lower, 0x5410);
          tAttr_fp8_u64(i).y = __byte_perm(upper, lower, 0x7632);
        }

        // Y = PV
        wait_barrier(readable_v[ismem_read], phase);

        constexpr int kNumTilePerWarp = 32 / kTileTransN;

        auto tOr = make_fragment_like(tYr);
        warpgroup_fence_operand(tOr);
        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;
#pragma unroll
        for (int in = 0; in < size<2>(tVTr); in++) {
#pragma unroll
          for (int iin = 0; iin < kNumTilePerWarp; iin++) {
            smem_trans_and_interleave0189_nm_nm(
                sV_tile(_, _, iwarp_in_warpgroup, in * kNumTilePerWarp + iin, ismem_read),
                sVT_tile(_, _, iwarp_in_warpgroup, in * kNumTilePerWarp + iin,
                         iwarpgroup * kStage + ismem_read),
                idx_in_warp);
          }
          cutlass::arch::fence_view_async_shared();

          warpgroup_arrive();
          cute::gemm(tiled_mma_pv, tAttA_fp8(_, _, in),
                     tVTr(_, _, in, iwarpgroup * kStage + ismem_read), tOr);
          warpgroup_commit_batch();
          tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_wait<0>();
        warpgroup_fence_operand(tOr);
#pragma unroll
        for (int i = 0; i < size(tYr); ++i) {
          tYr(i) += tOr(i);
        }

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_v[ismem_read]);
        }

        ++ismem_read;
        if (ismem_read == kStage) {
          phase ^= 1;
          ismem_read = 0;
        }
      }

      // Release Q barrier to prevent deadlock if violated.
      if (num_tile_active == 0) {
        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_q);
        }
        phase_q ^= 1;
      }

      auto tYr_mn = retile_fragment(tYr);
      // final online softmax
      final_online_softmax(tYr_mn, gSum, kM);

      // to bfloat16
      auto tYr_bf16 = make_tensor_like<Tout>(tYr);

      float vscale_eff = vscale * kFp8PrefillPScaleInv;
#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        tYr_bf16(i) = Tout(tYr(i) * vscale_eff);
      }

      // Epilogue: write register-C to global memory
      using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>;
      auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma_pv);
      auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx);

      auto tYr4s = r2s_thr_copy.retile_S(tYr_bf16);
      auto tYs4r = r2s_thr_copy.partition_D(sY);

      cute::copy(r2s_tiled_copy, tYr4s, tYs4r);
      syncwarpgroup(iwarpgroup);
      tma_store_fence();

      // using TMA to store
      if (is_leader_in_warpgroup) {
        auto cY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
        auto btma_y = tma_y.get_slice(0);

        auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
        auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, head)

        auto *td_y = td_qkvy + ibatch * 4 + 3;
        cute::copy(tma_y.with(td_y), tYss(_, iwarpgroup, 0),
                   tYgg(_, itile_m * 2 + iwarpgroup, 0, ihead_q));
        tma_store_arrive();
      }
    }
  }
}

// =============================================================================
// Hybrid "a8c8-fp16pv": Q fp8 + K/V fp8 storage; Q@Kᵀ runs as FP8 WGMMA (like
// pure-fp8), but P@V stays FP16 WGMMA (fp32 accumulate) like a16c8. This splices
// pure-fp8's QK half (fp8 Q + per-token qscale, fp8 K, fp8 QK MMA, qscale·kscale
// applied per-element in softmax) onto a16c8's PV half (V dequant fp8->fp16 in a
// dedicated cvt warpgroup, P cast fp32->half, F32F16F16 PV MMA, V-scale post-mul
// in the bf16 epilogue). Topology, tiles, masking, and epilogue are a16c8's; the
// bf16-K stage + K-cvt block are dropped (K is consumed fp8 directly).
//
// MODE 21 (static): Q per-(token,head) fp8 + K/V per-tensor. kscale (scalar) is
// folded into the per-row qscale; vscale (scalar) post-muls tYr.
// Config is AttentionKVCachePrefillConfig<float_e4m3_t, ...>: its SLayoutQ /
// SLayoutK / SLayoutV are the fp8 Q / fp8 K stage / fp8 V stage; TiledMmaQK is
// the fp8 QK MMA; TiledMmaPV is the F32F16F16 PV MMA. SLayoutVFp16 (half, MN) is
// the dequantized V WGMMA operand; SLayoutQS holds the per-row qscale.
template <typename Config, typename TmaQ, typename TmaKFp8, typename TmaVFp8, typename TmaY,
          typename TmaQS, typename SLayoutVFp16, typename SLayoutQS>
__global__ void __launch_bounds__(512, 1)
    attention_with_kvcache_prefill_qfp8_kpertensor_vpertensor_fp16_pv_compute_warp_specialization_kernel(  // NOLINT(whitespace/line_length)
        cute::TmaDescriptor *td_qy, const __grid_constant__ TmaKFp8 tma_k_fp8,
        const __grid_constant__ TmaVFp8 tma_v_fp8, const __grid_constant__ TmaQS tma_qs,
        const float *qscale_ptr, const float *kscale_ptr, const float *vscale_ptr,
        const int *cu_seqlens_q_ptr, const int *seqlens_kvcache_ptr, const int *block_ids_ptr,
        int num_batch, int max_seq_q, int max_seq_q_pad, int num_dim_qk, int num_dim_v,
        int num_head_q, int num_head_kv, int num_kvcache_blocks, int block_size,
        int num_seq_max_blocks, float one_over_dk_log2e, cutlass::FastDivmod head_kv_divmod,
        cutlass::FastDivmod head_q_divmod, cutlass::FastDivmod tile_m_divmod) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;  // float_e4m3_t (Q/K/V fp8)
  using Tout = typename Config::Tout;
  using TinPV = cute::half_t;
  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;
  using SLayoutQ = typename Config::SLayoutQ;     // fp8 Q
  using SLayoutKFp8 = typename Config::SLayoutK;  // fp8 K stage
  using SLayoutVFp8 = typename Config::SLayoutV;  // fp8 V stage
  using SLayoutY = typename Config::SLayoutY;

  (void)qscale_ptr;
  float kscale_pertensor = kscale_ptr[0];
  float vscale_pertensor = vscale_ptr[0] * kFp8PrefillPScaleInv;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;
  constexpr int kBlockSize = Config::kBlockSize;

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  __shared__ uint64_t readable_q;
  __shared__ uint64_t writable_q;
  __shared__ uint64_t k_readable[kStage];
  __shared__ uint64_t k_writable[kStage];
  __shared__ uint64_t v_fp8_readable[kStage];
  __shared__ uint64_t v_fp8_writable[kStage];
  __shared__ uint64_t v_fp16_readable[kStage];
  __shared__ uint64_t v_fp16_writable[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  // smem: fp8 Q | fp16 V (compute) | bf16 Y | fp32 QS | fp8 K stage | fp8 V
  // stage | scheduler ints.
  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_v = reinterpret_cast<TinPV *>(shm_q + cosize(SLayoutQ{}));
  auto *shm_y = reinterpret_cast<Tout *>(shm_v + cosize(SLayoutVFp16{}));
  auto *shm_qs = reinterpret_cast<float *>(shm_y + cosize(SLayoutY{}));
  auto *shm_k_fp8 = reinterpret_cast<Tin *>(shm_qs + cosize(SLayoutQS{}));
  auto *shm_v_fp8 = shm_k_fp8 + cosize(SLayoutKFp8{});
  auto *shm_seqlens_q = reinterpret_cast<int *>(shm_v_fp8 + cosize(SLayoutVFp8{}));
  auto *shm_seqlens_kv = shm_seqlens_q + num_batch;
  auto *shm_seqlens_qstart = shm_seqlens_kv + num_batch;

  TmaQ tma_q;
  TmaY tma_y;

  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gK =
      tma_k_fp8.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks));
  auto gV =
      tma_v_fp8.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks));
  auto gQS = tma_qs.get_tma_tensor(make_shape(max_seq_q_pad, num_head_q, num_batch));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileV>{}), make_stride(Int<kTileV>{}, Int<1>{}));

  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutVFp16{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});
  auto sQS = make_tensor(make_smem_ptr(shm_qs), SLayoutQS{});
  auto sK_fp8 = make_tensor(make_smem_ptr(shm_k_fp8), SLayoutKFp8{});
  auto sV_fp8 = make_tensor(make_smem_ptr(shm_v_fp8), SLayoutVFp8{});

  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k_fp8.get_slice(0);
  auto btma_v = tma_v_fp8.get_slice(0);
  auto btma_qs = tma_qs.get_slice(0);

  auto tQg = btma_q.partition_S(gQ);
  auto tKg = btma_k.partition_S(gK);
  auto tVg = btma_v.partition_S(gV);
  auto tQSg = btma_qs.partition_S(gQS);

  auto tQs = btma_q.partition_D(sQ);
  auto tKs = btma_k.partition_D(sK_fp8);
  auto tVs = btma_v.partition_D(sV_fp8);
  auto tQSs = btma_qs.partition_D(sQS);

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;

  auto thr_mma_qk = tiled_mma_qk.get_slice(idx);
  auto thr_mma_pv = tiled_mma_pv.get_slice(idx);

  auto tQs4r = thr_mma_qk.partition_A(sQ);      // fp8 Q
  auto tKs4r = thr_mma_qk.partition_B(sK_fp8);  // fp8 K stage feeds fp8 WGMMA
  auto tVs4r = thr_mma_pv.partition_B(sV);      // fp16 V feeds fp16 WGMMA

  auto tQr = thr_mma_qk.make_fragment_A(tQs4r);
  auto tKr = thr_mma_qk.make_fragment_B(tKs4r);
  auto tVr = thr_mma_pv.make_fragment_B(tVs4r);

  auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
  auto tYr = thr_mma_pv.partition_fragment_C(gYY);

  if (is_leader_in_block) {
    initialize_barrier(readable_q, 1);
    initialize_barrier(writable_q, 2);
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(k_readable[i], 1);
      initialize_barrier(k_writable[i], 2);
      initialize_barrier(v_fp8_readable[i], 1);
      initialize_barrier(v_fp8_writable[i], 1);
      initialize_barrier(v_fp16_readable[i], 1);
      initialize_barrier(v_fp16_writable[i], 2);
    }
  }

  for (int i = idx; i < num_batch; i += blockDim.x) {
    int num_seq_ibatch = cu_seqlens_q_ptr[i + 1] - cu_seqlens_q_ptr[i];
    shm_seqlens_q[i] = num_seq_ibatch;
    shm_seqlens_kv[i] = seqlens_kvcache_ptr[i];
    shm_seqlens_qstart[i] = seqlens_kvcache_ptr[i] - num_seq_ibatch;
  }

  __syncthreads();

  int max_num_tile_m = (max_seq_q + kTileM - 1) / kTileM;
  int max_total_blocks = num_head_q * num_batch * max_num_tile_m;

  constexpr int kNumBlockPerTileN = kTileN / kBlockSize;

  // V cvt runs on the producer's 3 idle warps (warps 9-11) => 96 threads. This
  // shadows the file-level kVCvtThreads(128) used by the dedicated-warpgroup
  // kernels; here the cvt is fused into the 384-thread producer warpgroup.
  constexpr int kVCvtWarps = 3;
  constexpr int kVCvtThreadsPg = kVCvtWarps * 32;

  if (idx >= 256) {
    // Producer warpgroup (384-thread topology): warp 8 = TMA (fp8 Q + QS, fp8 K,
    // fp8 V); warps 9-11 do the V fp8->fp16 cvt (kVCvtWarps=3 idle warps
    // reclaimed — no dedicated 4th cvt warpgroup). v_fp16 ring stays kStage(2)
    // so cvt runs one tile ahead of the consumer (double-buffered hop hiding).
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= 256;

    int iwarp_pg = __shfl_sync(0xFFFFFFFF, idx / 32, 0);

    if (iwarp_pg == 0) {
      // warp 8: TMA producer.
      bool elected_in_warp = cute::elect_one_sync();
      int tid_in_warp = idx % 32;
      bool is_leader_in_load = ((tid_in_warp == 0) && elected_in_warp);
      if (is_leader_in_load) {
        int phase = 1;  // wait k_writable parity (writable side starts hot)
        int phase_q = 1;
        int ismem_write = 0;

        while (true) {
          if (iblock >= max_total_blocks) {
            break;
          }

          auto [itile_m, ihead_q, ibatch] =
              get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                    max_num_tile_m, head_q_divmod, tile_m_divmod);

          if (itile_m < 0) {
            continue;
          }

          int ihead_kv, res;
          head_kv_divmod(ihead_kv, res, ihead_q);

          auto *td_q = td_qy + ibatch * 2;
          int start_seq_q = shm_seqlens_qstart[ibatch];
          auto *block_ids_ibatch_ptr = block_ids_ptr + ibatch * num_seq_max_blocks;

          // Load Q (fp8) + qscale.
          wait_barrier(writable_q, phase_q);
          cute::copy(tma_q.with(td_q, readable_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
          cute::copy(tma_qs.with(readable_q), tQSg(_, itile_m, ihead_q, ibatch), tQSs(_, 0));
          set_barrier_transaction_bytes(
              readable_q, sizeof(Tin) * cosize(SLayoutQ{}) + sizeof(float) * cosize(SLayoutQS{}));
          phase_q ^= 1;

          int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
          constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;
          constexpr int kTransactionBytesV = sizeof(Tin) * kTileV * kTileN;

#pragma unroll 1
          for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
            wait_barrier(k_writable[ismem_write], phase);

            int iblock_ids[kNumBlockPerTileN];
#pragma unroll
            for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
              iblock_ids[iblock_kv] = 0;
              int iblock_id = itile_seq_kv * kNumBlockPerTileN + iblock_kv;
              if (iblock_id < num_seq_max_blocks) {
                iblock_ids[iblock_kv] = block_ids_ibatch_ptr[iblock_id];
              }
            }

#pragma unroll
            for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
              int iblock_true = iblock_ids[iblock_kv];
              cute::copy(tma_k_fp8.with(k_readable[ismem_write]),
                         tKg(_, 0, _, ihead_kv, iblock_true), tKs(_, iblock_kv, _, ismem_write));
            }
            set_barrier_transaction_bytes(k_readable[ismem_write], kTransactionBytesK);

            wait_barrier(v_fp8_writable[ismem_write], phase);
#pragma unroll
            for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
              int iblock_true = iblock_ids[iblock_kv];
              cute::copy(tma_v_fp8.with(v_fp8_readable[ismem_write]),
                         tVg(_, _, 0, ihead_kv, iblock_true), tVs(_, _, iblock_kv, ismem_write));
            }
            set_barrier_transaction_bytes(v_fp8_readable[ismem_write], kTransactionBytesV);

            ++ismem_write;
            if (ismem_write == kStage) {
              ismem_write = 0;
              phase ^= 1;
            }
          }
        }
      }
    } else if (iwarp_pg >= 1 && iwarp_pg <= kVCvtWarps) {
      // V fp8->fp16 cvt on the producer's idle warps 9-11 (kVCvtWarps=3).
      // Mirrors the old dedicated cvt warpgroup, but reuses the producer's
      // register-deallocated warps instead of a 4th warpgroup. v_fp16 ring is
      // kStage(2): cvt leads the consumer by one tile, hiding the hop latency.
      int idx_in_cvt = idx - 32;  // warp 9 -> 0, warp 10 -> 32, warp 11 -> 64
      bool is_cvt_leader = (idx_in_cvt == 0);

      int phase_in = 0;
      int phase_out = 1;
      int s = 0;

      while (true) {
        if (iblock >= max_total_blocks) {
          break;
        }

        auto [itile_m, ihead_q, ibatch] =
            get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                  max_num_tile_m, head_q_divmod, tile_m_divmod);
        if (itile_m < 0) {
          continue;
        }

        int start_seq_q = shm_seqlens_qstart[ibatch];
        int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;

#pragma unroll 1
        for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
          wait_barrier(v_fp8_readable[s], phase_in);
          wait_barrier(v_fp16_writable[s], phase_out);
          {
            auto sVfp8_stage = sV_fp8(_, _, s);
            auto sV_stage = sV(_, _, s);
            hpc::attention::prefill::fp8_smem_to_half_smem_tile_raw_vec16_mn<
                kTileV, kTileN, /*kThreads=*/kVCvtThreadsPg>(sVfp8_stage, sV_stage, idx_in_cvt);
          }
          cutlass::arch::fence_view_async_shared();
          bar_sync<kVCvtThreadsPg>(kVCvtBarrierId);
          if (is_cvt_leader) {
            arrive_barrier(v_fp8_writable[s]);
            arrive_barrier(v_fp16_readable[s]);
          }

          ++s;
          if (s == kStage) {
            s = 0;
            phase_in ^= 1;
            phase_out ^= 1;
          }
        }
      }
    }
  } else {
    // Consumer warpgroups (threads 0..255, 2x wg): fp8 QK WGMMA against the fp8
    // K stage, then fp16 PV WGMMA against the dequantized V stage.
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
    int phase = 0;  // wait *_readable parity (cold at startup)
    int phase_q = 0;

    float tQS[kM];

    while (true) {
      if (iblock >= max_total_blocks) {
        break;
      }

      auto [itile_m, ihead_q, ibatch] =
          get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                max_num_tile_m, head_q_divmod, tile_m_divmod);

      if (itile_m < 0) {
        continue;
      }

      int num_seq_kv = shm_seqlens_kv[ibatch];
      int start_seq_q = shm_seqlens_qstart[ibatch];

      int ihead_kv, res;
      head_kv_divmod(ihead_kv, res, ihead_q);
      int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
      int num_tile_full = (start_seq_q + itile_m * kTileM) / kTileN;

      clear(tYr);
      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());

      wait_barrier(readable_q, phase_q);

      auto tI_mn = retile_fragment(tI);
      // Per-row qscale (kscale per-tensor scalar folded in).
#pragma unroll
      for (int im = 0; im < kM; im++) {
        tQS[im] = sQS(get<0>(tI_mn(im, 0))) * kscale_pertensor;
      }

#pragma unroll 1
      for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        wait_barrier(k_readable[ismem_read], phase);

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
          arrive_barrier(k_writable[ismem_read]);
        }

        if (itile_seq_kv == (num_tile_kv - 1)) {
          if (elected_idx_in_warpgroup) {
            arrive_barrier(writable_q);
          }
          phase_q ^= 1;
        }

        // Causal mask + per-row qscale (pure-fp8 style).
        if (itile_seq_kv >= num_tile_full) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              int irow = start_seq_q + itile_m * kTileM + get<0>(tI_mn(im, in));
              int icol = itile_seq_kv * kTileN + get<1>(tI_mn(im, in));
              if ((icol > irow) || (icol >= num_seq_kv)) {
                tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
              } else {
                tAttr_mn(im, in) *= tQS[im];
              }
            }
          }
        } else {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              tAttr_mn(im, in) *= tQS[im];
            }
          }
        }

        auto tYr_mn = retile_fragment(tYr);
        online_softmax(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);

        // Scale P before the fp16 cast; epilogue folds the reciprocal into vscale.
#pragma unroll
        for (int i = 0; i < size(tAttr); ++i) {
          tAttr(i) *= kFp8PrefillPScale;
        }

        auto tAttAhalf = make_tensor_like<TinPV>(tAttA);
#pragma unroll
        for (int i = 0; i < size(tAttA); ++i) {
          tAttAhalf(i) = (TinPV)(tAttA(i));
        }

        // Wait for cvt-V warp to publish fp16 V stage.
        wait_barrier(v_fp16_readable[ismem_read], phase);

        warpgroup_fence_operand(tYr);
        warpgroup_arrive();
        cute::gemm(tiled_mma_pv, tAttAhalf, tVr(_, _, _, ismem_read), tYr);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        if (elected_idx_in_warpgroup) {
          arrive_barrier(v_fp16_writable[ismem_read]);
        }

        ++ismem_read;
        if (ismem_read == kStage) {
          phase ^= 1;
          ismem_read = 0;
        }
      }

      auto tYr_mn = retile_fragment(tYr);
      final_online_softmax(tYr_mn, gSum, kM);

      // Per-tensor vscale post-mul on tYr.
      auto tYr_bf16 = make_tensor_like<Tout>(tYr);
#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        Tout v{tYr(i) * vscale_pertensor};
        tYr_bf16(i) = v;
      }

      using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>;
      auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma_pv);
      auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx);

      auto tYr4s = r2s_thr_copy.retile_S(tYr_bf16);
      auto tYs4r = r2s_thr_copy.partition_D(sY);

      cute::copy(r2s_tiled_copy, tYr4s, tYs4r);
      asm volatile("barrier.sync %0, 128;\n" ::"r"(iwarpgroup) : "memory");
      tma_store_fence();

      if (is_leader_in_warpgroup) {
        auto cY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
        auto btma_y = tma_y.get_slice(0);

        auto tYss = btma_y.partition_S(sY);
        auto tYgg = btma_y.partition_D(cY);

        auto *td_y = td_qy + ibatch * 2 + 1;
        cute::copy(tma_y.with(td_y), tYss(_, iwarpgroup, 0),
                   tYgg(_, itile_m * 2 + iwarpgroup, 0, ihead_q));
        tma_store_arrive();
      }
    }
  }
}

// Hybrid a8c8-fp16pv MODE 20 (dynamic): Q per-(token,head) fp8 + K per-(token,
// head) + V per-head. Vs mode 21: producer warp 8 also TMAs KS (per-(token,head)
// K scale, 1 fp32 per K-token row, sharing the k_readable barrier); the consumer
// reads it into a per-column register and multiplies qscale·kscale per element in
// softmax (no K-cvt fold, since K stays fp8). V per-head scale post-muls tYr.
template <typename Config, typename TmaQ, typename TmaKFp8, typename TmaVFp8, typename TmaY,
          typename TmaQS, typename TmaKS, typename SLayoutVFp16, typename SLayoutQS,
          typename SLayoutKS, typename SLayoutKSC, typename CopyBoxKS>
__global__ void __launch_bounds__(512, 1)
    attention_with_kvcache_prefill_qfp8_kpertoken_perhead_vperhead_fp16_pv_compute_warp_specialization_kernel(  // NOLINT(whitespace/line_length)
        cute::TmaDescriptor *td_qy, const __grid_constant__ TmaKFp8 tma_k_fp8,
        const __grid_constant__ TmaVFp8 tma_v_fp8, const __grid_constant__ TmaQS tma_qs,
        const __grid_constant__ TmaKS tma_ks, const float *qscale_ptr, const float *kscale_ptr,
        const float *vscale_ptr, const int *cu_seqlens_q_ptr, const int *seqlens_kvcache_ptr,
        const int *block_ids_ptr, int num_batch, int max_seq_q, int max_seq_q_pad, int num_dim_qk,
        int num_dim_v, int num_dim_scale, int num_head_q, int num_head_kv, int num_kvcache_blocks,
        int num_scale_blocks, int block_size, int num_seq_max_blocks, float one_over_dk_log2e,
        cutlass::FastDivmod head_kv_divmod, cutlass::FastDivmod head_q_divmod,
        cutlass::FastDivmod tile_m_divmod) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;  // float_e4m3_t
  using Tout = typename Config::Tout;
  using TinPV = cute::half_t;
  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;
  using SLayoutQ = typename Config::SLayoutQ;
  using SLayoutKFp8 = typename Config::SLayoutK;
  using SLayoutVFp8 = typename Config::SLayoutV;
  using SLayoutY = typename Config::SLayoutY;

  (void)qscale_ptr;
  (void)kscale_ptr;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;
  constexpr int kBlockSize = Config::kBlockSize;

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  __shared__ uint64_t readable_q;
  __shared__ uint64_t writable_q;
  __shared__ uint64_t k_readable[kStage];
  __shared__ uint64_t k_writable[kStage];
  __shared__ uint64_t v_fp8_readable[kStage];
  __shared__ uint64_t v_fp8_writable[kStage];
  __shared__ uint64_t v_fp16_readable[kStage];
  __shared__ uint64_t v_fp16_writable[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  // smem: fp8 Q | fp16 V (compute) | bf16 Y | fp32 QS | fp8 K stage | fp8 V
  // stage | fp32 KS | scheduler ints.
  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_v = reinterpret_cast<TinPV *>(shm_q + cosize(SLayoutQ{}));
  auto *shm_y = reinterpret_cast<Tout *>(shm_v + cosize(SLayoutVFp16{}));
  auto *shm_qs = reinterpret_cast<float *>(shm_y + cosize(SLayoutY{}));
  auto *shm_k_fp8 = reinterpret_cast<Tin *>(shm_qs + cosize(SLayoutQS{}));
  auto *shm_v_fp8 = shm_k_fp8 + cosize(SLayoutKFp8{});
  auto *shm_ks = reinterpret_cast<float *>(shm_v_fp8 + cosize(SLayoutVFp8{}));
  auto *shm_seqlens_q = reinterpret_cast<int *>(shm_ks + cosize(SLayoutKSC{}));
  auto *shm_seqlens_kv = shm_seqlens_q + num_batch;
  auto *shm_seqlens_qstart = shm_seqlens_kv + num_batch;

  TmaQ tma_q;
  TmaY tma_y;

  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gK =
      tma_k_fp8.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks));
  auto gV =
      tma_v_fp8.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks));
  auto gQS = tma_qs.get_tma_tensor(make_shape(max_seq_q_pad, num_head_q, num_batch));
  auto gKS = tma_ks.get_tma_tensor(
      make_shape(block_size / num_dim_scale, num_dim_scale, num_head_kv, num_scale_blocks));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileV>{}), make_stride(Int<kTileV>{}, Int<1>{}));

  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutVFp16{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});
  auto sQS = make_tensor(make_smem_ptr(shm_qs), SLayoutQS{});
  auto sK_fp8 = make_tensor(make_smem_ptr(shm_k_fp8), SLayoutKFp8{});
  auto sV_fp8 = make_tensor(make_smem_ptr(shm_v_fp8), SLayoutVFp8{});
  // KS: 3D (kTileN/kTileScale, kTileScale, kStage) flattened to (kTileN, kStage)
  // for the consumer (1 fp32 per K-token row).
  auto sKS_C = make_tensor(make_smem_ptr(shm_ks), SLayoutKSC{});
  auto sKS = make_tensor(sKS_C.data(), make_layout(make_shape(Int<kTileN>{}, Int<kStage>{})));

  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k_fp8.get_slice(0);
  auto btma_v = tma_v_fp8.get_slice(0);
  auto btma_qs = tma_qs.get_slice(0);
  auto btma_ks = tma_ks.get_slice(0);

  auto tQg = btma_q.partition_S(gQ);
  auto tKg = btma_k.partition_S(gK);
  auto tVg = btma_v.partition_S(gV);
  auto tQSg = btma_qs.partition_S(gQS);
  auto tKSg = btma_ks.partition_S(gKS);

  auto tQs = btma_q.partition_D(sQ);
  auto tKs = btma_k.partition_D(sK_fp8);
  auto tVs = btma_v.partition_D(sV_fp8);
  auto tQSs = btma_qs.partition_D(sQS);
  auto tKSs = btma_ks.partition_D(sKS_C);

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;

  auto thr_mma_qk = tiled_mma_qk.get_slice(idx);
  auto thr_mma_pv = tiled_mma_pv.get_slice(idx);

  auto tQs4r = thr_mma_qk.partition_A(sQ);
  auto tKs4r = thr_mma_qk.partition_B(sK_fp8);
  auto tVs4r = thr_mma_pv.partition_B(sV);

  auto tQr = thr_mma_qk.make_fragment_A(tQs4r);
  auto tKr = thr_mma_qk.make_fragment_B(tKs4r);
  auto tVr = thr_mma_pv.make_fragment_B(tVs4r);

  auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
  auto tYr = thr_mma_pv.partition_fragment_C(gYY);

  if (is_leader_in_block) {
    initialize_barrier(readable_q, 1);
    initialize_barrier(writable_q, 2);
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(k_readable[i], 1);
      initialize_barrier(k_writable[i], 2);
      initialize_barrier(v_fp8_readable[i], 1);
      initialize_barrier(v_fp8_writable[i], 1);
      initialize_barrier(v_fp16_readable[i], 1);
      initialize_barrier(v_fp16_writable[i], 2);
    }
  }

  for (int i = idx; i < num_batch; i += blockDim.x) {
    int num_seq_ibatch = cu_seqlens_q_ptr[i + 1] - cu_seqlens_q_ptr[i];
    shm_seqlens_q[i] = num_seq_ibatch;
    shm_seqlens_kv[i] = seqlens_kvcache_ptr[i];
    shm_seqlens_qstart[i] = seqlens_kvcache_ptr[i] - num_seq_ibatch;
  }

  __syncthreads();

  int max_num_tile_m = (max_seq_q + kTileM - 1) / kTileM;
  int max_total_blocks = num_head_q * num_batch * max_num_tile_m;

  constexpr int kNumBlockPerTileN = kTileN / kBlockSize;

  // V cvt runs on the producer's 3 idle warps (warps 9-11) => 96 threads. This
  // shadows the file-level kVCvtThreads(128) used by the dedicated-warpgroup
  // kernels; here the cvt is fused into the 384-thread producer warpgroup.
  constexpr int kVCvtWarps = 3;
  constexpr int kVCvtThreadsPg = kVCvtWarps * 32;

  if (idx >= 256) {
    // Producer warpgroup (384-thread topology): warp 8 = TMA (fp8 Q + QS, fp8 K
    // + KS, fp8 V); warps 9-11 do the V fp8->fp16 cvt (kVCvtWarps=3 idle warps
    // reclaimed — no dedicated 4th cvt warpgroup). v_fp16 ring stays kStage(2)
    // so cvt runs one tile ahead of the consumer (double-buffered hop hiding).
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= 256;

    int iwarp_pg = __shfl_sync(0xFFFFFFFF, idx / 32, 0);

    if (iwarp_pg == 0) {
      bool elected_in_warp = cute::elect_one_sync();
      int tid_in_warp = idx % 32;
      bool is_leader_in_load = ((tid_in_warp == 0) && elected_in_warp);
      if (is_leader_in_load) {
        int phase = 1;
        int phase_q = 1;
        int ismem_write = 0;

        while (true) {
          if (iblock >= max_total_blocks) {
            break;
          }
          auto [itile_m, ihead_q, ibatch] =
              get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                    max_num_tile_m, head_q_divmod, tile_m_divmod);
          if (itile_m < 0) {
            continue;
          }

          int ihead_kv, res;
          head_kv_divmod(ihead_kv, res, ihead_q);

          auto *td_q = td_qy + ibatch * 2;
          int start_seq_q = shm_seqlens_qstart[ibatch];
          auto *block_ids_ibatch_ptr = block_ids_ptr + ibatch * num_seq_max_blocks;

          // Load Q (fp8) + qscale.
          wait_barrier(writable_q, phase_q);
          cute::copy(tma_q.with(td_q, readable_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
          cute::copy(tma_qs.with(readable_q), tQSg(_, itile_m, ihead_q, ibatch), tQSs(_, 0));
          set_barrier_transaction_bytes(
              readable_q, sizeof(Tin) * cosize(SLayoutQ{}) + sizeof(float) * cosize(SLayoutQS{}));
          phase_q ^= 1;

          int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
          // K barrier carries fp8 K + fp32 KS payloads jointly.
          constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK + sizeof(float) * kTileN;
          constexpr int kTransactionBytesV = sizeof(Tin) * kTileV * kTileN;

#pragma unroll 1
          for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
            wait_barrier(k_writable[ismem_write], phase);

            int iblock_ids[kNumBlockPerTileN];
#pragma unroll
            for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
              iblock_ids[iblock_kv] = 0;
              int iblock_id = itile_seq_kv * kNumBlockPerTileN + iblock_kv;
              if (iblock_id < num_seq_max_blocks) {
                iblock_ids[iblock_kv] = block_ids_ibatch_ptr[iblock_id];
              }
            }

#pragma unroll
            for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
              int iblock_true = iblock_ids[iblock_kv];
              cute::copy(tma_k_fp8.with(k_readable[ismem_write]),
                         tKg(_, 0, _, ihead_kv, iblock_true), tKs(_, iblock_kv, _, ismem_write));
              cute::copy(tma_ks.with(k_readable[ismem_write]), tKSg(_, 0, _, ihead_kv, iblock_true),
                         tKSs(_, iblock_kv, _, ismem_write));
            }
            set_barrier_transaction_bytes(k_readable[ismem_write], kTransactionBytesK);

            wait_barrier(v_fp8_writable[ismem_write], phase);
#pragma unroll
            for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
              int iblock_true = iblock_ids[iblock_kv];
              cute::copy(tma_v_fp8.with(v_fp8_readable[ismem_write]),
                         tVg(_, _, 0, ihead_kv, iblock_true), tVs(_, _, iblock_kv, ismem_write));
            }
            set_barrier_transaction_bytes(v_fp8_readable[ismem_write], kTransactionBytesV);

            ++ismem_write;
            if (ismem_write == kStage) {
              ismem_write = 0;
              phase ^= 1;
            }
          }
        }
      }
    } else if (iwarp_pg >= 1 && iwarp_pg <= kVCvtWarps) {
      // V fp8->fp16 cvt on the producer's idle warps 9-11 (kVCvtWarps=3).
      // Mirrors the old dedicated cvt warpgroup, but reuses the producer's
      // register-deallocated warps instead of a 4th warpgroup. v_fp16 ring is
      // kStage(2): cvt leads the consumer by one tile, hiding the hop latency.
      int idx_in_cvt = idx - 32;  // warp 9 -> 0, warp 10 -> 32, warp 11 -> 64
      bool is_cvt_leader = (idx_in_cvt == 0);

      int phase_in = 0;
      int phase_out = 1;
      int s = 0;

      while (true) {
        if (iblock >= max_total_blocks) {
          break;
        }

        auto [itile_m, ihead_q, ibatch] =
            get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                  max_num_tile_m, head_q_divmod, tile_m_divmod);
        if (itile_m < 0) {
          continue;
        }

        int start_seq_q = shm_seqlens_qstart[ibatch];
        int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;

#pragma unroll 1
        for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
          wait_barrier(v_fp8_readable[s], phase_in);
          wait_barrier(v_fp16_writable[s], phase_out);
          {
            auto sVfp8_stage = sV_fp8(_, _, s);
            auto sV_stage = sV(_, _, s);
            hpc::attention::prefill::fp8_smem_to_half_smem_tile_raw_vec16_mn<
                kTileV, kTileN, /*kThreads=*/kVCvtThreadsPg>(sVfp8_stage, sV_stage, idx_in_cvt);
          }
          cutlass::arch::fence_view_async_shared();
          bar_sync<kVCvtThreadsPg>(kVCvtBarrierId);
          if (is_cvt_leader) {
            arrive_barrier(v_fp8_writable[s]);
            arrive_barrier(v_fp16_readable[s]);
          }

          ++s;
          if (s == kStage) {
            s = 0;
            phase_in ^= 1;
            phase_out ^= 1;
          }
        }
      }
    }
  } else {
    // Consumer warpgroups: fp8 QK + qscale·kscale softmax, fp16 PV.
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

    float tQS[kM];
    float tKS[kN];

    while (true) {
      if (iblock >= max_total_blocks) {
        break;
      }

      auto [itile_m, ihead_q, ibatch] =
          get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                max_num_tile_m, head_q_divmod, tile_m_divmod);

      if (itile_m < 0) {
        continue;
      }

      int num_seq_kv = shm_seqlens_kv[ibatch];
      int start_seq_q = shm_seqlens_qstart[ibatch];

      int ihead_kv, res;
      head_kv_divmod(ihead_kv, res, ihead_q);
      float vscale_perhead = vscale_ptr[ihead_kv] * kFp8PrefillPScaleInv;
      int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
      int num_tile_full = (start_seq_q + itile_m * kTileM) / kTileN;

      clear(tYr);
      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());

      wait_barrier(readable_q, phase_q);

      auto tI_mn = retile_fragment(tI);
#pragma unroll
      for (int im = 0; im < kM; im++) {
        tQS[im] = sQS(get<0>(tI_mn(im, 0)));
      }

#pragma unroll 1
      for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        wait_barrier(k_readable[ismem_read], phase);

#pragma unroll
        for (int in = 0; in < kN; in++) {
          tKS[in] = sKS(get<1>(tI_mn(0, in)), ismem_read);
        }

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
          arrive_barrier(k_writable[ismem_read]);
        }

        if (itile_seq_kv == (num_tile_kv - 1)) {
          if (elected_idx_in_warpgroup) {
            arrive_barrier(writable_q);
          }
          phase_q ^= 1;
        }

        // Causal mask + per-row qscale × per-col kscale (pure-fp8 style).
        if (itile_seq_kv >= num_tile_full) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              int irow = start_seq_q + itile_m * kTileM + get<0>(tI_mn(im, in));
              int icol = itile_seq_kv * kTileN + get<1>(tI_mn(im, in));
              if ((icol > irow) || (icol >= num_seq_kv)) {
                tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
              } else {
                tAttr_mn(im, in) *= tQS[im] * tKS[in];
              }
            }
          }
        } else {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              tAttr_mn(im, in) *= tQS[im] * tKS[in];
            }
          }
        }

        auto tYr_mn = retile_fragment(tYr);
        online_softmax(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);

#pragma unroll
        for (int i = 0; i < size(tAttr); ++i) {
          tAttr(i) *= kFp8PrefillPScale;
        }

        auto tAttAhalf = make_tensor_like<TinPV>(tAttA);
#pragma unroll
        for (int i = 0; i < size(tAttA); ++i) {
          tAttAhalf(i) = (TinPV)(tAttA(i));
        }

        wait_barrier(v_fp16_readable[ismem_read], phase);

        warpgroup_fence_operand(tYr);
        warpgroup_arrive();
        cute::gemm(tiled_mma_pv, tAttAhalf, tVr(_, _, _, ismem_read), tYr);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        if (elected_idx_in_warpgroup) {
          arrive_barrier(v_fp16_writable[ismem_read]);
        }

        ++ismem_read;
        if (ismem_read == kStage) {
          phase ^= 1;
          ismem_read = 0;
        }
      }

      auto tYr_mn = retile_fragment(tYr);
      final_online_softmax(tYr_mn, gSum, kM);

      // Per-head V-scale post-mul on tYr.
      auto tYr_bf16 = make_tensor_like<Tout>(tYr);
#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        Tout v{tYr(i) * vscale_perhead};
        tYr_bf16(i) = v;
      }

      using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>;
      auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma_pv);
      auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx);

      auto tYr4s = r2s_thr_copy.retile_S(tYr_bf16);
      auto tYs4r = r2s_thr_copy.partition_D(sY);

      cute::copy(r2s_tiled_copy, tYr4s, tYs4r);
      asm volatile("barrier.sync %0, 128;\n" ::"r"(iwarpgroup) : "memory");
      tma_store_fence();

      if (is_leader_in_warpgroup) {
        auto cY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
        auto btma_y = tma_y.get_slice(0);

        auto tYss = btma_y.partition_S(sY);
        auto tYgg = btma_y.partition_D(cY);

        auto *td_y = td_qy + ibatch * 2 + 1;
        cute::copy(tma_y.with(td_y), tYss(_, iwarpgroup, 0),
                   tYgg(_, itile_m * 2 + iwarpgroup, 0, ihead_q));
        tma_store_arrive();
      }
    }
  }
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_PREFILL_KERNELS_CUH_
