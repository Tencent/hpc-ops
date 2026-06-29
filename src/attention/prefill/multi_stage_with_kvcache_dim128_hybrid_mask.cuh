// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_PREFILL_MULTI_STAGE_WITH_KVCACHE_DIM128_HYBRID_MASK_CUH_
#define SRC_ATTENTION_PREFILL_MULTI_STAGE_WITH_KVCACHE_DIM128_HYBRID_MASK_CUH_

#include <cuda.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/attention/prefill/hybrid_mask_common.cuh"
#include "src/attention/prefill/kernels.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY,
          bool kHybridMask = false, int kPackG = 1>
__global__ void attention_with_kvcache_prefill_bf16_multi_stage_hybrid_mask_kernel(
    cute::TmaDescriptor *td_qy, const __grid_constant__ TmaK tma_k,
    const __grid_constant__ TmaV tma_v, const int *cu_seqlens_q_ptr, const int *seqlens_kvcache_ptr,
    const int *block_ids_ptr, const int *mm_prefix_range_ptr, int max_spans, int num_batch,
    int max_seq_q, int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv,
    int num_kvcache_blocks, int block_size, int num_seq_max_blocks, float one_over_dk_log2e,
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
  constexpr int kBlockSize = Config::kBlockSize;
  constexpr bool kPackGQA = (kPackG > 1);
  constexpr int pack_factor = kPackG;

  int idx = threadIdx.x;
  int itile_m = gridDim.x - 1 - blockIdx.x;
  int qy_head = blockIdx.y;
  int ibatch = blockIdx.z;

  int num_seq_q = cu_seqlens_q_ptr[ibatch + 1] - cu_seqlens_q_ptr[ibatch];
  int num_seq_kv = seqlens_kvcache_ptr[ibatch];
  int m_extent = kPackGQA ? num_seq_q * pack_factor : num_seq_q;
  if (itile_m * kTileM >= m_extent) {
    return;
  }

  int ihead_kv;
  if constexpr (kPackGQA) {
    ihead_kv = blockIdx.y;
  } else {
    int res;
    HeadKV(ihead_kv, res, qy_head);
  }

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
  auto gQ = [&]() {
    if constexpr (kPackGQA) {
      return tma_q.get_tma_tensor(
          make_shape(make_shape(Int<kPackG>{}, max_seq_q), num_dim_qk, num_head_kv));
    } else {
      return tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
    }
  }();
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks));
  auto gV =
      tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks));
  auto gY = [&]() {
    if constexpr (kPackGQA) {
      return tma_y.get_tma_tensor(
          make_shape(make_shape(Int<kPackG>{}, max_seq_q), num_dim_v, num_head_kv));
    } else {
      return tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
    }
  }();

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

  auto tQs = btma_q.partition_D([&]() {
    if constexpr (kPackGQA) {
      return make_tensor(make_smem_ptr(shm_q),
                         logical_divide(SLayoutQ{}, make_tile(Int<kPackG>{}, Int<kTileK>{})));
    } else {
      return sQ;
    }
  }());                               // (TMA, _1, _1)
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
    cute::copy(tma_q.with(td_q, bar_q), tQg(_, itile_m, _, qy_head), tQs(_, 0, _));
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
  int first_seq = kPackGQA ? (itile_m * kTileM) / pack_factor : (itile_m * kTileM);
  int num_tile_full = (start_seq_q + first_seq) / kTileN;
  int gM[kM];
  get_bounds<kM, kHybridMask, kPackGQA>(mm_prefix_range_ptr, max_spans, ibatch, start_seq_q,
                                        itile_m * kTileM, pack_factor, num_seq_kv, tI, gM);
  int num_tile_kv = get_num_tile_kv<kTileM, kTileN, kHybridMask, kPackGQA>(
      mm_prefix_range_ptr, max_spans, ibatch, start_seq_q, itile_m, m_extent, num_seq_kv,
      pack_factor);

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

    // do mask (causal or hybrid); per-row bound precomputed in gM.
    auto tAttr_mn = retile_fragment(tAttr);
    auto tI_mn = retile_fragment(tI);
    if (itile_seq_kv >= num_tile_full) {
      int icol_base = itile_seq_kv * kTileN;
#pragma unroll
      for (int im = 0; im < kM; ++im) {
        int bound = gM[im];
#pragma unroll
        for (int in = 0; in < kN; ++in) {
          int icol = icol_base + get<1>(tI_mn(im, in));
          if (icol >= bound) {
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
    auto cY = [&]() {
      if constexpr (kPackGQA) {
        return tma_y.get_tma_tensor(
            make_shape(make_shape(Int<kPackG>{}, max_seq_q), num_dim_v, num_head_kv));
      } else {
        return tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
      }
    }();
    auto btma_y = tma_y.get_slice(0);

    auto tYss = btma_y.partition_S([&]() {
      if constexpr (kPackGQA) {
        return make_tensor(make_smem_ptr(shm_y),
                           logical_divide(SLayoutY{}, make_tile(Int<kPackG>{}, Int<kTileV>{})));
      } else {
        return sY;
      }
    }());                                // (TMA, TMA_M, TMA_N)
    auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

    cute::copy(tma_y.with(td_y), tYss(_, 0, 0), tYgg(_, itile_m, 0, qy_head));
  }
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_PREFILL_MULTI_STAGE_WITH_KVCACHE_DIM128_HYBRID_MASK_CUH_
