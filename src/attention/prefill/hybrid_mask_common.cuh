// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_PREFILL_HYBRID_MASK_COMMON_CUH_
#define SRC_ATTENTION_PREFILL_HYBRID_MASK_COMMON_CUH_

#include <cuda.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/attention/prefill/kernels.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

struct HybridMask {
  const int *spans_b;
  int max_spans;

  __device__ __forceinline__ int bound_of(int q_abs) const {
#pragma unroll 1
    for (int j = 0; j < max_spans; ++j) {
      int s = spans_b[2 * j];
      int e = spans_b[2 * j + 1];
      if (s <= q_abs && q_abs <= e) {
        return e + 1;
      }
    }
    return q_abs + 1;
  }
};

__device__ __forceinline__ HybridMask make_hybrid_mask(const int *mm_prefix_range_ptr,
                                                       int max_spans, int ibatch) {
  return HybridMask{mm_prefix_range_ptr + static_cast<int64_t>(ibatch) * max_spans * 2, max_spans};
}

template <bool kPackGQA>
__device__ __forceinline__ int row_to_q_abs(int row, int start_seq_q, int pack_factor) {
  if constexpr (kPackGQA) {
    return start_seq_q + row / pack_factor;
  } else {
    return start_seq_q + row;
  }
}

template <int kTileM, int kTileN, bool kHybridMask, bool kPackGQA = false>
__device__ __forceinline__ int get_num_tile_kv(const int *mm_prefix_range_ptr, int max_spans,
                                               int ibatch, int start_seq_q, int itile_m,
                                               int m_extent, int num_seq_kv, int pack_factor = 1) {
  if constexpr (kPackGQA) {
    int last_row = min((itile_m + 1) * kTileM - 1, m_extent - 1);
    int q_abs_last = row_to_q_abs<kPackGQA>(last_row, start_seq_q, pack_factor);
    int bound;
    if constexpr (kHybridMask) {
      auto mask = make_hybrid_mask(mm_prefix_range_ptr, max_spans, ibatch);
      bound = mask.bound_of(q_abs_last);
    } else {
      bound = q_abs_last + 1;
    }
    int max_bound = min(bound, num_seq_kv);
    return (max_bound + kTileN - 1) / kTileN;
  } else {
    if constexpr (kHybridMask) {
      auto mask = make_hybrid_mask(mm_prefix_range_ptr, max_spans, ibatch);
      int last_q_local = min((itile_m + 1) * kTileM - 1, m_extent - 1);
      int max_bound = min(mask.bound_of(start_seq_q + last_q_local), num_seq_kv);
      return (max_bound + kTileN - 1) / kTileN;
    } else {
      return (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
    }
  }
}

template <int kM, bool kHybridMask, bool kPackGQA = false, typename TI>
__device__ __forceinline__ void get_bounds(const int *mm_prefix_range_ptr, int max_spans,
                                           int ibatch, int start_seq_q, int m_tile_base,
                                           int pack_factor, int num_seq_kv, const TI &tI,
                                           int (&gM)[kM]) {
  auto tI_mn = retile_fragment(tI);
  if constexpr (kHybridMask) {
    auto mask = make_hybrid_mask(mm_prefix_range_ptr, max_spans, ibatch);
#pragma unroll
    for (int im = 0; im < kM; ++im) {
      int row = m_tile_base + cute::get<0>(tI_mn(im, cute::_0{}));
      gM[im] =
          min(mask.bound_of(row_to_q_abs<kPackGQA>(row, start_seq_q, pack_factor)), num_seq_kv);
    }
  } else {
#pragma unroll
    for (int im = 0; im < kM; ++im) {
      int row = m_tile_base + cute::get<0>(tI_mn(im, cute::_0{}));
      gM[im] = min(row_to_q_abs<kPackGQA>(row, start_seq_q, pack_factor) + 1, num_seq_kv);
    }
  }
}

template <typename T1, typename T2, typename TmaQ, typename TmaY, int kPackG = 1>
__global__ void update_batched_tma_with_kvcache_packg(const vec_t<cute::TmaDescriptor, 2> td_qy,
                                                      cute::TmaDescriptor *tma_qy, const T1 *q_ptr,
                                                      const T2 *y_ptr, const int *cu_seqlens_q_ptr,
                                                      int num_batch, int max_seq_q, int num_dim_qk,
                                                      int num_dim_v, int num_head_q,
                                                      int num_head_kv, int ldQ, int ldY) {
  using namespace cute;  // NOLINT
  constexpr bool kPackGQA = (kPackG > 1);

  int idx = threadIdx.x;
  int ibatch = blockIdx.x;

  __shared__ cute::TmaDescriptor smem_tma_desc[2];

  int num_seq = cu_seqlens_q_ptr[ibatch + 1] - cu_seqlens_q_ptr[ibatch];
  int cu_seqlen_q = cu_seqlens_q_ptr[ibatch];
  auto *q_ibatch_ptr = q_ptr + static_cast<int64_t>(cu_seqlen_q) * ldQ;
  auto *y_ibatch_ptr = y_ptr + static_cast<int64_t>(cu_seqlen_q) * ldY;

  if (idx < 2) {
    smem_tma_desc[idx] = td_qy[idx];
  }
  __syncwarp();

  // Q
  if (idx == 0) {
    auto gQ = [&]() {
      if constexpr (kPackGQA) {
        return make_tensor(
            make_gmem_ptr(q_ibatch_ptr),
            make_shape(make_shape(Int<kPackG>{}, num_seq), num_dim_qk, num_head_kv),
            make_stride(make_stride(num_dim_qk, ldQ), Int<1>{}, Int<kPackG>{} * num_dim_qk));
      } else {
        return make_tensor(make_gmem_ptr(q_ibatch_ptr), make_shape(num_seq, num_dim_qk, num_head_q),
                           make_stride(ldQ, Int<1>{}, num_dim_qk));
      }
    }();
    update_tma_gtensor<TmaQ>(smem_tma_desc[idx], gQ);
  }

  // Y
  if (idx == 1) {
    auto gY = [&]() {
      if constexpr (kPackGQA) {
        return make_tensor(
            make_gmem_ptr(y_ibatch_ptr),
            make_shape(make_shape(Int<kPackG>{}, num_seq), num_dim_v, num_head_kv),
            make_stride(make_stride(num_dim_v, ldY), Int<1>{}, Int<kPackG>{} * num_dim_v));
      } else {
        return make_tensor(make_gmem_ptr(y_ibatch_ptr), make_shape(num_seq, num_dim_v, num_head_q),
                           make_stride(ldY, Int<1>{}, num_dim_v));
      }
    }();
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

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_PREFILL_HYBRID_MASK_COMMON_CUH_
