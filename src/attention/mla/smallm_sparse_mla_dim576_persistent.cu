// Copyright 2025 hpc-ops authors

#include <cuda.h>

#include <type_traits>

#include "src/attention/mla/smallm_sparse_mla_dim576_persistent.h"
#include "src/attention/mla/smallm_sparse_mla_dim576_persistent_launch.h"

namespace hpc {
namespace attention {
namespace mla {

bool smallm_sparse_mla_dim576_persistent_async(
    void* y_ptr, const void* q_ptr, const void* kvcache_ptr, float* y_partial_ptr, float* lse_ptr,
    int* task_tensor_ptr, const int* block_ids_ptr, const int* topk_ids_ptr,
    const int* cu_seqlens_q_ptr, const float* sink_weight_ptr, int num_batch, int total_seq_q,
    int num_head_q, int qk_dim, int v_dim, int num_kvcache_blocks, int num_seq_max_blocks,
    int num_max_topk, int block_size, int ldY, int ldQ, int ldKV, float softmax_scale,
    cudaStream_t stream, bool task_tensor_prebuilt, bool splitk, bool prefill) {
  constexpr int kTileNope = 512;
  constexpr int kTileRope = 64;
  constexpr int kTileV = 512;
  constexpr int kBlockSize = 64;

  if (qk_dim != (kTileNope + kTileRope) || v_dim != kTileV) {
    return false;
  }
  if (block_size != kBlockSize) {
    return false;
  }
  if (num_max_topk > kSparseDim576MaxNumTopk) {
    return false;
  }
  if (num_head_q < 1 || num_head_q > 64) {
    return false;
  }

  auto dispatch = [&](auto kTileM_tag, auto kUseSink_tag, auto kNumMathWG_tag) {
    constexpr int kTileM = decltype(kTileM_tag)::value;
    constexpr bool kUseSink = decltype(kUseSink_tag)::value;
    constexpr int kNumMathWG = decltype(kNumMathWG_tag)::value;
    run_sparse_dim576_persistent<kTileM, kNumMathWG, kUseSink>(
        y_ptr, q_ptr, kvcache_ptr, y_partial_ptr, lse_ptr, task_tensor_ptr, block_ids_ptr,
        topk_ids_ptr, cu_seqlens_q_ptr, sink_weight_ptr, num_batch, total_seq_q, num_head_q, qk_dim,
        v_dim, num_kvcache_blocks, num_seq_max_blocks, num_max_topk, ldY, ldQ, ldKV, softmax_scale,
        stream, task_tensor_prebuilt, splitk, prefill);
  };

  auto dispatch_sink = [&](auto kTileM_tag, auto kNumMathWG_tag) {
    if (sink_weight_ptr != nullptr) {
      dispatch(kTileM_tag, std::true_type{}, kNumMathWG_tag);
    } else {
      dispatch(kTileM_tag, std::false_type{}, kNumMathWG_tag);
    }
  };

  // dispatch by interval; each kTileM tile picks the smallest kHeadsTotal that
  // covers num_head_q. The kernel's epilogue mask + TMA OOB zero-fill handle
  // any leftover head slots when num_head_q is not a power of 2.
  if (num_head_q > 32) {  // 33..64
    dispatch_sink(std::integral_constant<int, 32>{}, std::integral_constant<int, 2>{});
  } else if (num_head_q > 16) {  // 17..32
    dispatch_sink(std::integral_constant<int, 32>{}, std::integral_constant<int, 1>{});
  } else if (num_head_q > 8) {  // 9..16
    dispatch_sink(std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
  } else {  // 1..8
    dispatch_sink(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
  }
  return true;
}

}  // namespace mla
}  // namespace attention
}  // namespace hpc
