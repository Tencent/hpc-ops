// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_MLA_SMALLM_SPARSE_MLA_DIM576_PERSISTENT_LAUNCH_CUH_
#define SRC_ATTENTION_MLA_SMALLM_SPARSE_MLA_DIM576_PERSISTENT_LAUNCH_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <type_traits>

#include "cute/tensor.hpp"
#include "src/attention/mla/smallm_mla_dim576_persistent.h"  // scheduler / combine reuse
#include "src/attention/mla/smallm_mla_dim576_persistent_combine_kernels.cuh"
#include "src/attention/mla/smallm_mla_dim576_persistent_get_scheduler_map.cuh"
#include "src/attention/mla/smallm_sparse_mla_dim576_persistent.h"
#include "src/attention/mla/smallm_sparse_mla_dim576_persistent_kernels.cuh"
#include "src/attention/mla/smallm_sparse_mla_dim576_persistent_launch.h"
#include "src/utils/utils.h"

namespace hpc {
namespace attention {
namespace mla {

template <int kTileM, int kTileN, int kTileNope, int kTileRope, int kTileV, int kBlockSize,
          bool kUseSink, int kNumMathWG = 1>
void launch_sparse_dim576_persistent(void* y_ptr, const void* q_ptr, const void* kvcache_ptr,
                                     float* y_partial_ptr, float* lse_ptr, int* task_tensor_ptr,
                                     const int* block_ids_ptr, const int* topk_ids_ptr,
                                     const int* cu_seqlens_q_ptr, const float* sink_weight_ptr,
                                     int num_batch, int total_seq_q, int num_head_q, int qk_dim,
                                     int v_dim, int num_kvcache_blocks, int num_seq_max_blocks,
                                     int num_max_topk, int ldY, int ldQ, int ldKV,
                                     float softmax_scale, cudaStream_t stream,
                                     bool task_tensor_prebuilt, bool splitk, bool prefill) {
  using namespace cute;  // NOLINT
  constexpr int kStage = 2;
  constexpr int kQHeadsTotal = kTileM * kNumMathWG;
  static_assert(kTileNope + kTileRope == 576, "dim576 persistent expects 512+64 split");
  static_assert(kTileNope == kTileV, "sV aliases sK_nope; requires kTileNope == kTileV");
  static_assert(kNumMathWG == 1 || kNumMathWG == 2, "only single/dual math WG supported");

  using Tin = cute::bfloat16_t;

  int num_sm = hpc::get_sm_count();

  int* task_list = task_tensor_ptr + dim576_persistent_task_list_offset();
  int* cu_tasks = task_tensor_ptr + dim576_persistent_cu_tasks_offset(total_seq_q, num_sm);
  int* cu_splits = task_tensor_ptr + dim576_persistent_cu_splits_offset(total_seq_q, num_sm);

  if (!task_tensor_prebuilt) {
    dim576_persistent_get_scheduler_map_sparse_async(task_list, cu_tasks, cu_splits,
                                                     cu_seqlens_q_ptr, num_batch, total_seq_q,
                                                     num_max_topk, num_sm, stream, splitk);
  }

  auto Q_nope = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(q_ptr)),
                            make_shape(num_head_q, Int<kTileNope>{}, total_seq_q),
                            make_stride(qk_dim, Int<1>{}, ldQ));
  auto Q_rope = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(q_ptr) + kTileNope),
                            make_shape(num_head_q, Int<kTileRope>{}, total_seq_q),
                            make_stride(qk_dim, Int<1>{}, ldQ));

  auto slayout_q_nope = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                      make_shape(Int<kQHeadsTotal>{}, Int<kTileNope>{}));
  auto slayout_q_rope = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                      make_shape(Int<kQHeadsTotal>{}, Int<kTileRope>{}));
  auto slayout_k_nope = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                      make_shape(Int<kTileN>{}, Int<kTileNope>{}, Int<kStage>{}));
  auto slayout_k_rope = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                      make_shape(Int<kTileN>{}, Int<kTileRope>{}, Int<kStage>{}));
  auto slayout_v_one_stage = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{},
                                           make_shape(Int<kTileV>{}, Int<kTileN>{}), GenRowMajor{});
  auto slayout_v =
      tile_to_shape(slayout_v_one_stage, make_shape(Int<kTileV>{}, Int<kTileN>{}, Int<kStage>{}));
  auto slayout_p =
      tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{}, make_shape(Int<kTileN>{}, Int<kTileM>{}));
  auto slayout_s =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileN>{}));

  auto tma_copy_layout_q_nope = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                              make_shape(Int<kQHeadsTotal>{}, Int<kTileNope>{}));
  auto tma_copy_layout_q_rope = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                              make_shape(Int<kQHeadsTotal>{}, Int<kTileRope>{}));

  auto tma_q_nope = make_tma_copy(SM90_TMA_LOAD{}, Q_nope, tma_copy_layout_q_nope);
  auto tma_q_rope = make_tma_copy(SM90_TMA_LOAD{}, Q_rope, tma_copy_layout_q_rope);

  using WarpgroupLayout = decltype(make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})));
  using AtomQK =
      typename kernels::Dim576PersistentAtomSelector<kTileM>::template Mma<GMMA::Major::K,
                                                                           GMMA::Major::K>;
  using AtomSV =
      typename kernels::Dim576PersistentAtomSelector<kTileM>::template Mma<GMMA::Major::MN,
                                                                           GMMA::Major::K>;
  using TiledMmaQK = decltype(make_tiled_mma(AtomQK{}, WarpgroupLayout{}));
  using TiledMmaSV = decltype(make_tiled_mma(AtomSV{}, WarpgroupLayout{}));

  dim3 attn_block(size(TiledMmaQK{}) * kNumMathWG + 128);
  dim3 attn_grid(num_sm);

  constexpr int kWarpsPerWrapGroup = 4;
  int shm_size = (cosize(slayout_q_nope) + cosize(slayout_q_rope) + cosize(slayout_k_nope) +
                  cosize(slayout_k_rope) + cosize(slayout_p) * kNumMathWG) *
                     sizeof(Tin) +
                 sizeof(float) * kQHeadsTotal * kWarpsPerWrapGroup * 2;

  constexpr float kLog2e = 1.4426950408889634f;
  float scale = softmax_scale > 0.f ? softmax_scale : (1.f / sqrtf(static_cast<float>(qk_dim)));
  float one_over_dk_log2e = scale * kLog2e;

  cudaLaunchAttribute pdl_attr[1];
  pdl_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl_attr[0].val.programmaticStreamSerializationAllowed = 1;

  auto attn_kernel = kernels::attention_sparse_mla_dim576_persistent_kernel<
      Tin, /*Tout=*/Tin, kTileM, kTileN, kTileNope, kTileRope, kTileV, TiledMmaQK, TiledMmaSV,
      decltype(tma_q_nope), decltype(tma_q_rope), decltype(slayout_q_nope),
      decltype(slayout_q_rope), decltype(slayout_k_nope), decltype(slayout_k_rope),
      decltype(slayout_p), decltype(slayout_s), decltype(slayout_v), kBlockSize, kStage, kUseSink,
      kNumMathWG>;

  cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  cudaLaunchConfig_t attn_cfg{};
  attn_cfg.gridDim = attn_grid;
  attn_cfg.blockDim = attn_block;
  attn_cfg.dynamicSmemBytes = shm_size;
  attn_cfg.stream = stream;
  attn_cfg.attrs = pdl_attr;
  attn_cfg.numAttrs = 1;

  cudaLaunchKernelEx(&attn_cfg, attn_kernel, tma_q_nope, tma_q_rope,
                     reinterpret_cast<const Tin*>(kvcache_ptr), y_partial_ptr, lse_ptr,
                     reinterpret_cast<Tin*>(y_ptr), sink_weight_ptr, block_ids_ptr, topk_ids_ptr,
                     cu_tasks, reinterpret_cast<int4*>(task_list), cu_splits, num_batch,
                     total_seq_q, num_head_q, qk_dim, v_dim, num_kvcache_blocks, num_seq_max_blocks,
                     num_max_topk, ldKV, ldY, one_over_dk_log2e);

  // splitk=False guarantees every batch lives on one SM (num_splits=1), so the
  // math kernel's single_split_epilogue path already writes the final bf16 y.
  // Skip the combine launch entirely
  if (!splitk) {
    return;
  }

  constexpr int kMinWavesPerSM = 4;
  constexpr int kCandidates[] = {128, 64, 32, 16, 8};
  int floor_ctas = kMinWavesPerSM * num_sm;
  int v_chunk_pick = 8;
  for (int kVChunk : kCandidates) {
    if (v_dim % kVChunk != 0) {
      continue;
    }
    int total_ctas = total_seq_q * num_head_q * (v_dim / kVChunk);
    if (total_ctas >= floor_ctas) {
      v_chunk_pick = kVChunk;
      break;
    }
  }

  auto launch_combine = [&](auto v_chunk_tag) {
    constexpr int kVChunk = decltype(v_chunk_tag)::value;
    constexpr int kMaxSplits = kDim576PersistentMaxNumSm;
    static_assert(kTileV % kVChunk == 0, "kTileV must be a multiple of kVChunk");

    constexpr int kNumVChunks = kTileV / kVChunk;
    constexpr int kWarpsPerBlock = (kNumVChunks % 4 == 0) ? 4 : (kNumVChunks % 2 == 0) ? 2 : 1;
    int v_chunks = v_dim / kVChunk;
    dim3 combine_grid(total_seq_q, num_head_q, v_chunks / kWarpsPerBlock);
    dim3 combine_block(32 * kWarpsPerBlock);
    auto combine_kernel = kernels::attention_mla_dim576_persistent_combine_kernel<
        __nv_bfloat16, kVChunk, kMaxSplits, kUseSink, /*kUsePDL=*/true, kWarpsPerBlock>;
    cudaLaunchConfig_t combine_cfg{};
    combine_cfg.gridDim = combine_grid;
    combine_cfg.blockDim = combine_block;
    combine_cfg.dynamicSmemBytes = 0;
    combine_cfg.stream = stream;
    combine_cfg.attrs = pdl_attr;
    combine_cfg.numAttrs = 1;
    cudaLaunchKernelEx(&combine_cfg, combine_kernel, reinterpret_cast<__nv_bfloat16*>(y_ptr),
                       y_partial_ptr, lse_ptr, sink_weight_ptr, cu_splits, num_head_q, v_dim,
                       total_seq_q, ldY);
  };

  switch (v_chunk_pick) {
    case 8:
      launch_combine(std::integral_constant<int, 8>{});
      break;
    case 16:
      launch_combine(std::integral_constant<int, 16>{});
      break;
    case 32:
      launch_combine(std::integral_constant<int, 32>{});
      break;
    case 64:
      launch_combine(std::integral_constant<int, 64>{});
      break;
    case 128:
      launch_combine(std::integral_constant<int, 128>{});
      break;
    default:
      launch_combine(std::integral_constant<int, 8>{});
      break;
  }
}

template <int kTileM, int kNumMathWG, bool kUseSink>
void run_sparse_dim576_persistent(void* y_ptr, const void* q_ptr, const void* kvcache_ptr,
                                  float* y_partial_ptr, float* lse_ptr, int* task_tensor_ptr,
                                  const int* block_ids_ptr, const int* topk_ids_ptr,
                                  const int* cu_seqlens_q_ptr, const float* sink_weight_ptr,
                                  int num_batch, int total_seq_q, int num_head_q, int qk_dim,
                                  int v_dim, int num_kvcache_blocks, int num_seq_max_blocks,
                                  int num_max_topk, int ldY, int ldQ, int ldKV, float softmax_scale,
                                  cudaStream_t stream, bool task_tensor_prebuilt, bool splitk,
                                  bool prefill) {
  launch_sparse_dim576_persistent<kTileM, /*kTileN=*/64, /*kTileNope=*/512, /*kTileRope=*/64,
                                  /*kTileV=*/512, /*kBlockSize=*/64, kUseSink, kNumMathWG>(
      y_ptr, q_ptr, kvcache_ptr, y_partial_ptr, lse_ptr, task_tensor_ptr, block_ids_ptr,
      topk_ids_ptr, cu_seqlens_q_ptr, sink_weight_ptr, num_batch, total_seq_q, num_head_q, qk_dim,
      v_dim, num_kvcache_blocks, num_seq_max_blocks, num_max_topk, ldY, ldQ, ldKV, softmax_scale,
      stream, task_tensor_prebuilt, splitk, prefill);
}

}  // namespace mla
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_MLA_SMALLM_SPARSE_MLA_DIM576_PERSISTENT_LAUNCH_CUH_
