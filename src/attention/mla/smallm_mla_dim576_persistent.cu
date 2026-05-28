// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <mutex>
#include <type_traits>

#include "cute/tensor.hpp"
#include "src/attention/mla/smallm_mla_dim576_persistent.h"
#include "src/attention/mla/smallm_mla_dim576_persistent_combine_kernels.cuh"
#include "src/attention/mla/smallm_mla_dim576_persistent_get_scheduler_map.cuh"
#include "src/attention/mla/smallm_mla_dim576_persistent_kernels.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace attention {
namespace mla {

template <int kTileM>
struct Dim576PersistentAtomSelector;

template <>
struct Dim576PersistentAtomSelector<32> {
  template <cute::GMMA::Major A, cute::GMMA::Major B>
  using Mma = cute::SM90_64x32x16_F32BF16BF16_SS<A, B>;
};

template <>
struct Dim576PersistentAtomSelector<16> {
  template <cute::GMMA::Major A, cute::GMMA::Major B>
  using Mma = cute::SM90_64x16x16_F32BF16BF16_SS<A, B>;
};

template <>
struct Dim576PersistentAtomSelector<8> {
  template <cute::GMMA::Major A, cute::GMMA::Major B>
  using Mma = cute::SM90_64x8x16_F32BF16BF16_SS<A, B>;
};

template <int kTileM, int kTileN, int kTileNope, int kTileRope, int kTileV, int kBlockSize,
          bool kUseSink>
static void launch_dim576_persistent(void* y_ptr, const void* q_ptr, const void* kvcache_ptr,
                                     float* y_partial_ptr, float* lse_ptr, int* task_tensor_ptr,
                                     const int* block_ids_ptr, const int* cu_seqlens_q_ptr,
                                     const int* num_seq_kv_ptr, const float* sink_weight_ptr,
                                     int num_batch, int total_seq_q, int num_head_q, int qk_dim,
                                     int v_dim, int num_kvcache_blocks, int num_seq_max_blocks,
                                     int ldY, int ldQ, int ldKV, float softmax_scale,
                                     cudaStream_t stream, bool task_tensor_prebuilt) {
  using namespace cute;  // NOLINT
  constexpr int kStage = 2;
  static_assert(kTileNope + kTileRope == 576, "dim576 persistent expects 512+64 split");
  static_assert(kTileNope == kTileV, "sV aliases sK_nope; requires kTileNope == kTileV");

  using Tin = cute::bfloat16_t;

  int num_sm = hpc::get_sm_count();

  // task tensor layout:
  //   [ task_list (max_num_jobs*4) | cu_tasks (num_sm+1) | cu_splits (num_batch+1) ]
  int4* task_list = reinterpret_cast<int4*>(task_tensor_ptr + dim576_persistent_task_list_offset());
  int* cu_tasks = task_tensor_ptr + dim576_persistent_cu_tasks_offset(num_batch, num_sm);
  int* cu_splits = task_tensor_ptr + dim576_persistent_cu_splits_offset(num_batch, num_sm);

  if (!task_tensor_prebuilt) {
    auto kernel = kernels::get_scheduler_map_kernel<kTileN>;
    kernel<<<1, kernels::kDim576PersistentSchedulerMapThreads, 0, stream>>>(
        task_list, cu_tasks, cu_splits, num_seq_kv_ptr, num_batch, num_sm);
  }

  auto Q_nope = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(q_ptr)),
                            make_shape(num_head_q, Int<kTileNope>{}, total_seq_q),
                            make_stride(qk_dim, Int<1>{}, ldQ));
  auto Q_rope = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(q_ptr) + kTileNope),
                            make_shape(num_head_q, Int<kTileRope>{}, total_seq_q),
                            make_stride(qk_dim, Int<1>{}, ldQ));
  auto KV_nope = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(kvcache_ptr)),
                             make_shape(kBlockSize, Int<kTileNope>{}, 1, num_kvcache_blocks),
                             make_stride(qk_dim, Int<1>{}, qk_dim, ldKV));
  auto KV_rope = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(kvcache_ptr) + kTileNope),
                             make_shape(kBlockSize, Int<kTileRope>{}, 1, num_kvcache_blocks),
                             make_stride(qk_dim, Int<1>{}, qk_dim, ldKV));

  auto slayout_q_nope =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileNope>{}));
  auto slayout_q_rope =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileRope>{}));
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

  auto tma_copy_layout_q_nope =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileNope>{}));
  auto tma_copy_layout_q_rope =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileRope>{}));
  auto tma_copy_layout_k_nope = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                              make_shape(Int<kBlockSize>{}, Int<kTileNope>{}));
  auto tma_copy_layout_k_rope = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                              make_shape(Int<kBlockSize>{}, Int<kTileRope>{}));

  auto tma_q_nope = make_tma_copy(SM90_TMA_LOAD{}, Q_nope, tma_copy_layout_q_nope);
  auto tma_q_rope = make_tma_copy(SM90_TMA_LOAD{}, Q_rope, tma_copy_layout_q_rope);
  auto tma_k_nope = make_tma_copy(SM90_TMA_LOAD{}, KV_nope, tma_copy_layout_k_nope);
  auto tma_k_rope = make_tma_copy(SM90_TMA_LOAD{}, KV_rope, tma_copy_layout_k_rope);

  using WarpgroupLayout = decltype(make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})));
  using AtomQK =
      typename Dim576PersistentAtomSelector<kTileM>::template Mma<GMMA::Major::K, GMMA::Major::K>;
  using AtomSV =
      typename Dim576PersistentAtomSelector<kTileM>::template Mma<GMMA::Major::MN, GMMA::Major::K>;
  using TiledMmaQK = decltype(make_tiled_mma(AtomQK{}, WarpgroupLayout{}));
  using TiledMmaSV = decltype(make_tiled_mma(AtomSV{}, WarpgroupLayout{}));

  dim3 attn_block(size(TiledMmaQK{}) + 128);
  dim3 attn_grid(num_sm);

  constexpr int kWarpsPerWrapGroup = 4;
  int shm_size = (cosize(slayout_q_nope) + cosize(slayout_q_rope) + cosize(slayout_k_nope) +
                  cosize(slayout_k_rope) + cosize(slayout_p)) *
                     sizeof(Tin) +
                 sizeof(float) * kTileM * kWarpsPerWrapGroup;

  constexpr float kLog2e = 1.4426950408889634f;
  float scale = softmax_scale > 0.f ? softmax_scale : (1.f / sqrtf(float(qk_dim)));
  float one_over_dk_log2e = scale * kLog2e;

  cudaLaunchAttribute pdl_attr[1];
  pdl_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl_attr[0].val.programmaticStreamSerializationAllowed = 1;

  auto attn_kernel = kernels::attention_mla_dim576_persistent_kernel<
      Tin, kTileM, kTileN, kTileNope, kTileRope, kTileV, TiledMmaQK, TiledMmaSV,
      decltype(tma_q_nope), decltype(tma_q_rope), decltype(tma_k_nope), decltype(tma_k_rope),
      decltype(slayout_q_nope), decltype(slayout_q_rope), decltype(slayout_k_nope),
      decltype(slayout_k_rope), decltype(slayout_p), decltype(slayout_s), decltype(slayout_v),
      kBlockSize, kStage>;

  cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  cudaLaunchConfig_t attn_cfg{};
  attn_cfg.gridDim = attn_grid;
  attn_cfg.blockDim = attn_block;
  attn_cfg.dynamicSmemBytes = shm_size;
  attn_cfg.stream = stream;
  attn_cfg.attrs = pdl_attr;
  attn_cfg.numAttrs = 1;

  cudaLaunchKernelEx(&attn_cfg, attn_kernel, tma_q_nope, tma_q_rope, tma_k_nope, tma_k_rope,
                     y_partial_ptr, lse_ptr, block_ids_ptr, cu_seqlens_q_ptr, num_seq_kv_ptr,
                     cu_tasks, task_list, cu_splits, num_batch, total_seq_q, num_head_q, qk_dim,
                     v_dim, num_kvcache_blocks, num_seq_max_blocks, one_over_dk_log2e);

  constexpr int kMinWavesPerSM = 4;
  constexpr int kCandidates[] = {128, 64, 32, 16, 8};
  int floor_ctas = kMinWavesPerSM * num_sm;
  int v_chunk_pick = 8;  // fallback
  for (int kVChunk : kCandidates) {
    if (v_dim % kVChunk != 0) {
      continue;  // skip non-divisors (defensive)
    }
    int total_ctas = num_batch * num_head_q * (v_dim / kVChunk);
    if (total_ctas >= floor_ctas) {
      v_chunk_pick = kVChunk;
      break;
    }
  }

  auto launch_combine = [&](auto v_chunk_tag) {
    constexpr int kVChunk = decltype(v_chunk_tag)::value;
    // kMaxSplits is sized by the upper-bound `kDim576PersistentMaxNumSm` so
    // a single binary covers all SM counts we care about (sm90/sm100).
    constexpr int kMaxSplits = kDim576PersistentMaxNumSm;
    static_assert(kTileV % kVChunk == 0, "kTileV must be a multiple of kVChunk");
    dim3 combine_grid(num_batch, num_head_q, v_dim / kVChunk);
    dim3 combine_block(32);
    auto combine_kernel =
        kernels::attention_mla_dim576_persistent_combine_kernel<__nv_bfloat16, kVChunk, kMaxSplits,
                                                                kUseSink, /*kUsePDL=*/true>;
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

bool smallm_mla_dim576_persistent_async(void* y_ptr, const void* q_ptr, const void* kvcache_ptr,
                                        float* y_partial_ptr, float* lse_ptr, int* task_tensor_ptr,
                                        const int* block_ids_ptr, const int* cu_seqlens_q_ptr,
                                        const int* num_seq_kv_ptr, const float* sink_weight_ptr,
                                        int num_batch, int total_seq_q, int num_head_q, int qk_dim,
                                        int v_dim, int num_kvcache_blocks, int num_seq_max_blocks,
                                        int ldY, int ldQ, int ldKV, float softmax_scale,
                                        cudaStream_t stream, bool task_tensor_prebuilt) {
  constexpr int kTileN = 64;
  constexpr int kTileNope = 512;
  constexpr int kTileRope = 64;
  constexpr int kTileV = 512;
  constexpr int kBlockSize = 64;

  // num_head_q ∈ {1, 2, 4, 8} all map to kTileM = 8 (smallest GMMA-N atom);
  // 16 and 32 map to native atoms.
  if (qk_dim != (kTileNope + kTileRope) || v_dim != kTileV) {
    return false;
  }
  if (num_head_q != 1 && num_head_q != 2 && num_head_q != 4 && num_head_q != 8 &&
      num_head_q != 16 && num_head_q != 32) {
    return false;
  }

  auto dispatch = [&](auto kTileM_tag, auto kUseSink_tag) {
    constexpr int kTileM = decltype(kTileM_tag)::value;
    constexpr bool kUseSink = decltype(kUseSink_tag)::value;
    launch_dim576_persistent<kTileM, kTileN, kTileNope, kTileRope, kTileV, kBlockSize, kUseSink>(
        y_ptr, q_ptr, kvcache_ptr, y_partial_ptr, lse_ptr, task_tensor_ptr, block_ids_ptr,
        cu_seqlens_q_ptr, num_seq_kv_ptr, sink_weight_ptr, num_batch, total_seq_q, num_head_q,
        qk_dim, v_dim, num_kvcache_blocks, num_seq_max_blocks, ldY, ldQ, ldKV, softmax_scale,
        stream, task_tensor_prebuilt);
  };

  auto dispatch_sink = [&](auto kTileM_tag) {
    if (sink_weight_ptr != nullptr) {
      dispatch(kTileM_tag, std::true_type{});
    } else {
      dispatch(kTileM_tag, std::false_type{});
    }
  };

  if (num_head_q == 32) {
    dispatch_sink(std::integral_constant<int, 32>{});
  } else if (num_head_q == 16) {
    dispatch_sink(std::integral_constant<int, 16>{});
  } else {
    dispatch_sink(std::integral_constant<int, 8>{});
  }
  return true;
}

bool dim576_persistent_get_scheduler_map_async(int* task_tensor_ptr, const int* num_seq_kv_ptr,
                                               int num_batch, cudaStream_t stream) {
  constexpr int kTileN = 64;
  if (num_batch <= 0) {
    return true;  // empty: nothing to do.
  }
  int num_sm = hpc::get_sm_count();
  // Slice the int32 workspace into the three regions the kernel writes.
  int4* task_list = reinterpret_cast<int4*>(task_tensor_ptr + dim576_persistent_task_list_offset());
  int* cu_tasks = task_tensor_ptr + dim576_persistent_cu_tasks_offset(num_batch, num_sm);
  int* cu_splits = task_tensor_ptr + dim576_persistent_cu_splits_offset(num_batch, num_sm);
  auto kernel = kernels::get_scheduler_map_kernel<kTileN>;
  kernel<<<1, kernels::kDim576PersistentSchedulerMapThreads, 0, stream>>>(
      task_list, cu_tasks, cu_splits, num_seq_kv_ptr, num_batch, num_sm);
  return true;
}

}  // namespace mla
}  // namespace attention
}  // namespace hpc
