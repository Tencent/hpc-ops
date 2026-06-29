// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/attention/prefill/config.h"
#include "src/attention/prefill/hybrid_mask_common.cuh"
#include "src/attention/prefill/kernels.cuh"
#include "src/attention/prefill/multi_stage_with_kvcache_dim128_hybrid_mask.cuh"
#include "src/attention/prefill/multi_stage_with_kvcache_dim128_hybrid_mask.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace prefill {

template <int kBlockSize, bool kHybridMask = false, int kPackG = 1>
void launch_multi_stage_with_kvcache_dim128_hybrid_mask(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    const void *mm_prefix_range_ptr, int max_spans, void *tmas_ptr, int num_batch, int total_seq_q,
    int max_seq_q, int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv,
    int num_kvcache_blocks, int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK,
    int ldK1, int ldK2, int ldV, int ldV1, int ldV2, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::bfloat16_t;
  using Tout = cute::bfloat16_t;

  constexpr bool kPackGQA = (kPackG > 1);
  const int m_q = kPackGQA ? max_seq_q * kPackG : max_seq_q;
  const int grid_y = kPackGQA ? num_head_kv : num_head_q;

  using TiledMmaQK = SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>;
  using TiledMmaPV = SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>;
  using Config = AttentionKVCachePrefillConfig<Tin, Tout, TiledMmaQK, TiledMmaPV, 64, 64, 128, 128,
                                               kBlockSize, 1, 1, 1, 128, 128, 128, 128>;
  Config config;

  auto K = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(kcache_ptr)),
                       make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks),
                       make_stride(ldK1, Int<1>{}, ldK2, ldK));
  auto V = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(vcache_ptr)),
                       make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks),
                       make_stride(Int<1>{}, ldV1, ldV2, ldV));

  auto *tma_qy = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;

  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K, typename Config::CopyBoxK{});
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, V, typename Config::CopyBoxV{});

  auto tma_q = [&]() {
    if constexpr (kPackGQA) {
      auto Qh = make_tensor(
          make_gmem_ptr(reinterpret_cast<const Tin *>(q_ptr)),
          make_shape(make_shape(Int<kPackG>{}, max_seq_q), num_dim_qk, num_head_kv),
          make_stride(make_stride(num_dim_qk, ldQ), Int<1>{}, Int<kPackG>{} * num_dim_qk));
      auto slq_h = logical_divide(typename Config::SLayoutQ{},
                                  make_tile(Int<kPackG>{}, Int<Config::kTileK>{}));
      return make_tma_copy(SM90_TMA_LOAD{}, Qh, slq_h);
    } else {
      auto Q = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(q_ptr)),
                           make_shape(max_seq_q, num_dim_qk, num_head_q),
                           make_stride(ldQ, Int<1>{}, num_dim_qk));
      return make_tma_copy(SM90_TMA_LOAD{}, Q, typename Config::SLayoutQ{});
    }
  }();
  auto tma_y = [&]() {
    if constexpr (kPackGQA) {
      auto Yh = make_tensor(
          make_gmem_ptr(reinterpret_cast<const Tout *>(y_ptr)),
          make_shape(make_shape(Int<kPackG>{}, max_seq_q), num_dim_v, num_head_kv),
          make_stride(make_stride(num_dim_v, ldY), Int<1>{}, Int<kPackG>{} * num_dim_v));
      auto sly_h = logical_divide(typename Config::SLayoutY{},
                                  make_tile(Int<kPackG>{}, Int<Config::kTileV>{}));
      return make_tma_copy(SM90_TMA_STORE{}, Yh, sly_h);
    } else {
      auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<const Tout *>(y_ptr)),
                           make_shape(max_seq_q, num_dim_v, num_head_q),
                           make_stride(ldY, Int<1>{}, num_dim_v));
      return make_tma_copy(SM90_TMA_STORE{}, Y, typename Config::CopyBoxY{});
    }
  }();

  // 0. update tma
  {
    vec_t<cute::TmaDescriptor, 2> td_qy{
        *tma_q.get_tma_descriptor(),
        *tma_y.get_tma_descriptor(),
    };
    kernels::update_batched_tma_with_kvcache_packg<Tin, Tout, decltype(tma_q), decltype(tma_y),
                                                   kPackG><<<num_batch, 32, 0, stream>>>(
        td_qy, tma_qy, (const Tin *)q_ptr, (const Tout *)y_ptr, (const int *)cu_seqlens_q_ptr,
        num_batch, max_seq_q, num_dim_qk, num_dim_v, num_head_q, num_head_kv, ldQ, ldY);
  }

  // 1. compute attention
  {
    int kv_group = num_head_q / num_head_kv;
    cutlass::FastDivmod head_kv_divmod(kv_group);

    int shm_size = config.get_shm_size();
    shm_size += sizeof(int) * num_batch * 3;

    dim3 block(128);
    dim3 grid((m_q + Config::kTileM - 1) / Config::kTileM, grid_y, num_batch);
    auto kernel = kernels::attention_with_kvcache_prefill_bf16_multi_stage_hybrid_mask_kernel<
        decltype(config), decltype(tma_q), decltype(tma_k), decltype(tma_v), decltype(tma_y),
        kHybridMask, kPackG>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    kernel<<<grid, block, shm_size, stream>>>(
        tma_qy, tma_k, tma_v, (const int *)cu_seqlens_q_ptr, (const int *)seqlens_kvcache_ptr,
        (const int *)block_ids_ptr, (const int *)mm_prefix_range_ptr, max_spans, num_batch,
        max_seq_q, num_dim_qk, num_dim_v, num_head_q, num_head_kv, num_kvcache_blocks, block_size,
        num_seq_max_blocks, one_over_dk_log2e, head_kv_divmod);
  }
}

static inline int pack_gqa_factor(int num_head_q, int num_head_kv) {
  if (num_head_kv <= 0 || num_head_q % num_head_kv != 0) {
    return 1;
  }
  int g = num_head_q / num_head_kv;
  if ((g == 2 || g == 4 || g == 8) && (64 % g == 0)) {
    return g;
  }
  return 1;
}

template <bool kHybridMask>
static void dispatch_multi_stage_with_kvcache_dim128_hybrid_mask(
    int pack_g, void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    const void *mm_prefix_range_ptr, int max_spans, void *tmas_ptr, int num_batch, int total_seq_q,
    int max_seq_q, int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv,
    int num_kvcache_blocks, int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK,
    int ldK1, int ldK2, int ldV, int ldV1, int ldV2, cudaStream_t stream) {
  auto do_launch = [&](auto pack_c) {
    constexpr int kPackG = decltype(pack_c)::value;
    auto run = [&](auto bs_c) {
      constexpr int kBlockSize = decltype(bs_c)::value;
      launch_multi_stage_with_kvcache_dim128_hybrid_mask<kBlockSize, kHybridMask, kPackG>(
          y_ptr, q_ptr, kcache_ptr, vcache_ptr, cu_seqlens_q_ptr, block_ids_ptr,
          seqlens_kvcache_ptr, mm_prefix_range_ptr, max_spans, tmas_ptr, num_batch, total_seq_q,
          max_seq_q, num_dim_qk, num_dim_v, num_head_q, num_head_kv, num_kvcache_blocks, block_size,
          num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1, ldV2, stream);
    };
    if (block_size == 16) {
      run(std::integral_constant<int, 16>{});
    } else if (block_size == 32) {
      run(std::integral_constant<int, 32>{});
    } else if (block_size == 64) {
      run(std::integral_constant<int, 64>{});
    }
  };

  switch (pack_g) {
    case 2:
      do_launch(std::integral_constant<int, 2>{});
      break;
    case 4:
      do_launch(std::integral_constant<int, 4>{});
      break;
    case 8:
      do_launch(std::integral_constant<int, 8>{});
      break;
    default:
      do_launch(std::integral_constant<int, 1>{});
      break;
  }
}

void multi_stage_with_kvcache_dim128_hybrid_mask_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    const void *mm_prefix_range_ptr, int max_spans, void *tmas_ptr, int num_batch, int total_seq_q,
    int max_seq_q, int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv,
    int num_kvcache_blocks, int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK,
    int ldK1, int ldK2, int ldV, int ldV1, int ldV2, cudaStream_t stream) {
  dispatch_multi_stage_with_kvcache_dim128_hybrid_mask</*kHybridMask=*/true>(
      pack_gqa_factor(num_head_q, num_head_kv), y_ptr, q_ptr, kcache_ptr, vcache_ptr,
      cu_seqlens_q_ptr, block_ids_ptr, seqlens_kvcache_ptr, mm_prefix_range_ptr, max_spans,
      tmas_ptr, num_batch, total_seq_q, max_seq_q, num_dim_qk, num_dim_v, num_head_q, num_head_kv,
      num_kvcache_blocks, block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1,
      ldV2, stream);
}

}  // namespace prefill
}  // namespace attention
}  // namespace hpc
