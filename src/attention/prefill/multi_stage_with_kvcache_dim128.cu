// Copyright (C) 2026 Tencent.

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/attention/prefill/config.h"
#include "src/attention/prefill/kernels.cuh"
#include "src/attention/prefill/multi_stage_with_kvcache_dim128.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace prefill {

template <int kBlockSize>
void launch_multi_stage_with_kvcache_dim128(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int num_dim_qk, int num_dim_v,
    int num_head_q, int num_head_kv, int num_kvcache_blocks, int block_size, int num_seq_max_blocks,
    int ldY, int ldQ, int ldK, int ldV, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::bfloat16_t;
  using Tout = cute::bfloat16_t;

  auto Q = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(q_ptr)),
                       make_shape(max_seq_q, num_dim_qk, num_head_q),
                       make_stride(ldQ, Int<1>{}, num_dim_qk));
  auto K = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(kcache_ptr)),
                       make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks),
                       make_stride(num_dim_qk * num_head_kv, Int<1>{}, num_dim_qk, ldK));
  auto V = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(vcache_ptr)),
                       make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks),
                       make_stride(Int<1>{}, num_head_kv * num_dim_v, num_dim_v, ldV));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<const Tout *>(y_ptr)),
                       make_shape(max_seq_q, num_dim_v, num_head_q),
                       make_stride(ldY, Int<1>{}, num_dim_v));

  auto *tma_qy = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;

  using TiledMmaQK = SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>;
  using TiledMmaPV = SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>;
  using Config = AttentionKVCachePrefillConfig<Tin, Tout, TiledMmaQK, TiledMmaPV, 64, 64, 128, 128,
                                               kBlockSize, 1, 1, 1, 128, 128, 128, 128>;

  Config config;
  auto [tma_q, tma_k, tma_v, tma_y] = config.get_tma(Q, K, V, Y);

  // 0. update tma
  {
    vec_t<cute::TmaDescriptor, 2> td_qy{
        *tma_q.get_tma_descriptor(),
        *tma_y.get_tma_descriptor(),
    };
    kernels::update_batched_tma_with_kvcache<Tin, Tout, decltype(tma_q), decltype(tma_y)>
        <<<num_batch, 32, 0, stream>>>(td_qy, tma_qy, (const Tin *)q_ptr, (const Tout *)y_ptr,
                                       (const int *)cu_seqlens_q_ptr, num_batch, max_seq_q,
                                       num_dim_qk, num_dim_v, num_head_q, ldQ, ldY);
  }

  // 1. compute attention
  {
    int kv_group = num_head_q / num_head_kv;
    cutlass::FastDivmod head_kv_divmod(kv_group);

    int shm_size = config.get_shm_size();
    shm_size += sizeof(int) * num_batch * 3;

    dim3 block(128);
    dim3 grid((max_seq_q + Config::kTileM - 1) / Config::kTileM, num_head_q, num_batch);
    auto kernel = kernels::attention_with_kvcache_prefill_bf16_multi_stage_kernel<
        decltype(config), decltype(tma_q), decltype(tma_k), decltype(tma_v), decltype(tma_y)>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    kernel<<<grid, block, shm_size, stream>>>(
        tma_qy, tma_k, tma_v, (const int *)cu_seqlens_q_ptr, (const int *)seqlens_kvcache_ptr,
        (const int *)block_ids_ptr, num_batch, max_seq_q, num_dim_qk, num_dim_v, num_head_q,
        num_head_kv, num_kvcache_blocks, block_size, num_seq_max_blocks, one_over_dk_log2e,
        head_kv_divmod);
  }
}
void multi_stage_with_kvcache_dim128_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int num_dim_qk, int num_dim_v,
    int num_head_q, int num_head_kv, int num_kvcache_blocks, int block_size, int num_seq_max_blocks,
    int ldY, int ldQ, int ldK, int ldV, cudaStream_t stream) {
  if (block_size == 32) {
    constexpr int kBlockSize = 32;
    launch_multi_stage_with_kvcache_dim128<kBlockSize>(
        y_ptr, q_ptr, kcache_ptr, vcache_ptr, cu_seqlens_q_ptr, block_ids_ptr, seqlens_kvcache_ptr,
        tmas_ptr, num_batch, total_seq_q, max_seq_q, num_dim_qk, num_dim_v, num_head_q, num_head_kv,
        num_kvcache_blocks, block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldV, stream);
  } else if (block_size == 64) {
    constexpr int kBlockSize = 64;
    launch_multi_stage_with_kvcache_dim128<kBlockSize>(
        y_ptr, q_ptr, kcache_ptr, vcache_ptr, cu_seqlens_q_ptr, block_ids_ptr, seqlens_kvcache_ptr,
        tmas_ptr, num_batch, total_seq_q, max_seq_q, num_dim_qk, num_dim_v, num_head_q, num_head_kv,
        num_kvcache_blocks, block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldV, stream);
  }
}

}  // namespace prefill
}  // namespace attention
}  // namespace hpc
