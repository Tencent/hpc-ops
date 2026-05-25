// Copyright (C) 2026 Tencent.

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/attention/prefill/config.h"
#include "src/attention/prefill/kernels.cuh"
#include "src/attention/prefill/warp_spec_with_kvcache_blocksparse_fp8_dim128.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace prefill {

template <int kBlockSize, bool kHasMask>
void launch_warp_spec_with_kvcache_blocksparse_qpertoken_perhead_kvpertensor_fp8_dim128(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK, int ldK1, int ldK2, int ldV,
    int ldV1, int ldV2, const void *block_mask_ptr, int num_tile_kv_in_mask, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  auto Q = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(q_ptr)),
                       make_shape(max_seq_q, num_dim_qk, num_head_q),
                       make_stride(ldQ, Int<1>{}, num_dim_qk));
  auto K = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(kcache_ptr)),
                       make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks),
                       make_stride(ldK1, Int<1>{}, ldK2, ldK));
  auto V = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(vcache_ptr)),
                       make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks),
                       make_stride(Int<1>{}, ldV1, ldV2, ldV));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<const Tout *>(y_ptr)),
                       make_shape(max_seq_q, num_dim_v, num_head_q),
                       make_stride(ldY, Int<1>{}, num_dim_v));
  auto QS = make_tensor(make_gmem_ptr(reinterpret_cast<const float *>(qscale_ptr)),
                        make_shape(max_seq_q_pad, num_head_q, num_batch),
                        make_stride(Int<1>{}, max_seq_q_pad, num_head_q * max_seq_q_pad));

  auto *tma_qy = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;

  using TiledMmaQK = SM90_64x64x32_F32E4M3E4M3_SS_TN<>;
  using TiledMmaPV = SM90_64x128x32_F32E4M3E4M3_RS_TN<>;
  using Config = AttentionKVCachePrefillQTokenKVTensorFp8Config<Tin, Tout, TiledMmaQK, TiledMmaPV,
                                                                128, 128, 128, 128, kBlockSize, 2,
                                                                2, 1, 128, 128, 128, 128>;

  Config config;
  auto [tma_q, tma_k, tma_v, tma_y, tma_qs] = config.get_tma(Q, K, V, Y, QS);

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

  // 1. compute block-sparse attention
  {
    int kv_group = num_head_q / num_head_kv;
    cutlass::FastDivmod head_kv_divmod(kv_group);
    cutlass::FastDivmod head_q_divmod(num_head_q);
    cutlass::FastDivmod tile_m_divmod(num_batch * num_head_q);

    int shm_size = config.get_shm_size();
    shm_size += sizeof(int) * num_batch * 3;
    if constexpr (kHasMask) {
      shm_size += (num_tile_kv_in_mask + 2) * sizeof(int);
      shm_size = (shm_size + 15) & ~15;
    }

    dim3 block(384);
    dim3 grid(get_sm_count());
    auto kernel = kernels::
        attention_with_kvcache_blocksparse_qpertoken_perhead_kvpertensor_prefill_fp8_warp_specialization_kernel<  // NOLINT
            decltype(config), decltype(tma_q), decltype(tma_k), decltype(tma_v), decltype(tma_y),
            decltype(tma_qs), kHasMask>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    kernel<<<grid, block, shm_size, stream>>>(
        tma_qy, tma_k, tma_v, tma_qs, (const float *)kscale_ptr, (const float *)vscale_ptr,
        (const int *)cu_seqlens_q_ptr, (const int *)seqlens_kvcache_ptr, (const int *)block_ids_ptr,
        num_batch, max_seq_q, max_seq_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv,
        num_kvcache_blocks, num_seq_max_blocks, one_over_dk_log2e, head_kv_divmod, head_q_divmod,
        tile_m_divmod, (const uint8_t *)block_mask_ptr, num_tile_kv_in_mask);
  }
}

template <int kBlockSize, bool kHasMask>
void launch_warp_spec_with_kvcache_blocksparse_qkpertoken_perhead_vperhead_fp8_dim128(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int scale_block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK,
    int ldK1, int ldK2, int ldV, int ldV1, int ldV2, int ldKS, int ldKS1, int ldKS2,
    const void *block_mask_ptr, int num_tile_kv_in_mask, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  int num_dim_scale = num_dim_v / sizeof(float);
  int num_scale_blocks = num_kvcache_blocks;

  constexpr int kTileM = 128;
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;
  constexpr int kTileV = 128;
  constexpr int kTileS = kTileK / sizeof(float);
  constexpr int kScaleBlockSize = kBlockSize / kTileS;

  auto Q = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(q_ptr)),
                       make_shape(max_seq_q, num_dim_qk, num_head_q),
                       make_stride(ldQ, Int<1>{}, num_dim_qk));
  auto K = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(kcache_ptr)),
                       make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks),
                       make_stride(ldK1, Int<1>{}, ldK2, ldK));
  auto V = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(vcache_ptr)),
                       make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks),
                       make_stride(Int<1>{}, ldV1, ldV2, ldV));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<const Tout *>(y_ptr)),
                       make_shape(max_seq_q, num_dim_v, num_head_q),
                       make_stride(ldY, Int<1>{}, num_dim_v));
  auto QS = make_tensor(make_gmem_ptr(reinterpret_cast<const float *>(qscale_ptr)),
                        make_shape(max_seq_q_pad, num_head_q, num_batch),
                        make_stride(Int<1>{}, max_seq_q_pad, num_head_q * max_seq_q_pad));
  auto KS = make_tensor(make_gmem_ptr(reinterpret_cast<const float *>(kscale_ptr)),
                        make_shape(kScaleBlockSize, num_dim_scale, num_head_kv, num_scale_blocks),
                        make_stride(ldKS1, Int<1>{}, ldKS2, ldKS));

  auto *tma_qy = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;

  using TiledMmaQK = SM90_64x64x32_F32E4M3E4M3_SS_TN<>;
  using TiledMmaPV = SM90_64x128x32_F32E4M3E4M3_RS_TN<>;
  using Config =
      AttentionKVCachePrefillQKTokenVTensorFp8Config<Tin, Tout, TiledMmaQK, TiledMmaPV, kTileM,
                                                     kTileN, kTileK, kTileV, kTileS, kBlockSize,
                                                     kScaleBlockSize, 2, 2, 1, 128, 128, 128, 128>;

  Config config;
  auto [tma_q, tma_k, tma_v, tma_y, tma_qs, tma_ks] = config.get_tma(Q, K, V, Y, QS, KS);

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

  // 1. compute block-sparse attention
  {
    int kv_group = num_head_q / num_head_kv;
    cutlass::FastDivmod head_kv_divmod(kv_group);
    cutlass::FastDivmod head_q_divmod(num_head_q);
    cutlass::FastDivmod tile_m_divmod(num_batch * num_head_q);

    int shm_size = config.get_shm_size();
    shm_size += sizeof(int) * num_batch * 3;
    if constexpr (kHasMask) {
      shm_size += (num_tile_kv_in_mask + 2) * sizeof(int);
      shm_size = (shm_size + 15) & ~15;
    }

    dim3 block(384);
    dim3 grid(get_sm_count());
    auto kernel = kernels::
        attention_with_kvcache_blocksparse_qkpertoken_perhead_vperhead_prefill_fp8_warp_specialization_kernel<  // NOLINT(whitespace/line_length)
            decltype(config), decltype(tma_q), decltype(tma_k), decltype(tma_v), decltype(tma_y),
            decltype(tma_qs), decltype(tma_ks), kHasMask>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    kernel<<<grid, block, shm_size, stream>>>(
        tma_qy, tma_k, tma_v, tma_qs, tma_ks, (const float *)qscale_ptr, (const float *)kscale_ptr,
        (const float *)vscale_ptr, (const int *)cu_seqlens_q_ptr, (const int *)seqlens_kvcache_ptr,
        (const int *)block_ids_ptr, num_batch, max_seq_q, max_seq_q_pad, num_dim_qk, num_dim_v,
        num_dim_scale, num_head_q, num_head_kv, num_kvcache_blocks, num_scale_blocks,
        num_seq_max_blocks, one_over_dk_log2e, head_kv_divmod, head_q_divmod, tile_m_divmod,
        (const uint8_t *)block_mask_ptr, num_tile_kv_in_mask);
  }
}

void warp_spec_with_kvcache_blocksparse_qpertoken_perhead_kvpertensor_fp8_dim128_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK, int ldK1, int ldK2, int ldV,
    int ldV1, int ldV2, const void *block_mask_ptr, int num_tile_kv_in_mask, cudaStream_t stream) {
  bool has_mask = block_mask_ptr != nullptr;
  if (block_size == 32) {
    constexpr int kBlockSize = 32;
    if (has_mask) {
      launch_warp_spec_with_kvcache_blocksparse_qpertoken_perhead_kvpertensor_fp8_dim128<kBlockSize,
                                                                                         true>(
          y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr,
          cu_seqlens_q_ptr, block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q,
          max_seq_q, max_seq_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv,
          num_kvcache_blocks, block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1,
          ldV2, block_mask_ptr, num_tile_kv_in_mask, stream);
    } else {
      launch_warp_spec_with_kvcache_blocksparse_qpertoken_perhead_kvpertensor_fp8_dim128<kBlockSize,
                                                                                         false>(
          y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr,
          cu_seqlens_q_ptr, block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q,
          max_seq_q, max_seq_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv,
          num_kvcache_blocks, block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1,
          ldV2, nullptr, 0, stream);
    }
  } else if (block_size == 64) {
    constexpr int kBlockSize = 64;
    if (has_mask) {
      launch_warp_spec_with_kvcache_blocksparse_qpertoken_perhead_kvpertensor_fp8_dim128<kBlockSize,
                                                                                         true>(
          y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr,
          cu_seqlens_q_ptr, block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q,
          max_seq_q, max_seq_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv,
          num_kvcache_blocks, block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1,
          ldV2, block_mask_ptr, num_tile_kv_in_mask, stream);
    } else {
      launch_warp_spec_with_kvcache_blocksparse_qpertoken_perhead_kvpertensor_fp8_dim128<kBlockSize,
                                                                                         false>(
          y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr,
          cu_seqlens_q_ptr, block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q,
          max_seq_q, max_seq_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv,
          num_kvcache_blocks, block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1,
          ldV2, nullptr, 0, stream);
    }
  }
}

void warp_spec_with_kvcache_blocksparse_qkpertoken_perhead_vperhead_fp8_dim128_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int scale_block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK,
    int ldK1, int ldK2, int ldV, int ldV1, int ldV2, int ldKS, int ldKS1, int ldKS2,
    const void *block_mask_ptr, int num_tile_kv_in_mask, cudaStream_t stream) {
  bool has_mask = block_mask_ptr != nullptr;
  if (block_size == 32) {
    constexpr int kBlockSize = 32;
    if (has_mask) {
      launch_warp_spec_with_kvcache_blocksparse_qkpertoken_perhead_vperhead_fp8_dim128<kBlockSize,
                                                                                       true>(
          y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr,
          cu_seqlens_q_ptr, block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q,
          max_seq_q, max_seq_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv,
          num_kvcache_blocks, block_size, scale_block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1,
          ldK2, ldV, ldV1, ldV2, ldKS, ldKS1, ldKS2, block_mask_ptr, num_tile_kv_in_mask, stream);
    } else {
      launch_warp_spec_with_kvcache_blocksparse_qkpertoken_perhead_vperhead_fp8_dim128<kBlockSize,
                                                                                       false>(
          y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr,
          cu_seqlens_q_ptr, block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q,
          max_seq_q, max_seq_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv,
          num_kvcache_blocks, block_size, scale_block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1,
          ldK2, ldV, ldV1, ldV2, ldKS, ldKS1, ldKS2, nullptr, 0, stream);
    }
  } else if (block_size == 64) {
    constexpr int kBlockSize = 64;
    if (has_mask) {
      launch_warp_spec_with_kvcache_blocksparse_qkpertoken_perhead_vperhead_fp8_dim128<kBlockSize,
                                                                                       true>(
          y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr,
          cu_seqlens_q_ptr, block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q,
          max_seq_q, max_seq_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv,
          num_kvcache_blocks, block_size, scale_block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1,
          ldK2, ldV, ldV1, ldV2, ldKS, ldKS1, ldKS2, block_mask_ptr, num_tile_kv_in_mask, stream);
    } else {
      launch_warp_spec_with_kvcache_blocksparse_qkpertoken_perhead_vperhead_fp8_dim128<kBlockSize,
                                                                                       false>(
          y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr,
          cu_seqlens_q_ptr, block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q,
          max_seq_q, max_seq_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv,
          num_kvcache_blocks, block_size, scale_block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1,
          ldK2, ldV, ldV1, ldV2, ldKS, ldKS1, ldKS2, nullptr, 0, stream);
    }
  }
}

}  // namespace prefill
}  // namespace attention
}  // namespace hpc
