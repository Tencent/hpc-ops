// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/attention/prefill/config.h"
#include "src/attention/prefill/kernels.cuh"
#include "src/attention/prefill/warp_spec_blocksparse_fp8_dim192.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace prefill {

template <bool kHasMask>
void launch_warp_spec_blocksparse_fp8_dim192(
    void *y_ptr, const void *q_ptr, const void *k_ptr, const void *v_ptr,
    const void *cu_seqlens_q_ptr, const void *cu_seqlens_kv_ptr, void *tmas_ptr, int num_batch,
    int total_seq_q, int max_seq_q, int max_seq_kv, int num_dim_qk, int num_dim_v, int num_head_q,
    int num_head_kv, int ldY, int ldQ, int ldK, int ldV, const void *block_mask_ptr,
    int num_tile_kv_in_mask, float softmax_qkscale, float vscale, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  auto Q = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(q_ptr)),
                       make_shape(max_seq_q, num_dim_qk, num_head_q),
                       make_stride(ldQ, Int<1>{}, num_dim_qk));
  auto K = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(k_ptr)),
                       make_shape(max_seq_kv, num_dim_qk, num_head_kv),
                       make_stride(ldK, Int<1>{}, num_dim_qk));
  auto V = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(v_ptr)),
                       make_shape(num_dim_v, max_seq_kv, num_head_kv),
                       make_stride(Int<1>{}, ldV, num_dim_v));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<const Tout *>(y_ptr)),
                       make_shape(max_seq_q, num_dim_v, num_head_q),
                       make_stride(ldY, Int<1>{}, num_dim_v));

  auto *tma_qkvy = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = softmax_qkscale * kLog2e;

  using TiledMmaQK = SM90_64x64x32_F32E4M3E4M3_SS_TN<>;
  using TiledMmaPV = SM90_64x128x32_F32E4M3E4M3_RS_TN<>;
  using Config = AttentionPrefillFp8Config<Tin, Tout, TiledMmaQK, TiledMmaPV, 128, 128, 192, 128, 2,
                                           2, 1, 64, 64, 128, 128>;

  Config config;
  // Use std::get to dodge a GCC 13 tuple_element bug on fp8+dim192 TiledCopy types.
  auto tmas = config.get_tma(Q, K, V, Y);
  auto tma_q = std::get<0>(tmas);
  auto tma_k = std::get<1>(tmas);
  auto tma_v = std::get<2>(tmas);
  auto tma_y = std::get<3>(tmas);

  // 0. update tma
  {
    vec_t<cute::TmaDescriptor, 4> td_qkvy{
        *tma_q.get_tma_descriptor(),
        *tma_k.get_tma_descriptor(),
        *tma_v.get_tma_descriptor(),
        *tma_y.get_tma_descriptor(),
    };
    kernels::update_batched_tma_fp8<Tin, Tout, decltype(tma_q), decltype(tma_k), decltype(tma_v),
                                    decltype(tma_y)><<<num_batch, 32, 0, stream>>>(
        td_qkvy, tma_qkvy, (const Tin *)q_ptr, (const Tin *)k_ptr, (const Tin *)v_ptr,
        (const Tout *)y_ptr, (const int *)cu_seqlens_q_ptr, (const int *)cu_seqlens_kv_ptr,
        num_batch, max_seq_q, max_seq_kv, num_dim_qk, num_dim_v, num_head_q, num_head_kv, ldQ, ldK,
        ldV, ldY);
  }

  // 1. compute attention
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
    auto kernel = kernels::attention_blocksparse_prefill_fp8_warp_specialization_kernel<
        decltype(config), decltype(tma_q), decltype(tma_k), decltype(tma_v), decltype(tma_y),
        kHasMask>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    kernel<<<grid, block, shm_size, stream>>>(
        tma_qkvy, (const int *)cu_seqlens_q_ptr, (const int *)cu_seqlens_kv_ptr, num_batch,
        max_seq_q, max_seq_kv, num_dim_qk, num_dim_v, num_head_q, num_head_kv, one_over_dk_log2e,
        vscale, head_kv_divmod, head_q_divmod, tile_m_divmod, (const uint8_t *)block_mask_ptr,
        num_tile_kv_in_mask);
  }
}

void warp_spec_blocksparse_fp8_dim192_async(
    void *y_ptr, const void *q_ptr, const void *k_ptr, const void *v_ptr,
    const void *cu_seqlens_q_ptr, const void *cu_seqlens_kv_ptr, void *tmas_ptr, int num_batch,
    int total_seq_q, int max_seq_q, int max_seq_kv, int num_dim_qk, int num_dim_v, int num_head_q,
    int num_head_kv, int ldY, int ldQ, int ldK, int ldV, const void *block_mask_ptr,
    int num_tile_kv_in_mask, float softmax_qkscale, float vscale, cudaStream_t stream) {
  bool has_mask = block_mask_ptr != nullptr;
  if (has_mask) {
    launch_warp_spec_blocksparse_fp8_dim192<true>(
        y_ptr, q_ptr, k_ptr, v_ptr, cu_seqlens_q_ptr, cu_seqlens_kv_ptr, tmas_ptr, num_batch,
        total_seq_q, max_seq_q, max_seq_kv, num_dim_qk, num_dim_v, num_head_q, num_head_kv, ldY,
        ldQ, ldK, ldV, block_mask_ptr, num_tile_kv_in_mask, softmax_qkscale, vscale, stream);
  } else {
    launch_warp_spec_blocksparse_fp8_dim192<false>(
        y_ptr, q_ptr, k_ptr, v_ptr, cu_seqlens_q_ptr, cu_seqlens_kv_ptr, tmas_ptr, num_batch,
        total_seq_q, max_seq_q, max_seq_kv, num_dim_qk, num_dim_v, num_head_q, num_head_kv, ldY,
        ldQ, ldK, ldV, nullptr, 0, softmax_qkscale, vscale, stream);
  }
}

}  // namespace prefill
}  // namespace attention
}  // namespace hpc
