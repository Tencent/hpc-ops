// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/attention/prefill/config.h"
#include "src/attention/prefill/mla_kernels.cuh"
#include "src/attention/prefill/warp_spec_mla.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace prefill {

void warp_spec_mla_async(void *y_ptr, const void *q_ptr, const void *kv_ptr,
                         const void *seqlens_q_ptr, const void *cu_seqlens_q_ptr, void *tmas_ptr,
                         int num_batch, int total_seq_q, int max_seq_q, int num_dim_qk,
                         int num_dim_v, int num_head_q, int num_head_kv, int ldY, int ldQ, int ldKV,
                         cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::bfloat16_t;
  using Tout = cute::bfloat16_t;

  auto Q = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(q_ptr)),
                       make_shape(max_seq_q, num_dim_qk, num_head_q),
                       make_stride(ldQ, Int<1>{}, num_dim_qk));
  auto KV = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(kv_ptr)),
                        make_shape(max_seq_q, num_dim_qk, num_head_kv),
                        make_stride(ldKV, Int<1>{}, num_dim_qk));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<const Tout *>(y_ptr)),
                       make_shape(max_seq_q, num_dim_v, num_head_q),
                       make_stride(ldY, Int<1>{}, num_dim_v));

  auto *tma_qkvy = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;

  using TiledMmaQK = SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>;
  using TiledMmaPV = SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>;
  using Config = MLAPrefillConfig<Tin, Tout, TiledMmaQK, TiledMmaPV, 128, 64, 192, 128, 2, 2, 1,
                                  128, 128, 128, 128>;

  Config config;
  auto [tma_q, tma_kv, tma_y] = config.get_tma(Q, KV, Y);

  // 0. update tma
  {
    vec_t<cute::TmaDescriptor, 3> td_qkvy{
        *tma_q.get_tma_descriptor(),
        *tma_kv.get_tma_descriptor(),
        *tma_y.get_tma_descriptor(),
    };
    kernels::update_batched_tma<Tin, decltype(tma_q), decltype(tma_kv), decltype(tma_y)>
        <<<num_batch, 32, 0, stream>>>(
            td_qkvy, tma_qkvy, (const Tin *)q_ptr, (const Tin *)kv_ptr, (const Tout *)y_ptr,
            (const int *)seqlens_q_ptr, (const int *)cu_seqlens_q_ptr, num_batch, max_seq_q,
            num_dim_qk, num_dim_v, num_head_q, num_head_kv, ldQ, ldKV, ldY);
  }

  // 1. compute attention
  {
    int kv_group = num_head_q / num_head_kv;
    cutlass::FastDivmod head_kv_divmod(kv_group);
    cutlass::FastDivmod head_q_divmod(num_head_q);
    cutlass::FastDivmod tile_m_divmod(num_batch * num_head_q);

    int shm_size = config.get_shm_size();

    dim3 block(384);
    dim3 grid(get_sm_count());
    auto kernel =
        kernels::mla_prefill_bf16_warp_specialization_kernel<decltype(config), decltype(tma_q),
                                                             decltype(tma_kv), decltype(tma_y)>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    kernel<<<grid, block, shm_size, stream>>>(
        tma_qkvy, (int *)seqlens_q_ptr, num_batch, max_seq_q, num_dim_qk, num_dim_v, num_head_q,
        num_head_kv, one_over_dk_log2e, head_kv_divmod, head_q_divmod, tile_m_divmod);
  }
}

}  // namespace prefill
}  // namespace attention
}  // namespace hpc
