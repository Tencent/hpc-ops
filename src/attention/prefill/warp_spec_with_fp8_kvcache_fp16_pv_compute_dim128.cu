// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>

#include "cute/tensor.hpp"
#include "src/attention/prefill/config.h"
#include "src/attention/prefill/kernels.cuh"
#include "src/attention/prefill/warp_spec_with_fp8_kvcache_fp16_pv_compute_dim128.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace prefill {

// Hybrid a8c8-fp16pv prefill: fp8 QK WGMMA + fp16 PV WGMMA. Topology mirrors
// a16c8 (kTileM=128, kTileN=64, kTileK=128, kTileV=128, kStage=2) but Tin is fp8
// and the QK MMA is the fp8 SS atom.

// Mode 21: Q per-(token, head) fp8 + K/V per-tensor warp-spec launcher.
template <int kBlockSize>
static void launch_warp_spec_with_fp8_kvcache_fp16_pv_compute_dim128_mode_kpertensor_vpertensor(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int max_seq_q, int max_seq_q_pad, int num_dim_qk, int num_dim_v,
    int num_head_q, int num_head_kv, int num_kvcache_blocks, int /*block_size*/,
    int num_seq_max_blocks, int ldY, int ldQ, int ldK, int ldK1, int ldK2, int ldV, int ldV1,
    int ldV2, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using TinQKV = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;
  using TinPV = cute::half_t;

  auto Q = make_tensor(make_gmem_ptr(reinterpret_cast<const TinQKV *>(q_ptr)),
                       make_shape(max_seq_q, num_dim_qk, num_head_q),
                       make_stride(ldQ, Int<1>{}, num_dim_qk));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<const Tout *>(y_ptr)),
                       make_shape(max_seq_q, num_dim_v, num_head_q),
                       make_stride(ldY, Int<1>{}, num_dim_v));
  auto K = make_tensor(make_gmem_ptr(reinterpret_cast<const TinQKV *>(kcache_ptr)),
                       make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks),
                       make_stride(ldK1, Int<1>{}, ldK2, ldK));
  auto V = make_tensor(make_gmem_ptr(reinterpret_cast<const TinQKV *>(vcache_ptr)),
                       make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks),
                       make_stride(Int<1>{}, ldV1, ldV2, ldV));
  auto QS = make_tensor(make_gmem_ptr(reinterpret_cast<const float *>(qscale_ptr)),
                        make_shape(max_seq_q_pad, num_head_q, num_batch),
                        make_stride(Int<1>{}, max_seq_q_pad, num_head_q * max_seq_q_pad));

  auto *tma_qy = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;

  // fp8 QK WGMMA, fp16 PV WGMMA. Config Tin is fp8 -> SLayoutQ/K/V are fp8.
  using TiledMmaQK = SM90_64x64x32_F32E4M3E4M3_SS_TN<>;
  using TiledMmaPV = SM90_64x128x16_F32F16F16_RS<GMMA::Major::K, GMMA::Major::MN>;
  using Config = AttentionKVCachePrefillConfig<TinQKV, Tout, TiledMmaQK, TiledMmaPV, 128, 64, 128,
                                               128, kBlockSize, 2, 2, 1, 128, 128, 128, 128>;

  // fp16 V compute stage (MN-major SW128, half) — same shape as the fp8 V stage.
  using SLayoutVFp16 = decltype(tile_to_shape(
      GMMA::Layout_MN_SW128_Atom<TinPV>{},
      make_shape(Int<Config::kTileV>{}, Int<Config::kTileN>{}, Int<Config::kStage>{})));
  // Per-row qscale (1 fp32 per Q-token row in the tile).
  using SLayoutQS = decltype(make_layout(make_shape(Int<Config::kTileM>{}), make_stride(Int<1>{})));
  // fp8 K/V TMA copy boxes (one paged block tile).
  using CopyBoxKFp8 = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<TinQKV>{},
                                             make_shape(Int<kBlockSize>{}, Int<Config::kTileK>{})));
  using CopyBoxVFp8 = decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<TinQKV>{},
                                             make_shape(Int<Config::kTileV>{}, Int<kBlockSize>{})));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, typename Config::SLayoutQ{});
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, typename Config::CopyBoxY{});
  auto tma_k_fp8 = make_tma_copy(SM90_TMA_LOAD{}, K, CopyBoxKFp8{});
  auto tma_v_fp8 = make_tma_copy(SM90_TMA_LOAD{}, V, CopyBoxVFp8{});
  auto tma_qs = make_tma_copy(SM90_TMA_LOAD{}, QS, SLayoutQS{});

  // 0. Update Q/Y TMA descriptors per batch.
  {
    vec_t<cute::TmaDescriptor, 2> td_qy{
        *tma_q.get_tma_descriptor(),
        *tma_y.get_tma_descriptor(),
    };
    kernels::update_batched_tma_with_kvcache<TinQKV, Tout, decltype(tma_q), decltype(tma_y)>
        <<<num_batch, 32, 0, stream>>>(td_qy, tma_qy, (const TinQKV *)q_ptr, (const Tout *)y_ptr,
                                       (const int *)cu_seqlens_q_ptr, num_batch, max_seq_q,
                                       num_dim_qk, num_dim_v, num_head_q, ldQ, ldY);
  }

  // 1. Compute attention.
  {
    int kv_group = num_head_q / num_head_kv;
    cutlass::FastDivmod head_kv_divmod(kv_group);
    cutlass::FastDivmod head_q_divmod(num_head_q);
    cutlass::FastDivmod tile_m_divmod(num_batch * num_head_q);

    // shm: fp8 Q | fp16 V | bf16 Y | fp32 QS | fp8 K stage | fp8 V stage | ints.
    int shm_size = static_cast<int>(cosize(typename Config::SLayoutQ{}) * sizeof(TinQKV));
    shm_size += static_cast<int>(cosize(SLayoutVFp16{}) * sizeof(TinPV));
    shm_size += static_cast<int>(cosize(typename Config::SLayoutY{}) * sizeof(Tout));
    shm_size += static_cast<int>(cosize(SLayoutQS{}) * sizeof(float));
    shm_size += static_cast<int>(
        (cosize(typename Config::SLayoutK{}) + cosize(typename Config::SLayoutV{})) *
        sizeof(TinQKV));
    shm_size += sizeof(int) * num_batch * 3;

    // 384 threads: 2 consumer warpgroups (256) + 1 producer warpgroup (128).
    // The V fp8->fp16 cvt runs on the producer's idle warps 9-11 (no dedicated
    // 4th cvt warpgroup). See kernels.cuh mode-21 kernel for the topology.
    dim3 block(384);
    dim3 grid(get_sm_count());
    auto kernel = kernels::
        attention_with_kvcache_prefill_qfp8_kpertensor_vpertensor_fp16_pv_compute_warp_specialization_kernel<  // NOLINT(whitespace/line_length)
            Config, decltype(tma_q), decltype(tma_k_fp8), decltype(tma_v_fp8), decltype(tma_y),
            decltype(tma_qs), SLayoutVFp16, SLayoutQS>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    kernel<<<grid, block, shm_size, stream>>>(
        tma_qy, tma_k_fp8, tma_v_fp8, tma_qs, reinterpret_cast<const float *>(qscale_ptr),
        reinterpret_cast<const float *>(kscale_ptr), reinterpret_cast<const float *>(vscale_ptr),
        (const int *)cu_seqlens_q_ptr, (const int *)seqlens_kvcache_ptr, (const int *)block_ids_ptr,
        num_batch, max_seq_q, max_seq_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv,
        num_kvcache_blocks, kBlockSize, num_seq_max_blocks, one_over_dk_log2e, head_kv_divmod,
        head_q_divmod, tile_m_divmod);
  }
}

// Mode 20: Q per-(token, head) fp8 + K per-(token, head) + V per-head launcher.
template <int kBlockSize>
static void
launch_warp_spec_with_fp8_kvcache_fp16_pv_compute_dim128_mode_kpertoken_perhead_vperhead(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int max_seq_q, int max_seq_q_pad, int num_dim_qk, int num_dim_v,
    int num_head_q, int num_head_kv, int num_kvcache_blocks, int /*block_size*/,
    int num_seq_max_blocks, int ldY, int ldQ, int ldK, int ldK1, int ldK2, int ldV, int ldV1,
    int ldV2, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using TinQKV = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;
  using TinPV = cute::half_t;

  auto Q = make_tensor(make_gmem_ptr(reinterpret_cast<const TinQKV *>(q_ptr)),
                       make_shape(max_seq_q, num_dim_qk, num_head_q),
                       make_stride(ldQ, Int<1>{}, num_dim_qk));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<const Tout *>(y_ptr)),
                       make_shape(max_seq_q, num_dim_v, num_head_q),
                       make_stride(ldY, Int<1>{}, num_dim_v));
  auto K = make_tensor(make_gmem_ptr(reinterpret_cast<const TinQKV *>(kcache_ptr)),
                       make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks),
                       make_stride(ldK1, Int<1>{}, ldK2, ldK));
  auto V = make_tensor(make_gmem_ptr(reinterpret_cast<const TinQKV *>(vcache_ptr)),
                       make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks),
                       make_stride(Int<1>{}, ldV1, ldV2, ldV));
  auto QS = make_tensor(make_gmem_ptr(reinterpret_cast<const float *>(qscale_ptr)),
                        make_shape(max_seq_q_pad, num_head_q, num_batch),
                        make_stride(Int<1>{}, max_seq_q_pad, num_head_q * max_seq_q_pad));

  auto *tma_qy = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;

  using TiledMmaQK = SM90_64x64x32_F32E4M3E4M3_SS_TN<>;
  using TiledMmaPV = SM90_64x128x16_F32F16F16_RS<GMMA::Major::K, GMMA::Major::MN>;
  using Config = AttentionKVCachePrefillConfig<TinQKV, Tout, TiledMmaQK, TiledMmaPV, 128, 64, 128,
                                               128, kBlockSize, 2, 2, 1, 128, 128, 128, 128>;

  using SLayoutVFp16 = decltype(tile_to_shape(
      GMMA::Layout_MN_SW128_Atom<TinPV>{},
      make_shape(Int<Config::kTileV>{}, Int<Config::kTileN>{}, Int<Config::kStage>{})));
  using SLayoutQS = decltype(make_layout(make_shape(Int<Config::kTileM>{}), make_stride(Int<1>{})));
  using CopyBoxKFp8 = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<TinQKV>{},
                                             make_shape(Int<kBlockSize>{}, Int<Config::kTileK>{})));
  using CopyBoxVFp8 = decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<TinQKV>{},
                                             make_shape(Int<Config::kTileV>{}, Int<kBlockSize>{})));

  // Per-(K-token, head) K scale: 1 fp32 per (block, K-token, KV-head); 4D fp8-
  // byte-packed presentation (KS strides = K strides / sizeof(float)).
  constexpr int kScaleByteSize = sizeof(float);
  constexpr int kTileScale = Config::kTileK / kScaleByteSize;
  int num_dim_scale = num_dim_qk / kScaleByteSize;
  int num_scale_blocks = num_kvcache_blocks;
  auto KS = make_tensor(
      make_gmem_ptr(reinterpret_cast<const float *>(kscale_ptr)),
      make_shape(kBlockSize / num_dim_scale, num_dim_scale, num_head_kv, num_scale_blocks),
      make_stride(ldK1 / kScaleByteSize, Int<1>{}, ldK2 / kScaleByteSize, ldK / kScaleByteSize));

  // KS smem layout: (kTileN/kTileScale, kTileScale, kStage); cosize/stage = kTileN.
  using SLayoutKS = decltype(make_layout(
      make_shape(Int<Config::kTileN / kTileScale>{}, Int<kTileScale>{}, Int<Config::kStage>{}),
      make_stride(Int<kTileScale>{}, Int<1>{}, Int<Config::kTileN>{})));
  using SLayoutKSC = SLayoutKS;
  using CopyBoxKS =
      decltype(make_layout(make_shape(Int<kBlockSize / kTileScale>{}, Int<kTileScale>{}),
                           make_stride(Int<kTileScale>{}, Int<1>{})));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, typename Config::SLayoutQ{});
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, typename Config::CopyBoxY{});
  auto tma_k_fp8 = make_tma_copy(SM90_TMA_LOAD{}, K, CopyBoxKFp8{});
  auto tma_v_fp8 = make_tma_copy(SM90_TMA_LOAD{}, V, CopyBoxVFp8{});
  auto tma_qs = make_tma_copy(SM90_TMA_LOAD{}, QS, SLayoutQS{});
  auto tma_ks = make_tma_copy(SM90_TMA_LOAD{}, KS, CopyBoxKS{});

  {
    vec_t<cute::TmaDescriptor, 2> td_qy{
        *tma_q.get_tma_descriptor(),
        *tma_y.get_tma_descriptor(),
    };
    kernels::update_batched_tma_with_kvcache<TinQKV, Tout, decltype(tma_q), decltype(tma_y)>
        <<<num_batch, 32, 0, stream>>>(td_qy, tma_qy, (const TinQKV *)q_ptr, (const Tout *)y_ptr,
                                       (const int *)cu_seqlens_q_ptr, num_batch, max_seq_q,
                                       num_dim_qk, num_dim_v, num_head_q, ldQ, ldY);
  }

  {
    int kv_group = num_head_q / num_head_kv;
    cutlass::FastDivmod head_kv_divmod(kv_group);
    cutlass::FastDivmod head_q_divmod(num_head_q);
    cutlass::FastDivmod tile_m_divmod(num_batch * num_head_q);

    // shm: fp8 Q | fp16 V | bf16 Y | fp32 QS | fp8 K | fp8 V | fp32 KS | ints.
    int shm_size = static_cast<int>(cosize(typename Config::SLayoutQ{}) * sizeof(TinQKV));
    shm_size += static_cast<int>(cosize(SLayoutVFp16{}) * sizeof(TinPV));
    shm_size += static_cast<int>(cosize(typename Config::SLayoutY{}) * sizeof(Tout));
    shm_size += static_cast<int>(cosize(SLayoutQS{}) * sizeof(float));
    shm_size += static_cast<int>(
        (cosize(typename Config::SLayoutK{}) + cosize(typename Config::SLayoutV{})) *
        sizeof(TinQKV));
    shm_size += static_cast<int>(cosize(SLayoutKSC{}) * sizeof(float));
    shm_size += sizeof(int) * num_batch * 3;

    // 384 threads: 2 consumer warpgroups (256) + 1 producer warpgroup (128).
    // The V fp8->fp16 cvt runs on the producer's idle warps 9-11 (no dedicated
    // 4th cvt warpgroup). See kernels.cuh mode-20 kernel for the topology.
    dim3 block(384);
    dim3 grid(get_sm_count());
    auto kernel = kernels::
        attention_with_kvcache_prefill_qfp8_kpertoken_perhead_vperhead_fp16_pv_compute_warp_specialization_kernel<  // NOLINT(whitespace/line_length)
            Config, decltype(tma_q), decltype(tma_k_fp8), decltype(tma_v_fp8), decltype(tma_y),
            decltype(tma_qs), decltype(tma_ks), SLayoutVFp16, SLayoutQS, SLayoutKS, SLayoutKSC,
            CopyBoxKS>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    kernel<<<grid, block, shm_size, stream>>>(
        tma_qy, tma_k_fp8, tma_v_fp8, tma_qs, tma_ks, reinterpret_cast<const float *>(qscale_ptr),
        reinterpret_cast<const float *>(kscale_ptr), reinterpret_cast<const float *>(vscale_ptr),
        (const int *)cu_seqlens_q_ptr, (const int *)seqlens_kvcache_ptr, (const int *)block_ids_ptr,
        num_batch, max_seq_q, max_seq_q_pad, num_dim_qk, num_dim_v, num_dim_scale, num_head_q,
        num_head_kv, num_kvcache_blocks, num_scale_blocks, kBlockSize, num_seq_max_blocks,
        one_over_dk_log2e, head_kv_divmod, head_q_divmod, tile_m_divmod);
  }
}

void warp_spec_with_fp8_kvcache_fp16_pv_compute_dim128_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK, int ldK1, int ldK2, int ldV,
    int ldV1, int ldV2, int quant_type, cudaStream_t stream) {
  (void)total_seq_q;

  if (quant_type == 20) {
    if (block_size == 32) {
      launch_warp_spec_with_fp8_kvcache_fp16_pv_compute_dim128_mode_kpertoken_perhead_vperhead<32>(
          y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr,
          cu_seqlens_q_ptr, block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, max_seq_q,
          max_seq_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv, num_kvcache_blocks,
          block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1, ldV2, stream);
    } else if (block_size == 64) {
      launch_warp_spec_with_fp8_kvcache_fp16_pv_compute_dim128_mode_kpertoken_perhead_vperhead<64>(
          y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr,
          cu_seqlens_q_ptr, block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, max_seq_q,
          max_seq_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv, num_kvcache_blocks,
          block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1, ldV2, stream);
    }
    return;
  }

  // quant_type == 21 (static, K/V per-tensor).
  if (block_size == 32) {
    launch_warp_spec_with_fp8_kvcache_fp16_pv_compute_dim128_mode_kpertensor_vpertensor<32>(
        y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, cu_seqlens_q_ptr,
        block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, max_seq_q, max_seq_q_pad,
        num_dim_qk, num_dim_v, num_head_q, num_head_kv, num_kvcache_blocks, block_size,
        num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1, ldV2, stream);
  } else if (block_size == 64) {
    launch_warp_spec_with_fp8_kvcache_fp16_pv_compute_dim128_mode_kpertensor_vpertensor<64>(
        y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, cu_seqlens_q_ptr,
        block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, max_seq_q, max_seq_q_pad,
        num_dim_qk, num_dim_v, num_head_q, num_head_kv, num_kvcache_blocks, block_size,
        num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1, ldV2, stream);
  }
}

}  // namespace prefill
}  // namespace attention
}  // namespace hpc
