// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "cute/tensor.hpp"
#include "src/attention/decode/sm100/static/smallm_fp8_qpertoken_perhead_kvpertensor_dim128_static_clc_nosplitk_kernels.cuh"
#include "src/attention/decode/smallm_dim128.h"
#include "src/attention/decode/splitk_combine_kernels.cuh"

namespace hpc {
namespace attention {
namespace decode {

template <int kTileM, int kTileN, int kTileK, int kTileV, int kBlockSize>
void launch_smallm_fp8_qpertoken_perhead_kvpertensor_dim128_static_clc_nosplitk_kernel(
    void *y_ptr, void *lse_ptr, void *splitk_out_ptr, const int *task_map_ptr, const void *q_ptr,
    void *kcache_ptr, void *vcache_ptr, const int *block_ids_ptr, const int *num_seq_kvcache_ptr,
    const float *qscale_ptr, const float *kscale_ptr, const float *vscale_ptr, int *split_flag_ptr,
    bool new_kv_included, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int heads_per_group, int num_dim_qk, int num_dim_v, int num_kvcache_blocks,
    int block_size, int num_seq_max_blocks, int qscale_pad_stride, int ldY, int ldQ,
    int64_t kcache_block_stride, int64_t kcache_token_stride, int64_t kcache_head_stride,
    int64_t vcache_block_stride, int64_t vcache_token_stride, int64_t vcache_head_stride,
    cudaStream_t stream) {
  using namespace cute;  // NOLINT

  constexpr int kStageQ = 4;
  constexpr int kStageK = 4;
  constexpr int kStageP = 4;
  constexpr int kHeadsPerGroup = 8;

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  auto Q = make_tensor(
      make_gmem_ptr(reinterpret_cast<const Tin *>(q_ptr)),
      make_shape(heads_per_group, num_dim_qk, num_head_k, num_seq_q, num_batch),
      make_stride(num_dim_qk, Int<1>{}, heads_per_group * num_dim_qk, ldQ, ldQ * num_seq_q));

  auto K = make_tensor(
      make_gmem_ptr(reinterpret_cast<const Tin *>(kcache_ptr)),
      make_shape(kBlockSize, num_dim_qk, num_head_k, num_kvcache_blocks),
      make_stride(kcache_token_stride, Int<1>{}, kcache_head_stride, kcache_block_stride));

  auto V = make_tensor(
      make_gmem_ptr(reinterpret_cast<const Tin *>(vcache_ptr)),
      make_shape(num_dim_v, kBlockSize, num_head_v, num_kvcache_blocks),
      make_stride(Int<1>{}, vcache_token_stride, vcache_head_stride, vcache_block_stride));

  auto Y = make_tensor(
      make_gmem_ptr(reinterpret_cast<const Tout *>(y_ptr)),
      make_shape(num_dim_v, heads_per_group, num_head_k, num_seq_q, num_batch),
      make_stride(Int<1>{}, num_dim_v, heads_per_group * num_dim_v, ldY, ldY * num_seq_q));

  auto slayout_q = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStageQ>{}));

  auto slayout_k = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStageK>{}));

  auto slayout_p = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileN>{}, Int<kTileM>{}, Int<kStageP>{}));

  auto slayout_s = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileM>{}, Int<kTileN>{}, Int<kStageP>{}));

  auto slayout_v = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileV>{}, Int<kTileN>{}, Int<kStageK>{}));

  auto slayout_y =
      tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tout>{}, make_shape(Int<kTileV>{}, Int<kTileM>{}));

  auto tma_copy_layout_q = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                         make_shape(Int<kHeadsPerGroup>{}, Int<kTileK>{}));
  auto tma_copy_layout_k =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kBlockSize>{}, Int<kTileK>{}));
  auto tma_copy_layout_v = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{},
                                         make_shape(Int<kTileV>{}, Int<kBlockSize>{}));
  auto tma_copy_layout_y = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tout>{},
                                         make_shape(Int<kTileV>{}, Int<kHeadsPerGroup>{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, tma_copy_layout_q);
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K, tma_copy_layout_k);
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, V, tma_copy_layout_v);
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, tma_copy_layout_y);

  auto qk_tiled_mma = make_tiled_mma(
      MMA_Traits<SM100_MMA_F8F6F4_SS, Tin, Tin, float, cute::C<kTileN>, cute::C<kTileM>,
                 cute::integral_constant<UMMA::Major, UMMA::Major::K>,
                 cute::integral_constant<UMMA::Major, UMMA::Major::K>,
                 cute::integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                 cute::integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One> >{});

  auto sv_tiled_mma = make_tiled_mma(
      MMA_Traits<SM100_MMA_F8F6F4_SS, Tin, Tin, float, cute::C<kTileV>, cute::C<kTileM>,
                 cute::integral_constant<UMMA::Major, UMMA::Major::MN>,
                 cute::integral_constant<UMMA::Major, UMMA::Major::K>,
                 cute::integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                 cute::integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One> >{});
  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;
  int pad_heads_per_group = ((heads_per_group + 7) / 8) * 8;

  static constexpr int shm_qkpv =
      (cosize(slayout_q) + cosize(slayout_k) + cosize(slayout_v) + cosize(slayout_p)) * sizeof(Tin);
  static constexpr int shm_y = cosize(slayout_y) * sizeof(Tout);
  //   int shm_blk_ids = sizeof(int) * num_seq_max_blocks * kStageQ;
  int shm_size = shm_qkpv + shm_y;

  cutlass::FastDivmod splitk_head_kv_divider(num_head_k);

  constexpr int kClusterM = 1;
  constexpr int kClusterN = 1;
  constexpr int kClusterK = 1;
  constexpr int kClusters = kClusterM * kClusterN * kClusterK;
  constexpr int kMmaSM = 1;

  dim3 grid(num_head_k * num_batch);
  dim3 block(512);

  auto kernel = kernels::
      smallm_attention_decode_fp8_qpertoken_perhead_kvpertensor_static_1sm_clc_nosplitk_kernel<
          Tout, Tin, kTileM, kTileN, kTileK, kTileV, kHeadsPerGroup, decltype(qk_tiled_mma),
          decltype(sv_tiled_mma), decltype(tma_q), decltype(tma_k), decltype(tma_v),
          decltype(tma_y), decltype(slayout_q), decltype(slayout_k), decltype(slayout_p),
          decltype(slayout_s), decltype(slayout_v), decltype(slayout_y), kClusterM, kClusterN,
          kClusterK, kMmaSM, kBlockSize, kStageQ, kStageK, kStageP>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  cudaLaunchConfig_t config;
  memset(&config, 0, sizeof(config));

  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = shm_size;

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = kClusters;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  config.numAttrs = 1;
  config.attrs = attribute;

  config.stream = stream;

  cudaLaunchKernelEx(
      &config, kernel, tma_q, tma_k, tma_v, tma_y, block_ids_ptr, num_seq_kvcache_ptr, qscale_ptr,
      kscale_ptr, vscale_ptr, new_kv_included, num_batch, num_seq_q, num_dim_qk, num_dim_v,
      num_head_q, num_head_k, num_head_v, heads_per_group, pad_heads_per_group, num_kvcache_blocks,
      num_seq_max_blocks, qscale_pad_stride, one_over_dk_log2e, splitk_head_kv_divider);
}

bool smallm_fp8_qpertoken_perhead_kvpertensor_dim128_static_async(
    void *y_ptr, void *lse_ptr, void *splitk_out_ptr, const int *task_map_ptr, const void *q_ptr,
    void *kcache_ptr, void *vcache_ptr, const int *block_ids_ptr, const int *num_seq_kvcache_ptr,
    const float *qscale_ptr, const float *kscale_ptr, const float *vscale_ptr, int *split_flag_ptr,
    bool new_kv_included, int splitk, int splitk_min_len, int consumers, int num_batch,
    int num_seq_q, int num_head_q, int num_head_k, int num_head_v, int num_dim_qk, int num_dim_v,
    int num_kvcache_blocks, int block_size, int num_seq_max_blocks, int qscale_pad_stride, int ldY,
    int ldQ, int64_t kcache_block_stride, int64_t kcache_token_stride, int64_t kcache_head_stride,
    int64_t vcache_block_stride, int64_t vcache_token_stride, int64_t vcache_head_stride,
    cudaStream_t stream) {
  using namespace cute;  // NOLINT

  constexpr int kTileN = 128;
  constexpr int kTileK = 128;
  constexpr int kTileV = 128;

  if (num_dim_qk != kTileK || num_dim_v != kTileV || block_size != 64) {
    std::cout << "launch launch_attention_decode_bf16_dim128_smallm_fp8 failed with "
              << "  num_dim_qk: " << num_dim_qk << ", num_dim_v: " << num_dim_v
              << ", block_size:" << block_size << std::endl;
    return false;
  }

  int heads_per_group = num_head_q / num_head_k;
  if (heads_per_group != 8) {
    std::cout << "launch launch_attention_decode_bf16_dim128_smallm_fp8 failed with "
              << " heads_per_group:" << heads_per_group << ", num_head_q:" << num_head_q
              << ", num_head_k:" << num_head_k << std::endl;
    return false;
  }

  auto launch = [&](auto tilem_tag) {
    constexpr int kTileM = decltype(tilem_tag)::value;
    constexpr int kBlockSize = 64;

    launch_smallm_fp8_qpertoken_perhead_kvpertensor_dim128_static_clc_nosplitk_kernel<
        kTileM, kTileN, kTileK, kTileV, kBlockSize>(
        y_ptr, lse_ptr, splitk_out_ptr, task_map_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
        num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr, new_kv_included,
        num_batch, num_seq_q, num_head_q, num_head_k, num_head_v, heads_per_group, num_dim_qk,
        num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks, qscale_pad_stride, ldY, ldQ,
        kcache_block_stride, kcache_token_stride, kcache_head_stride, vcache_block_stride,
        vcache_token_stride, vcache_head_stride, stream);
  };

  if (num_seq_q == 1) {
    launch(std::integral_constant<int, 8>{});
  } else if (num_seq_q == 2) {
    launch(std::integral_constant<int, 16>{});
  } else if (num_seq_q == 3 || num_seq_q == 4) {
    launch(std::integral_constant<int, 32>{});
  }

  return true;
}

}  // namespace decode
}  // namespace attention
}  // namespace hpc
