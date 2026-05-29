// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "cute/tensor.hpp"
#include "src/attention/decode/sm90/dynamic/smallm_fp8_qpertoken_perhead_kvpertensor_dim128_dynamic_splitk_kernels.cuh"
#include "src/attention/decode/smallm_dim128.h"
#include "src/attention/decode/splitk_combine_kernels.cuh"

namespace hpc {
namespace attention {
namespace decode {

template <int kTileM, bool kAInReg>
static constexpr auto mma_selector_fp8() {
  using namespace cute;  // NOLINT
  if constexpr (kAInReg) {
    if constexpr (kTileM == 8) {
      return SM90_64x8x32_F32E4M3E4M3_RS_TN<>{};
    } else if constexpr (kTileM == 16) {
      return SM90_64x16x32_F32E4M3E4M3_RS_TN<>{};
    } else if constexpr (kTileM == 24) {
      return SM90_64x24x32_F32E4M3E4M3_RS_TN<>{};
    } else if constexpr (kTileM == 32) {
      return SM90_64x32x32_F32E4M3E4M3_RS_TN<>{};
    }
  } else {
    if constexpr (kTileM == 8) {
      return SM90_64x8x32_F32E4M3E4M3_SS_TN<>{};
    } else if constexpr (kTileM == 16) {
      return SM90_64x16x32_F32E4M3E4M3_SS_TN<>{};
    } else if constexpr (kTileM == 24) {
      return SM90_64x24x32_F32E4M3E4M3_SS_TN<>{};
    } else if constexpr (kTileM == 32) {
      return SM90_64x32x32_F32E4M3E4M3_SS_TN<>{};
    }
  }
}

template <int kTileM, int kTileN, int kTileK, int kTileV, int kBlockSize, int kMaxSplitK>
static void launch_smallm_fp8_qpertoken_perhead_kvpertensor_dim128_dynamic_splitk_kernel(
    void *y_ptr, void *splitk_out_ptr, void *lse_ptr, const int *task_map_ptr, const void *q_ptr,
    void *kcache_ptr, void *vcache_ptr, const int *block_ids_ptr, const float *qscale_ptr,
    const float *kscale_ptr, const float *vscale_ptr, int num_batch, int num_seq_q, int num_head_q,
    int num_head_k, int num_head_v, int heads_per_group, int num_dim_qk, int num_dim_v,
    int num_kvcache_blocks, int num_seq_max_blocks, int qscale_pad_stride, int ldQ,
    int64_t kcache_block_stride, int64_t kcache_token_stride, int64_t kcache_head_stride,
    int64_t vcache_block_stride, int64_t vcache_token_stride, int64_t vcache_head_stride,
    cudaStream_t stream) {
  using namespace cute;  // NOLINT

  constexpr int kStage = 2;
  constexpr int kHeadsPerGroup = 8;
  constexpr int kWarpGroupN = 1;

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

  auto splitY = make_tensor(
      make_gmem_ptr(reinterpret_cast<float *>(splitk_out_ptr)),
      make_shape(num_dim_v, heads_per_group, num_head_k, num_seq_q, kMaxSplitK, num_batch),
      make_stride(Int<1>{}, num_dim_v, heads_per_group * num_dim_v, num_dim_v * num_head_q,
                  num_dim_v * num_head_q * num_seq_q,
                  num_dim_v * num_head_q * num_seq_q * kMaxSplitK));

  auto slayout_q =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto slayout_k = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_p = tile_to_shape(GMMA::Layout_MN_SW64_Atom<Tin>{},
                                 make_shape(Int<kTileN>{}, Int<kTileM>{}, Int<kWarpGroupN>{}));
  auto slayout_s = tile_to_shape(GMMA::Layout_K_SW64_Atom<Tin>{},
                                 make_shape(Int<kTileM>{}, Int<kTileN>{}, Int<kWarpGroupN>{}));
  auto slayout_vtma = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{},
                                    make_shape(Int<kTileV>{}, Int<kTileN>{}, Int<kStage>{}));
  auto slayout_splity = tile_to_shape(GMMA::Layout_MN_SW128_Atom<float>{},
                                      make_shape(Int<kTileV>{}, Int<kTileM>{}, Int<kWarpGroupN>{}));

  auto tma_copy_layout_q = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                         make_shape(Int<kHeadsPerGroup>{}, Int<kTileK>{}));
  auto tma_copy_layout_k =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kBlockSize>{}, Int<kTileK>{}));
  auto tma_copy_layout_v = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{},
                                         make_shape(Int<kTileV>{}, Int<kBlockSize>{}));
  auto tma_copy_layout_splity = tile_to_shape(GMMA::Layout_MN_SW128_Atom<float>{},
                                              make_shape(Int<kTileV>{}, Int<kHeadsPerGroup>{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, tma_copy_layout_q);
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K, tma_copy_layout_k);
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, V, tma_copy_layout_v);
  auto tma_splity = make_tma_copy(SM90_TMA_STORE{}, splitY, tma_copy_layout_splity);

  auto qk_mma_atom = mma_selector_fp8<kTileM, false>();
  auto sv_mma_atom = mma_selector_fp8<kTileM, true>();

  using TiledMmaQK = decltype(make_tiled_mma(qk_mma_atom));
  using TiledMmaSV = decltype(make_tiled_mma(sv_mma_atom));

  constexpr int kWarpsPerWrapGroup = 4;
  int shm_qkv = (cosize(slayout_q) + cosize(slayout_k) + cosize(slayout_vtma) + cosize(slayout_p)) *
                    sizeof(Tin) +
                sizeof(float) * kTileM * kWarpsPerWrapGroup * kWarpGroupN;
  int shm_blk_ids = sizeof(int) * num_seq_max_blocks;
  int shm_splity = cosize(slayout_splity) * sizeof(float);
  int shm_size = std::max(shm_qkv + shm_blk_ids, shm_splity);

  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;

  int num_sm = get_sm_count();
  int num_total_ctas = num_sm * dynamic::kCtaPerSmMap.at(9)[num_seq_q - 1];

  dim3 grid(num_total_ctas);
  dim3 block(size(TiledMmaQK{}) * kWarpGroupN + 32);

  auto kernel =
      kernels::smallm_attention_decode_fp8_qpertoken_perhead_kvpertensor_dynamic_splitk_kernel<
          Tin, Tout, kTileM, kTileN, kTileK, kTileV, kHeadsPerGroup, TiledMmaQK, TiledMmaSV,
          decltype(tma_q), decltype(tma_k), decltype(tma_v), decltype(tma_splity),
          decltype(slayout_q), decltype(slayout_k), decltype(slayout_p), decltype(slayout_s),
          decltype(slayout_vtma), decltype(slayout_splity), kBlockSize, kStage, kMaxSplitK>;

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  int pad_heads_per_group = ((heads_per_group + 7) / 8) * 8;

  cudaLaunchConfig_t attn_config;
  memset(&attn_config, 0, sizeof(attn_config));
  attn_config.gridDim = grid;
  attn_config.blockDim = block;
  attn_config.dynamicSmemBytes = shm_size;
  cudaLaunchAttribute attn_attrs[1];
  attn_attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attn_attrs[0].val.programmaticStreamSerializationAllowed = 1;
  attn_config.numAttrs = 1;
  attn_config.attrs = attn_attrs;
  attn_config.stream = stream;

  cudaLaunchKernelEx(&attn_config, kernel, tma_q, tma_k, tma_v, tma_splity,
                     reinterpret_cast<float *>(splitk_out_ptr), reinterpret_cast<float *>(lse_ptr),
                     task_map_ptr, block_ids_ptr, qscale_ptr, kscale_ptr, vscale_ptr, num_batch,
                     num_seq_q, num_dim_qk, num_dim_v, num_head_q, num_head_k, num_head_v,
                     heads_per_group, pad_heads_per_group, num_kvcache_blocks, num_seq_max_blocks,
                     qscale_pad_stride, one_over_dk_log2e);

  // ---- combine kernel: LSE-weighted reduction across chunks ----
  constexpr int kCombineWarps = 4;
  dim3 combine_grid(num_head_q / kCombineWarps, num_seq_q, num_batch);
  dim3 combine_block(kCombineWarps * 32);
  cutlass::FastDivmod hpg_divider(heads_per_group);

  auto combine_kernel =
      kernels::attention_decode_dynamic_splitk_combine_kernel<__nv_bfloat16, kCombineWarps,
                                                              kMaxSplitK>;

  cudaLaunchConfig_t combine_config;
  memset(&combine_config, 0, sizeof(combine_config));
  combine_config.gridDim = combine_grid;
  combine_config.blockDim = combine_block;
  combine_config.dynamicSmemBytes = 0;
  cudaLaunchAttribute combine_attrs[1];
  combine_attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  combine_attrs[0].val.programmaticStreamSerializationAllowed = 1;
  combine_config.numAttrs = 1;
  combine_config.attrs = combine_attrs;
  combine_config.stream = stream;

  cudaLaunchKernelEx(&combine_config, combine_kernel, reinterpret_cast<__nv_bfloat16 *>(y_ptr),
                     reinterpret_cast<const float *>(splitk_out_ptr),
                     reinterpret_cast<const float *>(lse_ptr), task_map_ptr, num_total_ctas,
                     num_batch, num_seq_q, num_head_q, num_head_k, pad_heads_per_group, num_dim_v,
                     hpg_divider);
}

bool smallm_fp8_qpertoken_perhead_kvpertensor_dim128_dynamic_async(
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

  constexpr int kTileN = 64;
  constexpr int kTileK = 128;
  constexpr int kTileV = 128;

  if (num_dim_qk != kTileK || num_dim_v != kTileV || (block_size != 32 && block_size != 64) ||
      (consumers != 1 && consumers != 2)) {
    std::cout << "launch launch_attention_decode_bf16_dim128_smallm_fp8 failed with "
              << "  num_dim_qk: " << num_dim_qk << ", num_dim_v: " << num_dim_v
              << ", block_size:" << block_size << std::endl;
    return false;
  }

  int heads_per_group = num_head_q / num_head_k;
  if (heads_per_group != 8 && heads_per_group != 4) {
    std::cout << "launch launch_attention_decode_bf16_dim128_smallm_fp8 failed with "
              << " heads_per_group:" << heads_per_group << ", num_head_q:" << num_head_q
              << ", num_head_k:" << num_head_k << std::endl;
    return false;
  }

  auto launch = [&](auto splitk_tag, auto tilem_tag, auto block_size_tag) {
    constexpr int kTileM = decltype(tilem_tag)::value;
    constexpr int kBlockSize = decltype(block_size_tag)::value;
    constexpr int kSplitK = decltype(splitk_tag)::value;
    launch_smallm_fp8_qpertoken_perhead_kvpertensor_dim128_dynamic_splitk_kernel<
        kTileM, kTileN, kTileK, kTileV, kBlockSize, kSplitK>(
        y_ptr, splitk_out_ptr, lse_ptr, task_map_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
        qscale_ptr, kscale_ptr, vscale_ptr, num_batch, num_seq_q, num_head_q, num_head_k,
        num_head_v, heads_per_group, num_dim_qk, num_dim_v, num_kvcache_blocks, num_seq_max_blocks,
        qscale_pad_stride, ldQ, kcache_block_stride, kcache_token_stride, kcache_head_stride,
        vcache_block_stride, vcache_token_stride, vcache_head_stride, stream);
  };
  auto dispatch_block_size = [&](auto splitk_tag, auto tilem_tag) {
    if (block_size == 32) {
      launch(splitk_tag, tilem_tag, std::integral_constant<int, 32>{});
    } else if (block_size == 64) {
      launch(splitk_tag, tilem_tag, std::integral_constant<int, 64>{});
    }
  };

  auto dispatch_mtp = [&](auto splitk_tag) {
    if (num_seq_q == 1) {
      dispatch_block_size(splitk_tag, std::integral_constant<int, 8>{});
    } else if (num_seq_q == 2) {
      dispatch_block_size(splitk_tag, std::integral_constant<int, 16>{});
    } else if (num_seq_q == 3) {
      dispatch_block_size(splitk_tag, std::integral_constant<int, 24>{});
    } else if (num_seq_q == 4) {
      dispatch_block_size(splitk_tag, std::integral_constant<int, 32>{});
    }
  };

  if (splitk == 78 * 4) {
    dispatch_mtp(std::integral_constant<int, 78 * 4>{});
  } else if (splitk == 78 * 3) {
    dispatch_mtp(std::integral_constant<int, 78 * 3>{});
  } else if (splitk == 78 * 2) {
    dispatch_mtp(std::integral_constant<int, 78 * 2>{});
  } else {
    dispatch_mtp(std::integral_constant<int, 64>{});
  }

  return true;
}

}  // namespace decode
}  // namespace attention
}  // namespace hpc
