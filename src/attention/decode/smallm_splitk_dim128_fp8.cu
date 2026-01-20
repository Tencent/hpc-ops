// Copyright (C) 2026 Tencent.

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "cute/tensor.hpp"
#include "src/attention/decode/smallm_dim128.h"
#include "src/attention/decode/smallm_splitk_combine_kernels.cuh"
#include "src/attention/decode/smallm_splitk_fp8_kernels.cuh"

namespace hpc {
namespace attention {
namespace decode {

template <int kHeadsPerGroup, int kCopyBoxN, int kTileM, int kTileN, int kTileK, int kTileV,
          int kBlockSize, int kSplitK, int kSplitMinLen, int kConsumers>
void launch_attention_decode_fp8_dim128_smallm_splitk(
    void *y_ptr, void *lse_ptr, void *splitk_out_ptr, const void *q_ptr, void *kcache_ptr,
    void *vcache_ptr, const int *block_ids_ptr, const int *num_seq_kvcache_ptr,
    const float *qscale_ptr, const float *kscale_ptr, const float *vscale_ptr, int *split_flag_ptr,
    bool new_kv_included, int num_batch, int num_head_q, int num_head_k, int num_head_v,
    int heads_per_group, int num_dim_qk, int num_dim_v, int num_kvcache_blocks, int block_size,
    int num_seq_max_blocks, int qscale_pad_stride, int ldY, int ldQ, int ldK, int ldV,
    cudaStream_t stream) {
  using namespace cute;  // NOLINT
  constexpr int kStage = 4;

  static_assert(kStage % kConsumers == 0);

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  auto Q = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(q_ptr)),
                       make_shape(num_head_q, num_dim_qk, num_batch),
                       make_stride(num_dim_qk, Int<1>{}, ldQ));

  auto K = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(kcache_ptr)),
                       make_shape(kBlockSize, num_dim_qk, num_head_k, num_kvcache_blocks),
                       make_stride(num_dim_qk * num_head_k, Int<1>{}, num_dim_qk, ldK));

  auto V = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(vcache_ptr)),
                       make_shape(num_dim_v, kBlockSize, num_head_v, num_kvcache_blocks),
                       make_stride(Int<1>{}, num_head_v * num_dim_v, num_dim_v, ldV));

  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)),
                       make_shape(num_dim_v, num_head_q, num_batch),
                       make_stride(Int<1>{}, num_dim_v, ldY));

  auto splitY = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(splitk_out_ptr)),
                            make_shape(num_dim_v, num_head_q, kSplitK * kConsumers, num_batch),
                            make_stride(Int<1>{}, num_dim_v, num_dim_v * num_head_q,
                                        num_dim_v * num_head_q * kSplitK * kConsumers));

  auto slayout_q =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileN>{}, Int<kTileK>{}));
  auto slayout_k = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{}));

  auto slayout_p = tile_to_shape(GMMA::Layout_MN_SW64_Atom<Tin>{},
                                 make_shape(Int<kTileM>{}, Int<kTileN>{}, Int<kConsumers>{}));

  auto slayout_s = tile_to_shape(GMMA::Layout_K_SW64_Atom<Tin>{},
                                 make_shape(Int<kTileN>{}, Int<kTileM>{}, Int<kConsumers>{}));

  auto slayout_vtma = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{},
                                    make_shape(Int<kTileV>{}, Int<kTileM>{}, Int<kStage>{}));

  auto slayout_y =
      tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tout>{}, make_shape(Int<kTileV>{}, Int<kTileN>{}));

  auto slayout_splity = tile_to_shape(GMMA::Layout_MN_SW128_Atom<float>{},
                                      make_shape(Int<kTileV>{}, Int<kTileN>{}, Int<kConsumers>{}));

  auto tma_copy_layout_q =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kCopyBoxN>{}, Int<kTileK>{}));
  auto tma_copy_layout_k =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kBlockSize>{}, Int<kTileK>{}));
  auto tma_copy_layout_v = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{},
                                         make_shape(Int<kTileV>{}, Int<kBlockSize>{}));
  auto tma_copy_layout_y = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tout>{},
                                         make_shape(Int<kTileV>{}, Int<kCopyBoxN>{}));
  auto tma_copy_layout_splity = tile_to_shape(GMMA::Layout_MN_SW128_Atom<float>{},
                                              make_shape(Int<kTileV>{}, Int<kCopyBoxN>{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, tma_copy_layout_q);
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K, tma_copy_layout_k);
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, V, tma_copy_layout_v);
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, tma_copy_layout_y);
  auto tma_splity = make_tma_copy(SM90_TMA_STORE{}, splitY, tma_copy_layout_splity);

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, float>;
  auto r2s_tiled_copy =
      make_tiled_copy(R2SCopyAtomC{},
                      make_layout(make_shape(make_shape(Int<8>{}, Int<4>{}), Int<4>{}),
                                  make_stride(make_stride(Int<4>{}, Int<32>{}), Int<1>{})),
                      make_layout(make_shape(make_shape(Int<2>{}, Int<1>{}), Int<2>{}),
                                  make_stride(make_stride(Int<2>{}, Int<0>{}), Int<1>{})));

  using R2SCopyAtomP = Copy_Atom<cute::SM90_U16x2_STSM_T, Tin>;
  auto tiled_copy_P_r2s = make_tiled_copy(
      R2SCopyAtomP{}, Layout<Shape<Shape<_8, _4>, _4>, Stride<Stride<_4, _32>, _1>>{},
      Layout<Shape<Shape<_2, _1>, _2>, Stride<Stride<_1, _0>, _2>>{});

  using TiledMmaQK = decltype(make_tiled_mma(SM90_64x8x32_F32E4M3E4M3_SS_TN<>{}));
  using TiledMmaSV = decltype(make_tiled_mma(SM90_64x8x32_F32E4M3E4M3_RS_TN<>{}));

  dim3 block(size(TiledMmaQK{}) * kConsumers + 32);
  // dim3 grid(kSplitK, num_head_k, num_batch);
  dim3 grid(num_head_k, num_batch, kSplitK);

  constexpr int kWarpsPerWrapGroup = 4;
  int shm_qkv = (cosize(slayout_q) + cosize(slayout_k) + cosize(slayout_vtma) + cosize(slayout_p)) *
                    sizeof(Tin) +
                sizeof(float) * kTileN * kWarpsPerWrapGroup * kConsumers;
  int shm_blk_ids = sizeof(int) * num_seq_max_blocks;
  int shm_y = std::max(cosize(slayout_y) * sizeof(Tout), cosize(slayout_splity) * sizeof(float));
  int shm_size = std::max(shm_qkv + shm_blk_ids, shm_y);

  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;

  auto kernel = kernels::attention_decode_fp8_multistage_ws_smallm_splitk_kernel<
      Tout, Tin, kTileM, kTileN, kTileK, kTileV, TiledMmaQK, TiledMmaSV, decltype(tma_q),
      decltype(tma_k), decltype(tma_v), decltype(tma_y), decltype(tma_splity),
      decltype(tiled_copy_P_r2s), decltype(r2s_tiled_copy), decltype(slayout_q),
      decltype(slayout_k), decltype(slayout_p), decltype(slayout_s), decltype(slayout_vtma),
      decltype(slayout_y), decltype(slayout_splity), kHeadsPerGroup, kBlockSize, kStage, kConsumers,
      kSplitK, kSplitMinLen>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  kernel<<<grid, block, shm_size, stream>>>(
      tma_q, tma_k, tma_v, tma_y, tma_splity, reinterpret_cast<Tout *>(y_ptr),
      reinterpret_cast<float *>(splitk_out_ptr), reinterpret_cast<float *>(lse_ptr), block_ids_ptr,
      num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr, new_kv_included,
      num_batch, num_dim_qk, num_dim_v, num_head_q, num_head_k, num_head_v, heads_per_group,
      num_kvcache_blocks, num_seq_max_blocks, qscale_pad_stride, one_over_dk_log2e);
}

bool smallm_splitk_dim128_fp8_async(
    void *y_ptr, void *lse_ptr, void *splitk_out_ptr, const void *q_ptr, void *kcache_ptr,
    void *vcache_ptr, const int *block_ids_ptr, const int *num_seq_kvcache_ptr,
    const float *qscale_ptr, const float *kscale_ptr, const float *vscale_ptr, int *split_flag_ptr,
    bool new_kv_included, int splitk, int splitk_min_len, int consumers, int num_batch,
    int num_head_q, int num_head_k, int num_head_v, int num_dim_qk, int num_dim_v,
    int num_kvcache_blocks, int block_size, int num_seq_max_blocks, int qscale_pad_stride, int ldY,
    int ldQ, int ldK, int ldV, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  constexpr int kTileM = 64;
  constexpr int kTileN = 8;
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
  if (heads_per_group != 8 && (heads_per_group != 4 || (heads_per_group == 4 && num_head_k != 1))) {
    std::cout << "launch launch_attention_decode_bf16_dim128_smallm_fp8 failed with "
              << " heads_per_group:" << heads_per_group << ", num_head_q:" << num_head_q
              << ", num_head_k:" << num_head_k << std::endl;
    return false;
  }

  if (!(splitk == 4 && splitk_min_len == 4096) && !(splitk == 4 && splitk_min_len == 512)) {
    std::cout << "launch launch_attention_decode_bf16_dim128_smallm_fp8 failed with "
              << " splitk:" << splitk << ", splitk_min_len:" << splitk_min_len << std::endl;
    return false;
  }

  if (consumers == 2) {
    constexpr int kConsumers = 2;
    if (splitk == 4) {
      constexpr int kSplitK = 4;
      if (splitk_min_len == 4096) {
        constexpr int kSplitMinLen = 4096;
        if (heads_per_group == 8) {
          constexpr int kHeadsPerGroup = 8;
          constexpr int kCopyBoxN = 8;
          if (block_size == 32) {
            constexpr int kBlockSize = 32;
            launch_attention_decode_fp8_dim128_smallm_splitk<kHeadsPerGroup, kCopyBoxN, kTileM,
                                                             kTileN, kTileK, kTileV, kBlockSize,
                                                             kSplitK, kSplitMinLen, kConsumers>(
                y_ptr, lse_ptr, splitk_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
                num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
                new_kv_included, num_batch, num_head_q, num_head_k, num_head_v, heads_per_group,
                num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
                qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
          } else if (block_size == 64) {
            constexpr int kBlockSize = 64;
            launch_attention_decode_fp8_dim128_smallm_splitk<kHeadsPerGroup, kCopyBoxN, kTileM,
                                                             kTileN, kTileK, kTileV, kBlockSize,
                                                             kSplitK, kSplitMinLen, kConsumers>(
                y_ptr, lse_ptr, splitk_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
                num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
                new_kv_included, num_batch, num_head_q, num_head_k, num_head_v, heads_per_group,
                num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
                qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
          }
        } else if (heads_per_group == 4) {
          constexpr int kHeadsPerGroup = 4;
          constexpr int kCopyBoxN = 8;
          if (block_size == 32) {
            constexpr int kBlockSize = 32;
            launch_attention_decode_fp8_dim128_smallm_splitk<kHeadsPerGroup, kCopyBoxN, kTileM,
                                                             kTileN, kTileK, kTileV, kBlockSize,
                                                             kSplitK, kSplitMinLen, kConsumers>(
                y_ptr, lse_ptr, splitk_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
                num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
                new_kv_included, num_batch, num_head_q, num_head_k, num_head_v, heads_per_group,
                num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
                qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
          } else if (block_size == 64) {
            constexpr int kBlockSize = 64;
            launch_attention_decode_fp8_dim128_smallm_splitk<kHeadsPerGroup, kCopyBoxN, kTileM,
                                                             kTileN, kTileK, kTileV, kBlockSize,
                                                             kSplitK, kSplitMinLen, kConsumers>(
                y_ptr, lse_ptr, splitk_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
                num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
                new_kv_included, num_batch, num_head_q, num_head_k, num_head_v, heads_per_group,
                num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
                qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
          }
        }
      } else if (splitk_min_len == 512) {
        constexpr int kSplitMinLen = 512;
        if (heads_per_group == 8) {
          constexpr int kHeadsPerGroup = 8;
          constexpr int kCopyBoxN = 8;
          if (block_size == 32) {
            constexpr int kBlockSize = 32;
            launch_attention_decode_fp8_dim128_smallm_splitk<kHeadsPerGroup, kCopyBoxN, kTileM,
                                                             kTileN, kTileK, kTileV, kBlockSize,
                                                             kSplitK, kSplitMinLen, kConsumers>(
                y_ptr, lse_ptr, splitk_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
                num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
                new_kv_included, num_batch, num_head_q, num_head_k, num_head_v, heads_per_group,
                num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
                qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
          } else if (block_size == 64) {
            constexpr int kBlockSize = 64;
            launch_attention_decode_fp8_dim128_smallm_splitk<kHeadsPerGroup, kCopyBoxN, kTileM,
                                                             kTileN, kTileK, kTileV, kBlockSize,
                                                             kSplitK, kSplitMinLen, kConsumers>(
                y_ptr, lse_ptr, splitk_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
                num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
                new_kv_included, num_batch, num_head_q, num_head_k, num_head_v, heads_per_group,
                num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
                qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
          }
        } else if (heads_per_group == 4) {
          constexpr int kHeadsPerGroup = 4;
          constexpr int kCopyBoxN = 8;
          if (block_size == 32) {
            constexpr int kBlockSize = 32;
            launch_attention_decode_fp8_dim128_smallm_splitk<kHeadsPerGroup, kCopyBoxN, kTileM,
                                                             kTileN, kTileK, kTileV, kBlockSize,
                                                             kSplitK, kSplitMinLen, kConsumers>(
                y_ptr, lse_ptr, splitk_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
                num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
                new_kv_included, num_batch, num_head_q, num_head_k, num_head_v, heads_per_group,
                num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
                qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
          } else if (block_size == 64) {
            constexpr int kBlockSize = 64;
            launch_attention_decode_fp8_dim128_smallm_splitk<kHeadsPerGroup, kCopyBoxN, kTileM,
                                                             kTileN, kTileK, kTileV, kBlockSize,
                                                             kSplitK, kSplitMinLen, kConsumers>(
                y_ptr, lse_ptr, splitk_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
                num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
                new_kv_included, num_batch, num_head_q, num_head_k, num_head_v, heads_per_group,
                num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
                qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
          }
        }
      }
    }
  } else if (consumers == 1) {
    constexpr int kConsumers = 1;
    if (splitk == 4) {
      constexpr int kSplitK = 4;
      if (splitk_min_len == 4096) {
        constexpr int kSplitMinLen = 4096;
        if (heads_per_group == 8) {
          constexpr int kHeadsPerGroup = 8;
          constexpr int kCopyBoxN = 8;
          if (block_size == 32) {
            constexpr int kBlockSize = 32;
            launch_attention_decode_fp8_dim128_smallm_splitk<kHeadsPerGroup, kCopyBoxN, kTileM,
                                                             kTileN, kTileK, kTileV, kBlockSize,
                                                             kSplitK, kSplitMinLen, kConsumers>(
                y_ptr, lse_ptr, splitk_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
                num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
                new_kv_included, num_batch, num_head_q, num_head_k, num_head_v, heads_per_group,
                num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
                qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
          } else if (block_size == 64) {
            constexpr int kBlockSize = 64;
            launch_attention_decode_fp8_dim128_smallm_splitk<kHeadsPerGroup, kCopyBoxN, kTileM,
                                                             kTileN, kTileK, kTileV, kBlockSize,
                                                             kSplitK, kSplitMinLen, kConsumers>(
                y_ptr, lse_ptr, splitk_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
                num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
                new_kv_included, num_batch, num_head_q, num_head_k, num_head_v, heads_per_group,
                num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
                qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
          }
        } else if (heads_per_group == 4) {
          constexpr int kHeadsPerGroup = 4;
          constexpr int kCopyBoxN = 8;
          if (block_size == 32) {
            constexpr int kBlockSize = 32;
            launch_attention_decode_fp8_dim128_smallm_splitk<kHeadsPerGroup, kCopyBoxN, kTileM,
                                                             kTileN, kTileK, kTileV, kBlockSize,
                                                             kSplitK, kSplitMinLen, kConsumers>(
                y_ptr, lse_ptr, splitk_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
                num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
                new_kv_included, num_batch, num_head_q, num_head_k, num_head_v, heads_per_group,
                num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
                qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
          } else if (block_size == 64) {
            constexpr int kBlockSize = 64;
            launch_attention_decode_fp8_dim128_smallm_splitk<kHeadsPerGroup, kCopyBoxN, kTileM,
                                                             kTileN, kTileK, kTileV, kBlockSize,
                                                             kSplitK, kSplitMinLen, kConsumers>(
                y_ptr, lse_ptr, splitk_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
                num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
                new_kv_included, num_batch, num_head_q, num_head_k, num_head_v, heads_per_group,
                num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
                qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
          }
        }
      } else if (splitk_min_len == 512) {
        constexpr int kSplitMinLen = 512;
        if (heads_per_group == 8) {
          constexpr int kHeadsPerGroup = 8;
          constexpr int kCopyBoxN = 8;
          if (block_size == 32) {
            constexpr int kBlockSize = 32;
            launch_attention_decode_fp8_dim128_smallm_splitk<kHeadsPerGroup, kCopyBoxN, kTileM,
                                                             kTileN, kTileK, kTileV, kBlockSize,
                                                             kSplitK, kSplitMinLen, kConsumers>(
                y_ptr, lse_ptr, splitk_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
                num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
                new_kv_included, num_batch, num_head_q, num_head_k, num_head_v, heads_per_group,
                num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
                qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
          } else if (block_size == 64) {
            constexpr int kBlockSize = 64;
            launch_attention_decode_fp8_dim128_smallm_splitk<kHeadsPerGroup, kCopyBoxN, kTileM,
                                                             kTileN, kTileK, kTileV, kBlockSize,
                                                             kSplitK, kSplitMinLen, kConsumers>(
                y_ptr, lse_ptr, splitk_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
                num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
                new_kv_included, num_batch, num_head_q, num_head_k, num_head_v, heads_per_group,
                num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
                qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
          }
        } else if (heads_per_group == 4) {
          constexpr int kHeadsPerGroup = 4;
          constexpr int kCopyBoxN = 8;
          if (block_size == 32) {
            constexpr int kBlockSize = 32;
            launch_attention_decode_fp8_dim128_smallm_splitk<kHeadsPerGroup, kCopyBoxN, kTileM,
                                                             kTileN, kTileK, kTileV, kBlockSize,
                                                             kSplitK, kSplitMinLen, kConsumers>(
                y_ptr, lse_ptr, splitk_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
                num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
                new_kv_included, num_batch, num_head_q, num_head_k, num_head_v, heads_per_group,
                num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
                qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
          } else if (block_size == 64) {
            constexpr int kBlockSize = 64;
            launch_attention_decode_fp8_dim128_smallm_splitk<kHeadsPerGroup, kCopyBoxN, kTileM,
                                                             kTileN, kTileK, kTileV, kBlockSize,
                                                             kSplitK, kSplitMinLen, kConsumers>(
                y_ptr, lse_ptr, splitk_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
                num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
                new_kv_included, num_batch, num_head_q, num_head_k, num_head_v, heads_per_group,
                num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
                qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
          }
        }
      }
    }
  }

  return true;
}

}  // namespace decode
}  // namespace attention
}  // namespace hpc
