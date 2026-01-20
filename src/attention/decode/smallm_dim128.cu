// Copyright (C) 2026 Tencent.

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "cute/tensor.hpp"
#include "src/attention/decode/smallm_dim128.h"
#include "src/attention/decode/smallm_kernels.cuh"

namespace hpc {
namespace attention {
namespace decode {

template <int kHeadsPerGroup, int kTileM, int kTileN, int kTileK, int kTileV, int kBlockSize>
void launch_attention_decode_bf16_dim128_smallm(
    void *y_ptr, const void *q_ptr, void *kcache_ptr, void *vcache_ptr, const int *block_ids_ptr,
    const int *num_seq_kvcache_ptr, bool new_kv_included, int num_batch, int num_head_q,
    int num_head_k, int num_head_v, int heads_per_group, int num_dim_qk, int num_dim_v,
    int num_kvcache_blocks, int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK,
    int ldV, cudaStream_t stream) {
  using namespace cute;  // NOLINT
  constexpr int kStage = 2;

  using Tin = cute::bfloat16_t;
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

  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<const Tout *>(y_ptr)),
                       make_shape(num_dim_v, num_head_q, num_batch),
                       make_stride(Int<1>{}, num_dim_v, ldY));

  auto slayout_q =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileN>{}, Int<kTileK>{}));
  auto slayout_k = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{}));

  auto slayout_p =
      tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileN>{}));

  auto slayout_s =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileN>{}, Int<kTileM>{}));

  auto slayout_v = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileV>{}, Int<kTileM>{}, Int<kStage>{}));
  auto slayout_y =
      tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tout>{}, make_shape(Int<kTileV>{}, Int<kTileN>{}));

  auto tma_copy_layout_q = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                         make_shape(Int<kHeadsPerGroup>{}, Int<kTileK>{}));
  auto tma_copy_layout_k =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kBlockSize>{}, Int<kTileK>{}));
  auto tma_copy_layout_v = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{},
                                         make_shape(Int<kTileV>{}, Int<kBlockSize>{}));
  auto tma_copy_layout_y = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{},
                                         make_shape(Int<kTileV>{}, Int<kHeadsPerGroup>{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, tma_copy_layout_q);
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K, tma_copy_layout_k);
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, V, tma_copy_layout_v);
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, tma_copy_layout_y);

  using TiledMmaQK =
      decltype(make_tiled_mma(SM90_64x8x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
  using TiledMmaSV =
      decltype(make_tiled_mma(SM90_64x8x16_F32BF16BF16_SS<GMMA::Major::MN, GMMA::Major::K>{}));

  dim3 block(size(TiledMmaQK{}) + 32);
  dim3 grid(num_head_k, num_batch);

  constexpr int kWarpsPerWrapGroup = 4;
  int shm_qkv = (cosize(slayout_q) + cosize(slayout_k) + cosize(slayout_p) + cosize(slayout_v)) *
                    sizeof(Tin) +
                sizeof(float) * kTileN * kWarpsPerWrapGroup;
  int shm_blk_ids = sizeof(int) * num_seq_max_blocks;
  int shm_y = cosize(slayout_y) * sizeof(Tout);
  int shm_size = std::max(shm_qkv + shm_blk_ids, shm_y);

  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;

  auto kernel = kernels::attention_decode_bf16_multistage_ws_smallm_kernel<
      Tout, Tin, kTileM, kTileN, kTileK, kTileV, TiledMmaQK, TiledMmaSV, decltype(tma_q),
      decltype(tma_k), decltype(tma_v), decltype(tma_y), decltype(slayout_q), decltype(slayout_k),
      decltype(slayout_p), decltype(slayout_s), decltype(slayout_v), decltype(slayout_y),
      kBlockSize, kStage>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  kernel<<<grid, block, shm_size, stream>>>(
      tma_q, tma_k, tma_v, tma_y, block_ids_ptr, num_seq_kvcache_ptr, new_kv_included, num_batch,
      num_dim_qk, num_dim_v, num_head_q, num_head_k, num_head_v, heads_per_group,
      num_kvcache_blocks, num_seq_max_blocks, one_over_dk_log2e);
}

bool smallm_dim128_async(void *y_ptr, const void *q_ptr, void *kcache_ptr, void *vcache_ptr,
                         const int *block_ids_ptr, const int *num_seq_kvcache_ptr,
                         bool new_kv_included, int num_batch, int num_head_q, int num_head_k,
                         int num_head_v, int num_dim_qk, int num_dim_v, int num_kvcache_blocks,
                         int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK, int ldV,
                         cudaStream_t stream) {
  using namespace cute;  // NOLINT

  constexpr int kTileM = 64;
  constexpr int kTileN = 8;
  constexpr int kTileK = 128;
  constexpr int kTileV = 128;

  if (num_dim_qk != kTileK || num_dim_v != kTileV || (block_size != 32 && block_size != 64)) {
    std::cout << "launch launch_attention_decode_bf16_dim128_smallm failed with "
              << "  num_dim_qk: " << num_dim_qk << ", num_dim_v: " << num_dim_v
              << ", block_size:" << block_size << std::endl;
    return false;
  }

  int heads_per_group = num_head_q / num_head_k;
  if (heads_per_group == 8 || heads_per_group == 4) {
    constexpr int kHeadsPerGroup = 8;
    if (block_size == 32) {
      constexpr int kBlockSize = 32;
      launch_attention_decode_bf16_dim128_smallm<kHeadsPerGroup, kTileM, kTileN, kTileK, kTileV,
                                                 kBlockSize>(
          y_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr, num_seq_kvcache_ptr, new_kv_included,
          num_batch, num_head_q, num_head_k, num_head_v, heads_per_group, num_dim_qk, num_dim_v,
          num_kvcache_blocks, block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldV, stream);
    } else if (block_size == 64) {
      constexpr int kBlockSize = 64;
      launch_attention_decode_bf16_dim128_smallm<kHeadsPerGroup, kTileM, kTileN, kTileK, kTileV,
                                                 kBlockSize>(
          y_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr, num_seq_kvcache_ptr, new_kv_included,
          num_batch, num_head_q, num_head_k, num_head_v, heads_per_group, num_dim_qk, num_dim_v,
          num_kvcache_blocks, block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldV, stream);
    }
  }

  return true;
}

}  // namespace decode
}  // namespace attention
}  // namespace hpc
