// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "cute/tensor.hpp"
#include "src/attention/attention.h"
#include "src/attention/attention_decode.cuh"

namespace hpc {
namespace attention {

template <int kHeadsPerGroup, int kTileM, int kTileN, int kTileK, int kTileV, int kBlockSize>
void launch_attention_decode_bf16_dim128(void *y_ptr, const void *q_ptr, void *kvcache_ptr,
                                         const int *block_ids_ptr, const int *cache_lens_ptr,
                                         int num_batch, int num_seq_q, int num_head_q,
                                         int num_head_kv, int heads_per_group, int num_dim_qk,
                                         int num_dim_v, int num_blocks, int block_size,
                                         int max_num_blocks, int ldY, int ldQ,
                                         cudaStream_t stream) {
  using namespace cute;  // NOLINT
  constexpr int kStage = 2;

  using Tin = cute::bfloat16_t;
  using Tout = cute::bfloat16_t;

  auto Q = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(q_ptr)),
                       make_shape(num_head_q, num_dim_qk, num_seq_q, num_batch),
                       make_stride(num_dim_qk, Int<1>{}, ldQ, num_seq_q * ldQ));

  auto K = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(kvcache_ptr)),
                       make_shape(kBlockSize, num_dim_qk, num_head_kv, num_blocks),
                       make_stride(num_dim_qk * num_head_kv, Int<1>{}, num_dim_qk,
                                   num_dim_qk * num_head_kv * kBlockSize * 2));

  auto V = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(kvcache_ptr) +
                                     kBlockSize * num_dim_qk * num_head_kv),
                       make_shape(num_dim_v, kBlockSize, num_head_kv, num_blocks),
                       make_stride(Int<1>{}, num_head_kv * num_dim_v, num_dim_v,
                                   num_dim_v * num_head_kv * kBlockSize * 2));

  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<const Tout *>(y_ptr)),
                       make_shape(num_head_q, num_dim_v, num_seq_q, num_batch),
                       make_stride(num_dim_v, Int<1>{}, ldY, num_seq_q * ldY));

  auto slayout_q =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto slayout_k = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_v = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileV>{}, Int<kTileN>{}, Int<kStage>{}));
  auto slayout_y =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tout>{}, make_shape(Int<kTileM>{}, Int<kTileV>{}));

  auto tma_copy_layout_q = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                         make_shape(Int<kHeadsPerGroup>{}, Int<kTileK>{}));
  auto tma_copy_layout_k =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kBlockSize>{}, Int<kTileK>{}));
  auto tma_copy_layout_v = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{},
                                         make_shape(Int<kTileV>{}, Int<kBlockSize>{}));
  auto tma_copy_layout_y = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                         make_shape(Int<kHeadsPerGroup>{}, Int<kTileK>{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, tma_copy_layout_q);
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K, tma_copy_layout_k);
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, V, tma_copy_layout_v);
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, tma_copy_layout_y);

  using TiledMmaQK =
      decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
  using TiledMmaPV =
      decltype(make_tiled_mma(SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));

  dim3 block(size(TiledMmaQK{}) + 32);
  dim3 grid(num_head_kv, num_batch);

  int shm_qkv = (cosize(slayout_q) + cosize(slayout_k) + cosize(slayout_v)) * sizeof(Tin);
  int shm_blk_ids = sizeof(int) * max_num_blocks;
  int shm_y = cosize(slayout_y) * sizeof(Tout);
  int shm_size = std::max(shm_qkv + shm_blk_ids, shm_y);

  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;

  auto kernel = kernels::attention_decode_bf16_multistage_ws_kernel<
      Tout, Tin, kTileM, kTileN, kTileK, kTileV, TiledMmaQK, TiledMmaPV, decltype(tma_q),
      decltype(tma_k), decltype(tma_v), decltype(tma_y), decltype(slayout_q), decltype(slayout_k),
      decltype(slayout_v), decltype(slayout_y), kBlockSize, kStage>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  kernel<<<grid, block, shm_size, stream>>>(tma_q, tma_k, tma_v, tma_y, block_ids_ptr,
                                            cache_lens_ptr, num_batch, num_seq_q, num_dim_qk,
                                            num_dim_v, num_head_q, num_head_kv, heads_per_group,
                                            num_blocks, max_num_blocks, one_over_dk_log2e);
}

bool attention_decode_bf16_headdim128_async(void *y_ptr, const void *q_ptr, void *kvcache_ptr,
                                            const int *block_ids_ptr, const int *cache_lens_ptr,
                                            int num_batch, int num_seq_q, int num_head_q,
                                            int num_head_kv, int heads_per_group, int num_dim_qk,
                                            int num_dim_v, int num_blocks, int block_size,
                                            int max_num_blocks, int ldY, int ldQ,
                                            cudaStream_t stream) {
  using namespace cute;  // NOLINT

  constexpr int kTileM = 64;
  constexpr int kTileN = 32;
  constexpr int kTileK = 128;
  constexpr int kTileV = 128;
  constexpr int kBlockSize = 32;

  if (num_dim_qk != kTileK || num_dim_v != kTileV || kBlockSize != block_size) {
    std::cout << "launch attention_decode_bf16_headdim128 failed with "
              << "  num_dim_qk: " << num_dim_qk << ", num_dim_v: " << num_dim_v
              << ", block_size:" << block_size << std::endl;
    return false;
  }

  if (heads_per_group == 8 || heads_per_group == 4) {
    constexpr int kHeadsPerGroup = 8;
    launch_attention_decode_bf16_dim128<kHeadsPerGroup, kTileM, kTileN, kTileK, kTileV, kBlockSize>(
        y_ptr, q_ptr, kvcache_ptr, block_ids_ptr, cache_lens_ptr, num_batch, num_seq_q, num_head_q,
        num_head_kv, heads_per_group, num_dim_qk, num_dim_v, num_blocks, block_size, max_num_blocks,
        ldY, ldQ, stream);
  }

  return true;
}
}  // namespace attention
}  // namespace hpc
