// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "cute/tensor.hpp"
#include "src/attention/mla/smallm_sparse_mla_dim512.h"
#include "src/attention/mla/sparse_mla_kernels.cuh"

namespace hpc {
namespace attention {
namespace mla {

template <int kTileM, int kTileN, int kTileK, int kTileV, int kBlockSize>
void launch_attention_sparse_mla_with_kvcache_bf16_dim512_smallm(
    void* y_ptr, const void* q_ptr, void* win_kvcache_ptr, const int* win_block_ids_ptr,
    const int* win_topk_ids_ptr, void* compress_kvcache_ptr, const int* compress_block_ids_ptr,
    const int* compress_topk_ids_ptr, const int* cu_seqlens_q_ptr, const void* sink_weight_ptr,
    float softmax_scale, int num_batch, int total_seq_q, int num_head_q, int head_dim,
    int num_win_kvcache_blocks, int num_compress_kvcache_blocks, int num_win_seq_max_blocks,
    int num_compress_seq_max_blocks, int num_win_max_topk, int num_compress_max_topk, int ldY,
    int ldQ, int ldWinKV, int ldCompressKV, cudaStream_t stream) {
  using namespace cute;  // NOLINT
  constexpr int kStage = 2;

  using Tin = cute::bfloat16_t;
  using Tout = cute::bfloat16_t;

  auto Q = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(q_ptr)),
                       make_shape(num_head_q, head_dim, total_seq_q),
                       make_stride(head_dim, Int<1>{}, ldQ));

  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<const Tout*>(y_ptr)),
                       make_shape(head_dim, num_head_q, total_seq_q),
                       make_stride(Int<1>{}, head_dim, ldY));

  auto slayout_q =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto slayout_k = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_p =
      tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{}, make_shape(Int<kTileN>{}, Int<kTileM>{}));

  auto slayout_s =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileN>{}));

  auto slayout_v_one_stage =
      composition(slayout_k, Layout<Shape<Int<kTileK>, Int<kTileN>>, Stride<Int<kTileN>, _1>>{});

  auto slayout_v =
      tile_to_shape(slayout_v_one_stage, make_shape(Int<kTileV>{}, Int<kTileN>{}, Int<kStage>{}));

  auto slayout_y =
      tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tout>{}, make_shape(Int<kTileV>{}, Int<kTileM>{}));

  auto tma_copy_layout_q =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto tma_copy_layout_y =
      tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tin>{}, make_shape(Int<kTileV>{}, Int<kTileM>{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, tma_copy_layout_q);
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, tma_copy_layout_y);

  using WarpgroupLayout = decltype(make_layout(make_shape(Int<1>{}, Int<2>{}, Int<1>{})));

  using TiledMmaQK = decltype(make_tiled_mma(
      SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}, WarpgroupLayout{}));
  using TiledMmaSV = decltype(make_tiled_mma(
      SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::MN, GMMA::Major::K>{}, WarpgroupLayout{}));

  dim3 block(size(TiledMmaQK{}) + 128);
  dim3 grid(num_head_q / kTileM, total_seq_q);

  constexpr int kWarpsPerWrapGroup = 4;
  int shm_qkv = (cosize(slayout_q) + cosize(slayout_k) + cosize(slayout_p)) * sizeof(Tin) +
                sizeof(float) * kTileM * kWarpsPerWrapGroup;
  int shm_blk_ids = sizeof(int) * (num_win_seq_max_blocks + num_compress_seq_max_blocks);
  int shm_y = cosize(slayout_y) * sizeof(Tout);
  int shm_size = std::max(shm_qkv + shm_blk_ids, shm_y);

  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = softmax_scale * kLog2e;

  auto kernel = kernels::attention_sparse_mla_with_kvcache_bf16_multistage_ws_smallm_kernel<
      Tout, Tin, kTileM, kTileN, kTileK, kTileV, TiledMmaQK, TiledMmaSV, decltype(tma_q),
      decltype(tma_y), decltype(slayout_q), decltype(slayout_k), decltype(slayout_p),
      decltype(slayout_s), decltype(slayout_v), decltype(slayout_y), kBlockSize, kStage>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  kernel<<<grid, block, shm_size, stream>>>(
      tma_q, tma_y, win_kvcache_ptr, win_block_ids_ptr, win_topk_ids_ptr, compress_kvcache_ptr,
      compress_block_ids_ptr, compress_topk_ids_ptr, cu_seqlens_q_ptr,
      reinterpret_cast<const float*>(sink_weight_ptr), num_batch, total_seq_q, num_head_q, head_dim,
      num_win_kvcache_blocks, num_compress_kvcache_blocks, num_win_seq_max_blocks,
      num_compress_seq_max_blocks, num_win_max_topk, num_compress_max_topk, one_over_dk_log2e);
}

bool smallm_sparse_mla_dim512_async(void* y_ptr, const void* q_ptr, void* win_kvcache_ptr,
                                    const int* win_block_ids_ptr, const int* win_topk_ids_ptr,
                                    void* compress_kvcache_ptr, const int* compress_block_ids_ptr,
                                    const int* compress_topk_ids_ptr, const int* cu_seqlens_q_ptr,
                                    const void* sink_weight_ptr, float softmax_scale, int num_batch,
                                    int total_seq_q, int num_head_q, int head_dim,
                                    int num_win_kvcache_blocks, int num_compress_kvcache_blocks,
                                    int num_win_seq_max_blocks, int num_compress_seq_max_blocks,
                                    int block_size, int num_win_max_topk, int num_compress_max_topk,
                                    int ldY, int ldQ, int ldWinKV, int ldCompressKV,
                                    cudaStream_t stream) {
  using namespace cute;  // NOLINT

  constexpr int kTileM = 64;
  constexpr int kTileN = 64;
  constexpr int kTileK = 512;
  constexpr int kTileV = 512;

  if (head_dim != kTileV) {
    std::cout << "launch sparse_mla_with_kvcache_bf16 failed with head_dim: " << head_dim
              << std::endl;
    return false;
  }

  constexpr int kBlockSize = 64;
  if (block_size != kBlockSize) {
    std::cout << "launch sparse_mla_with_kvcache_bf16 failed with block size: " << block_size
              << std::endl;
    return false;
  }

  launch_attention_sparse_mla_with_kvcache_bf16_dim512_smallm<kTileM, kTileN, kTileK, kTileV,
                                                              kBlockSize>(
      y_ptr, q_ptr, win_kvcache_ptr, win_block_ids_ptr, win_topk_ids_ptr, compress_kvcache_ptr,
      compress_block_ids_ptr, compress_topk_ids_ptr, cu_seqlens_q_ptr, sink_weight_ptr,
      softmax_scale, num_batch, total_seq_q, num_head_q, head_dim, num_win_kvcache_blocks,
      num_compress_kvcache_blocks, num_win_seq_max_blocks, num_compress_seq_max_blocks,
      num_win_max_topk, num_compress_max_topk, ldY, ldQ, ldWinKV, ldCompressKV, stream);

  return true;
}

}  // namespace mla
}  // namespace attention
}  // namespace hpc
