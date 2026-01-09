// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/indexer/kernels.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace indexer {

template <int kRatio>
void launch_warp_spec_mqa_indexer_logits_fp8(
    void* y_ptr, const void* q_ptr, const void* w_ptr, const void* kvcache_ptr,
    const void* cu_seqlens_q_ptr, const void* seqlens_kv_ptr, const void* block_ids_ptr,
    const int& num_batch, const int& total_seq_q, const int& num_head_q, const int& head_dim,
    const int& num_kvcache_blocks, const int& num_seq_max_blocks, const int& max_context_len,
    const int& num_split, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = float;

  constexpr int kTileM = 128;
  constexpr int kTileN = 64;
  constexpr int kTileK = 128;
  constexpr int kBlockSize = 64;
  constexpr int kStageK = 2;

  auto Q = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(q_ptr)),
                       make_shape(num_head_q, head_dim, total_seq_q),
                       make_stride(head_dim, Int<1>{}, head_dim * num_head_q));

  auto W = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(w_ptr)),
                       make_shape(num_head_q, total_seq_q), make_stride(Int<1>{}, num_head_q));

  auto K = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(kvcache_ptr)),
                       make_shape(kBlockSize, head_dim, num_kvcache_blocks),
                       make_stride(head_dim, Int<1>{}, head_dim * kBlockSize));

  auto Y =
      make_tensor(make_gmem_ptr(reinterpret_cast<const Tout*>(y_ptr)),
                  make_shape(max_context_len, total_seq_q), make_stride(Int<1>{}, max_context_len));

  auto slayout_q =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileN>{}, Int<kTileK>{}));

  auto slayout_k = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStageK>{}));

  auto tma_copy_layout_k =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kBlockSize>{}, Int<kTileK>{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, slayout_q);
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K, tma_copy_layout_k);

  using TiledMmaAtom = SM90_64x64x32_F32E4M3E4M3_SS_TN<>;
  using WarpgroupLayout = decltype(make_layout(make_shape(Int<2>{}, Int<1>{}, Int<1>{})));
  using TiledMma = decltype(make_tiled_mma(TiledMmaAtom{}, WarpgroupLayout{}));

  auto shm_size =
      (cosize(slayout_q) + cosize(slayout_k)) * sizeof(Tin) + kTileN * sizeof(cute::bfloat16_t);

  dim3 block(384);
  dim3 grid(get_sm_count());

  cutlass::FastDivmod num_split_divider(num_split);

  auto kernel =
      kernels::mqa_indexer_logits_fp8_kernel<Tin, Tout, kTileM, kTileN, kTileK, kBlockSize, kStageK,
                                             kRatio, TiledMma, decltype(tma_q), decltype(tma_k),
                                             decltype(slayout_q), decltype(slayout_k)>;

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
  kernel<<<grid, block, shm_size, stream>>>(
      tma_q, tma_k, reinterpret_cast<const cute::bfloat16_t*>(w_ptr),
      reinterpret_cast<Tout*>(y_ptr), reinterpret_cast<const int*>(cu_seqlens_q_ptr),
      reinterpret_cast<const int*>(seqlens_kv_ptr), reinterpret_cast<const int*>(block_ids_ptr),
      num_batch, total_seq_q, num_head_q, head_dim, num_kvcache_blocks, num_seq_max_blocks,
      max_context_len, num_split_divider);
}

void mqa_indexer_logits_fp8_async(void* y_ptr, const void* q_ptr, const void* kvcache_ptr,
                                  const void* w_ptr, const void* cu_seqlens_q_ptr,
                                  const void* seqlens_kv_ptr, const void* block_ids_ptr,
                                  const int& num_batch, const int& total_seq_q,
                                  const int& num_head_q, const int& head_dim,
                                  const int& num_kvcache_blocks, const int& block_size,
                                  const int& num_seq_max_blocks, const int& max_context_len,
                                  const int& ratio, const int& num_split, cudaStream_t stream) {
  constexpr int kRatio = 4;
  launch_warp_spec_mqa_indexer_logits_fp8<kRatio>(
      y_ptr, q_ptr, w_ptr, kvcache_ptr, cu_seqlens_q_ptr, seqlens_kv_ptr, block_ids_ptr, num_batch,
      total_seq_q, num_head_q, head_dim, num_kvcache_blocks, num_seq_max_blocks, max_context_len,
      num_split, stream);
}

}  // namespace indexer
}  // namespace hpc
