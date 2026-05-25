// Copyright (C) 2026 Tencent.

#include <cuda.h>

#include <algorithm>
#include <type_traits>

#include "cute/tensor.hpp"
#include "src/stem/stem_kernels.cuh"
#include "src/stem/stem_oam_gemm_dim128.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace stem {

template <int kStemBlockSize, int kStride, int kDimQK>
void launch_stem_oam_gemm_dim128(void *block_logits_ptr, const void *qflat_ptr,
                                 const void *kflat_ptr, const void *vbias_ptr,
                                 const void *q_seq_lens_ptr, const void *kv_seq_lens_ptr,
                                 int num_batch, int num_head_q, int num_head_kv, int max_num_qb,
                                 int max_num_kb, bool causal, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tbf16 = cute::bfloat16_t;

  constexpr int kFlatDim = kStride * kDimQK;
  constexpr int kTileM = 128;
  constexpr int kTileN = 128;
  constexpr int kTileK = 64;
  constexpr int kStage = 4;

  // SMEM layouts: K-major SW128 for BF16
  auto slayout_q = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tbf16>{},
                                 make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_k = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tbf16>{},
                                 make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{}));

  auto cpbox_logits = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tbf16>{},
                                    make_shape(Int<kTileM / 2>{}, Int<kTileN>{}));
  auto slayout_logits =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tbf16>{}, make_shape(Int<kTileM>{}, Int<kTileN>{}));

  // SMEM size
  int shm_qk = sizeof(Tbf16) * (cosize(slayout_q) + cosize(slayout_k));
  int shm_logits = sizeof(Tbf16) * cosize(slayout_logits);
  int shm_size = shm_qk + shm_logits;

  // TMA Load: use REAL qflat/kflat shapes. TMA hardware zero-fills OOB reads.
  // Qflat: [B, Hq, max_num_qb, kFlatDim] — real shape, real stride
  auto gQflat = make_tensor(make_gmem_ptr(reinterpret_cast<const Tbf16 *>(qflat_ptr)),
                            make_shape(max_num_qb, kFlatDim, num_head_q, num_batch),
                            make_stride(static_cast<int64_t>(kFlatDim), Int<1>{},
                                        static_cast<int64_t>(max_num_qb) * kFlatDim,
                                        static_cast<int64_t>(num_head_q) * max_num_qb * kFlatDim));

  // Kflat: [B, Hkv, max_num_kb, kFlatDim] — real shape, real stride
  auto gKflat = make_tensor(make_gmem_ptr(reinterpret_cast<const Tbf16 *>(kflat_ptr)),
                            make_shape(max_num_kb, kFlatDim, num_head_kv, num_batch),
                            make_stride(static_cast<int64_t>(kFlatDim), Int<1>{},
                                        static_cast<int64_t>(max_num_kb) * kFlatDim,
                                        static_cast<int64_t>(num_head_kv) * max_num_kb * kFlatDim));

  // TMA Store: block_logits needs minimal padding for TMA copy box [64, 128].
  //   logits_qb: >= 64 (copy box M/2), multiple of 64
  //   logits_kb: >= 128 (copy box N), multiple of 64 (for 128B stride alignment)
  // entry.cc allocates block_logits with exactly these dimensions.
  int logits_qb = std::max(((max_num_qb + 63) / 64) * 64, 64);
  int logits_kb = std::max(((max_num_kb + 63) / 64) * 64, 128);

  auto gLogits = make_tensor(make_gmem_ptr(reinterpret_cast<Tbf16 *>(block_logits_ptr)),
                             make_shape(logits_qb, logits_kb, num_head_q, num_batch),
                             make_stride(static_cast<int64_t>(logits_kb), Int<1>{},
                                         static_cast<int64_t>(logits_qb) * logits_kb,
                                         static_cast<int64_t>(num_head_q) * logits_qb * logits_kb));

  // TMA descriptors
  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, gQflat, slayout_q(_, _, 0),
                             Tile<Int<kTileM>, Int<kTileK>>{}, Int<1>{});
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, gKflat, slayout_k(_, _, 0),
                             Tile<Int<kTileN>, Int<kTileK>>{}, Int<1>{});
  auto tma_logits = make_tma_copy(SM90_TMA_STORE{}, gLogits, cpbox_logits);

  // MMA configuration: BF16 SS, 2 warpgroups along M
  auto warpgroup_layout = make_layout(make_shape(Int<2>{}, Int<1>{}, Int<1>{}));
  auto tiled_mma = make_tiled_mma(SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{},
                                  warpgroup_layout);

  // Tile dispatch covers the output allocation range.
  int num_q_tiles = (logits_qb + kTileM - 1) / kTileM;
  int num_k_tiles = (logits_kb + kTileN - 1) / kTileN;
  int tiles_per_head = num_q_tiles * num_k_tiles;
  int total_tiles = num_batch * num_head_q * tiles_per_head;

  cutlass::FastDivmod qk_tile_divmod(tiles_per_head);
  cutlass::FastDivmod k_tile_divmod(num_k_tiles);
  cutlass::FastDivmod head_q_divmod(num_head_q);

  int num_heads_per_kv = num_head_q / num_head_kv;

  dim3 block(size(tiled_mma) + 128);
  dim3 grid(std::min(get_sm_count(), total_tiles));

  // Causal dispatch: two template instantiations
  auto do_launch = [&](auto causal_tag) {
    constexpr bool kCausal = decltype(causal_tag)::value;
    auto kernel =
        kernels::stem_oam_gemm_kernel<kCausal, decltype(tiled_mma), decltype(tma_q),
                                      decltype(tma_k), decltype(tma_logits), kTileM, kTileN, kStage,
                                      kStemBlockSize, kStride, kDimQK, decltype(slayout_q),
                                      decltype(slayout_k), decltype(slayout_logits)>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    kernel<<<grid, block, shm_size, stream>>>(
        tma_q, tma_k, tma_logits, reinterpret_cast<const int *>(q_seq_lens_ptr),
        reinterpret_cast<const int *>(kv_seq_lens_ptr), reinterpret_cast<const float *>(vbias_ptr),
        num_batch, num_head_q, num_heads_per_kv, max_num_qb, max_num_kb, total_tiles,
        qk_tile_divmod, k_tile_divmod, head_q_divmod);
  };

  if (causal) {
    do_launch(std::true_type{});
  } else {
    do_launch(std::false_type{});
  }
}

void stem_oam_gemm_dim128_async(void *block_logits_ptr, const void *qflat_ptr,
                                const void *kflat_ptr, const void *vbias_ptr,
                                const void *q_seq_lens_ptr, const void *kv_seq_lens_ptr,
                                int num_batch, int num_head_q, int num_head_kv, int max_num_qb,
                                int max_num_kb, int stem_block_size, int stem_stride, bool causal,
                                cudaStream_t stream) {
  launch_stem_oam_gemm_dim128<128, 16, 128>(block_logits_ptr, qflat_ptr, kflat_ptr, vbias_ptr,
                                            q_seq_lens_ptr, kv_seq_lens_ptr, num_batch, num_head_q,
                                            num_head_kv, max_num_qb, max_num_kb, causal, stream);
}

}  // namespace stem
}  // namespace hpc
