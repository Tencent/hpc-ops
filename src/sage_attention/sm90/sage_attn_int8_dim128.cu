// Copyright 2025 hpc-ops authors

#include <cuda.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/sage_attention/sm90/sage_attn_int8_config.h"
#include "src/sage_attention/sm90/sage_attn_int8_dim128.h"
#include "src/sage_attention/sm90/sage_attn_int8_kernel.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace sage_attention {

void sage_attn_int8_fp8_dim128_sm90_async(void *y_ptr, const void *q_ptr, const void *k_ptr,
                                          const void *v_ptr, const float *q_scale_ptr,
                                          const float *k_scale_ptr, const float *v_scale_ptr,
                                          int num_batch, int qo_len, int kv_len, int head_dim,
                                          int num_head_q, int num_head_kv, int stride_bz_y,
                                          int stride_seq_y, int stride_h_y, int stride_bz_q,
                                          int stride_seq_q, int stride_h_q, int stride_bz_k,
                                          int stride_seq_k, int stride_h_k, int tensor_layout,
                                          int is_causal, float sm_scale, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using TinQK = int8_t;
  using TinV = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  static constexpr int kTileM = 64;
  static constexpr int kTileN = 128;
  static constexpr int kTileK = 128;
  static constexpr int kTileV = 128;
  static constexpr int kStage = 2;

  using TiledMmaQKAtom = SM90_64x128x32_S32S8S8_SS_TN;
  using TiledMmaPVAtom = SM90_64x128x32_F32E4M3E4M3_RS_TN<>;

  auto Q = make_tensor(make_gmem_ptr(reinterpret_cast<const TinQK *>(q_ptr)),
                       make_shape(qo_len, head_dim, num_head_q, num_batch),
                       make_stride(stride_seq_q, Int<1>{}, stride_h_q, stride_bz_q));

  auto K = make_tensor(make_gmem_ptr(reinterpret_cast<const TinQK *>(k_ptr)),
                       make_shape(kv_len, head_dim, num_head_kv, num_batch),
                       make_stride(stride_seq_k, Int<1>{}, stride_h_k, stride_bz_k));

  int padded_kv_len = ((kv_len + kTileN - 1) / kTileN) * kTileN;
  auto V = [&] {
    if (tensor_layout == 0) {
      return make_tensor(make_gmem_ptr(reinterpret_cast<const TinV *>(v_ptr)),
                         make_shape(head_dim, padded_kv_len, num_head_kv, num_batch),
                         make_stride(num_head_kv * padded_kv_len, Int<1>{}, padded_kv_len,
                                     head_dim * num_head_kv * padded_kv_len));
    }
    return make_tensor(make_gmem_ptr(reinterpret_cast<const TinV *>(v_ptr)),
                       make_shape(head_dim, padded_kv_len, num_head_kv, num_batch),
                       make_stride(padded_kv_len, Int<1>{}, head_dim * padded_kv_len,
                                   num_head_kv * head_dim * padded_kv_len));
  }();

  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)),
                       make_shape(qo_len, head_dim, num_head_q, num_batch),
                       make_stride(stride_seq_y, Int<1>{}, stride_h_y, stride_bz_y));

  using Config = SageAttentionInt8Sm90Config<TiledMmaQKAtom, TiledMmaPVAtom, kTileM, kTileN, kTileK,
                                             kTileV, kStage>;
  Config config;

  auto [tma_q, tma_k, tma_v, tma_y] = config.get_tma(Q, K, V, Y);

  int num_tile_m = (qo_len + kTileM - 1) / kTileM;

  constexpr float kLog2e = 1.4426950408889634f;
  float sm_scale_log2e = sm_scale * kLog2e;

  int shm_size = config.get_shm_size();
  int num_kv_groups = num_head_q / num_head_kv;

  // Single warpgroup (128 threads) so __launch_bounds__(128, 2) can schedule
  // 2 CTAs per SM, doubling math-warpgroup occupancy vs the old WS layout.
  dim3 block(128);
  dim3 grid(num_tile_m, num_head_q, num_batch);

#define LAUNCH_SM90_ATTN(GROUPS, CAUSAL)                                                          \
  do {                                                                                            \
    auto kernel =                                                                                 \
        sage_attention_int8_kernel_sm90<Config, GROUPS, CAUSAL, decltype(tma_q), decltype(tma_k), \
                                        decltype(tma_v), decltype(tma_y)>;                        \
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);          \
    kernel<<<grid, block, shm_size, stream>>>(tma_q, tma_k, tma_v, tma_y, q_scale_ptr,            \
                                              k_scale_ptr, v_scale_ptr, qo_len, kv_len,           \
                                              num_head_q, num_head_kv, sm_scale_log2e);           \
  } while (0)

#define DISPATCH_CAUSAL(GROUPS)      \
  if (is_causal) {                   \
    LAUNCH_SM90_ATTN(GROUPS, true);  \
  } else {                           \
    LAUNCH_SM90_ATTN(GROUPS, false); \
  }

  switch (num_kv_groups) {
    case 1:
      DISPATCH_CAUSAL(1);
      break;
    case 2:
      DISPATCH_CAUSAL(2);
      break;
    case 4:
      DISPATCH_CAUSAL(4);
      break;
    case 8:
      DISPATCH_CAUSAL(8);
      break;
    default:
      DISPATCH_CAUSAL(0);
      break;
  }
#undef DISPATCH_CAUSAL
#undef LAUNCH_SM90_ATTN
}

}  // namespace sage_attention
}  // namespace hpc
