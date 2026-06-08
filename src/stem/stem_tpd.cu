// Copyright (C) 2026 Tencent.

#include <cuda.h>

#include "src/stem/stem_kernels.cuh"
#include "src/stem/stem_tpd.h"

namespace hpc {
namespace stem {

// kEPT: Elements per thread
// kWPR: Warps per row
// kWPC: Warps per CTA
//
//   max_Kb    kEPT   kWPR   kWPC  Threads  kRowsPerCTA  Coverage
//   ≤ 1024      32     1     8      256         8         1024
//   ≤ 2048      32     2     2       64         1         2048
//   ≤ 4096      32     4     4      128         1         4096
//   ≤ 8192      32     8     8      256         1         8192
//   > 8192     128     8     8      256         1        32768
//
//   kWPR = 1  →  kWPC = 8     (8 independent rows/CTA, warp-shuffle only, no sync)
//   kWPR > 1  →  kWPC = kWPR  (1 cooperative row/CTA, cross-warp via __syncthreads)
template <int kEPT, int kWPR, int kWPC>
void launch_stem_tpd(void *mask_ptr, const void *block_logits_ptr, const void *q_seq_lens_ptr,
                     const void *kv_seq_lens_ptr, const void *num_prompt_tokens_ptr, int num_batch,
                     int num_heads, int max_Qb, int max_Kb, int block_size, float alpha,
                     int initial_blocks, int window_size, float k_block_num_rate_medium,
                     int k_block_num_bias_medium, float k_block_num_rate_large,
                     int k_block_num_bias_large, cudaStream_t stream) {
  constexpr int kRowsPerCTA = kWPC / kWPR;

  dim3 grid((max_Qb + kRowsPerCTA - 1) / kRowsPerCTA, num_heads, num_batch);
  dim3 block(kWPC * 32);

  kernels::stem_tpd_kernel<kEPT, kWPR, kWPC><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(block_logits_ptr),
      reinterpret_cast<uint8_t *>(mask_ptr), reinterpret_cast<const int *>(q_seq_lens_ptr),
      reinterpret_cast<const int *>(kv_seq_lens_ptr),
      reinterpret_cast<const int *>(num_prompt_tokens_ptr), alpha, block_size, initial_blocks,
      window_size, k_block_num_rate_medium, k_block_num_bias_medium, k_block_num_rate_large,
      k_block_num_bias_large, num_heads, max_Qb, max_Kb);
}

void stem_tpd_async(void *mask_ptr, const void *block_logits_ptr, const void *q_seq_lens_ptr,
                    const void *kv_seq_lens_ptr, const void *num_prompt_tokens_ptr, int num_batch,
                    int num_heads, int max_Qb, int max_Kb, int block_size, float alpha,
                    int initial_blocks, int window_size, float k_block_num_rate_medium,
                    int k_block_num_bias_medium, float k_block_num_rate_large,
                    int k_block_num_bias_large, cudaStream_t stream) {
  if (max_Kb <= 1024) {
    launch_stem_tpd<32, 1, 8>(mask_ptr, block_logits_ptr, q_seq_lens_ptr, kv_seq_lens_ptr,
                              num_prompt_tokens_ptr, num_batch, num_heads, max_Qb, max_Kb,
                              block_size, alpha, initial_blocks, window_size,
                              k_block_num_rate_medium, k_block_num_bias_medium,
                              k_block_num_rate_large, k_block_num_bias_large, stream);
  } else if (max_Kb <= 2048) {
    launch_stem_tpd<32, 2, 2>(mask_ptr, block_logits_ptr, q_seq_lens_ptr, kv_seq_lens_ptr,
                              num_prompt_tokens_ptr, num_batch, num_heads, max_Qb, max_Kb,
                              block_size, alpha, initial_blocks, window_size,
                              k_block_num_rate_medium, k_block_num_bias_medium,
                              k_block_num_rate_large, k_block_num_bias_large, stream);
  } else if (max_Kb <= 4096) {
    launch_stem_tpd<32, 4, 4>(mask_ptr, block_logits_ptr, q_seq_lens_ptr, kv_seq_lens_ptr,
                              num_prompt_tokens_ptr, num_batch, num_heads, max_Qb, max_Kb,
                              block_size, alpha, initial_blocks, window_size,
                              k_block_num_rate_medium, k_block_num_bias_medium,
                              k_block_num_rate_large, k_block_num_bias_large, stream);
  } else if (max_Kb <= 8192) {
    launch_stem_tpd<32, 8, 8>(mask_ptr, block_logits_ptr, q_seq_lens_ptr, kv_seq_lens_ptr,
                              num_prompt_tokens_ptr, num_batch, num_heads, max_Qb, max_Kb,
                              block_size, alpha, initial_blocks, window_size,
                              k_block_num_rate_medium, k_block_num_bias_medium,
                              k_block_num_rate_large, k_block_num_bias_large, stream);
  } else {
    launch_stem_tpd<128, 8, 8>(mask_ptr, block_logits_ptr, q_seq_lens_ptr, kv_seq_lens_ptr,
                               num_prompt_tokens_ptr, num_batch, num_heads, max_Qb, max_Kb,
                               block_size, alpha, initial_blocks, window_size,
                               k_block_num_rate_medium, k_block_num_bias_medium,
                               k_block_num_rate_large, k_block_num_bias_large, stream);
  }
}

}  // namespace stem
}  // namespace hpc
