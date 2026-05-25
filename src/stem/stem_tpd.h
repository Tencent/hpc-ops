// Copyright 2025 hpc-ops authors

#ifndef SRC_STEM_STEM_TPD_H_
#define SRC_STEM_STEM_TPD_H_

#include <cuda_runtime_api.h>

namespace hpc {
namespace stem {

void stem_tpd_async(void *mask_ptr, const void *block_logits_ptr, const void *q_seq_lens_ptr,
                    const void *kv_seq_lens_ptr, const void *num_prompt_tokens_ptr, int num_batch,
                    int num_heads, int max_Qb, int max_Kb, int block_size, float alpha,
                    int initial_blocks, int window_size, float k_block_num_rate_medium,
                    int k_block_num_bias_medium, float k_block_num_rate_large,
                    int k_block_num_bias_large, cudaStream_t stream);

}  // namespace stem
}  // namespace hpc

#endif  // SRC_STEM_STEM_TPD_H_
