// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_PREFILL_WARP_SPEC_WITH_FP8_KVCACHE_FP16_PV_COMPUTE_DIM128_H_
#define SRC_ATTENTION_PREFILL_WARP_SPEC_WITH_FP8_KVCACHE_FP16_PV_COMPUTE_DIM128_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace attention {
namespace prefill {

// Warp-specialized hybrid prefill attention: Q/K/V all FP8 storage; Q@Kᵀ runs as
// FP8 WGMMA while P@V stays FP16 WGMMA (fp32 accumulate), output bf16. Q is fp8
// with a per-(token, head) qscale (required). quant_type in {20, 21}
// (see hpc::QuantType): 20 = K per-token+head, V per-head (dynamic);
// 21 = K/V per-tensor (static).
void warp_spec_with_fp8_kvcache_fp16_pv_compute_dim128_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK, int ldK1, int ldK2, int ldV,
    int ldV1, int ldV2, int quant_type, cudaStream_t stream);

}  // namespace prefill
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_PREFILL_WARP_SPEC_WITH_FP8_KVCACHE_FP16_PV_COMPUTE_DIM128_H_
