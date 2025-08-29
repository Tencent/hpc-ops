// Copyright 2025 hpc-ops authors

#ifndef SRC_GEM3_GEM3_H_
#define SRC_GEM3_GEM3_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace gem3 {

void gem3_async(void *y_ptr, const void *q_ptr, const void *k_ptr, const void *v_ptr,
                const void *qscale_ptr, const void *kscale_ptr, int num_batch, int num_seq,
                int num_qk_dim, int num_v_dim, cudaStream_t stream);

}  // namespace gem3
}  // namespace hpc

#endif  // SRC_GEM3_GEM3_H_
