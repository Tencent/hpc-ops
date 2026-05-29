// Copyright 2025 hpc-ops authors

#ifndef SRC_COMM_GEMM_COMM_GEMM_H_
#define SRC_COMM_GEMM_COMM_GEMM_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace comm_gemm {
void fuse_gemm_reduce_scatter_fp8_async(void *y_ptr, const void *x_ptr, const void *weight_ptr,
                                        const void *x_scale_ptr, const void *weight_scale_ptr,
                                        const void *bias_ptr, int m, int n, int k, int m_pad,
                                        int num_block_k, int num_block_n, cudaStream_t stream,
                                        void *signal_ptr, void *multimem_output_ptr,
                                        void *multimem_signal_ptr, int num_comp_sm, int num_comm_sm,
                                        int rank, int world_size);

}  // namespace comm_gemm
}  // namespace hpc

#endif  // SRC_COMM_GEMM_COMM_GEMM_H_
