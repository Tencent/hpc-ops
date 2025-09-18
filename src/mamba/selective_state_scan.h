// Copyright 2025 hpc-ops authors

#ifndef SRC_MAMBA_SELECTIVE_STATE_SCAN_H_
#define SRC_MAMBA_SELECTIVE_STATE_SCAN_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace mamba {

bool exp_dA_chunked_cumsum_async(float* yscale_ptr, float* x_scale_ptr, void* tma_desc_ptr,
                                 const void* y_ptr, const void* zxbcdt_ptr, const float* A_ptr,
                                 const float* bias_ptr, const int* split_metadata_ptr,
                                 int chunk_size, int max_chunks, int batch_size, int nheads,
                                 int head_dim, int ngroups, int dstate, int zxbcdt_row_stride,
                                 int dt_offset, int padded_seqlen_stride, int tma_desc_count,
                                 cudaStream_t stream);

bool chunk_states_bmm_async(float* out_ptr, const void* zxbcdt_ptr, const void* scaled_x_ptr,
                            const void* tensormaps, const int* split_metadata, int batch_size,
                            int total_chunks, int nheads, int ngroups, int head_dim, int dstate,
                            int zxbcdt_row_stride, int chunk_size, int tma_desc_count,
                            cudaStream_t stream);

bool chunk_states_passing_async(void* chunked_states_cumsum_ptr, float* ssm_states,
                                const int* indices_ptr, const float* chunked_states_ptr,
                                const float* yscale_ptr, const int* split_metadata_ptr,
                                int chunk_size, int total_padded_seqlen, int total_chunks,
                                int batch_size, int nheads, int head_dim, int dstate,
                                cudaStream_t stream);

bool pre_y_bmm_async(float* out_ptr, const void* zxbcdt_ptr, const void* chunked_states_cumsum,
                     const void* tensormaps, const int* split_metadata, int batch_size,
                     int total_chunks, int nheads, int ngroups, int head_dim, int dstate,
                     int zxbcdt_row_stride, int chunk_size, int tma_desc_count,
                     cudaStream_t stream);

bool chunk_scan_gem3_async(void* out_ptr, const void* zxbcdt_ptr, const void* pre_y,
                           const float* xscale, const float* yscale, const float* D_ptr,
                           const void* tensormaps, const int* split_metadata, int batch_size,
                           int total_chunks, int total_padded_seqlen, int nheads, int ngroups,
                           int head_dim, int dstate, int zxbcdt_row_stride, int chunk_size,
                           int tma_desc_count, cudaStream_t stream);

}  // namespace mamba
}  // namespace hpc

#endif  // SRC_MAMBA_SELECTIVE_STATE_SCAN_H_
