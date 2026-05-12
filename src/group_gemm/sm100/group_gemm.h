// Copyright 2025 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM100_GROUP_GEMM_H_
#define SRC_GROUP_GEMM_SM100_GROUP_GEMM_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace group_gemm {

void group_gemm_fp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                          const void *seqlens_ptr, const void *cu_seqlens_ptr, const void *y_scale,
                          void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, void *task_map_ptr,
                          int num_waves, int num_group, int m, int n, int k,
                          int num_seq_per_group_avg, bool update_tma, bool use_pdl,
                          cudaStream_t stream);

void group_gemm_cp_async_fp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                   const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                   const void *y_scale, void *tmas_ptr, void *tiles_ptr,
                                   void *cu_tiles_ptr, void *task_map_ptr, int num_waves,
                                   int num_group, int m, int n, int k, int num_seq_per_group_avg,
                                   bool update_tma, bool use_pdl, cudaStream_t stream,
                                   const void *x_row_map_ptr = nullptr, int x_num_rows = 0);

void group_gemm_cp_async_fp8_act_mul_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                           const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                           const void *gate_up_scale_ptr,
                                           const void *act_mul_scale_ptr, void *tmas_ptr,
                                           void *tiles_ptr, void *cu_tiles_ptr, int num_group,
                                           int m, int n, int k, int num_seq_per_group_avg,
                                           bool update_tma, bool use_pdl, cudaStream_t stream,
                                           const void *x_row_map_ptr = nullptr, int x_num_rows = 0);

void group_gemm_blockwise_fp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                    const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                    const void *xscale_ptr, const void *wscale_ptr, void *tmas_ptr,
                                    void *tiles_ptr, void *cu_tiles_ptr, void *task_map_ptr,
                                    int num_waves, int num_group, int m, int n, int k, int m_pad,
                                    int num_block_k_pad4, int num_seq_per_group_avg,
                                    bool update_tma, bool use_pdl, cudaStream_t stream);

void reformat_x_scale_async(void *output_ptr, const void *xscale_ptr, const void *seqlens_ptr,
                            const void *cu_seqlens_ptr, int num_group, int m, int n, int tilem,
                            cudaStream_t stream);

}  // namespace group_gemm
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM100_GROUP_GEMM_H_
