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

void group_gemm_fp8_with_reduce_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                      const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                      const void *y_scale, void *tmas_ptr, void *tiles_ptr,
                                      void *cu_tiles_ptr, void *task_map_ptr, void *x_row_map_ptr,
                                      void *topk_scale_row_map_ptr, int num_waves, int num_group,
                                      int m, int n, int k, int num_seq_per_group_avg,
                                      bool update_tma, bool use_pdl, cudaStream_t stream);

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

void prepack_mxfp8_w_scale_async(void *sfw_packed_ptr, const void *sfw_ptr, int num_group, int n,
                                 int k, bool is_2sm, cudaStream_t stream);

void prepack_mxfp8_x_scale_async(void *sfx_packed_ptr, const void *sfx_ptr,
                                 const void *cu_seqlens_ptr, int num_group, int m_total, int k,
                                 int kTileM, bool is_smallm, cudaStream_t stream);

void group_gemm_mxfp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                            const void *sfx_packed_ptr, const void *sfw_packed_ptr,
                            const void *seqlens_ptr, const void *cu_seqlens_ptr, void *tmas_ptr,
                            void *tiles_ptr, void *cu_tiles_ptr, int num_group, int m, int n, int k,
                            int num_seq_per_group_avg, bool update_tma, cudaStream_t stream,
                            bool use_pdl = false);

void group_gemm_cp_async_mxfp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                     const void *sfx_packed_ptr, const void *sfw_packed_ptr,
                                     const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                     void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr,
                                     int num_group, int m, int n, int k, int num_seq_per_group_avg,
                                     bool update_tma, cudaStream_t stream,
                                     const void *x_row_map_ptr = nullptr, int x_num_rows = 0,
                                     bool use_pdl = false);

inline int mxfp8_dispatch_kTileM(int num_seq_per_group_avg, int n) {
  bool use_2sm = (n % 256 == 0) && (num_seq_per_group_avg > 32);
  if (use_2sm) {
    // 2SM ladder (group_gemm_mxfp8.cu LAUNCH_2SM_MXFP8 ladder).
    if (num_seq_per_group_avg <= 32) {
      return 32;
    }
    if (num_seq_per_group_avg <= 64) {
      return 64;
    }
    if (num_seq_per_group_avg <= 96) {
      return 96;
    }
    if (num_seq_per_group_avg <= 128) {
      return 128;
    }
    if (num_seq_per_group_avg <= 160) {
      return 160;
    }
    if (num_seq_per_group_avg <= 192) {
      return 192;
    }
    return 256;
  }
  // 1SM ladder (group_gemm_mxfp8.cu LAUNCH_1SM_MXFP8 ladder).
  if (num_seq_per_group_avg <= 16) {
    return 16;
  }
  if (num_seq_per_group_avg <= 32) {
    return 32;
  }
  if (num_seq_per_group_avg <= 48) {
    return 48;
  }
  if (num_seq_per_group_avg <= 64) {
    return 64;
  }
  if (num_seq_per_group_avg <= 128) {
    return 128;
  }
  return 256;
}

inline int mxfp8_dispatch_kTileM_cp_async(int num_seq_per_group_avg) {
  if (num_seq_per_group_avg <= 16) {
    return 16;
  }
  if (num_seq_per_group_avg <= 32) {
    return 32;
  }
  if (num_seq_per_group_avg <= 48) {
    return 48;
  }
  if (num_seq_per_group_avg <= 64) {
    return 64;
  }
  if (num_seq_per_group_avg <= 128) {
    return 128;
  }
  return 256;
}

}  // namespace group_gemm
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM100_GROUP_GEMM_H_
