// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/fuse_moe/fuse_moe.h"

namespace hpc {
namespace fuse_moe {

void fuse_moe_async(void *output_ptr, const void *input_ptr, void *gate_up_input_ptr,
                    void *gate_up_output_ptr, const void *gate_up_weight_ptr,
                    const void *gate_up_scale_ptr, void *gate_up_tmas_ptr,
                    const void *act_and_mul_scale_ptr, void *down_input_ptr, void *down_output_ptr,
                    const void *down_weight_ptr, const void *down_scale_ptr, void *down_tmas_ptr,
                    const void *topk_ids_ptr, const void *topk_scale_ptr, void *topk_pos_ptr,
                    void *seqlens_ptr, void *cu_seqlens_ptr, void *tiles_ptr, void *cu_tiles_ptr,
                    int num_seq, int hidden_size, int intermediate_size, int num_topk,
                    int num_expert_total, int num_expert_local, int rank_ep, bool use_bf16_mul,
                    cudaStream_t stream) {
  int total_num_seq = num_seq * num_topk;
  int num_seq_per_group_avg = total_num_seq / num_expert_total;
  using T1 = __nv_bfloat16;
  using T2 = __nv_fp8_e4m3;

  // 0. call count_and_gather_async
  count_and_gather_async(gate_up_input_ptr, gate_up_output_ptr, down_input_ptr, down_output_ptr,
                         input_ptr, topk_ids_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr,
                         gate_up_tmas_ptr, down_tmas_ptr, tiles_ptr, cu_tiles_ptr, num_seq,
                         hidden_size, intermediate_size, num_topk, num_expert_local, rank_ep,
                         num_seq_per_group_avg, stream);

  // 1. call gate_up linear
  group_gemm::group_gemm_fp8_async(
      gate_up_output_ptr, gate_up_input_ptr, gate_up_weight_ptr, seqlens_ptr, cu_seqlens_ptr,
      gate_up_scale_ptr, gate_up_tmas_ptr, tiles_ptr, cu_tiles_ptr, num_expert_local, total_num_seq,
      intermediate_size, hidden_size, num_seq_per_group_avg, false, stream);
  // 2. call act and mul ??? EP seq_len
  activation::act_mul_and_quant_async((T2 *)down_input_ptr, (const T1 *)gate_up_output_ptr,
                                      (const float *)act_and_mul_scale_ptr, total_num_seq,
                                      intermediate_size, use_bf16_mul, stream);

  // 3. call down linear
  group_gemm::group_gemm_fp8_async(down_output_ptr, down_input_ptr, down_weight_ptr, seqlens_ptr,
                                   cu_seqlens_ptr, down_scale_ptr, down_tmas_ptr, tiles_ptr,
                                   cu_tiles_ptr, num_expert_local, total_num_seq, hidden_size,
                                   intermediate_size / 2, num_seq_per_group_avg, false, stream);

  // 4. call reduce //delete total_num_seq
  reduce_async(output_ptr, down_output_ptr, topk_pos_ptr, topk_scale_ptr, total_num_seq, num_seq,
               hidden_size, num_topk, stream);
}

}  // namespace fuse_moe
}  // namespace hpc
