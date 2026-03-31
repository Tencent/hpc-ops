// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>

#include <algorithm>

#include "src/ucl_comm/fuse_allreduce_combine.h"
#include "src/utils/utils.cuh"
#include "ucl/shmem.h"

namespace hpc {
namespace ucl_comm {
namespace kernels {

template <int kVecSize = 8, int kHiddenSize, int kNumThreadPerBlcok>
__global__ void fuse_allreduce_combine_kernel(
    const __nv_bfloat16 *__restrict__ input_ptr, const __nv_bfloat16 *__restrict__ mc_input_ptr,
    const __nv_bfloat16 *__restrict__ mn_input_ptr, __nv_bfloat16 *__restrict__ output_ptr,
    __nv_bfloat16 *__restrict__ mc_output_ptr, __nv_bfloat16 *__restrict__ mn_output_ptr,
    uint64_t **signal, uint64_t *__restrict__ output_multinode_signal_ptr, int rank, int local_size,
    int world_size, int attn_dp_size, int attn_tp_size, int moe_ep_size, int moe_tp_size,
    const int num_tokens, int world_rank, int batch_size, int num_qp) {
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;
  const int warp_id = thread_id / 32;
  const int lane_id = thread_id % 32;
  const int qp_id = (block_id * attn_tp_size + warp_id) % num_qp;

  using T = __nv_bfloat162;
  constexpr int kN = kVecSize / 2;

  const int rank_in_tp_group = rank % attn_tp_size;
  const int num_tokens_real = rank_in_tp_group < (batch_size / num_tokens)
                                  ? num_tokens
                                  : max(0, batch_size - rank_in_tp_group * num_tokens);

  const int div_num = num_tokens_real / gridDim.x;
  const int modulo_num = num_tokens_real % gridDim.x;
  const int itoken_start = min(div_num * block_id + min(block_id, modulo_num), num_tokens_real);
  const int itoken_end =
      min(itoken_start + div_num + (block_id < modulo_num ? 1 : 0), num_tokens_real);

  const int node_id = world_rank / local_size;
  const int peer_start = node_id * local_size;
  const int segment_gap = (local_size / attn_tp_size) * batch_size * kHiddenSize;

  const int div_sms_reduce = gridDim.x / num_tokens;
  const int modulo_sms_reduce = gridDim.x % num_tokens;

  const int base_block_id_reduce = block_id % num_tokens;
  const int idx_in_sms_group_reduce = block_id / num_tokens;
  const int max_num_of_sms_group_reduce =
      div_sms_reduce + (base_block_id_reduce < modulo_sms_reduce ? 1 : 0);
  const int num_of_sms_group_reduce = min(max_num_of_sms_group_reduce, attn_tp_size);

  //   sync remote blocks
  if (thread_id < local_size) {
    int peer_world_rank_intra = peer_start + thread_id;  // target peer's world_rank
    put_signal_relaxed_64(signal[peer_world_rank_intra] + block_id * local_size + rank);
    wait_signal_relaxed_64(signal[world_rank] + block_id * local_size + thread_id);
  }

  __syncthreads();

#pragma unroll 1
  for (int itoken = itoken_start; itoken < itoken_end; itoken++) {
    const int offset = itoken * kHiddenSize + thread_id * kVecSize;
    auto *mcptr_in = mc_input_ptr + offset;
    auto *mcptr_out = mc_output_ptr + offset;

    // First Segment
    auto in_sum = multi_load_reduce_add<T, kN>(mcptr_in);
    multi_store(mcptr_out, in_sum);

    // Second Segment
    in_sum = multi_load_reduce_add<T, kN>(mcptr_in + segment_gap);
    multi_store(mcptr_out + segment_gap, in_sum);
  }

  __syncthreads();

  if (gridDim.x <= num_tokens) {
    if (thread_id < local_size) {
      int peer_world_rank_intra = peer_start + thread_id;  // target peer's world_rank
      put_signal_release_64(signal[peer_world_rank_intra] + block_id * local_size + rank);
      wait_signal_acquire_64(signal[world_rank] + block_id * local_size + thread_id);
    }
  } else {
    if (block_id < num_tokens) {
      for (int sig_idx = thread_id; sig_idx < num_of_sms_group_reduce * local_size;
           sig_idx += kNumThreadPerBlcok) {
        int peer_world_rank_intra = peer_start + sig_idx % local_size;  // target peer's world_rank
        put_signal_release_64(
            signal[peer_world_rank_intra] +
            (base_block_id_reduce + (sig_idx / local_size) * num_tokens) * local_size + rank);
      }
    }
    if (block_id < num_tokens * attn_tp_size && thread_id < local_size) {
      wait_signal_acquire_64(signal[world_rank] + block_id * local_size + thread_id);
    }
  }

  __syncthreads();

  const int peer_world_rank = (world_rank + local_size) % world_size;

  const int peer_node_idx = (node_id + 1) % (world_size / local_size);
  const int peer_multinode_offset =
      peer_node_idx * (local_size / attn_tp_size) * batch_size * kHiddenSize;
  const int local_multinode_offset =
      node_id * (local_size / attn_tp_size) * batch_size * kHiddenSize;

  auto *source_multinode = mn_input_ptr + peer_multinode_offset;
  auto *dest_multinode = mn_output_ptr;
  auto *sig_addr = output_multinode_signal_ptr;

  // Simulate to calculate real token of each rank in tp group
  const int warp_for_tp_rank = warp_id % attn_tp_size;
  const int warp_of_tp_group = warp_id / attn_tp_size;
  const int num_tokens_real_warp = warp_for_tp_rank < (batch_size / num_tokens)
                                       ? num_tokens
                                       : max(0, batch_size - warp_for_tp_rank * num_tokens);

  const int div_num_warp = num_tokens_real_warp / gridDim.x;
  const int modulo_num_warp = num_tokens_real_warp % gridDim.x;
  const int itoken_start_warp =
      min(div_num_warp * block_id + min(block_id, modulo_num_warp), num_tokens_real_warp);
  const int itoken_end_warp =
      min(itoken_start_warp + div_num_warp + (block_id < modulo_num_warp ? 1 : 0),
          num_tokens_real_warp);

  const size_t token_bytes = kHiddenSize * sizeof(__nv_bfloat16);
  const int warp_token_count = itoken_end_warp - itoken_start_warp;
  const size_t warp_msg_size = warp_token_count * token_bytes;

  size_t warp_offset =
      warp_for_tp_rank * num_tokens < batch_size
          ? (warp_of_tp_group * batch_size + warp_for_tp_rank * num_tokens + itoken_start_warp) *
                token_bytes
          : (warp_of_tp_group * batch_size + batch_size) * token_bytes;

  if (warp_id < attn_tp_size) {
    if (warp_token_count > 0) {
      ucl::shmemx_net_put_nbi_warp(reinterpret_cast<uint64_t>(dest_multinode) + warp_offset,
                                   reinterpret_cast<uint64_t>(source_multinode) + warp_offset,
                                   warp_msg_size, peer_world_rank, qp_id, lane_id, 0);
    }
    if (lane_id == 0) {
      ucl::shmemx_net_amo_nonfetch_add(sig_addr + block_id * attn_tp_size + warp_id, 1,
                                       peer_world_rank, qp_id);
    }
  }

  assert(kHiddenSize == kVecSize * kNumThreadPerBlcok);
  __nv_bfloat16 reg_in_1[kVecSize];
  __nv_bfloat16 reg_in_2[kVecSize];
  __nv_bfloat16 reg_out[kVecSize];

  if (gridDim.x <= num_tokens) {
    if (warp_id < attn_tp_size && lane_id == 0) {
      ucl::shmem_signal_wait_until(sig_addr + block_id * attn_tp_size + warp_id, ucl::SHMEM_CMP_EQ,
                                   static_cast<uint64_t>(1));

      *(sig_addr + block_id * attn_tp_size + warp_id) = 0;
    }

    __syncthreads();

    for (int iphase = 0; iphase < attn_tp_size; iphase++) {
      int num_tokens_real_phase = iphase < (batch_size / num_tokens)
                                      ? num_tokens
                                      : max(0, batch_size - iphase * num_tokens);
      int div_num_phase = num_tokens_real_phase / gridDim.x;
      int modulo_num_phase = num_tokens_real_phase % gridDim.x;
      int itoken_start_phase =
          min(div_num_phase * block_id + min(block_id, modulo_num_phase), num_tokens_real_phase);
      int itoken_end_phase =
          min(itoken_start_phase + div_num_phase + (block_id < modulo_num_phase ? 1 : 0),
              num_tokens_real_phase);

      for (int itoken = iphase * num_tokens + itoken_start_phase;
           itoken < iphase * num_tokens + itoken_end_phase; itoken++) {
        auto *source_reduce = mn_input_ptr + local_multinode_offset + itoken * kHiddenSize;
        auto *dest_reduce = mn_output_ptr + itoken * kHiddenSize;
        size_t i = thread_id * kVecSize;
        *reinterpret_cast<__uint128_t *>(reg_in_1) =
            *reinterpret_cast<__uint128_t const *>(source_reduce + i);
        *reinterpret_cast<__uint128_t *>(reg_in_2) =
            *reinterpret_cast<__uint128_t const *>(dest_reduce + i);
#pragma unroll
        for (int j = 0; j < kVecSize; ++j) {
          reg_out[j] = static_cast<__nv_bfloat16>(static_cast<float>(reg_in_1[j]) +
                                                  static_cast<float>(reg_in_2[j]));
        }
        *reinterpret_cast<__uint128_t *>(dest_reduce + i) =
            *reinterpret_cast<__uint128_t const *>(reg_out);
      }
    }
  } else {
    const int div_phases_sms_reduce = attn_tp_size / num_of_sms_group_reduce;
    const int modulo_phases_sms_reduce = attn_tp_size % num_of_sms_group_reduce;
    const int iphase_start_reduce = min(div_phases_sms_reduce * idx_in_sms_group_reduce +
                                            min(idx_in_sms_group_reduce, modulo_phases_sms_reduce),
                                        attn_tp_size);
    const int iphase_end_reduce =
        min(iphase_start_reduce + div_phases_sms_reduce +
                (idx_in_sms_group_reduce < modulo_phases_sms_reduce ? 1 : 0),
            attn_tp_size);

    if (iphase_end_reduce > iphase_start_reduce) {
      if (warp_id >= iphase_start_reduce && warp_id < iphase_end_reduce && lane_id == 0) {
        ucl::shmem_signal_wait_until(sig_addr + base_block_id_reduce * attn_tp_size + warp_id,
                                     ucl::SHMEM_CMP_EQ, static_cast<uint64_t>(1));

        *(sig_addr + base_block_id_reduce * attn_tp_size + warp_id) = 0;
      }
      __syncthreads();

      // if itoken > batch_size in phase == attn_tp_size - 1, extra data will be dropped in python
      for (int iphase = iphase_start_reduce; iphase < iphase_end_reduce; iphase++) {
        const int itoken = iphase * num_tokens + base_block_id_reduce;
        auto *source_reduce = mn_input_ptr + local_multinode_offset + itoken * kHiddenSize;
        auto *dest_reduce = mn_output_ptr + itoken * kHiddenSize;
        size_t i = thread_id * kVecSize;
        *reinterpret_cast<__uint128_t *>(reg_in_1) =
            *reinterpret_cast<__uint128_t const *>(source_reduce + i);
        *reinterpret_cast<__uint128_t *>(reg_in_2) =
            *reinterpret_cast<__uint128_t const *>(dest_reduce + i);
#pragma unroll
        for (int j = 0; j < kVecSize; ++j) {
          reg_out[j] = static_cast<__nv_bfloat16>(static_cast<float>(reg_in_1[j]) +
                                                  static_cast<float>(reg_in_2[j]));
        }
        *reinterpret_cast<__uint128_t *>(dest_reduce + i) =
            *reinterpret_cast<__uint128_t const *>(reg_out);
      }
    }
  }
}

}  // namespace kernels

void fuse_allreduce_combine_async(const void *input_ptr, const void *mc_input_ptr,
                                  const void *mn_input_ptr, void *output_ptr, void *mc_output_ptr,
                                  void *mn_output_ptr, void *signal_ptr,
                                  void *output_multinode_signal_ptr, int64_t rank,
                                  int64_t local_size, int64_t world_size, int64_t attn_dp_size,
                                  int64_t attn_tp_size, int64_t moe_ep_size, int64_t moe_tp_size,
                                  int64_t num_max_blocks, int num_tokens, int hidden_size,
                                  int world_rank, int batch_size, int num_qp, cudaStream_t stream) {
  constexpr int kVecSize = 8;
  if (hidden_size == 7168) {
    constexpr int kNumThreadPerBlcok = 896;
    constexpr int kHiddenSize = 7168;
    dim3 grid(num_max_blocks);
    dim3 block(kNumThreadPerBlcok);
    assert(kNumThreadPerBlcok >= 32 * attn_tp_size);

    kernels::fuse_allreduce_combine_kernel<kVecSize, kHiddenSize, kNumThreadPerBlcok>
        <<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(input_ptr),
            static_cast<const __nv_bfloat16 *>(mc_input_ptr),
            static_cast<const __nv_bfloat16 *>(mn_input_ptr),
            static_cast<__nv_bfloat16 *>(output_ptr), static_cast<__nv_bfloat16 *>(mc_output_ptr),
            static_cast<__nv_bfloat16 *>(mn_output_ptr), reinterpret_cast<uint64_t **>(signal_ptr),
            static_cast<uint64_t *>(output_multinode_signal_ptr), static_cast<int>(rank),
            static_cast<int>(local_size), static_cast<int>(world_size),
            static_cast<int>(attn_dp_size), static_cast<int>(attn_tp_size),
            static_cast<int>(moe_ep_size), static_cast<int>(moe_tp_size),
            static_cast<int>(num_tokens), static_cast<int>(world_rank),
            static_cast<int>(batch_size), static_cast<int>(num_qp));
  } else if (hidden_size == 4096) {
    constexpr int kNumThreadPerBlcok = 512;
    constexpr int kHiddenSize = 4096;
    dim3 grid(num_max_blocks);
    dim3 block(kNumThreadPerBlcok);
    assert(kNumThreadPerBlcok >= 32 * attn_tp_size);

    kernels::fuse_allreduce_combine_kernel<kVecSize, kHiddenSize, kNumThreadPerBlcok>
        <<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(input_ptr),
            static_cast<const __nv_bfloat16 *>(mc_input_ptr),
            static_cast<const __nv_bfloat16 *>(mn_input_ptr),
            static_cast<__nv_bfloat16 *>(output_ptr), static_cast<__nv_bfloat16 *>(mc_output_ptr),
            static_cast<__nv_bfloat16 *>(mn_output_ptr), reinterpret_cast<uint64_t **>(signal_ptr),
            static_cast<uint64_t *>(output_multinode_signal_ptr), static_cast<int>(rank),
            static_cast<int>(local_size), static_cast<int>(world_size),
            static_cast<int>(attn_dp_size), static_cast<int>(attn_tp_size),
            static_cast<int>(moe_ep_size), static_cast<int>(moe_tp_size),
            static_cast<int>(num_tokens), static_cast<int>(world_rank),
            static_cast<int>(batch_size), static_cast<int>(num_qp));
  }
}

}  // namespace ucl_comm
}  // namespace hpc
