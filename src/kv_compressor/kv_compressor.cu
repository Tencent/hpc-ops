// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include <string>

#include "cutlass/fast_math.h"
#include "src/kv_compressor/kv_compressor.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace kv_compressor {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

namespace kernels {

// copy one tile by for-loop
template <typename DType, int kItemPerThread, int kValidRowDim, int kRowStride, int kMaxNumRow>
__device__ void copy_tile_to_state(DType* dst, const DType* src, int valid_num_row, int tidx) {
  static_assert(kValidRowDim % kItemPerThread == 0);
  if (tidx >= kValidRowDim / kItemPerThread) {
    return;
  }
#pragma unroll
  for (int irow = 0; irow < kMaxNumRow; irow++) {
    if (irow < valid_num_row) {
      auto* src_row_start = src + irow * kRowStride;
      auto* dst_row_start = dst + irow * kRowStride;
      auto src_load = load<DType, kItemPerThread>(src_row_start + tidx * kItemPerThread);
      store(dst_row_start + tidx * kItemPerThread, src_load);
    }
  }
}
// copy one tile by for-loop
template <typename DType, int kItemPerThread, int kValidRowDim, int kRowStride, int kMaxNumRow>
__device__ void add_copy_tile_to_state(DType* dst, const DType* src1, const DType* src2,
                                       int valid_num_row, int tidx) {
  static_assert(kValidRowDim % kItemPerThread == 0);
  if (tidx >= kValidRowDim / kItemPerThread) {
    return;
  }
#pragma unroll
  for (int irow = 0; irow < kMaxNumRow; irow++) {
    if (irow < valid_num_row) {
      auto* src1_row_start = src1 + irow * kRowStride;
      auto* src2_row_start = src2 + irow * kRowStride;
      auto* dst_row_start = dst + irow * kRowStride;
      auto src1_load = load<DType, kItemPerThread>(src1_row_start + tidx * kItemPerThread);
      auto src2_load = load<DType, kItemPerThread>(src2_row_start + tidx * kItemPerThread);
#pragma unroll
      for (int i = 0; i < kItemPerThread; i++) {
        src1_load[i] += src2_load[i];
      }
      store(dst_row_start + tidx * kItemPerThread, src1_load);
    }
  }
}

template <typename DType, int kItemPerThread, int kRatio = 128, int kHeadDim, int kRowStride,
          bool kOverlap = false>
__device__ void online_compress(DType* compressed_kv_ptr, const DType* partial_kv_ptr,
                                const DType* partial_score_ptr, const DType* ape_ptr, int tidx) {
  if (tidx >= kHeadDim / kItemPerThread) {
    return;
  }

  vec_t<DType, kItemPerThread> max_score;
  vec_t<DType, kItemPerThread> denominator = {0};
  vec_t<DType, kItemPerThread> numerator = {0};
// init online softmax
#pragma unroll
  for (int i = 0; i < kItemPerThread; i++) {
    max_score[i] = kNegInf;
  }

#pragma unroll
  for (int irow = 0; irow < kRatio; irow++) {
    auto* kv_row_start = partial_kv_ptr + irow * kRowStride;
    auto* score_row_start = partial_score_ptr + irow * kRowStride;
    auto kv_load_vec = load<DType, kItemPerThread>(kv_row_start + tidx * kItemPerThread);
    auto score_load_vec = load<DType, kItemPerThread>(score_row_start + tidx * kItemPerThread);
    auto ape_load_vec =
        load<DType, kItemPerThread>(ape_ptr + irow * kRowStride + tidx * kItemPerThread);
#pragma unroll
    for (int i = 0; i < kItemPerThread; i++) {
      score_load_vec[i] += ape_load_vec[i];
      if (score_load_vec[i] > max_score[i]) {
        float scale = expf(max_score[i] - score_load_vec[i]);
        denominator[i] = fmaf(denominator[i], scale, 1.0f);
        numerator[i] = fmaf(numerator[i], scale, kv_load_vec[i]);
        max_score[i] = score_load_vec[i];
      } else {
        float w = expf(score_load_vec[i] - max_score[i]);
        denominator[i] += w;
        numerator[i] = fmaf(kv_load_vec[i], w, numerator[i]);
      }
    }
  }

  if constexpr (kOverlap) {
    auto* down_right_partial_kv_ptr = partial_kv_ptr + kRatio * kRowStride + kHeadDim;
    auto* down_right_partial_score_ptr = partial_score_ptr + kRatio * kRowStride + kHeadDim;
    auto* right_ape_ptr = ape_ptr + kHeadDim;
#pragma unroll
    for (int irow = 0; irow < kRatio; irow++) {
      auto* kv_row_start = down_right_partial_kv_ptr + irow * kRowStride;
      auto* score_row_start = down_right_partial_score_ptr + irow * kRowStride;
      auto kv_load_vec = load<DType, kItemPerThread>(kv_row_start + tidx * kItemPerThread);
      auto score_load_vec = load<DType, kItemPerThread>(score_row_start + tidx * kItemPerThread);
      auto ape_load_vec =
          load<DType, kItemPerThread>(right_ape_ptr + irow * kRowStride + tidx * kItemPerThread);
#pragma unroll
      for (int i = 0; i < kItemPerThread; i++) {
        score_load_vec[i] += ape_load_vec[i];
        if (score_load_vec[i] > max_score[i]) {
          float scale = expf(max_score[i] - score_load_vec[i]);
          denominator[i] = fmaf(denominator[i], scale, 1.0f);
          numerator[i] = fmaf(numerator[i], scale, kv_load_vec[i]);
          max_score[i] = score_load_vec[i];
        } else {
          float w = expf(score_load_vec[i] - max_score[i]);
          denominator[i] += w;
          numerator[i] = fmaf(kv_load_vec[i], w, numerator[i]);
        }
      }
    }
  }

  vec_t<DType, kItemPerThread> softmax_val = {0};
#pragma unroll
  for (int i = 0; i < kItemPerThread; i++) {
    softmax_val[i] = denominator[i] > 0 ? numerator[i] / denominator[i] : -kNegInf;
    compressed_kv_ptr[tidx * kItemPerThread + i] = softmax_val[i];
  }
}

__forceinline__ __device__ void get_batch_id_of_this_block(const int* cu_compressed_seqlens_ptr,
                                                           int num_batch, int bidx,
                                                           int& batch_id_of_this_block,
                                                           bool& should_compress_this_block,
                                                           int& icompress_offset,
                                                           int& cu_compressed_seqlens_this_block) {
  int used_remainder_blocks = 0;
  int used_cutoff_blocks = 0;
  for (int ibatch = 0; ibatch < num_batch; ibatch++) {
    int pre_used = used_remainder_blocks + used_cutoff_blocks;
    used_remainder_blocks += 1;
    used_cutoff_blocks += cu_compressed_seqlens_ptr[ibatch + 1] - cu_compressed_seqlens_ptr[ibatch];
    if (pre_used <= bidx && bidx < used_remainder_blocks + used_cutoff_blocks) {
      batch_id_of_this_block = ibatch;
      should_compress_this_block =
          (bidx < (used_remainder_blocks + used_cutoff_blocks - 1));  // -1 for len to index
      icompress_offset = bidx - pre_used;
      cu_compressed_seqlens_this_block = cu_compressed_seqlens_ptr[ibatch];
      return;
    }
  }
  batch_id_of_this_block = -1;  // this block is not used neither for cutoff nor for remainder
}

template <typename DType, int kItemPerThread, int kRatio = 128, int kHeadDim = 512,
          bool kOverlap = false>
__global__ void kv_compressor_prefill(float* compressed_kv_ptr, const float* kv_ptr,
                                      const float* score_ptr, const int* cu_seqlens_ptr,
                                      const int* cu_compressed_seqlens_ptr, float* kv_states_ptr,
                                      float* score_states_ptr, const int* state_index_ptr,
                                      const int* start_pos_ptr, const float* ape_ptr, int num_batch,
                                      int total_seqlen) {
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;

  __shared__ int batch_id_of_this_block;
  __shared__ bool should_compress_this_block;
  __shared__ int icompress_offset;
  __shared__ int cu_compressed_seqlens_this_block;
  if (tidx == 0) {  // used for task assign
    get_batch_id_of_this_block(cu_compressed_seqlens_ptr, num_batch, bidx, batch_id_of_this_block,
                               should_compress_this_block, icompress_offset,
                               cu_compressed_seqlens_this_block);
  }
  __syncthreads();

  if (batch_id_of_this_block == -1) {
    // this block is not used neither for cutoff nor for remainder
    return;
  }
  constexpr int kKVStrideElem = kHeadDim * (1 + int(kOverlap));
  auto* kv_for_this_block_ptr =
      kv_ptr + (cu_seqlens_ptr[batch_id_of_this_block] + icompress_offset * kRatio) * kKVStrideElem;
  auto* score_this_block_ptr =
      score_ptr +
      (cu_seqlens_ptr[batch_id_of_this_block] + icompress_offset * kRatio) * kKVStrideElem;

  if (should_compress_this_block) {
    auto* compressed_kv_this_block_ptr =
        compressed_kv_ptr + (cu_compressed_seqlens_this_block + icompress_offset) * kHeadDim;
    if (kOverlap) {
      if (icompress_offset == 0) {  // first compress block, should not add upper block
        online_compress<DType, kItemPerThread, kRatio, kHeadDim, kKVStrideElem,
                        false>(  // act like not overlap
            compressed_kv_this_block_ptr, kv_for_this_block_ptr + kHeadDim,
            score_this_block_ptr + kHeadDim, ape_ptr + kHeadDim, tidx);
      } else {  // not first compress block, give the upper block ptr
        auto* kv_upper_block_ptr =
            kv_ptr + (cu_seqlens_ptr[batch_id_of_this_block] + (icompress_offset - 1) * kRatio) *
                         kKVStrideElem;
        auto* score_upper_block_ptr =
            score_ptr + (cu_seqlens_ptr[batch_id_of_this_block] + (icompress_offset - 1) * kRatio) *
                            kKVStrideElem;
        online_compress<DType, kItemPerThread, kRatio, kHeadDim, kKVStrideElem, kOverlap>(
            compressed_kv_this_block_ptr, kv_upper_block_ptr, score_upper_block_ptr, ape_ptr, tidx);
      }
    } else {  // not overlap
      online_compress<DType, kItemPerThread, kRatio, kHeadDim, kKVStrideElem, kOverlap>(
          compressed_kv_this_block_ptr, kv_for_this_block_ptr, score_this_block_ptr, ape_ptr, tidx);
    }
  } else {
    int seqlen_this_block =
        cu_seqlens_ptr[batch_id_of_this_block + 1] - cu_seqlens_ptr[batch_id_of_this_block];
    int remainder = seqlen_this_block % kRatio;
    auto* kv_states_this_block_ptr = kv_states_ptr + state_index_ptr[batch_id_of_this_block] *
                                                         kKVStrideElem * (1 + int(kOverlap)) *
                                                         kRatio;
    auto* score_states_this_block_ptr = score_states_ptr + state_index_ptr[batch_id_of_this_block] *
                                                               kKVStrideElem * (1 + int(kOverlap)) *
                                                               kRatio;
    if constexpr (kOverlap) {
      if (icompress_offset > 0) {  // cutoff > 0, should copy the upper states
                                   // copy upper left state
        auto* kv_upper_block_ptr =
            kv_ptr + (cu_seqlens_ptr[batch_id_of_this_block] + (icompress_offset - 1) * kRatio) *
                         kKVStrideElem;
        auto* score_upper_block_ptr =
            score_ptr + (cu_seqlens_ptr[batch_id_of_this_block] + (icompress_offset - 1) * kRatio) *
                            kKVStrideElem;
        copy_tile_to_state<DType, kItemPerThread, kHeadDim, kKVStrideElem, kRatio>(
            kv_states_this_block_ptr, kv_upper_block_ptr, kRatio, tidx);
        add_copy_tile_to_state<DType, kItemPerThread, kHeadDim, kKVStrideElem, kRatio>(
            score_states_this_block_ptr, score_upper_block_ptr, ape_ptr, kRatio, tidx);
      }
      // copy down left state
      copy_tile_to_state<DType, kItemPerThread, kHeadDim, kKVStrideElem, kRatio>(
          kv_states_this_block_ptr + kRatio * kKVStrideElem, kv_for_this_block_ptr, remainder,
          tidx);
      add_copy_tile_to_state<DType, kItemPerThread, kHeadDim, kKVStrideElem, kRatio>(
          score_states_this_block_ptr + kRatio * kKVStrideElem, score_this_block_ptr, ape_ptr,
          remainder, tidx);
      // copy down right state
      copy_tile_to_state<DType, kItemPerThread, kHeadDim, kKVStrideElem, kRatio>(
          kv_states_this_block_ptr + kRatio * kKVStrideElem + kHeadDim,
          kv_for_this_block_ptr + kHeadDim, remainder, tidx);
      add_copy_tile_to_state<DType, kItemPerThread, kHeadDim, kKVStrideElem, kRatio>(
          score_states_this_block_ptr + kRatio * kKVStrideElem + kHeadDim,
          score_this_block_ptr + kHeadDim, ape_ptr + kHeadDim, remainder, tidx);
    } else {  // not overlap
      copy_tile_to_state<DType, kItemPerThread, kHeadDim, kKVStrideElem, kRatio>(
          kv_states_this_block_ptr, kv_for_this_block_ptr, remainder, tidx);
      add_copy_tile_to_state<DType, kItemPerThread, kHeadDim, kKVStrideElem, kRatio>(
          score_states_this_block_ptr, score_this_block_ptr, ape_ptr, remainder, tidx);
    }
  }
}

}  // namespace kernels

bool kv_compressor_fp32_async(float* compressed_kv_ptr, const float* kv_ptr, const float* score_ptr,
                              const int* cu_seqlens_ptr, const int* cu_compressed_seqlens_ptr,
                              float* kv_states_ptr, float* score_states_ptr,
                              const int* state_index_ptr, const int* start_pos_ptr,
                              const float* ape_ptr, int num_batch, int total_seqlen, int ratio,
                              bool overlap, int head_dim, bool is_prefill, cudaStream_t stream) {
  using DType = float;
  if (is_prefill) {
    if (ratio == 4) {
      constexpr int kRatio = 4;
      constexpr int kOverlap = true;
      if (!overlap) {
        return false;
      }
      if (head_dim == 128) {
        constexpr int kItemPerThread = 1;
        constexpr int kHeadDim = 128;
        dim3 block(128);
        dim3 grid(total_seqlen / kRatio + num_batch);  // some unused blocks in tail, but it's ok
        kernels::kv_compressor_prefill<DType, kItemPerThread, kRatio, kHeadDim, kOverlap>
            <<<grid, block, 0, stream>>>(compressed_kv_ptr, kv_ptr, score_ptr, cu_seqlens_ptr,
                                         cu_compressed_seqlens_ptr, kv_states_ptr, score_states_ptr,
                                         state_index_ptr, start_pos_ptr, ape_ptr, num_batch,
                                         total_seqlen);
      } else if (head_dim == 512) {
        constexpr int kItemPerThread = 4;
        constexpr int kHeadDim = 512;
        dim3 block(128);
        dim3 grid(total_seqlen / kRatio + num_batch);  // some unused blocks in tail, but it's ok
        kernels::kv_compressor_prefill<DType, kItemPerThread, kRatio, kHeadDim, kOverlap>
            <<<grid, block, 0, stream>>>(compressed_kv_ptr, kv_ptr, score_ptr, cu_seqlens_ptr,
                                         cu_compressed_seqlens_ptr, kv_states_ptr, score_states_ptr,
                                         state_index_ptr, start_pos_ptr, ape_ptr, num_batch,
                                         total_seqlen);
      }

    } else if (ratio == 128) {
      constexpr int kOverlap = false;
      constexpr int kRatio = 128;
      constexpr int kItemPerThread = 4;
      constexpr int kHeadDim = 512;
      dim3 block(128);
      dim3 grid(total_seqlen / kRatio + num_batch);  // some unused blocks in tail, but it's ok
      kernels::kv_compressor_prefill<DType, kItemPerThread, kRatio, kHeadDim, kOverlap>
          <<<grid, block, 0, stream>>>(compressed_kv_ptr, kv_ptr, score_ptr, cu_seqlens_ptr,
                                       cu_compressed_seqlens_ptr, kv_states_ptr, score_states_ptr,
                                       state_index_ptr, start_pos_ptr, ape_ptr, num_batch,
                                       total_seqlen);
    }
    return true;
  } else {
    return false;
  }
}
}  // namespace kv_compressor
}  // namespace hpc
