// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include <string>

#include "cutlass/fast_math.h"
#include "src/compressor/compressor.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace compressor {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

namespace kernels {

// copy one tile by for-loop
template <typename DType, int kItemPerThread, int kValidRowDim, int kMaxNumRow>
__forceinline__ __device__ void copy_tile_to_state(DType* dst, const DType* src, int valid_num_row,
                                                   int dst_stride, int src_stride, int tidx) {
  static_assert(kValidRowDim % kItemPerThread == 0);
  if (tidx >= kValidRowDim / kItemPerThread) {
    return;
  }
#pragma unroll
  for (int irow = 0; irow < kMaxNumRow; irow++) {
    if (irow < valid_num_row) {
      auto* src_irow = src + irow * src_stride;
      auto* dst_irow = dst + irow * dst_stride;
      auto src_load = load<DType, kItemPerThread>(src_irow + tidx * kItemPerThread);
      store(dst_irow + tidx * kItemPerThread, src_load);
    }
  }
}
// copy one tile by for-loop
template <typename DType, int kItemPerThread, int kValidRowDim, int kMaxNumRow>
__forceinline__ __device__ void add_copy_tile_to_state(DType* dst, const DType* src1,
                                                       const DType* src2, int valid_num_row,
                                                       int dst_stride, int src1_stride,
                                                       int src2_stride, int tidx) {
  static_assert(kValidRowDim % kItemPerThread == 0);
  if (tidx >= kValidRowDim / kItemPerThread) {
    return;
  }
#pragma unroll
  for (int irow = 0; irow < kMaxNumRow; irow++) {
    if (irow < valid_num_row) {
      auto* src1_irow = src1 + irow * src1_stride;
      auto* src2_irow = src2 + irow * src2_stride;
      auto* dst_irow = dst + irow * dst_stride;
      auto src1_vec = load<DType, kItemPerThread>(src1_irow + tidx * kItemPerThread);
      auto src2_vec = load<DType, kItemPerThread>(src2_irow + tidx * kItemPerThread);
#pragma unroll
      for (int i = 0; i < size(src1_vec); i++) {
        src1_vec[i] += src2_vec[i];
      }
      store(dst_irow + tidx * kItemPerThread, src1_vec);
    }
  }
}

template <typename DType, int kItemPerThread, int kRatio = 128, int kHeadDim, int kWidth,
          bool kOverlap = false, bool kAddUpperApe = true, bool kAddDownApe = true>
__forceinline__ __device__ void online_compress(DType* compressed_kv_ptr, const DType* kv_state_ptr,
                                                const DType* score_state_ptr,
                                                const DType* partial_kv_ptr,
                                                const DType* partial_score_ptr,
                                                const DType* ape_ptr, int state_stride,
                                                int kv_stride, int tidx) {
  if (tidx >= kHeadDim / kItemPerThread) {
    return;
  }

  vec_t<DType, kItemPerThread> max_score;
  // softmax_sum = a/b
  vec_t<DType, kItemPerThread> b = {0};  // b = sum(expf(score))
  vec_t<DType, kItemPerThread> a = {0};  // a = sum(kv * expf(score))
// init online softmax
#pragma unroll
  for (int i = 0; i < size(max_score); i++) {
    max_score[i] = kNegInf;
  }

  constexpr int kPreloadRow = 1;  // preload has no contribute
#pragma unroll
  for (int iround = 0; iround < kRatio / kPreloadRow; iround++) {
    vec_t<DType, kItemPerThread> kv_lvecs[kPreloadRow];
    vec_t<DType, kItemPerThread> score_vecs[kPreloadRow];
    vec_t<DType, kItemPerThread> ape_vecs[kPreloadRow];
    // load
#pragma unroll
    for (int i = 0; i < kPreloadRow; i++) {
      int irow = iround * kPreloadRow + i;
      auto* kv_irow = kv_state_ptr + irow * state_stride;
      auto* score_irow = score_state_ptr + irow * state_stride;
      auto* ape_irow = ape_ptr + irow * kWidth;
      kv_lvecs[i] = load<DType, kItemPerThread>(kv_irow + tidx * kItemPerThread);
      score_vecs[i] = load<DType, kItemPerThread>(score_irow + tidx * kItemPerThread);
      ape_vecs[i] = load<DType, kItemPerThread>(ape_irow + tidx * kItemPerThread);
    }
    // process
    for (int i = 0; i < kPreloadRow; i++) {
#pragma unroll
      for (int j = 0; j < size(max_score); j++) {
        if constexpr (kAddUpperApe) {  // if read from state, do not add ape
          score_vecs[i][j] += ape_vecs[i][j];
        }
        DType m = fmaxf(max_score[j], score_vecs[i][j]);
        float s = expf_ftz(max_score[j] - m);
        float w = expf_ftz(score_vecs[i][j] - m);
        b[j] = fmaf(b[j], s, w);
        a[j] = fmaf(kv_lvecs[i][j], w, a[j] * s);
        max_score[j] = m;
      }
    }
  }

  if constexpr (kOverlap) {
    auto* right_ape_ptr = ape_ptr + kHeadDim;
#pragma unroll
    for (int irow = 0; irow < kRatio; irow++) {
      auto* kv_irow = partial_kv_ptr + irow * (int64_t)kv_stride;
      auto* score_irow = partial_score_ptr + irow * (int64_t)kv_stride;
      auto* ape_irow = right_ape_ptr + irow * kWidth;
      auto kv_lvecs = load<DType, kItemPerThread>(kv_irow + tidx * kItemPerThread);
      auto score_vecs = load<DType, kItemPerThread>(score_irow + tidx * kItemPerThread);
      auto ape_vec = load<DType, kItemPerThread>(ape_irow + tidx * kItemPerThread);
#pragma unroll
      for (int i = 0; i < size(max_score); i++) {
        if constexpr (kAddDownApe) {  // if read from state, do not add ape
          score_vecs[i] += ape_vec[i];
        }
        DType m = fmaxf(max_score[i], score_vecs[i]);
        float s = expf_ftz(max_score[i] - m);
        float w = expf_ftz(score_vecs[i] - m);
        b[i] = fmaf(b[i], s, w);
        a[i] = fmaf(kv_lvecs[i], w, a[i] * s);
        max_score[i] = m;
      }
    }
  }

  vec_t<DType, kItemPerThread> s = {0};  // final softmax_sum
#pragma unroll
  for (int i = 0; i < size(s); i++) {
    // float inv_b = rcpf_ftz(b[i]);
    s[i] = b[i] > 0 ? a[i] / b[i] : -kNegInf;
    compressed_kv_ptr[tidx * kItemPerThread + i] = s[i];
  }
}

template <typename DType, int kItemPerThread, int kRatio = 128, int kHeadDim, int kWidth,
          bool kOverlap = false, bool kAddUpperApe = true, bool kAddDownApe = true>
__device__ void online_compress(DType* compressed_kv_ptr, const DType* partial_kv_ptr,
                                const DType* partial_score_ptr, const DType* ape_ptr, int kv_stride,
                                int tidx) {
  online_compress<DType, kItemPerThread, kRatio, kHeadDim, kWidth, kOverlap, kAddUpperApe,
                  kAddDownApe>(compressed_kv_ptr, partial_kv_ptr, partial_score_ptr,
                               partial_kv_ptr + kRatio * kv_stride + kHeadDim,
                               partial_score_ptr + kRatio * kv_stride + kHeadDim, ape_ptr,
                               kv_stride, kv_stride, tidx);
}

template <int kRatio>
__forceinline__ __device__ void get_batch_id_of_this_block(const int* cu_compressed_seqlens_ptr,
                                                           const int* cu_seqlens_ptr, int num_batch,
                                                           int bidx, int& batch_id,
                                                           bool& should_compress, int& icompress,
                                                           int& cu_compressed_seqlen, int& seqlen) {
  int r = 0;  // used remainder blocks
  int c = 0;  // used cutoff blocks
  int s = 0;
  for (int ibatch = 0; ibatch < num_batch; ibatch++) {
    int pre_used = r + c;
    r += 1;
    s = cu_seqlens_ptr[ibatch + 1] - cu_seqlens_ptr[ibatch];
    if (s >= kRatio) {
      c += cu_compressed_seqlens_ptr[ibatch + 1] - cu_compressed_seqlens_ptr[ibatch];
    }
    if (pre_used <= bidx && bidx < r + c) {
      batch_id = ibatch;
      should_compress = (bidx < (r + c - 1));  // -1 for len to index
      icompress = bidx - pre_used;
      cu_compressed_seqlen = cu_compressed_seqlens_ptr[ibatch];
      seqlen = s;
      return;
    }
  }
  batch_id = -1;  // this block is not used neither for cutoff nor for remainder
}

template <typename DType, int kItemPerThread, int kRatio = 128, int kHeadDim = 512,
          bool kOverlap = false>
__global__ void kv_compressor_prefill(float* compressed_kv_ptr, const float* kv_ptr,
                                      const float* score_ptr, const int* cu_seqlens_ptr,
                                      const int* cu_compressed_seqlens_ptr, float* kv_states_ptr,
                                      float* score_states_ptr, const int* state_index_ptr,
                                      const int* start_pos_ptr, const float* ape_ptr, int kv_stride,
                                      int num_batch, int total_seqlen) {
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;

  __shared__ int batch_id;
  __shared__ bool should_compress;
  __shared__ int icompress;
  __shared__ int cu_compressed_seqlen;
  __shared__ int seqlen;
  if (tidx == 0) {  // used for task assign
    get_batch_id_of_this_block<kRatio>(cu_compressed_seqlens_ptr, cu_seqlens_ptr, num_batch, bidx,
                                       batch_id, should_compress, icompress, cu_compressed_seqlen,
                                       seqlen);
  }
  __syncthreads();
  if (batch_id == -1) {
    // this block is not used neither for cutoff nor for remainder
    return;
  }
  constexpr int kWidth = kHeadDim * (1 + int(kOverlap));
  auto* kv = kv_ptr + (cu_seqlens_ptr[batch_id] + icompress * kRatio) * (int64_t)kv_stride;
  auto* score = score_ptr + (cu_seqlens_ptr[batch_id] + icompress * kRatio) * (int64_t)kv_stride;

  auto* kv_state =
      kv_states_ptr + state_index_ptr[batch_id] * kWidth * (1 + int(kOverlap)) * kRatio;
  auto* score_state =
      score_states_ptr + state_index_ptr[batch_id] * kWidth * (1 + int(kOverlap)) * kRatio;

  if (seqlen < kRatio) {
    if (icompress > 0) {  // block-level return
      return;
    }
    // decode seq
    int start_pos = start_pos_ptr[batch_id];
    for (int i = 0; i < seqlen; i++) {
      int cur_pos = start_pos + i;
      int local_pos = cur_pos % kRatio;
      auto* kv_row = kv + i * (int64_t)kv_stride;
      auto* score_row = score + i * (int64_t)kv_stride;
      auto* ape_row = ape_ptr + local_pos * kWidth;

      // copy down left state
      constexpr int kNumRow = 1;  // 1 row for each iter
      if constexpr (kOverlap) {
        auto* kv_state_row = kv_state + (kRatio + local_pos) * kWidth;
        auto* score_state_row = score_state + (kRatio + local_pos) * kWidth;
        copy_tile_to_state<DType, kItemPerThread, kHeadDim, kNumRow>(kv_state_row, kv_row, kNumRow,
                                                                     kWidth, kv_stride, tidx);
        add_copy_tile_to_state<DType, kItemPerThread, kHeadDim, kNumRow>(
            score_state_row, score_row, ape_row, kNumRow, kWidth, kv_stride, kWidth, tidx);
        // copy down right state
        copy_tile_to_state<DType, kItemPerThread, kHeadDim, kNumRow>(
            kv_state_row + kHeadDim, kv_row + kHeadDim, kNumRow, kWidth, kv_stride, tidx);
        add_copy_tile_to_state<DType, kItemPerThread, kHeadDim, kNumRow>(
            score_state_row + kHeadDim, score_row + kHeadDim, ape_row + kHeadDim, kNumRow, kWidth,
            kv_stride, kWidth, tidx);
      } else {
        auto* kv_state_row = kv_state + local_pos * kWidth;
        auto* score_state_row = score_state + local_pos * kWidth;
        copy_tile_to_state<DType, kItemPerThread, kHeadDim, kNumRow>(kv_state_row, kv_row, kNumRow,
                                                                     kWidth, kv_stride, tidx);
        add_copy_tile_to_state<DType, kItemPerThread, kHeadDim, kNumRow>(
            score_state_row, score_row, ape_row, kNumRow, kWidth, kv_stride, kWidth, tidx);
      }

      if ((cur_pos + 1) % kRatio == 0) {  // should compress
        auto* out_ptr = compressed_kv_ptr + cu_compressed_seqlen * kHeadDim;
        if constexpr (kOverlap) {
          if (cur_pos >= kRatio) {
            online_compress<DType, kItemPerThread, kRatio, kHeadDim, kWidth, kOverlap, false,
                            false>(out_ptr, kv_state, score_state, ape_ptr, kWidth, tidx);
          } else {
            // first compress block of whole request, should not add upper block
            online_compress<DType, kItemPerThread, kRatio, kHeadDim, kWidth, false, false,
                            false>(  // act like not overlap
                out_ptr, kv_state + kRatio * kWidth + kHeadDim,
                score_state + kRatio * kWidth + kHeadDim, ape_ptr + kHeadDim, kWidth, tidx);
          }

          // copy down left state to upper left state
          copy_tile_to_state<DType, kItemPerThread, kHeadDim, kRatio>(
              kv_state, kv_state + kRatio * kWidth, kRatio, kWidth, kWidth, tidx);
          copy_tile_to_state<DType, kItemPerThread, kHeadDim, kRatio>(
              score_state, score_state + kRatio * kWidth, kRatio, kWidth, kWidth, tidx);
        } else {
          online_compress<DType, kItemPerThread, kRatio, kHeadDim, kWidth, kOverlap, false, false>(
              out_ptr, kv_state, score_state, ape_ptr, kWidth, tidx);
        }
      }
    }

  } else {
    if (should_compress) {
      auto* out_ptr = compressed_kv_ptr + (cu_compressed_seqlen + icompress) * kHeadDim;
      if constexpr (kOverlap) {
        if (icompress == 0) {  // first compress block of this chunk
          if (start_pos_ptr[batch_id] == 0) {
            // first compress block of whole request, should not add upper block
            online_compress<DType, kItemPerThread, kRatio, kHeadDim, kWidth,
                            false>(  // act like not overlap
                out_ptr, kv + kHeadDim, score + kHeadDim, ape_ptr + kHeadDim, kv_stride, tidx);
          } else {  // chunk prefill
            // compress with upper states
            online_compress<DType, kItemPerThread, kRatio, kHeadDim, kWidth, kOverlap, false>(
                out_ptr, kv_state, score_state, kv + kHeadDim, score + kHeadDim, ape_ptr, kWidth,
                kv_stride, tidx);
          }

          // copy upper left state from the last compressed block
          int icompress_last = (cu_seqlens_ptr[batch_id + 1] - cu_seqlens_ptr[batch_id]) / kRatio;
          auto* kv_upper = kv_ptr + (cu_seqlens_ptr[batch_id] + (icompress_last - 1) * kRatio) *
                                        (int64_t)kv_stride;
          auto* score_upper =
              score_ptr +
              (cu_seqlens_ptr[batch_id] + (icompress_last - 1) * kRatio) * (int64_t)kv_stride;
          copy_tile_to_state<DType, kItemPerThread, kHeadDim, kRatio>(kv_state, kv_upper, kRatio,
                                                                      kWidth, kv_stride, tidx);
          add_copy_tile_to_state<DType, kItemPerThread, kHeadDim, kRatio>(
              score_state, score_upper, ape_ptr, kRatio, kWidth, kv_stride, kWidth, tidx);

        } else {  // not first compress block, give the upper block ptr
          auto* kv_upper =
              kv_ptr + (cu_seqlens_ptr[batch_id] + (icompress - 1) * kRatio) * (int64_t)kv_stride;
          auto* score_upper = score_ptr + (cu_seqlens_ptr[batch_id] + (icompress - 1) * kRatio) *
                                              (int64_t)kv_stride;
          online_compress<DType, kItemPerThread, kRatio, kHeadDim, kWidth, kOverlap>(
              out_ptr, kv_upper, score_upper, ape_ptr, kv_stride, tidx);
        }
      } else {  // not overlap
        online_compress<DType, kItemPerThread, kRatio, kHeadDim, kWidth, kOverlap>(
            out_ptr, kv, score, ape_ptr, kv_stride, tidx);
      }
    } else {
      int remainder = seqlen % kRatio;

      if constexpr (kOverlap) {
        // copy down left state
        copy_tile_to_state<DType, kItemPerThread, kHeadDim, kRatio>(
            kv_state + kRatio * kWidth, kv, remainder, kWidth, kv_stride, tidx);
        add_copy_tile_to_state<DType, kItemPerThread, kHeadDim, kRatio>(
            score_state + kRatio * kWidth, score, ape_ptr, remainder, kWidth, kv_stride, kWidth,
            tidx);
        // copy down right state
        copy_tile_to_state<DType, kItemPerThread, kHeadDim, kRatio>(
            kv_state + kRatio * kWidth + kHeadDim, kv + kHeadDim, remainder, kWidth, kv_stride,
            tidx);
        add_copy_tile_to_state<DType, kItemPerThread, kHeadDim, kRatio>(
            score_state + kRatio * kWidth + kHeadDim, score + kHeadDim, ape_ptr + kHeadDim,
            remainder, kWidth, kv_stride, kWidth, tidx);
      } else {  // not overlap
        copy_tile_to_state<DType, kItemPerThread, kHeadDim, kRatio>(kv_state, kv, remainder, kWidth,
                                                                    kv_stride, tidx);
        add_copy_tile_to_state<DType, kItemPerThread, kHeadDim, kRatio>(
            score_state, score, ape_ptr, remainder, kWidth, kv_stride, kWidth, tidx);
      }
    }
  }
}

}  // namespace kernels

bool kv_compressor_fp32_async(float* compressed_kv_ptr, const float* kv_ptr, const float* score_ptr,
                              const int* cu_seqlens_ptr, const int* cu_compressed_seqlens_ptr,
                              float* kv_states_ptr, float* score_states_ptr,
                              const int* state_index_ptr, const int* start_pos_ptr,
                              const float* ape_ptr, int num_batch, int total_seqlen, int kv_stride,
                              int ratio, bool overlap, int head_dim, bool is_prefill,
                              cudaStream_t stream) {
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
                                         state_index_ptr, start_pos_ptr, ape_ptr, kv_stride,
                                         num_batch, total_seqlen);
      } else if (head_dim == 512) {
        constexpr int kItemPerThread = 4;
        constexpr int kHeadDim = 512;
        dim3 block(128);
        dim3 grid(total_seqlen / kRatio + num_batch);  // some unused blocks in tail, but it's ok
        kernels::kv_compressor_prefill<DType, kItemPerThread, kRatio, kHeadDim, kOverlap>
            <<<grid, block, 0, stream>>>(compressed_kv_ptr, kv_ptr, score_ptr, cu_seqlens_ptr,
                                         cu_compressed_seqlens_ptr, kv_states_ptr, score_states_ptr,
                                         state_index_ptr, start_pos_ptr, ape_ptr, kv_stride,
                                         num_batch, total_seqlen);
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
                                       state_index_ptr, start_pos_ptr, ape_ptr, kv_stride,
                                       num_batch, total_seqlen);
    }
    return true;
  } else {
    return false;
  }
}
}  // namespace compressor
}  // namespace hpc
