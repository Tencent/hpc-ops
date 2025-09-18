// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include "cute/tensor.hpp"
#include "src/mamba/selective_state_scan.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace mamba {
namespace kernels {

template <int kThreadsPerBlock, int kChunkSize, int Stage, typename T>
__global__ void chunk_states_passing(T* chunked_states_cumsum_ptr, float* ssm_states_ptr,
                                     const int* indices_ptr, const float* chunked_states_ptr,
                                     const float* yscale_ptr, const int* split_metadata_ptr,
                                     int batch_size, int total_padded_seqlen, int chunk_size,
                                     int nchunksxheaddimxdstate, int headdimxdstate, int dstate) {
  constexpr int kWarpSize = 32;
  constexpr int kRowsPerBlock = kThreadsPerBlock / kWarpSize;
  constexpr int kItemsPerThread = 4;

  const auto* cu_padded_seqlens_ptr = split_metadata_ptr + 1 * (batch_size + 1);
  const auto* cu_chunks_ptr = split_metadata_ptr + 2 * (batch_size + 1);
  const auto* seqlens_ptr = split_metadata_ptr + 3 * (batch_size + 1);
  const auto* nchunks_ptr = split_metadata_ptr + 5 * (batch_size + 1);

  int nbatch = gridDim.z;
  int nheads = gridDim.y;
  int ihead = blockIdx.y;
  int ibatch = blockIdx.z;
  int iwarp = threadIdx.x / 32;
  int ilane = threadIdx.x % 32;
  int irow = blockIdx.x * kRowsPerBlock + iwarp;
  int icol = ilane * kItemsPerThread;
  int nchunks = nchunks_ptr[ibatch];
  int ichunk_start = cu_chunks_ptr[ibatch];
  int seqlen = seqlens_ptr[ibatch];
  int cu_padded_seqlen = cu_padded_seqlens_ptr[ibatch];

  const auto* yscale_batch = yscale_ptr + ihead * total_padded_seqlen + cu_padded_seqlen;
  const auto* chunked_states_batch = chunked_states_ptr + ihead * nchunksxheaddimxdstate +
                                     ichunk_start * headdimxdstate + irow * dstate + icol;
  auto* chunked_states_cumsum_batch =
      chunked_states_cumsum_ptr + ihead * (nchunksxheaddimxdstate - nbatch * headdimxdstate) +
      (ichunk_start - ibatch) * headdimxdstate + irow * dstate + icol;
  auto* ssm_states_batch = ssm_states_ptr + indices_ptr[ibatch] * headdimxdstate * nheads +
                           ihead * headdimxdstate + irow * dstate + icol;

  vec_t<float, kItemsPerThread> pre_chunked_states = load<float, 4>(chunked_states_batch);
  vec_t<float, kItemsPerThread> chunked_states[Stage];
  float yscale[Stage];

  for (int ichunk = 1; ichunk < (nchunks + Stage - 1) / Stage * Stage; ichunk += Stage) {
#pragma unroll
    for (int si = 0; si < Stage; si++) {
      int iload = ichunk + si;
      if (iload < nchunks) {
        chunked_states[si] = load<float, 4>(chunked_states_batch + iload * headdimxdstate);
        yscale[si] = expf_ftz(yscale_batch[umin((iload + 1) * chunk_size, seqlen) - 1]);
      }
    }

#pragma unroll
    for (int si = 0; si < Stage; si++) {
      int istore = ichunk + si;
      if (istore < nchunks) {
        if (istore <= nchunks - 1) {
          if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            store(chunked_states_cumsum_batch + (istore - 1) * headdimxdstate,
                  to<__nv_bfloat162>(pre_chunked_states));
          }
        }
#pragma unroll
        for (int i = 0; i < kItemsPerThread; i++) {
          chunked_states[si][i] = chunked_states[si][i] + pre_chunked_states[i] * yscale[si];
        }
        pre_chunked_states = chunked_states[si];
      }
    }
  }

  store(ssm_states_batch, pre_chunked_states);
}

}  // namespace kernels

bool chunk_states_passing_async(void* chunked_states_cumsum_ptr, float* ssm_states,
                                const int* indices_ptr, const float* chunked_states_ptr,
                                const float* yscale_ptr, const int* split_metadata_ptr,
                                int chunk_size, int total_padded_seqlen, int total_chunks,
                                int batch_size, int nheads, int head_dim, int dstate,
                                cudaStream_t stream) {
  constexpr int kThreadsPerBlock = 128;
  constexpr int kRowsPerBlock = kThreadsPerBlock / 32;
  constexpr int kStage = 8;
  constexpr int kChunkSize = 256;

  if (chunk_size != kChunkSize) {
    printf("chunk_states_passing_async Only support chunk_size:%d\n", kChunkSize);
    return false;
  }

  dim3 block(kThreadsPerBlock);
  dim3 grid((head_dim + kRowsPerBlock - 1) / kRowsPerBlock, nheads, batch_size);

  int headdimxdstate = head_dim * dstate;
  kernels::chunk_states_passing<kThreadsPerBlock, kChunkSize, kStage><<<grid, block, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(chunked_states_cumsum_ptr), ssm_states, indices_ptr,
      chunked_states_ptr, yscale_ptr, split_metadata_ptr, batch_size, total_padded_seqlen,
      chunk_size, total_chunks * headdimxdstate, headdimxdstate, dstate);
  return true;
}

}  // namespace mamba
}  // namespace hpc
