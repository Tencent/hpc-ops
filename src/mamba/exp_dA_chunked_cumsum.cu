// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "src/mamba/selective_state_scan.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace mamba {
namespace kernels {

template <int kChunkSize, typename T, int kTmaDescCount, typename TmaZ, typename TmaX,
          typename TmaB, typename TmaC, typename TmaXS, typename TmaQYS, typename TmaKYS,
          typename TmaCQ, typename TmaBK, typename TmaXV, typename TmaY>
__global__ void exp_dA_chunked_cumsum(const vec_t<cute::TmaDescriptor, kTmaDescCount> td_global,
                                      cute::TmaDescriptor* tensormaps, float* yscale_ptr,
                                      float* xscale_ptr, const T* y_ptr, const T* zxbcdt,
                                      const int* split_metadata, const float* A_ptr,
                                      const float* dt_bias_ptr, uint32_t nheads, uint32_t head_dim,
                                      uint32_t ngroups, uint32_t dstate, uint32_t zxbcdt_row_stride,
                                      uint32_t dt_offset, uint32_t padded_seqlen_stride) {
  int ichunk = blockIdx.x;
  int iwarp = threadIdx.x / 32;
  int ilane = threadIdx.x % 32;
  int ihead = blockIdx.y;
  int ibatch = blockIdx.z;
  int num_batch = gridDim.z;

  const auto* cu_seqlens = split_metadata;
  const auto* cumsum_padded_seqlens = split_metadata + num_batch + 1;
  const auto* seqlens = split_metadata + 3 * (num_batch + 1);

  int ibatch_seq_start = cu_seqlens[ibatch];
  int ibatch_padded_start = cumsum_padded_seqlens[ibatch];
  uint32_t seqlen = seqlens[ibatch];
  int ichunk_seq_start = ichunk * kChunkSize;
  typedef cub::BlockScan<float, kChunkSize> BlockScan;

  __shared__ cute::TmaDescriptor smem_tma_desc[kTmaDescCount] alignas(128);
  __shared__ typename BlockScan::TempStorage temp_storage;
  BlockScan block_scan(temp_storage);

  if (iwarp < kChunkSize / 32) {
    if (ichunk_seq_start >= seqlen) {
      return;
    }

    int iseq_start = ibatch_seq_start + ichunk_seq_start + threadIdx.x;
    int ibatch_seq_end = ibatch_seq_start + seqlens[ibatch];
    int ipadded_seq_start = ibatch_padded_start + ichunk_seq_start + threadIdx.x;

    float A = A_ptr[ihead];
    float dt_bias = dt_bias_ptr[ihead];
    vec_t<float, 1> dt;
    float dA_cumsum = 0;

    if (iseq_start < ibatch_seq_end) {
      int idx = iseq_start * zxbcdt_row_stride + dt_offset + ihead;
      dt = to<float>(load<T, 1>(zxbcdt + idx));
      dt[0] += dt_bias;
      dt[0] = softplus(dt[0]);
      dA_cumsum = dt[0] * A;
    }

    block_scan.InclusiveSum(dA_cumsum, dA_cumsum);

    if (iseq_start < ibatch_seq_end) {
      store(yscale_ptr + ihead * padded_seqlen_stride + ipadded_seq_start, dA_cumsum);
      store(xscale_ptr + ihead * padded_seqlen_stride + ipadded_seq_start, dt[0]);
    }
  } else {
    if (blockIdx.x == 0) {
      // update tma desc
      for (int i = ihead; i < kTmaDescCount; i += nheads) {
        smem_tma_desc[i] = td_global[i];
      }

      int cu_seqlen = cu_seqlens[ibatch];
      int row_offset = cu_seqlen * zxbcdt_row_stride;
      const T* z_ptr = zxbcdt;
      const T* x_ptr = zxbcdt + nheads * head_dim;
      const T* B_ptr = x_ptr + nheads * head_dim;
      const T* C_ptr = B_ptr + ngroups * dstate;

      const auto* z_batch = z_ptr + row_offset;
      const auto* x_batch = x_ptr + row_offset;
      const auto* B_batch = B_ptr + row_offset;
      const auto* C_batch = C_ptr + row_offset;
      const auto* XS_batch = xscale_ptr + ibatch_padded_start;
      const auto* YS_batch = yscale_ptr + ibatch_padded_start;
      const auto* y_batch = y_ptr + cu_seqlen * nheads * head_dim;

      if (ilane == 0) {
        if (ihead == 0) {
          auto gZ = cute::make_tensor(cute::make_gmem_ptr(z_batch),
                                      cute::make_shape(seqlen, head_dim, nheads),
                                      cute::make_stride(zxbcdt_row_stride, 1, head_dim));
          hpc::update_tma_gtensor<TmaZ>(smem_tma_desc[ihead], gZ);

          auto gBK = cute::make_tensor(cute::make_gmem_ptr(B_batch),
                                       cute::make_shape(seqlen, dstate, ngroups),
                                       cute::make_stride(zxbcdt_row_stride, 1, dstate));
          hpc::update_tma_gtensor<TmaBK>(smem_tma_desc[ihead + nheads], gBK);
        }

        if (ihead == 1) {
          auto gX = cute::make_tensor(cute::make_gmem_ptr(y_batch),
                                      cute::make_shape(head_dim, seqlen, nheads),
                                      cute::make_stride(1, nheads * head_dim, head_dim));
          hpc::update_tma_gtensor<TmaX>(smem_tma_desc[ihead], gX);

          auto gXV = cute::make_tensor(cute::make_gmem_ptr(x_batch),
                                       cute::make_shape(head_dim, seqlen, nheads),
                                       cute::make_stride(1, zxbcdt_row_stride, head_dim));
          hpc::update_tma_gtensor<TmaXV>(smem_tma_desc[ihead + nheads], gXV);
        }

        if (ihead == 2) {
          auto gB = cute::make_tensor(cute::make_gmem_ptr(B_batch),
                                      cute::make_shape(dstate, seqlen, ngroups),
                                      cute::make_stride(1, zxbcdt_row_stride, dstate));
          hpc::update_tma_gtensor<TmaB>(smem_tma_desc[ihead], gB);
          auto gY = cute::make_tensor(cute::make_gmem_ptr(y_batch),
                                      cute::make_shape(seqlen, head_dim, nheads),
                                      cute::make_stride(nheads * head_dim, 1, head_dim));
          hpc::update_tma_gtensor<TmaY>(smem_tma_desc[ihead + nheads], gY);
        }

        if (ihead == 3) {
          auto gC = cute::make_tensor(cute::make_gmem_ptr(C_batch),
                                      cute::make_shape(seqlen, dstate, ngroups),
                                      cute::make_stride(zxbcdt_row_stride, 1, dstate));
          hpc::update_tma_gtensor<TmaC>(smem_tma_desc[ihead], gC);
        }

        if (ihead == 4) {
          auto gXS =
              cute::make_tensor(cute::make_gmem_ptr(XS_batch), cute::make_shape(seqlen, nheads),
                                cute::make_stride(1, padded_seqlen_stride));
          hpc::update_tma_gtensor<TmaXS>(smem_tma_desc[ihead], gXS);
        }

        if (ihead == 5) {
          auto gQYS =
              cute::make_tensor(cute::make_gmem_ptr(YS_batch), cute::make_shape(seqlen, nheads),
                                cute::make_stride(1, padded_seqlen_stride));
          hpc::update_tma_gtensor<TmaQYS>(smem_tma_desc[ihead], gQYS);
        }

        if (ihead == 6) {
          auto gKYS =
              cute::make_tensor(cute::make_gmem_ptr(YS_batch), cute::make_shape(seqlen, nheads),
                                cute::make_stride(1, padded_seqlen_stride));
          hpc::update_tma_gtensor<TmaKYS>(smem_tma_desc[ihead], gKYS);
        }

        if (ihead == 7) {
          auto gCQ = cute::make_tensor(cute::make_gmem_ptr(C_batch),
                                       cute::make_shape(seqlen, dstate, ngroups),
                                       cute::make_stride(zxbcdt_row_stride, 1, dstate));
          hpc::update_tma_gtensor<TmaCQ>(smem_tma_desc[ihead], gCQ);
        }
      }
      __syncwarp();
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
      for (int i = ihead; i < kTmaDescCount; i += nheads) {
        cute::tma_descriptor_cp_fence_release(&tensormaps[kTmaDescCount * ibatch + i],
                                              smem_tma_desc[i]);
      }
    }
  }
}

}  // namespace kernels

bool exp_dA_chunked_cumsum_async(float* yscale_ptr, float* xscale_ptr, void* tma_desc_ptr,
                                 const void* y_ptr, const void* zxbcdt_ptr, const float* A_ptr,
                                 const float* bias_ptr, const int* split_metadata_ptr,
                                 int chunk_size, int max_chunks, int batch_size, int nheads,
                                 int head_dim, int ngroups, int dstate, int zxbcdt_row_stride,
                                 int dt_offset, int padded_seqlen_stride, int tma_desc_count,
                                 cudaStream_t stream) {
  using namespace cute;  // NOLINT

  constexpr int kTileM = 64;
  constexpr int kTileN = 80;
  constexpr int kTileK = 16;
  constexpr int kStage = 8;

  constexpr int kTileM2 = 64;
  constexpr int kTileK2 = 64;
  constexpr int kStage2 = 2;

  constexpr int kTileM3 = 64;
  constexpr int kTileN3 = 64;
  constexpr int kTileV = 80;
  constexpr int kTileK3 = 128;

  constexpr int kChunkSize = 256;
  constexpr int kTmaDescCount = 11;

  if (chunk_size != kChunkSize || tma_desc_count != kTmaDescCount) {
    printf("exp_dA_chunked_cumsum_async Only support chunk_size:%d, tma_desc_count:%d\n",
           kChunkSize, kTmaDescCount);
    return false;
  }

  // gTensor
  auto Z =
      cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<const cute::bfloat16_t*>(zxbcdt_ptr)),
                        cute::make_shape(kChunkSize, head_dim, nheads),
                        cute::make_stride(zxbcdt_row_stride, cute::Int<1>{}, head_dim));
  auto X =
      cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<const cute::bfloat16_t*>(zxbcdt_ptr)),
                        cute::make_shape(head_dim, kChunkSize, nheads),
                        cute::make_stride(cute::Int<1>{}, nheads * head_dim, head_dim));
  auto B =
      cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<const cute::bfloat16_t*>(zxbcdt_ptr)),
                        cute::make_shape(dstate, kChunkSize, ngroups),
                        cute::make_stride(cute::Int<1>{}, zxbcdt_row_stride, dstate));
  auto C =
      cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<const cute::bfloat16_t*>(zxbcdt_ptr)),
                        cute::make_shape(kChunkSize, dstate, ngroups),
                        cute::make_stride(zxbcdt_row_stride, cute::Int<1>{}, dstate));
  auto XS = cute::make_tensor(cute::make_gmem_ptr(xscale_ptr), cute::make_shape(kChunkSize, nheads),
                              cute::make_stride(cute::Int<1>{}, padded_seqlen_stride));
  auto YS = cute::make_tensor(cute::make_gmem_ptr(yscale_ptr), cute::make_shape(kChunkSize, nheads),
                              cute::make_stride(cute::Int<1>{}, padded_seqlen_stride));
  auto CQ =
      cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<const cute::bfloat16_t*>(zxbcdt_ptr)),
                        cute::make_shape(kChunkSize, dstate, ngroups),
                        cute::make_stride(zxbcdt_row_stride, cute::Int<1>{}, dstate));
  auto BK =
      cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<const cute::bfloat16_t*>(zxbcdt_ptr)),
                        cute::make_shape(kChunkSize, dstate, ngroups),
                        cute::make_stride(zxbcdt_row_stride, cute::Int<1>{}, dstate));
  auto XV =
      cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<const cute::bfloat16_t*>(zxbcdt_ptr)),
                        cute::make_shape(head_dim, kChunkSize, nheads),
                        cute::make_stride(cute::Int<1>{}, zxbcdt_row_stride, head_dim));

  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<const cute::bfloat16_t*>(y_ptr)),
                       make_shape(kChunkSize, head_dim, nheads),
                       make_stride(nheads * head_dim, Int<1>{}, head_dim));

  // smem layout
  auto sZ_layout = cute::tile_to_shape(cute::GMMA::Layout_K_SW32_Atom<cute::bfloat16_t>{},
                                       cute::make_shape(cute::Int<kTileN3>{}, cute::Int<kTileV>{}));
  auto sX_layout = cute::tile_to_shape(
      cute::GMMA::Layout_MN_SW32_Atom<cute::bfloat16_t>{},
      cute::make_shape(cute::Int<kTileN>{}, cute::Int<kTileK>{}, cute::Int<kStage>{}));
  auto sB_layout = cute::tile_to_shape(
      cute::GMMA::Layout_MN_SW128_Atom<cute::bfloat16_t>{},
      cute::make_shape(cute::Int<kTileM>{}, cute::Int<kTileK>{}, cute::Int<kStage>{}));
  auto sC_layout = cute::tile_to_shape(
      cute::GMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{},
      cute::make_shape(cute::Int<kTileM2>{}, cute::Int<kTileK2>{}, cute::Int<kStage2>{}));
  auto sQS_layout =
      cute::make_layout(cute::make_shape(cute::Int<kTileM3>{}), cute::make_stride(cute::Int<1>{}));
  auto sKS_layout =
      cute::make_layout(cute::make_shape(cute::Int<kTileN3>{}), cute::make_stride(cute::Int<1>{}));
  auto sCQ_layout =
      cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{},
                          cute::make_shape(cute::Int<kTileM3>{}, cute::Int<kTileK3>{}));
  auto sBK_layout =
      cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{},
                          cute::make_shape(cute::Int<kTileN3>{}, cute::Int<kTileK3>{}));
  auto sXV_layout =
      cute::tile_to_shape(cute::GMMA::Layout_MN_SW32_Atom<cute::bfloat16_t>{},
                          cute::make_shape(cute::Int<kTileV>{}, cute::Int<kTileN3>{}));
  auto sY_layout = cute::tile_to_shape(cute::GMMA::Layout_K_SW32_Atom<cute::bfloat16_t>{},
                                       cute::make_shape(cute::Int<kTileM3>{}, cute::Int<kTileV>{}));

  // TMA
  auto tma_z = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, Z, sZ_layout);
  auto tma_x = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, X, cute::take<0, 2>(sX_layout));
  auto tma_b = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, B, cute::take<0, 2>(sB_layout));
  auto tma_c = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, C, cute::take<0, 2>(sC_layout));
  auto tma_xs = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, XS, sKS_layout);
  auto tma_qys = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, YS, sQS_layout);
  auto tma_kys = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, YS, sKS_layout);
  auto tma_cq = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, CQ, sCQ_layout);
  auto tma_bk = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, BK, sBK_layout);
  auto tma_xv = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, XV, sXV_layout);
  auto tma_y = cute::make_tma_copy(cute::SM90_TMA_STORE{}, Y, sY_layout);

  vec_t<cute::TmaDescriptor, kTmaDescCount> td_global{
      *tma_z.get_tma_descriptor(),   *tma_x.get_tma_descriptor(),  *tma_b.get_tma_descriptor(),
      *tma_c.get_tma_descriptor(),   *tma_xs.get_tma_descriptor(), *tma_qys.get_tma_descriptor(),
      *tma_kys.get_tma_descriptor(), *tma_cq.get_tma_descriptor(), *tma_bk.get_tma_descriptor(),
      *tma_xv.get_tma_descriptor(),  *tma_y.get_tma_descriptor(),
  };

  dim3 block(kChunkSize + 32);
  dim3 grid(max_chunks, nheads, batch_size);

  kernels::exp_dA_chunked_cumsum<
      kChunkSize, __nv_bfloat16, kTmaDescCount, decltype(tma_z), decltype(tma_x), decltype(tma_b),
      decltype(tma_c), decltype(tma_xs), decltype(tma_qys), decltype(tma_kys), decltype(tma_cq),
      decltype(tma_bk), decltype(tma_xv), decltype(tma_y)><<<grid, block, 0, stream>>>(
      td_global, reinterpret_cast<cute::TmaDescriptor*>(tma_desc_ptr), yscale_ptr, xscale_ptr,
      reinterpret_cast<const __nv_bfloat16*>(y_ptr),
      reinterpret_cast<const __nv_bfloat16*>(zxbcdt_ptr), split_metadata_ptr, A_ptr, bias_ptr,
      nheads, head_dim, ngroups, dstate, zxbcdt_row_stride, dt_offset, padded_seqlen_stride);
  return true;
}

}  // namespace mamba
}  // namespace hpc
