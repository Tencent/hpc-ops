// Copyright 2026 Tencent

// sm90 host dispatch for the decode-shaped iHC pre path.
//
// Routes by token count. The shared row-per-block kernel
// (fuse_ihc_pre_or_head_kernel, kPreRowsPerBlock = 1) has a grid of exactly
// num_batch, so on an H20's 78 SMs it is latency-flat at ~12-13us from 1 to 64
// tokens; splitting K instead fills the device and is 3.8x faster at one token.
// The two converge once the row grid is large enough, so this path claims
// num_batch < kDecodeTokenThreshold and defers above it.
//
// End-to-end through torch.ops.hpc.fuse_ihc_pre on H20, hidden 6144, bf16,
// CUDA Graph (us per call), against the two Triton implementations:
//
//     tokens    vllm triton    flaggems triton    hpc (this path)
//          1           4.20               3.87               3.18
//          4           4.72               4.30               3.59
//         16           6.64               5.28               5.10
//         32          10.01               6.40               6.47
//         64          16.91               8.52              10.51

#include <cuda_runtime_api.h>

#include <algorithm>
#include <type_traits>

#include "src/iHC/iHC.h"
#include "src/iHC/sm90/ihc_decode.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace iHC {
namespace sm90 {

namespace {

constexpr int kStage1Threads = 128;
constexpr int kStage2Threads = 256;
constexpr int kElementsPerThread = 8;

// Token count at or above which this path declines and the shared
// row-per-block kernel runs instead. Measured end-to-end on H20 under CUDA
// Graph (us per call, hidden 6144):
//
//     tokens    row-per-block    k-split
//          1            12.0        3.18
//         32            12.9        6.47
//         64            12.9       10.51
//        128            19.1       19.10
//        255            32.8       32.77
//        256            33.6       33.60   (parity; declined from here)
//
// The two converge around 128-256 because by then the row grid fills the device
// too. 256 is where the shared kernel is reliably no worse, and it coincides
// with the Triton implementation's own _DECODE_N_THRESHOLD.
constexpr int kDecodeTokenThreshold = 256;

// BLOCK_K by token count: tiny batches need K cut finely to expose enough
// blocks, larger ones prefer fewer and wider chunks because the per-block
// reduction tail stops being free once the machine is full.
//
// Measured on H20 at hidden 6144 under CUDA Graph (us per call), sweeping every
// legal BLOCK_K -- the table follows the measured minimum, not the Triton
// implementation's _DECODE_BLOCK_K, whose stage1 does different per-program work:
//
//     tokens   1024   2048   3072   4096   best
//          1   3.76   4.00   4.71   5.37   1024
//          4   4.24   4.20   5.01   5.55   2048
//         16   5.60   5.41   5.93   6.73   2048
//         32   7.85   7.08   7.45   7.88   2048
//         63  12.47  11.63  11.21  11.09   4096
//
// Each entry must divide hc_mult * hidden_dim and be a multiple of
// kStage1Threads * kElementsPerThread (= 1024); hidden 4096 and 6144 with
// hc_mult 4 give kHCDim 16384 and 24576, both multiples of 1024.
constexpr int decode_block_k(int num_batch) {
  if (num_batch < 2) {
    return 1024;
  }
  if (num_batch < 48) {
    return 2048;
  }
  return 4096;
}

template <int kHiddenDim, int kBlockK>
void launch_decode(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                   const __nv_bfloat16* x_ptr, const float* w_ptr, const float* hc_scale_ptr,
                   const float* hc_base_ptr, int num_batch, float norm_eps, float hc_eps,
                   float magnitude, float* scratch_ptr, cudaStream_t stream, bool use_pdl) {
  constexpr int kHCMult = 4;
  constexpr int kHCDim = kHCMult * kHiddenDim;
  constexpr int kNumSplits = kHCDim / kBlockK;

  auto launch = [&](auto pdl_tag) {
    constexpr bool kUsePDL = decltype(pdl_tag)::value;

    auto stage1 = ihc_pre_stage1_decode_kernel<kHCMult, kHiddenDim, kElementsPerThread,
                                               kStage1Threads, kBlockK, kUsePDL>;
    cudaLaunchConfig_t cfg1{};
    cfg1.gridDim = dim3(num_batch, kNumSplits);
    cfg1.blockDim = dim3(kStage1Threads);
    cfg1.stream = stream;
    cudaLaunchAttribute attr1[1];
    cfg1.numAttrs = 0;
    if constexpr (kUsePDL) {
      attr1[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr1[0].val.programmaticStreamSerializationAllowed = 1;
      cfg1.attrs = attr1;
      cfg1.numAttrs = 1;
    }
    cudaLaunchKernelEx(&cfg1, stage1, scratch_ptr, x_ptr, w_ptr, num_batch);

    // Split the hidden dim across blocks as well as the tokens: at one token a
    // single block doing all 6144 lanes measured 4.9us, against 1.7us spread
    // over six.
    constexpr int kBlockD = kStage2Threads * kElementsPerThread;
    constexpr int kHiddenBlocks = kHiddenDim / kBlockD;

    auto stage2 = ihc_pre_stage2_decode_kernel<kHCMult, kHiddenDim, kElementsPerThread,
                                               kStage2Threads, kNumSplits, kUsePDL>;
    cudaLaunchConfig_t cfg2{};
    cfg2.gridDim = dim3(num_batch, kHiddenBlocks);
    cfg2.blockDim = dim3(kStage2Threads);
    cfg2.stream = stream;
    cudaLaunchAttribute attr2[1];
    cfg2.numAttrs = 0;
    if constexpr (kUsePDL) {
      attr2[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr2[0].val.programmaticStreamSerializationAllowed = 1;
      cfg2.attrs = attr2;
      cfg2.numAttrs = 1;
    }
    cudaLaunchKernelEx(&cfg2, stage2, output_y_ptr, output_H_post_ptr, x_ptr,
                       const_cast<const float*>(scratch_ptr), hc_scale_ptr, hc_base_ptr, num_batch,
                       norm_eps, hc_eps, magnitude);
  };

  if (use_pdl) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

template <int kHiddenDim>
void launch_decode_by_block_k(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                              const __nv_bfloat16* x_ptr, const float* w_ptr,
                              const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                              float norm_eps, float hc_eps, float magnitude, float* scratch_ptr,
                              cudaStream_t stream, bool use_pdl) {
  switch (decode_block_k(num_batch)) {
    case 1024:
      launch_decode<kHiddenDim, 1024>(output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr,
                                      hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude,
                                      scratch_ptr, stream, use_pdl);
      break;
    case 2048:
      launch_decode<kHiddenDim, 2048>(output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr,
                                      hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude,
                                      scratch_ptr, stream, use_pdl);
      break;
    default:
      launch_decode<kHiddenDim, 4096>(output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr,
                                      hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude,
                                      scratch_ptr, stream, use_pdl);
      break;
  }
}

}  // namespace

bool ihc_pre_decode_supported(int num_batch, int hc_mult, int hidden_dim) {
  if (num_batch <= 0 || num_batch >= kDecodeTokenThreshold) {
    return false;
  }
  if (hc_mult != 4) {
    return false;
  }
  return hidden_dim == 4096 || hidden_dim == 6144;
}

size_t ihc_pre_decode_scratch_floats(int num_batch, int hc_mult, int hidden_dim) {
  if (!ihc_pre_decode_supported(num_batch, hc_mult, hidden_dim)) {
    return 0;
  }
  const int num_splits = (hc_mult * hidden_dim) / decode_block_k(num_batch);
  return static_cast<size_t>(num_batch) * num_splits * (1 + 2 * hc_mult);
}

void fuse_ihc_pre_decode_async(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                               const __nv_bfloat16* x_ptr, const float* w_ptr,
                               const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                               int hc_mult, int hidden_dim, float norm_eps, float hc_eps,
                               float magnitude, float* scratch_ptr, cudaStream_t stream,
                               bool use_pdl) {
  if (hidden_dim == 6144) {
    launch_decode_by_block_k<6144>(output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr,
                                   hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, scratch_ptr,
                                   stream, use_pdl);
  } else if (hidden_dim == 4096) {
    launch_decode_by_block_k<4096>(output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr,
                                   hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, scratch_ptr,
                                   stream, use_pdl);
  }
}

}  // namespace sm90
}  // namespace iHC
}  // namespace hpc
