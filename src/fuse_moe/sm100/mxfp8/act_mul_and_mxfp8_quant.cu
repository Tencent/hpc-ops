// Copyright 2026 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdio.h>

#include <type_traits>

#include "src/fuse_moe/sm100/mxfp8/fuse_moe_mxfp8.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace fuse_moe {
namespace kernels_mxfp8 {

// UE8M0 scale calculation: SF = 2^k where k = ceil(log2(absmax) - log2(448)).
__device__ __forceinline__ uint8_t fp32_absmax_to_ue8m0(float absmax) {
  if (absmax == 0.f) {
    return 0;
  }
  uint32_t bits = __float_as_uint(absmax);
  int exp_biased = (bits >> 23) & 0xFF;
  uint32_t mant = bits & 0x7FFFFF;
  int sf_bits = exp_biased - 8 + (mant > 0x600000u ? 1 : 0);
  if (sf_bits < 0) {
    sf_bits = 0;
  }
  if (sf_bits > 255) {
    sf_bits = 255;
  }
  return static_cast<uint8_t>(sf_bits);
}

template <int kThreadPerBlock, int kTileM, int kSfLanes, bool kUsePDL = false>
__global__ void act_mul_mxfp8_kernel(const __nv_bfloat16 *__restrict__ gate_up_ptr,
                                     __nv_fp8_e4m3 *__restrict__ out_ptr,
                                     uint8_t *__restrict__ out_scale_packed_ptr,
                                     const int *__restrict__ valid_row_range_ptr,
                                     const int *__restrict__ cu_seqlens_ptr,
                                     const int *__restrict__ cu_tiles_ptr, int num_expert_local,
                                     int intermediate_size, int k_sf_tiles) {
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  constexpr int kSfVec = 32;
  constexpr int kElemsPerIter = 8;
  constexpr int kNumIter = kSfVec / kElemsPerIter;  // = 4

  const int K_sf = intermediate_size / kSfVec;

  int irow = blockIdx.y;
  int ikblock = blockIdx.x * kThreadPerBlock + threadIdx.x;

  // Load cu_seqlens and cu_tiles
  extern __shared__ int smem[];
  int *cu_seqlens_shm = smem;
  int *cu_tiles_shm = smem + num_expert_local + 1;
  for (int i = threadIdx.x; i <= num_expert_local; i += kThreadPerBlock) {
    cu_seqlens_shm[i] = cu_seqlens_ptr[i];
    cu_tiles_shm[i] = cu_tiles_ptr[i];
  }
  __syncthreads();

  if (ikblock >= K_sf) {
    return;
  }

  int valid_rows = *valid_row_range_ptr;
  if (irow >= valid_rows) {
    return;
  }

  const int k_base = ikblock * kSfVec;
  const __nv_bfloat16 *row_gate = gate_up_ptr + static_cast<uint64_t>(irow) * intermediate_size * 2;
  const __nv_bfloat16 *row_up = row_gate + intermediate_size;
  __nv_fp8_e4m3 *out_row = out_ptr + static_cast<uint64_t>(irow) * intermediate_size + k_base;

  using T = __nv_bfloat162;

  // Pass 1: compute y = silu(gate) * up, find absmax over 32 elements
  float y_buf[kSfVec];
  float absmax = 0.f;

#pragma unroll
  for (int iter = 0; iter < kNumIter; ++iter) {
    int offset = k_base + iter * kElemsPerIter;
    auto gate = to<float>(load<T, 4>(row_gate + offset));
    auto up = to<float>(load<T, 4>(row_up + offset));
#pragma unroll
    for (int i = 0; i < kElemsPerIter; ++i) {
      float v = silu(gate[i]) * up[i];
      y_buf[iter * kElemsPerIter + i] = v;
      absmax = fmaxf(absmax, fabsf(v));
    }
  }

  // Compute UE8M0 scale from absmax
  uint8_t sf_bits = fp32_absmax_to_ue8m0(absmax);

  float inv_sf;
  if (sf_bits == 0) {
    inv_sf = 0.f;
  } else {
    float sf = exp2f_ftz(static_cast<float>(sf_bits) - 127.f);
    inv_sf = 1.f / sf;
  }

  // Pass 2: quantize 32 fp8 elements and store as 2 x 16-byte writes
  vec_t<__nv_fp8_e4m3, 16> out_fp8_lo, out_fp8_hi;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out_fp8_lo[i] = __nv_fp8_e4m3(y_buf[i] * inv_sf);
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out_fp8_hi[i] = __nv_fp8_e4m3(y_buf[16 + i] * inv_sf);
  }
  store<__nv_fp8_e4m3, 16>(out_row, out_fp8_lo);
  store<__nv_fp8_e4m3, 16>(out_row + 16, out_fp8_hi);

  // Store scale directly to packed SFA layout (fused prepack)
  // Find which expert group this row belongs to (binary search in cu_seqlens)
  int lo = 0, hi = num_expert_local;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (cu_seqlens_shm[mid] <= irow) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  int group = lo;

  int local_row = irow - cu_seqlens_shm[group];
  int tile_in_group = local_row / kTileM;
  int row_in_tile = local_row - tile_in_group * kTileM;
  int global_tile = cu_tiles_shm[group] + tile_in_group;

  // Packed layout addressing:
  //   For kSfLanes=32: row_in_tile maps to (lane, q) as:
  //     lane = row_in_tile % 32,  q = row_in_tile / 32
  //   For kSfLanes=64: the mapping is interleaved:
  //     row_in_tile < 128: lane = row_in_tile % 32,      q = row_in_tile / 32
  //     row_in_tile >= 128: lane = row_in_tile % 32 + 32, q = (row_in_tile - 128) / 32
  //   dst_v index = global_tile * kSfLanes * k_sf_tiles + kt * kSfLanes + lane
  //   byte in vec = q * 4 + k_within
  int lane, q;
  if constexpr (kSfLanes == 32) {
    lane = row_in_tile & 31;
    q = row_in_tile >> 5;
  } else {
    // kSfLanes == 64
    if (row_in_tile < 128) {
      lane = row_in_tile & 31;
      q = row_in_tile >> 5;
    } else {
      lane = (row_in_tile & 31) + 32;
      q = (row_in_tile - 128) >> 5;
    }
  }
  int kt = ikblock >> 2;       // ikblock / 4
  int k_within = ikblock & 3;  // ikblock % 4

  int byte_offset = global_tile * (kSfLanes * k_sf_tiles * 16) + kt * (kSfLanes * 16) + lane * 16 +
                    q * 4 + k_within;
  out_scale_packed_ptr[byte_offset] = sf_bits;

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels_mxfp8

void act_mul_and_mxfp8_quant_async(void *out_ptr, void *out_scale_packed_ptr,
                                   const void *gate_up_ptr, const void *valid_row_range_ptr,
                                   const void *cu_seqlens_ptr, const void *cu_tiles_ptr,
                                   int num_expert_local, int total_num_seq, int intermediate_size,
                                   int kTileM, int k_sf_tiles, cudaStream_t stream, bool use_pdl) {
  constexpr int kSfVec = 32;
  constexpr int kThreadPerBlock = 128;

  int K_sf = intermediate_size / kSfVec;
  int gridx = (K_sf + kThreadPerBlock - 1) / kThreadPerBlock;

  dim3 grid(gridx, total_num_seq);
  dim3 block(kThreadPerBlock);
  int smem_size = 2 * (num_expert_local + 1) * sizeof(int);

  bool is_smallm = (kTileM <= 128);

  auto launch = [&](auto tilem_tag, auto sflanes_tag) {
    constexpr int KTM = decltype(tilem_tag)::value;
    constexpr int SFLANES = decltype(sflanes_tag)::value;
    if (use_pdl) {
      auto kernel = kernels_mxfp8::act_mul_mxfp8_kernel<kThreadPerBlock, KTM, SFLANES, true>;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = grid;
      cfg.blockDim = block;
      cfg.dynamicSmemBytes = smem_size;
      cfg.stream = stream;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      cudaLaunchKernelEx(&cfg, kernel, reinterpret_cast<const __nv_bfloat16 *>(gate_up_ptr),
                         reinterpret_cast<__nv_fp8_e4m3 *>(out_ptr),
                         reinterpret_cast<uint8_t *>(out_scale_packed_ptr),
                         reinterpret_cast<const int *>(valid_row_range_ptr),
                         reinterpret_cast<const int *>(cu_seqlens_ptr),
                         reinterpret_cast<const int *>(cu_tiles_ptr), num_expert_local,
                         intermediate_size, k_sf_tiles);
    } else {
      auto kernel = kernels_mxfp8::act_mul_mxfp8_kernel<kThreadPerBlock, KTM, SFLANES, false>;
      kernel<<<grid, block, smem_size, stream>>>(
          reinterpret_cast<const __nv_bfloat16 *>(gate_up_ptr),
          reinterpret_cast<__nv_fp8_e4m3 *>(out_ptr),
          reinterpret_cast<uint8_t *>(out_scale_packed_ptr),
          reinterpret_cast<const int *>(valid_row_range_ptr),
          reinterpret_cast<const int *>(cu_seqlens_ptr),
          reinterpret_cast<const int *>(cu_tiles_ptr), num_expert_local, intermediate_size,
          k_sf_tiles);
    }
  };

  using SF32 = std::integral_constant<int, 32>;
  using SF64 = std::integral_constant<int, 64>;

  if (is_smallm) {
    switch (kTileM) {
      case 16:
        return launch(std::integral_constant<int, 16>{}, SF32{});
      case 32:
        return launch(std::integral_constant<int, 32>{}, SF32{});
      case 48:
        return launch(std::integral_constant<int, 48>{}, SF32{});
      case 64:
        return launch(std::integral_constant<int, 64>{}, SF32{});
      case 96:
        return launch(std::integral_constant<int, 96>{}, SF32{});
      default:
        return launch(std::integral_constant<int, 128>{}, SF32{});
    }
  } else {
    switch (kTileM) {
      case 160:
        return launch(std::integral_constant<int, 160>{}, SF64{});
      case 192:
        return launch(std::integral_constant<int, 192>{}, SF64{});
      default:
        return launch(std::integral_constant<int, 256>{}, SF64{});
    }
  }
}

}  // namespace fuse_moe
}  // namespace hpc
