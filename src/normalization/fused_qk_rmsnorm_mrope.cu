// Copyright 2025 hpc-ops authors

#include <cuda.h>

#include "src/normalization/fused_qk_rmsnorm_mrope.h"
#include "src/utils/utils.cuh"
#include "torch/version.h"

namespace hpc {
namespace normalization {

namespace kernels {

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float round_bf16_to_float(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

// Keep inference-time fused RMSNorm bitwise-identical to the PyTorch training
// reference. These helpers intentionally follow PyTorch's operation order and
// intermediate bf16 rounding.
__device__ __forceinline__ float rmsnorm_weight_like_torch(float x, float rms,
                                                           __nv_bfloat16 weight) {
  float normed_bf16 = round_bf16_to_float(__fmul_rn(x, rms));
  return round_bf16_to_float(__fmul_rn(bf16_to_float(weight), normed_bf16));
}

// Match PyTorch's rsqrt path so train/inference comparisons can use exact
// equality instead of tolerances.
__device__ __forceinline__ float rsqrt_like_torch(float x) { return rsqrtf(x); }

// This reduction mirrors PyTorch's CUDA mean implementation so the fused
// inference kernel stays bitwise-identical to the PyTorch training reference.
// The reduction order matters for bf16 inputs because a 1-ULP float difference
// before the final cast can change the resulting bf16 value.
__device__ __forceinline__ float reduce_mean_square_128_like_torch(const __nv_bfloat16* data,
                                                                   int lane) {
  // PyTorch changed the dim=128 CUDA mean path in 2.10: 2.9 does scalar strided
  // loads, while 2.10 vectorizes input loads for this exact size.
  float values[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 10)
    float x = bf16_to_float(data[lane * 4 + i]);
#else
    float x = bf16_to_float(data[lane + i * 32]);
#endif
    values[i] = __fmul_rn(x, x);
  }
#pragma unroll
  for (int i = 1; i < 4; i++) {
    values[0] = __fadd_rn(values[0], values[i]);
  }
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 10)
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    values[0] = __fadd_rn(values[0], __shfl_down_sync(0xffffffff, values[0], offset));
  }
#else
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    values[0] = __fadd_rn(values[0], __shfl_down_sync(0xffffffff, values[0], offset));
  }
#endif
  return __shfl_sync(0xffffffff, values[0], 0);
}

// The _like_torch suffix documents that this helper intentionally matches PyTorch's
// operation order and rounding behavior for bitwise-identical results.
template <int kHalfDim, typename PositionsTensor, typename CosSinCacheTensor>
__device__ __forceinline__ float apply_mrope_element_like_torch(
    const float* smem_row, int elem_idx, int out_token, int mrope_section_h, int mrope_section_w,
    const PositionsTensor& positions, const CosSinCacheTensor& cos_sin_cache) {
  int half_idx = elem_idx % kHalfDim;
  int full_idx = half_idx + kHalfDim;

  int axis = 0;
  if ((half_idx % 3 == 1) && (half_idx < 3 * mrope_section_h)) {
    axis = 1;
  }
  if ((half_idx % 3 == 2) && (half_idx < 3 * mrope_section_w)) {
    axis = 2;
  }

  int64_t pos = positions(axis, out_token);
  float cos_val = cos_sin_cache(pos, half_idx);
  float sin_val = cos_sin_cache(pos, full_idx);

  float half_x = smem_row[half_idx];
  float full_x = smem_row[full_idx];
  bool is_first_half = elem_idx < kHalfDim;
  float lhs = is_first_half ? half_x : full_x;
  float rhs = is_first_half ? -full_x : half_x;
  return __fadd_rn(__fmul_rn(lhs, cos_val), __fmul_rn(rhs, sin_val));
}

template <int kDim>
__global__ void fused_qk_rmsnorm_mrope_kernel(
    __nv_bfloat16* __restrict__ out_q_ptr, __nv_bfloat16* __restrict__ out_k_ptr,
    __nv_bfloat16* __restrict__ out_v_ptr, const __nv_bfloat16* __restrict__ und_qkv_ptr,
    const __nv_bfloat16* __restrict__ gen_qkv_ptr,
    const __nv_bfloat16* __restrict__ und_q_weight_ptr,
    const __nv_bfloat16* __restrict__ und_k_weight_ptr,
    const __nv_bfloat16* __restrict__ gen_q_weight_ptr,
    const __nv_bfloat16* __restrict__ gen_k_weight_ptr, const int64_t* __restrict__ positions_ptr,
    const float* __restrict__ cos_sin_cache_ptr, const int64_t* __restrict__ cat_indices_ptr,
    int und_len, int total_tokens, int num_q_heads, int num_k_heads, int num_v_heads,
    int mrope_section_h, int mrope_section_w, float eps, int cos_sin_cache_stride) {
  constexpr int kN = 16 / sizeof(__nv_bfloat16);
  constexpr int kWarpsPerBlock = 4;
  constexpr int kRowsPerBlock = kWarpsPerBlock;
  constexpr float kInvDim = 1.0f / kDim;
  constexpr int kHalfDim = kDim / 2;

  const int gen_len = total_tokens - und_len;
  const int qkv_stride = num_q_heads * kDim + num_k_heads * kDim + num_v_heads * kDim;
  const int k_offset = num_q_heads * kDim;
  const int v_offset = (num_q_heads + num_k_heads) * kDim;

  auto positions = cute::make_tensor(cute::make_gmem_ptr(positions_ptr),
                                     cute::make_shape(cute::Int<3>{}, total_tokens),
                                     cute::make_stride(total_tokens, cute::Int<1>{}));
  auto cos_sin_cache = cute::make_tensor(cute::make_gmem_ptr(cos_sin_cache_ptr),
                                         cute::make_shape(total_tokens, cos_sin_cache_stride),
                                         cute::make_stride(cos_sin_cache_stride, cute::Int<1>{}));

  auto und_q = cute::make_tensor(cute::make_gmem_ptr(und_qkv_ptr),
                                 cute::make_shape(und_len, num_q_heads, cute::Int<kDim>{}),
                                 cute::make_stride(qkv_stride, cute::Int<kDim>{}, cute::Int<1>{}));
  auto und_k = cute::make_tensor(cute::make_gmem_ptr(und_qkv_ptr + k_offset),
                                 cute::make_shape(und_len, num_k_heads, cute::Int<kDim>{}),
                                 cute::make_stride(qkv_stride, cute::Int<kDim>{}, cute::Int<1>{}));
  auto und_v = cute::make_tensor(cute::make_gmem_ptr(und_qkv_ptr + v_offset),
                                 cute::make_shape(und_len, num_v_heads, cute::Int<kDim>{}),
                                 cute::make_stride(qkv_stride, cute::Int<kDim>{}, cute::Int<1>{}));
  auto gen_q = cute::make_tensor(cute::make_gmem_ptr(gen_qkv_ptr),
                                 cute::make_shape(gen_len, num_q_heads, cute::Int<kDim>{}),
                                 cute::make_stride(qkv_stride, cute::Int<kDim>{}, cute::Int<1>{}));
  auto gen_k = cute::make_tensor(cute::make_gmem_ptr(gen_qkv_ptr + k_offset),
                                 cute::make_shape(gen_len, num_k_heads, cute::Int<kDim>{}),
                                 cute::make_stride(qkv_stride, cute::Int<kDim>{}, cute::Int<1>{}));
  auto gen_v = cute::make_tensor(cute::make_gmem_ptr(gen_qkv_ptr + v_offset),
                                 cute::make_shape(gen_len, num_v_heads, cute::Int<kDim>{}),
                                 cute::make_stride(qkv_stride, cute::Int<kDim>{}, cute::Int<1>{}));
  auto out_q = cute::make_tensor(cute::make_gmem_ptr(out_q_ptr),
                                 cute::make_shape(total_tokens, num_q_heads, cute::Int<kDim>{}),
                                 cute::LayoutRight{});
  auto out_k = cute::make_tensor(cute::make_gmem_ptr(out_k_ptr),
                                 cute::make_shape(total_tokens, num_k_heads, cute::Int<kDim>{}),
                                 cute::LayoutRight{});
  auto out_v = cute::make_tensor(cute::make_gmem_ptr(out_v_ptr),
                                 cute::make_shape(total_tokens, num_v_heads, cute::Int<kDim>{}),
                                 cute::LayoutRight{});

  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int ilane = idx % 32;

  int irow = blockIdx.x * kRowsPerBlock + iwarp;
  int col = ilane * kN;
  bool valid_col = col < kDim;
  if (irow >= total_tokens * num_q_heads) {
    return;
  }

  int out_token = irow / num_q_heads;
  int head = irow % num_q_heads;

  // Resolve source token via cat_indices
  int64_t src_token = cat_indices_ptr[out_token];
  bool is_und = src_token < und_len;
  int64_t local_token = is_und ? src_token : src_token - und_len;
  const __nv_bfloat16* q_weight_ptr = is_und ? und_q_weight_ptr : gen_q_weight_ptr;
  const __nv_bfloat16* k_weight_ptr = is_und ? und_k_weight_ptr : gen_k_weight_ptr;

  // Process Q
  {
    const __nv_bfloat16* q_src =
        is_und ? cute::raw_pointer_cast(und_q.data() + und_q.layout()(local_token, head, 0))
               : cute::raw_pointer_cast(gen_q.data() + gen_q.layout()(local_token, head, 0));
    __nv_bfloat16* q_dst =
        cute::raw_pointer_cast(out_q.data() + out_q.layout()(out_token, head, 0));

    vec_t<float, kN> q_in;
#pragma unroll
    for (int i = 0; i < kN; i++) {
      q_in[i] = 0.0f;
    }
    if (valid_col) {
      auto q_bf16 = load<__nv_bfloat162, kN / 2>(q_src + col);
      q_in = to<float>(q_bf16);
    }

    // RMSNorm
    float rms = reduce_mean_square_128_like_torch(q_src, ilane);
    float denom = __fadd_rn(__fmul_rn(rms, kInvDim), eps);
    rms = rsqrt_like_torch(denom);
    vec_t<__nv_bfloat16, kN> q_w;
#pragma unroll
    for (int i = 0; i < kN; i++) {
      q_w[i] = 0;
    }
    if (valid_col) {
      q_w = load<__nv_bfloat16, kN>(q_weight_ptr + col);
    }
#pragma unroll
    for (int i = 0; i < kN; i++) {
      q_in[i] = rmsnorm_weight_like_torch(q_in[i], rms, q_w[i]);
    }

    // 3D MRoPE
    __shared__ float smem_data[kWarpsPerBlock][kDim];
#pragma unroll
    for (int i = 0; i < kN; i++) {
      if (valid_col) {
        smem_data[iwarp][col + i] = q_in[i];
      }
    }

    __syncwarp();
    vec_t<float, kN> q_out;
    if (valid_col) {
#pragma unroll
      for (int i = 0; i < kN; i++) {
        q_out[i] = apply_mrope_element_like_torch<kHalfDim>(smem_data[iwarp], col + i, out_token,
                                                            mrope_section_h, mrope_section_w,
                                                            positions, cos_sin_cache);
      }
    }

    if (valid_col) {
      store(q_dst + col, to<__nv_bfloat16>(q_out));
    }
  }

  // Process K
  if (head < num_k_heads) {
    const __nv_bfloat16* k_src =
        is_und ? cute::raw_pointer_cast(und_k.data() + und_k.layout()(local_token, head, 0))
               : cute::raw_pointer_cast(gen_k.data() + gen_k.layout()(local_token, head, 0));
    __nv_bfloat16* k_dst =
        cute::raw_pointer_cast(out_k.data() + out_k.layout()(out_token, head, 0));

    vec_t<float, kN> k_in;
#pragma unroll
    for (int i = 0; i < kN; i++) {
      k_in[i] = 0.0f;
    }
    if (valid_col) {
      auto k_bf16 = load<__nv_bfloat162, kN / 2>(k_src + col);
      k_in = to<float>(k_bf16);
    }

    float rms = reduce_mean_square_128_like_torch(k_src, ilane);
    float denom = __fadd_rn(__fmul_rn(rms, kInvDim), eps);
    rms = rsqrt_like_torch(denom);
    vec_t<__nv_bfloat16, kN> k_w;
#pragma unroll
    for (int i = 0; i < kN; i++) {
      k_w[i] = 0;
    }
    if (valid_col) {
      k_w = load<__nv_bfloat16, kN>(k_weight_ptr + col);
    }
#pragma unroll
    for (int i = 0; i < kN; i++) {
      k_in[i] = rmsnorm_weight_like_torch(k_in[i], rms, k_w[i]);
    }

    __shared__ float smem_k_data[kWarpsPerBlock][kDim];
#pragma unroll
    for (int i = 0; i < kN; i++) {
      if (valid_col) {
        smem_k_data[iwarp][col + i] = k_in[i];
      }
    }

    __syncwarp();
    vec_t<float, kN> k_out;
    if (valid_col) {
#pragma unroll
      for (int i = 0; i < kN; i++) {
        k_out[i] = apply_mrope_element_like_torch<kHalfDim>(smem_k_data[iwarp], col + i, out_token,
                                                            mrope_section_h, mrope_section_w,
                                                            positions, cos_sin_cache);
      }
    }
    if (valid_col) {
      store(k_dst + col, to<__nv_bfloat16>(k_out));
    }
  }

  // Process V
  if (head < num_v_heads) {
    const __nv_bfloat16* v_src =
        is_und ? cute::raw_pointer_cast(und_v.data() + und_v.layout()(local_token, head, 0))
               : cute::raw_pointer_cast(gen_v.data() + gen_v.layout()(local_token, head, 0));
    __nv_bfloat16* v_dst =
        cute::raw_pointer_cast(out_v.data() + out_v.layout()(out_token, head, 0));
    if (valid_col) {
      auto v_data = load<__nv_bfloat162, kN / 2>(v_src + col);
      store(v_dst + col, v_data);
    }
  }
}

}  // namespace kernels

void fused_qk_rmsnorm_mrope_async(
    __nv_bfloat16* out_q_ptr, __nv_bfloat16* out_k_ptr, __nv_bfloat16* out_v_ptr,
    const __nv_bfloat16* und_qkv_ptr, const __nv_bfloat16* gen_qkv_ptr,
    const __nv_bfloat16* und_q_weight_ptr, const __nv_bfloat16* und_k_weight_ptr,
    const __nv_bfloat16* gen_q_weight_ptr, const __nv_bfloat16* gen_k_weight_ptr,
    const int64_t* positions_ptr, const float* cos_sin_cache_ptr, const int64_t* cat_indices_ptr,
    int und_len, int total_tokens, int num_q_heads, int num_k_heads, int num_v_heads, int head_dim,
    int mrope_section_h, int mrope_section_w, float eps, int cos_sin_cache_stride,
    cudaStream_t stream) {
  constexpr int kWarpCount = 4;
  constexpr int kWarpSize = 32;
  dim3 block(kWarpSize * kWarpCount);

  if (head_dim == 128) {
    constexpr int kDim = 128;
    constexpr int kRowsPerBlock = kWarpCount;
    dim3 grid((total_tokens * num_q_heads + kRowsPerBlock - 1) / kRowsPerBlock);
    kernels::fused_qk_rmsnorm_mrope_kernel<kDim><<<grid, block, 0, stream>>>(
        out_q_ptr, out_k_ptr, out_v_ptr, und_qkv_ptr, gen_qkv_ptr, und_q_weight_ptr,
        und_k_weight_ptr, gen_q_weight_ptr, gen_k_weight_ptr, positions_ptr, cos_sin_cache_ptr,
        cat_indices_ptr, und_len, total_tokens, num_q_heads, num_k_heads, num_v_heads,
        mrope_section_h, mrope_section_w, eps, cos_sin_cache_stride);
  }
}

}  // namespace normalization
}  // namespace hpc
