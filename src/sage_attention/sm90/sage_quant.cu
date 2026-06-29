// Copyright 2025 hpc-ops authors
//
// SM90 SageAttention2 quantization.
//   - K is per-thread-quantized at CTA_K=128 with 4 scales/block, where each
//     scale covers the 32 K-token positions visited by lane%4 == s in
//     wgmma m64n128k32 C-layout.
//   - V is per-channel FP8 with the official SageAttention2 16-token
//     seq-dim permute `P = [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15]`
//     baked in (matches the permute applied by upstream SageAttention's
//     `transpose_pad_permute_cuda`).  This permute is required so that the
//     PV wgmma `m64n128k32.f32.e4m3.e4m3` SS-mode B-operand reads V in the
//     layout the kernel-internal byte_perm on P expects.  Without this
//     permute, PV output is wrong (~cos 0.5).
//   - V is padded to a multiple of 128 with explicit zero-fill (so OOB tokens
//     become exact 0, immune to FP8 NaN encodings).
//   - `KMeanVScaleFusedKernel` + `k_mean_v_scale_fused_cuda` (a pre-pass
//     that computes smooth-K mean and per-channel V scale) is bundled at
//     the end of this file.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/all.h>

#include "src/sage_attention/sm90/sage_quant.h"
#include "src/utils/utils.cuh"

#define SA_CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), "Tensor " #x " must be on CUDA")
#define SA_CHECK_DTYPE(x, true_dtype) \
  TORCH_CHECK(x.dtype() == true_dtype, "Tensor " #x " must have dtype (" #true_dtype ")")
#define SA_CHECK_DIMS(x, true_dim) \
  TORCH_CHECK(x.dim() == true_dim, "Tensor " #x " must have dimension number (" #true_dim ")")
#define SA_CHECK_SHAPE(x, ...)                                \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              "Tensor " #x " must have shape (" #__VA_ARGS__ ")")
#define SA_CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "Tensor " #x " must be contiguous")
#define SA_CHECK_LASTDIM_CONTIGUOUS(x) \
  TORCH_CHECK(x.stride(-1) == 1, "Tensor " #x " must be contiguous at the last dimension")

namespace hpc {
namespace sage_attention {

namespace {

__device__ __forceinline__ int8_t float_to_int8_rn(float v) {
  uint32_t bits = __float_as_uint(v);
  float rounded = v + __uint_as_float((bits & 0x80000000u) | 0x3f000000u);
  return static_cast<int8_t>(rounded);
}

__device__ __forceinline__ uint4 ldg_nc_128b(const void *ptr) {
  uint4 ret;
  asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
               : "l"(ptr));
  return ret;
}

}  // namespace

// -- Q quant ----------------------------------------------------------------
// Grid: (num_block_q, num_head_q, batch).  Block: 128 = 4 warps.
// Each warp handles a 16-row stripe of CTA_Q=64; lane%4 splits the head_dim
// into 4 stripes of 32 elements; lane/4 indexes 8 row-groups (each row-group
// covers 2 wgmma rows: r and r+8).
template <bool IS_NHD, int HEAD_DIM = 128>
__global__ __launch_bounds__(128, 4) void QuantQSm90Kernel(const nv_bfloat16 *__restrict__ q,
                                                           int8_t *__restrict__ q_int8,
                                                           float *__restrict__ q_scale,
                                                           uint32_t qo_len, uint32_t num_head_q,
                                                           uint32_t num_q_scale) {
  constexpr float kInvInt8Max = 1.0f / 127.0f;
  constexpr float kQuantEps = 1e-7f;

  uint32_t block_q = blockIdx.x;
  uint32_t head_id = blockIdx.y;
  uint32_t batch_id = blockIdx.z;

  uint32_t tid = threadIdx.x;
  uint32_t warp = tid >> 5;
  uint32_t lane = tid & 31;

  uint32_t row_group = lane >> 2;     // 0..7
  uint32_t lane_in_group = lane & 3;  // 0..3
  uint32_t row0 = block_q * 64 + warp * 16 + row_group;
  uint32_t row1 = row0 + 8;

  uint32_t batch_head = batch_id * num_head_q + head_id;

  uint32_t base_offset0 = IS_NHD ? (batch_id * qo_len * num_head_q + head_id) * HEAD_DIM
                                 : batch_head * qo_len * HEAD_DIM;
  const nv_bfloat16 *q_row0 = q + base_offset0 + row0 * (IS_NHD ? num_head_q * HEAD_DIM : HEAD_DIM);
  const nv_bfloat16 *q_row1 = q + base_offset0 + row1 * (IS_NHD ? num_head_q * HEAD_DIM : HEAD_DIM);

  bool valid0 = row0 < qo_len;
  bool valid1 = row1 < qo_len;

  float local_amax = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    uint32_t dd = lane_in_group * 8 + chunk * 32;
    if (valid0) {
      uint4 packed = *reinterpret_cast<const uint4 *>(q_row0 + dd);
      nv_bfloat162 *vals = reinterpret_cast<nv_bfloat162 *>(&packed);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        local_amax =
            fmaxf(local_amax, fmaxf(fabsf(__low2float(vals[i])), fabsf(__high2float(vals[i]))));
      }
    }
    if (valid1) {
      uint4 packed = *reinterpret_cast<const uint4 *>(q_row1 + dd);
      nv_bfloat162 *vals = reinterpret_cast<nv_bfloat162 *>(&packed);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        local_amax =
            fmaxf(local_amax, fmaxf(fabsf(__low2float(vals[i])), fabsf(__high2float(vals[i]))));
      }
    }
  }
  // Reduce across the 4 lanes that share the same row_group.
  local_amax = fmaxf(local_amax, __shfl_xor_sync(0xFFFFFFFF, local_amax, 1, 32));
  local_amax = fmaxf(local_amax, __shfl_xor_sync(0xFFFFFFFF, local_amax, 2, 32));

  float scale = fmaf(local_amax, kInvInt8Max, kQuantEps);
  if (lane_in_group == 0) {
    q_scale[batch_head * num_q_scale + block_q * 32 + warp * 8 + row_group] = scale;
  }
  float rcp = rcpf_ftz(scale);

#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    uint32_t dd = lane_in_group * 8 + chunk * 32;
    if (valid0) {
      uint4 packed = *reinterpret_cast<const uint4 *>(q_row0 + dd);
      nv_bfloat16 *vals = reinterpret_cast<nv_bfloat16 *>(&packed);
      int8_t qvals[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        qvals[i] = float_to_int8_rn(__bfloat162float(vals[i]) * rcp);
      }
      *reinterpret_cast<uint64_t *>(q_int8 + base_offset0 +
                                    row0 * (IS_NHD ? num_head_q * HEAD_DIM : HEAD_DIM) + dd) =
          *reinterpret_cast<uint64_t *>(&qvals[0]);
    }
    if (valid1) {
      uint4 packed = *reinterpret_cast<const uint4 *>(q_row1 + dd);
      nv_bfloat16 *vals = reinterpret_cast<nv_bfloat16 *>(&packed);
      int8_t qvals[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        qvals[i] = float_to_int8_rn(__bfloat162float(vals[i]) * rcp);
      }
      *reinterpret_cast<uint64_t *>(q_int8 + base_offset0 +
                                    row1 * (IS_NHD ? num_head_q * HEAD_DIM : HEAD_DIM) + dd) =
          *reinterpret_cast<uint64_t *>(&qvals[0]);
    }
  }
}

// -- K quant ----------------------------------------------------------------
// 1 block = (block_k, scale_idx, head_kv, batch) flattened into grid.x.
// Each block computes 1 scale (over 32 specific K tokens covered by
// lane%4 == s in wgmma C-layout) AND quantizes those 32 tokens.  Different
// (block_k, scale_idx) blocks write disjoint tokens, so no races.
//
// To stay within 128 registers (avoid spills), we do not cache values across
// passes -- amax is computed first, then K is re-loaded from L2 for the int8
// write.  L2 hit is essentially free vs. spill->local-mem round trip.
template <bool IS_NHD, int HEAD_DIM = 128>
__global__ __launch_bounds__(128, 4) void QuantKSm90Kernel(
    const nv_bfloat16 *__restrict__ k, const float *__restrict__ km, int8_t *__restrict__ k_int8,
    float *__restrict__ k_scale, uint32_t kv_len, uint32_t num_head_kv, uint32_t num_block_k) {
  constexpr float kInvInt8Max = 1.0f / 127.0f;
  constexpr float kQuantEps = 1e-7f;

  uint32_t task = blockIdx.x;
  uint32_t block_k = task >> 2;
  uint32_t s = task & 3;
  uint32_t head_id = blockIdx.y;
  uint32_t batch_id = blockIdx.z;

  uint32_t tid = threadIdx.x;
  uint32_t batch_head = batch_id * num_head_kv + head_id;

  // 1 thread per d position (HEAD_DIM == blockDim.x == 128).
  uint32_t d = tid;
  bool valid_d = d < HEAD_DIM;

  float km_val = (km != nullptr && valid_d) ? km[batch_head * HEAD_DIM + d] : 0.0f;

  uint32_t k_seq_stride = IS_NHD ? num_head_kv * HEAD_DIM : HEAD_DIM;
  uint32_t k_base_offset = IS_NHD ? (batch_id * kv_len * num_head_kv + head_id) * HEAD_DIM
                                  : batch_head * kv_len * HEAD_DIM;
  const nv_bfloat16 *k_base = k + k_base_offset;
  int8_t *kq_base = k_int8 + k_base_offset;

  // Pass 1 - amax across the 32 covered tokens (no caching).
  float local_amax = 0.0f;
  if (valid_d) {
#pragma unroll
    for (int atom = 0; atom < 16; ++atom) {
#pragma unroll
      for (int off = 0; off < 2; ++off) {
        uint32_t t = block_k * 128 + atom * 8 + s * 2 + off;
        if (t < kv_len) {
          float v = __bfloat162float(k_base[t * k_seq_stride + d]) - km_val;
          local_amax = fmaxf(local_amax, fabsf(v));
        }
      }
    }
  }

  // Block reduce max via warp shfl + smem aggregator.
  __shared__ float warp_amax[4];
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    local_amax = fmaxf(local_amax, __shfl_xor_sync(0xFFFFFFFF, local_amax, mask, 32));
  }
  if ((tid & 31) == 0) {
    warp_amax[tid >> 5] = local_amax;
  }
  __syncthreads();
  __shared__ float sh_rcp;
  if (tid == 0) {
    float block_amax = fmaxf(fmaxf(warp_amax[0], warp_amax[1]), fmaxf(warp_amax[2], warp_amax[3]));
    float scale = fmaf(block_amax, kInvInt8Max, kQuantEps);
    sh_rcp = rcpf_ftz(scale);
    k_scale[batch_head * (num_block_k * 4) + block_k * 4 + s] = scale;
  }
  __syncthreads();

  // Pass 2 - re-load K from L2, subtract km, scale, write int8.
  if (valid_d) {
    float rcp = sh_rcp;
#pragma unroll
    for (int atom = 0; atom < 16; ++atom) {
#pragma unroll
      for (int off = 0; off < 2; ++off) {
        uint32_t t = block_k * 128 + atom * 8 + s * 2 + off;
        if (t < kv_len) {
          float v = __bfloat162float(k_base[t * k_seq_stride + d]) - km_val;
          kq_base[t * k_seq_stride + d] = float_to_int8_rn(v * rcp);
        }
      }
    }
  }
}

// -- V quant ----------------------------------------------------------------
// Per-channel FP8.  Output shape:
//   HND : [B, H_kv, D, padded_kv_len_128]
//   NHD : [B, D,    H_kv, padded_kv_len_128]
// (kv is the contiguous last dim, matching the SM90 TMA tile shape.)
//
// Applies the official SageAttention2 16-token seq-dim permute
// `P = [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15]` (matches the permute used
// in upstream SageAttention's `transpose_pad_permute_cuda`), required so
// that wgmma `m64n128k32.f32.e4m3.e4m3` SS-mode B-operand reads V from smem
// in the layout the kernel-internal byte_perm on P expects.
//
// 1 block handles 1 (batch, head_kv, d_group=32, kv_tile=64).  Tokens beyond
// kv_len are written as exact 0 (NOT torch::empty garbage) so that OOB
// PV contributions from masked-out columns are guaranteed to be zero even
// after FP8 NaN-encoding pitfalls.
template <bool IS_NHD, int HEAD_DIM = 128, int V_TILE_S = 64, int V_D_GROUP = 32>
__global__ __launch_bounds__(256, 4) void QuantVPerChannelFp8Sm90Kernel(
    const nv_bfloat16 *__restrict__ v, const float *__restrict__ v_scale,
    int8_t *__restrict__ v_fp8, uint32_t kv_len, uint32_t num_head_kv, uint32_t v_num_tiles,
    uint32_t stride_bz_vo, uint32_t stride_d_vo, uint32_t stride_h_vo) {
  constexpr int kVecElems = 8;
  constexpr int kVecsPerToken = V_D_GROUP / kVecElems;  // 32/8 = 4

  uint32_t task = blockIdx.x;
  uint32_t v_tile = task % v_num_tiles;
  uint32_t tmp = task / v_num_tiles;
  uint32_t v_d_group = tmp % (HEAD_DIM / V_D_GROUP);
  tmp /= (HEAD_DIM / V_D_GROUP);
  uint32_t v_head_id = tmp % num_head_kv;
  uint32_t v_batch_id = tmp / num_head_kv;
  uint32_t v_d_base = v_d_group * V_D_GROUP;

  __shared__ uint8_t smem_fp8[V_D_GROUP][V_TILE_S];

  uint32_t tid = threadIdx.x;
  uint32_t token_local = tid / kVecsPerToken;            // 0..63
  uint32_t d_local = (tid % kVecsPerToken) * kVecElems;  // 0,8,16,24
  uint32_t v_token = v_tile * V_TILE_S + token_local;

  // 16-token seq-dim permute applied on the OUTPUT (smem→gmem) side so
  // that input loads stay coalesced.  out_token_local = P(token_local).
  //   r = token_local & 15
  //   P(r) = (r>>3)*2 + ((r>>1)&3)*4 + (r & 1)
  //        = [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15][r]
  uint32_t r = token_local & 15u;
  uint32_t out_token_local =
      (token_local & ~15u) + ((r >> 3) * 2u + ((r >> 1) & 3u) * 4u + (r & 1u));

  uint32_t v_stride_bz = IS_NHD ? kv_len * num_head_kv * HEAD_DIM : num_head_kv * kv_len * HEAD_DIM;
  uint32_t v_stride_h = IS_NHD ? HEAD_DIM : kv_len * HEAD_DIM;
  uint32_t v_stride_seq = IS_NHD ? num_head_kv * HEAD_DIM : HEAD_DIM;

  // Write to smem [d][permuted token]; XOR is for bank-conflict avoidance.
  if (v_token < kv_len) {
    const nv_bfloat16 *v_ptr = v + v_batch_id * v_stride_bz + v_head_id * v_stride_h +
                               v_token * v_stride_seq + v_d_base + d_local;
    uint4 packed = ldg_nc_128b(v_ptr);
    nv_bfloat16 *vals = reinterpret_cast<nv_bfloat16 *>(&packed);
    uint32_t vs_batch_head = v_batch_id * num_head_kv + v_head_id;
    const float *scale_ptr = v_scale + vs_batch_head * HEAD_DIM + v_d_base + d_local;
    float sv[kVecElems];
    *reinterpret_cast<float4 *>(&sv[0]) = *reinterpret_cast<const float4 *>(scale_ptr);
    *reinterpret_cast<float4 *>(&sv[4]) = *reinterpret_cast<const float4 *>(scale_ptr + 4);
#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
      uint32_t dd = d_local + i;
      float x = __bfloat162float(vals[i]);
      float rcp = __frcp_rn(sv[i]);
      smem_fp8[dd][out_token_local ^ ((dd & 24) << 1)] = __nv_fp8_e4m3(x * rcp).__x;
    }
  } else {
#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
      uint32_t dd = d_local + i;
      smem_fp8[dd][out_token_local ^ ((dd & 24) << 1)] = 0;
    }
  }
  __syncthreads();

  // Coalesced store from smem to gmem (kv contiguous).
  constexpr uint32_t kBytesPerStore = 8;
  constexpr uint32_t kStoreVecsPerD = V_TILE_S / kBytesPerStore;
  uint32_t store_d_local = tid / kStoreVecsPerD;
  uint32_t store_token_local = (tid % kStoreVecsPerD) * kBytesPerStore;
  int8_t *v_out_ptr = v_fp8 + v_batch_id * stride_bz_vo + v_head_id * stride_h_vo +
                      (v_d_base + store_d_local) * stride_d_vo + v_tile * V_TILE_S +
                      store_token_local;
  uint32_t smem_token = store_token_local ^ ((store_d_local & 24) << 1);
  *reinterpret_cast<uint64_t *>(v_out_ptr) =
      *reinterpret_cast<uint64_t *>(&smem_fp8[store_d_local][smem_token]);
}

void qkv_fused_sm90_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor km,
                         torch::Tensor v_scale, torch::Tensor q_int8, torch::Tensor q_scale,
                         torch::Tensor k_int8, torch::Tensor k_scale, torch::Tensor v_fp8,
                         int tensor_layout) {
  SA_CHECK_CUDA(q);
  SA_CHECK_CUDA(k);
  SA_CHECK_CUDA(v);
  SA_CHECK_CUDA(km);
  SA_CHECK_CUDA(v_scale);
  SA_CHECK_CUDA(q_int8);
  SA_CHECK_CUDA(q_scale);
  SA_CHECK_CUDA(k_int8);
  SA_CHECK_CUDA(k_scale);
  SA_CHECK_CUDA(v_fp8);
  SA_CHECK_LASTDIM_CONTIGUOUS(q);
  SA_CHECK_LASTDIM_CONTIGUOUS(k);
  SA_CHECK_LASTDIM_CONTIGUOUS(v);
  SA_CHECK_LASTDIM_CONTIGUOUS(km);
  SA_CHECK_CONTIGUOUS(q_int8);
  SA_CHECK_CONTIGUOUS(k_int8);
  SA_CHECK_CONTIGUOUS(v_fp8);
  SA_CHECK_CONTIGUOUS(q_scale);
  SA_CHECK_CONTIGUOUS(k_scale);
  SA_CHECK_CONTIGUOUS(v_scale);
  SA_CHECK_DTYPE(q_int8, torch::kInt8);
  SA_CHECK_DTYPE(k_int8, torch::kInt8);
  SA_CHECK_DTYPE(q_scale, torch::kFloat);
  SA_CHECK_DTYPE(k_scale, torch::kFloat);
  SA_CHECK_DTYPE(v_scale, torch::kFloat);
  TORCH_CHECK(v_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn);
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && k.scalar_type() == torch::kBFloat16 &&
              v.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(km.scalar_type() == torch::kFloat32);

  const int batch_size = q.size(0);
  const int head_dim = q.size(3);
  TORCH_CHECK(head_dim == 128, "qkv_fused_sm90_cuda only supports head_dim=128");

  int qo_len, kv_len, num_head_q, num_head_kv;
  int stride_bz_vo, stride_d_vo, stride_h_vo;
  if (tensor_layout == 0) {
    qo_len = q.size(1);
    num_head_q = q.size(2);
    kv_len = k.size(1);
    num_head_kv = k.size(2);
    stride_bz_vo = v_fp8.stride(0);
    stride_d_vo = v_fp8.stride(1);
    stride_h_vo = v_fp8.stride(2);
  } else {
    num_head_q = q.size(1);
    qo_len = q.size(2);
    num_head_kv = k.size(1);
    kv_len = k.size(2);
    stride_bz_vo = v_fp8.stride(0);
    stride_h_vo = v_fp8.stride(1);
    stride_d_vo = v_fp8.stride(2);
  }

  constexpr int HEAD_DIM = 128;
  constexpr int CTA_Q = 64;
  constexpr int CTA_K = 128;
  constexpr int V_TILE_S = 64;
  constexpr int V_D_GROUP = 32;

  uint32_t num_block_q = (qo_len + CTA_Q - 1) / CTA_Q;
  uint32_t num_q_scale = num_block_q * 32;
  uint32_t num_block_k = (kv_len + CTA_K - 1) / CTA_K;
  uint32_t v_padded = num_block_k * CTA_K;
  uint32_t v_num_tiles = v_padded / V_TILE_S;
  uint32_t v_d_groups = HEAD_DIM / V_D_GROUP;
  uint32_t v_tasks = batch_size * num_head_kv * v_d_groups * v_num_tiles;

  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

  // Q quant.
  {
    dim3 grid(num_block_q, num_head_q, batch_size);
    dim3 block(128);
    if (tensor_layout == 0) {
      QuantQSm90Kernel<true, HEAD_DIM><<<grid, block, 0, stream>>>(
          reinterpret_cast<nv_bfloat16 *>(q.data_ptr()), q_int8.data_ptr<int8_t>(),
          reinterpret_cast<float *>(q_scale.data_ptr()), qo_len, num_head_q, num_q_scale);
    } else {
      QuantQSm90Kernel<false, HEAD_DIM><<<grid, block, 0, stream>>>(
          reinterpret_cast<nv_bfloat16 *>(q.data_ptr()), q_int8.data_ptr<int8_t>(),
          reinterpret_cast<float *>(q_scale.data_ptr()), qo_len, num_head_q, num_q_scale);
    }
  }

  // K quant.
  {
    dim3 grid(num_block_k * 4, num_head_kv, batch_size);
    dim3 block(128);
    if (tensor_layout == 0) {
      QuantKSm90Kernel<true, HEAD_DIM><<<grid, block, 0, stream>>>(
          reinterpret_cast<nv_bfloat16 *>(k.data_ptr()), reinterpret_cast<float *>(km.data_ptr()),
          k_int8.data_ptr<int8_t>(), reinterpret_cast<float *>(k_scale.data_ptr()), kv_len,
          num_head_kv, num_block_k);
    } else {
      QuantKSm90Kernel<false, HEAD_DIM><<<grid, block, 0, stream>>>(
          reinterpret_cast<nv_bfloat16 *>(k.data_ptr()), reinterpret_cast<float *>(km.data_ptr()),
          k_int8.data_ptr<int8_t>(), reinterpret_cast<float *>(k_scale.data_ptr()), kv_len,
          num_head_kv, num_block_k);
    }
  }

  // V quant.
  {
    dim3 grid(v_tasks);
    dim3 block(256);
    if (tensor_layout == 0) {
      QuantVPerChannelFp8Sm90Kernel<true, HEAD_DIM, V_TILE_S, V_D_GROUP>
          <<<grid, block, 0, stream>>>(reinterpret_cast<nv_bfloat16 *>(v.data_ptr()),
                                       reinterpret_cast<float *>(v_scale.data_ptr()),
                                       reinterpret_cast<int8_t *>(v_fp8.data_ptr()), kv_len,
                                       num_head_kv, v_num_tiles, stride_bz_vo, stride_d_vo,
                                       stride_h_vo);
    } else {
      QuantVPerChannelFp8Sm90Kernel<false, HEAD_DIM, V_TILE_S, V_D_GROUP>
          <<<grid, block, 0, stream>>>(reinterpret_cast<nv_bfloat16 *>(v.data_ptr()),
                                       reinterpret_cast<float *>(v_scale.data_ptr()),
                                       reinterpret_cast<int8_t *>(v_fp8.data_ptr()), kv_len,
                                       num_head_kv, v_num_tiles, stride_bz_vo, stride_d_vo,
                                       stride_h_vo);
    }
  }
}

// ── K mean + per-channel V amax/scale (pre-pass for `qkv_fused_sm90_cuda`) ──
//
// One-pass fused kernel: half of the grid CTAs compute K mean over the kv
// dim, the other half compute per-channel V amax / 448 (per-channel FP8
// scale).  Outputs `km` (fp32) and `v_scale` (fp32) are consumed downstream
// by the Q/K/V quant kernels above.
template <int D_GROUP = 8, int THREADS = 128>
__global__ void KMeanVScaleFusedKernel(const nv_bfloat16 *__restrict__ k,
                                       const nv_bfloat16 *__restrict__ v, float *__restrict__ km,
                                       float *__restrict__ v_scale, const uint32_t kv_len,
                                       const uint32_t stride_bz_k, const uint32_t stride_seq_k,
                                       const uint32_t stride_h_k, const uint32_t stride_bz_v,
                                       const uint32_t stride_seq_v, const uint32_t stride_h_v,
                                       const uint32_t stride_bz_km, const uint32_t stride_h_km,
                                       const uint32_t stride_bz_vs, const uint32_t stride_h_vs) {
  constexpr uint32_t kVecElems = 8;
  constexpr float kInvFp8E4m3Max = 1.0f / 448.0f;
  constexpr uint32_t kNumDGroups = 128 / D_GROUP;
  constexpr uint32_t kNumWarps = THREADS / 32;

  uint32_t head_group = blockIdx.x;
  uint32_t batch_task = blockIdx.z;
  uint32_t d_group = head_group & (kNumDGroups - 1);
  uint32_t head_id = head_group >> 4;
  uint32_t batch_id = batch_task >> 1;
  uint32_t d_base = d_group * D_GROUP;
  uint32_t thread_id = threadIdx.x;
  uint32_t warp_id = thread_id / 32;
  uint32_t lane = thread_id & 31;

  uint32_t elems_per_iter = THREADS * kVecElems;
  uint32_t total_elems = kv_len * D_GROUP;

  if ((batch_task & 1) == 0) {
    float local_sum[D_GROUP] = {};

    for (uint32_t base = 0; base < total_elems; base += elems_per_iter) {
      uint32_t linear = base + thread_id * kVecElems;
      if (linear >= total_elems) {
        break;
      }
      uint32_t token = linear / D_GROUP;
      uint32_t d_local = linear % D_GROUP;
      const nv_bfloat16 *k_ptr = k + batch_id * stride_bz_k + head_id * stride_h_k +
                                 token * stride_seq_k + d_base + d_local;
      uint4 packed = *reinterpret_cast<const uint4 *>(k_ptr);
      nv_bfloat16 *vals = reinterpret_cast<nv_bfloat16 *>(&packed);
#pragma unroll
      for (int i = 0; i < kVecElems; ++i) {
        local_sum[d_local + i] += __bfloat162float(vals[i]);
      }
    }

#pragma unroll
    for (int ch = 0; ch < D_GROUP; ++ch) {
#pragma unroll
      for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum[ch] += __shfl_xor_sync(0xFFFFFFFF, local_sum[ch], mask);
      }
    }

    __shared__ float warp_sums[kNumWarps][D_GROUP];
    if (lane == 0) {
#pragma unroll
      for (int ch = 0; ch < D_GROUP; ++ch) {
        warp_sums[warp_id][ch] = local_sum[ch];
      }
    }
    __syncthreads();

    if (thread_id < D_GROUP) {
      float sum = 0.0f;
#pragma unroll
      for (uint32_t w = 0; w < kNumWarps; ++w) {
        sum += warp_sums[w][thread_id];
      }
      km[batch_id * stride_bz_km + head_id * stride_h_km + d_base + thread_id] =
          sum / static_cast<float>(kv_len);
    }
  } else {
    float local_max[D_GROUP] = {};

    for (uint32_t base = 0; base < total_elems; base += elems_per_iter) {
      uint32_t linear = base + thread_id * kVecElems;
      if (linear >= total_elems) {
        break;
      }
      uint32_t token = linear / D_GROUP;
      uint32_t d_local = linear % D_GROUP;
      const nv_bfloat16 *v_ptr = v + batch_id * stride_bz_v + head_id * stride_h_v +
                                 token * stride_seq_v + d_base + d_local;
      uint4 packed = *reinterpret_cast<const uint4 *>(v_ptr);
      nv_bfloat16 *vals = reinterpret_cast<nv_bfloat16 *>(&packed);
#pragma unroll
      for (int i = 0; i < kVecElems; ++i) {
        float abs_val = fabsf(__bfloat162float(vals[i]));
        local_max[d_local + i] = fmaxf(local_max[d_local + i], abs_val);
      }
    }

#pragma unroll
    for (int ch = 0; ch < D_GROUP; ++ch) {
#pragma unroll
      for (int mask = 16; mask > 0; mask >>= 1) {
        local_max[ch] = fmaxf(local_max[ch], __shfl_xor_sync(0xFFFFFFFF, local_max[ch], mask));
      }
    }

    __shared__ float warp_maxs[kNumWarps][D_GROUP];
    if (lane == 0) {
#pragma unroll
      for (int ch = 0; ch < D_GROUP; ++ch) {
        warp_maxs[warp_id][ch] = local_max[ch];
      }
    }
    __syncthreads();

    if (thread_id < D_GROUP) {
      float mx = 0.0f;
#pragma unroll
      for (uint32_t w = 0; w < kNumWarps; ++w) {
        mx = fmaxf(mx, warp_maxs[w][thread_id]);
      }
      v_scale[batch_id * stride_bz_vs + head_id * stride_h_vs + d_base + thread_id] =
          mx * kInvFp8E4m3Max;
    }
  }
}

void k_mean_v_scale_fused_cuda(torch::Tensor k, torch::Tensor v, torch::Tensor km,
                               torch::Tensor v_scale, int tensor_layout) {
  SA_CHECK_CUDA(k);
  SA_CHECK_CUDA(v);
  SA_CHECK_CUDA(km);
  SA_CHECK_CUDA(v_scale);
  SA_CHECK_LASTDIM_CONTIGUOUS(k);
  SA_CHECK_LASTDIM_CONTIGUOUS(v);
  SA_CHECK_LASTDIM_CONTIGUOUS(km);
  SA_CHECK_CONTIGUOUS(v_scale);
  SA_CHECK_DIMS(k, 4);
  SA_CHECK_DIMS(v, 4);
  SA_CHECK_DTYPE(km, torch::kFloat);
  SA_CHECK_DTYPE(v_scale, torch::kFloat);

  const int batch_size = k.size(0);
  const int head_dim = k.size(3);
  TORCH_CHECK(head_dim == 128, "fused K mean/V scale only supports head_dim=128");
  TORCH_CHECK(v.size(0) == batch_size && v.size(3) == head_dim, "k/v shape mismatch");

  int kv_len, num_head_kv;
  int stride_bz_k = k.stride(0), stride_seq_k, stride_h_k;
  int stride_bz_v = v.stride(0), stride_seq_v, stride_h_v;
  if (tensor_layout == 0) {
    kv_len = k.size(1);
    num_head_kv = k.size(2);
    stride_seq_k = k.stride(1);
    stride_h_k = k.stride(2);
    stride_seq_v = v.stride(1);
    stride_h_v = v.stride(2);
    SA_CHECK_SHAPE(v, batch_size, kv_len, num_head_kv, head_dim);
    SA_CHECK_SHAPE(km, batch_size, 1, num_head_kv, head_dim);
  } else {
    num_head_kv = k.size(1);
    kv_len = k.size(2);
    stride_h_k = k.stride(1);
    stride_seq_k = k.stride(2);
    stride_h_v = v.stride(1);
    stride_seq_v = v.stride(2);
    SA_CHECK_SHAPE(v, batch_size, num_head_kv, kv_len, head_dim);
    SA_CHECK_SHAPE(km, batch_size, num_head_kv, 1, head_dim);
  }
  SA_CHECK_SHAPE(v_scale, batch_size, num_head_kv, head_dim);

  int stride_bz_km = km.stride(0);
  int stride_h_km = tensor_layout == 0 ? km.stride(2) : km.stride(1);

  constexpr int D_GROUP = 8;
  constexpr int THREADS = 128;
  dim3 grid(num_head_kv * (head_dim / D_GROUP), 1, batch_size * 2);
  dim3 block(THREADS);

  auto stream = at::cuda::getCurrentCUDAStream(k.get_device());

  auto dtype = k.scalar_type();
  TORCH_CHECK(dtype == v.scalar_type(), "k/v must have the same dtype");
  TORCH_CHECK(dtype == torch::kBFloat16, "k_mean_v_scale_fused only supports bfloat16");
  KMeanVScaleFusedKernel<D_GROUP, THREADS><<<grid, block, 0, stream>>>(
      reinterpret_cast<nv_bfloat16 *>(k.data_ptr()), reinterpret_cast<nv_bfloat16 *>(v.data_ptr()),
      reinterpret_cast<float *>(km.data_ptr()), reinterpret_cast<float *>(v_scale.data_ptr()),
      kv_len, stride_bz_k, stride_seq_k, stride_h_k, stride_bz_v, stride_seq_v, stride_h_v,
      stride_bz_km, stride_h_km, v_scale.stride(0), v_scale.stride(1));
}

}  // namespace sage_attention
}  // namespace hpc
