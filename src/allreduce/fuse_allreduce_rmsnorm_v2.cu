// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <tuple>

#include "src/allreduce/fuse_allreduce_rmsnorm_v2.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace allreduce {
namespace kernels {

__device__ __forceinline__ bool is_neg_zero_f32(float v) {
  return __float_as_uint(v) == 0x80000000u;
}

__device__ __forceinline__ bool is_neg_zero_bf16(__nv_bfloat16 v) {
  return __bfloat16_as_ushort(v) == static_cast<uint16_t>(0x8000U);
}

__device__ __forceinline__ float4 lamport_init_f4() {
  return make_float4(-0.0f, -0.0f, -0.0f, -0.0f);
}

template <typename T1, typename T2>
__device__ __forceinline__ constexpr T1 ceil_div(const T1 x, const T2 y) noexcept {
  return (x + y - 1) / y;
}

template <typename T1, typename T2>
__device__ __forceinline__ constexpr T1 round_up(const T1 x, const T2 y) noexcept {
  return ceil_div(x, y) * y;
}

template <typename T_IN>
__device__ __forceinline__ void pipeline_async_copy_f4(T_IN* dst, T_IN const* src) {
  float4* dst4 = reinterpret_cast<float4*>(dst);
  float4 const* src4 = reinterpret_cast<float4 const*>(src);
  __pipeline_memcpy_async(dst4, src4, sizeof(float4));
}

enum TwoShotAllReduceStage : uint8_t {
  SCATTER = 0,
  BROADCAST = 1,
  NUM_STAGES = 2,
};

struct __attribute__((aligned(32))) LamportBufferScheduler {
 public:
  __device__ __forceinline__ explicit LamportBufferScheduler(uint32_t* buffer_flags,
                                                             uint32_t num_stages = 2)
      : buffer_flags_ptr(buffer_flags), flag_access_ptr(&buffer_flags[8]) {
    uint4 flag = reinterpret_cast<uint4*>(buffer_flags)[0];
    current_index = flag.x;
    dirty_index = flag.y;
    bytes_per_buffer = flag.z;
    two_shot_num_stages = num_stages;
    *reinterpret_cast<uint4*>(&bytes_to_clear) = reinterpret_cast<uint4*>(buffer_flags)[1];
  }

  __device__ __forceinline__ void* get_cur_lamport_buf(void* buffer_base_ptr,
                                                       int stage_idx = 0) const {
    return get_stage_ptr(buffer_base_ptr, current_index, stage_idx);
  }

  __device__ __forceinline__ void clear_dirty_lamport_buf(void* buffer_base_ptr, int stage_idx) {
    if (stage_idx >= two_shot_num_stages) {
      return;
    }
    uint32_t global_cta_idx = blockIdx.x * gridDim.y + blockIdx.y;
    uint32_t global_tid = global_cta_idx * blockDim.x + threadIdx.x;
    uint32_t num_threads = gridDim.x * gridDim.y * blockDim.x;

    uint32_t clear_boundary = ceil_div<uint32_t>(bytes_to_clear[stage_idx], sizeof(float4));
    auto* dst = reinterpret_cast<float4*>(get_stage_ptr(buffer_base_ptr, dirty_index, stage_idx));
    for (uint32_t packed_idx = global_tid; packed_idx < clear_boundary; packed_idx += num_threads) {
      dst[packed_idx] = lamport_init_f4();
    }
  }

  __device__ __forceinline__ void single_cta_arrive() {
    __syncthreads();
    if (threadIdx.x == 0) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
      asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" ::"l"(flag_access_ptr), "r"(1)
                   : "memory");
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
      asm volatile("red.release.global.gpu.add.u32 [%0], %1;" ::"l"(flag_access_ptr), "r"(1)
                   : "memory");
#else
      atomicAdd(flag_access_ptr, 1);
#endif
    }
  }

  __device__ __forceinline__ void wait_ctas_and_update(uint4 bytes_to_clear_per_stage) {
    bool is_last_cta_t0 =
        (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0);
    int target_count = gridDim.x * gridDim.y * gridDim.z;
    if (is_last_cta_t0) {
      uint4* flag_ptr = reinterpret_cast<uint4*>(buffer_flags_ptr);
      while (*reinterpret_cast<uint32_t volatile*>(flag_access_ptr) < target_count) {
      }
      flag_ptr[0] = {(current_index + 1) % 3,  // Current index
                     current_index,            // Dirty index
                     bytes_per_buffer,         // Two-Shot Lamport-Buffer size
                     two_shot_num_stages};     // Two-Shot Num Stages
      flag_ptr[1] = bytes_to_clear_per_stage;
      *flag_access_ptr = 0;
    }
  }

 private:
  __device__ __forceinline__ void* get_stage_ptr(void* buffer_base_ptr, uint32_t lamport_index,
                                                 uint32_t stage_index) const {
    return reinterpret_cast<void*>(
        reinterpret_cast<char*>(buffer_base_ptr) +
        static_cast<size_t>((lamport_index * two_shot_num_stages + stage_index) *
                            static_cast<size_t>(bytes_per_buffer / two_shot_num_stages)));
  }

  uint32_t* buffer_flags_ptr;
  uint32_t* flag_access_ptr;

  uint32_t current_index, dirty_index;
  alignas(16) std::array<uint32_t, 4> bytes_to_clear;

  uint32_t two_shot_num_stages;
  uint32_t bytes_per_buffer;
};

template <uint8_t kWorldSize, uint32_t kHiddenSize, uint32_t kBlockSize = 128>
__global__ __launch_bounds__(kBlockSize) void two_shot_allreduce_kernel(
    __nv_bfloat16* output_ptr, __nv_bfloat16 const* input_ptr, __nv_bfloat16** mn_buffer_ptrs,
    __nv_bfloat16* mc_ptr, uint32_t const num_tokens, uint32_t const rank, uint32_t* buffer_flags) {
  constexpr int kEltsPerThread = sizeof(float4) / sizeof(__nv_bfloat16);
  constexpr int kLamportEltsPerPacked = sizeof(float4) / sizeof(float);
  constexpr uint32_t kNumBlocksPerToken = kHiddenSize / (kBlockSize * kEltsPerThread);

  static_assert(kNumBlocksPerToken * kBlockSize * kEltsPerThread == kHiddenSize,
                "gridDim.y * blockDim.x * 8 must exactly equal kHiddenSize");
  static_assert(kWorldSize == 2 || kWorldSize == 4 || kWorldSize == 8,
                "kWorldSize must be one of {2, 4, 8}");

  int packed_idx = blockIdx.y * kBlockSize + threadIdx.x;
  int itoken = blockIdx.x;
  int thread_offset = itoken * kHiddenSize + packed_idx * kEltsPerThread;

  int dest_rank = itoken % kWorldSize;
  int dest_token_offset = itoken / kWorldSize;
  cudaGridDependencySynchronize();
  LamportBufferScheduler scheduler(buffer_flags, TwoShotAllReduceStage::NUM_STAGES);

  auto* scatter_buf_local = reinterpret_cast<__nv_bfloat16*>(
      scheduler.get_cur_lamport_buf(mn_buffer_ptrs[rank], TwoShotAllReduceStage::SCATTER));
  auto* scatter_buf_dest = reinterpret_cast<__nv_bfloat16*>(
      scheduler.get_cur_lamport_buf(mn_buffer_ptrs[dest_rank], TwoShotAllReduceStage::SCATTER));
  auto* broadcast_buf = reinterpret_cast<__nv_bfloat16*>(
      scheduler.get_cur_lamport_buf(mc_ptr, TwoShotAllReduceStage::BROADCAST));

  cudaTriggerProgrammaticLaunchCompletion();

  // 1. Scatter
  auto val = load<__nv_bfloat16, kEltsPerThread>(&input_ptr[thread_offset]);
#pragma unroll
  for (int i = 0; i < kEltsPerThread; i++) {
    if (is_neg_zero_bf16(val[i])) {
      val[i] = __float2bfloat16(0.F);
    }
  }

  store<__nv_bfloat16, kEltsPerThread>(
      &scatter_buf_dest[dest_token_offset * kHiddenSize * kWorldSize + rank * kHiddenSize +
                        packed_idx * kEltsPerThread],
      val);

  scheduler.clear_dirty_lamport_buf(mn_buffer_ptrs[rank], TwoShotAllReduceStage::SCATTER);

  // 2. Reduction and Broadcast
  if ((itoken % kWorldSize) == rank) {
    int local_token = itoken / kWorldSize;
    float accum[kEltsPerThread] = {0.F};

    vec_t<float, kLamportEltsPerPacked> values_lamport[kWorldSize];
    while (1) {
      bool valid = true;
#pragma unroll
      for (int r = 0; r < kWorldSize; r++) {
        *reinterpret_cast<float4*>(&values_lamport[r]) = load_global_volatile_f4(
            &scatter_buf_local[local_token * kHiddenSize * kWorldSize + r * kHiddenSize +
                               packed_idx * kEltsPerThread]);
#pragma unroll
        for (int i = 0; i < kLamportEltsPerPacked; i++) {
          valid &= !is_neg_zero_f32(values_lamport[r][i]);
        }
      }
      if (valid) {
        break;
      }
    }

    auto values = reinterpret_cast<vec_t<__nv_bfloat16, kEltsPerThread>*>(values_lamport);
#pragma unroll
    for (int r = 0; r < kWorldSize; r++) {
#pragma unroll
      for (int i = 0; i < kEltsPerThread; i++) {
        accum[i] += __bfloat162float(values[r][i]);
      }
    }

    vec_t<__nv_bfloat16, kEltsPerThread> packed_accum;
#pragma unroll
    for (int i = 0; i < kEltsPerThread; i++) {
      packed_accum[i] = __float2bfloat16(accum[i]);
    }
    store<__nv_bfloat16, kEltsPerThread>(
        &broadcast_buf[itoken * kHiddenSize + packed_idx * kEltsPerThread], packed_accum);
  }
  scheduler.clear_dirty_lamport_buf(mn_buffer_ptrs[rank], TwoShotAllReduceStage::BROADCAST);
}

template <uint32_t kHiddenSize>
__global__
__launch_bounds__(kHiddenSize / (sizeof(float4) / sizeof(__nv_bfloat16))) void rmsnorm_kernel(
    __nv_bfloat16* output_pre_norm, __nv_bfloat16* output_norm, __nv_bfloat16* buffer_input,
    __nv_bfloat16 const* gamma, float epsilon, __nv_bfloat16 const* residual, uint32_t num_tokens,
    uint32_t world_size, uint32_t* buffer_flags) {
  constexpr uint32_t kEltsPerLoad = sizeof(float4) / sizeof(__nv_bfloat16);
  constexpr uint32_t kBlockSize = kHiddenSize / kEltsPerLoad;
  constexpr uint32_t kNumWarps = kBlockSize / 32;
  constexpr uint32_t kSmemBufferSize = kHiddenSize * sizeof(__nv_bfloat16);
  constexpr float kInvHiddenSize = 1.0f / static_cast<float>(kHiddenSize);

  static_assert(kHiddenSize % kEltsPerLoad == 0,
                "kHiddenSize must be a multiple of kEltsPerLoad (8)");
  static_assert(kBlockSize % 32 == 0, "kBlockSize must be a multiple of warpSize (32)");
  static_assert(kBlockSize >= 32 && kBlockSize <= 1024, "kBlockSize out of CUDA block-size range");

  uint32_t const itoken = blockIdx.x;
  uint32_t const thread_offset = threadIdx.x;

  extern __shared__ uint8_t smem[];
  float rms_input[kEltsPerLoad];

  __nv_bfloat16* smem_input = reinterpret_cast<__nv_bfloat16*>(&smem[0]);
  __nv_bfloat16* smem_residual = reinterpret_cast<__nv_bfloat16*>(&smem[kSmemBufferSize]);
  __nv_bfloat16* smem_gamma = reinterpret_cast<__nv_bfloat16*>(&smem[2 * kSmemBufferSize]);

  LamportBufferScheduler scheduler(buffer_flags, TwoShotAllReduceStage::NUM_STAGES);
  __nv_bfloat16* input = reinterpret_cast<__nv_bfloat16*>(scheduler.get_cur_lamport_buf(
      reinterpret_cast<void*>(buffer_input), TwoShotAllReduceStage::BROADCAST));

  cudaTriggerProgrammaticLaunchCompletion();

  uint32_t const block_load_offset = itoken * kHiddenSize;
  uint32_t const thread_load_offset = thread_offset * kEltsPerLoad;
  uint32_t const offset = block_load_offset + thread_load_offset;

  pipeline_async_copy_f4(&smem_residual[thread_load_offset], &residual[offset]);
  __pipeline_commit();
  pipeline_async_copy_f4(&smem_gamma[thread_load_offset], &gamma[thread_load_offset]);
  __pipeline_commit();

  scheduler.single_cta_arrive();
  {
    float4* dst4 = reinterpret_cast<float4*>(&smem_input[thread_load_offset]);
    float4 const* src4 = reinterpret_cast<float4 const*>(&input[offset]);
    while (true) {
      float4 value = load_global_volatile_f4(src4);
      if (!is_neg_zero_f32(value.x)) {
        *dst4 = value;
        break;
      }
    }
  }

  __pipeline_wait_prior(1);
  __syncthreads();

  float thread_sum = 0.f;
  {
    auto inp = load<__nv_bfloat16, kEltsPerLoad>(&smem_input[thread_load_offset]);
    auto res = load<__nv_bfloat16, kEltsPerLoad>(&smem_residual[thread_load_offset]);

    vec_t<__nv_bfloat16, kEltsPerLoad> inp_plus_res;
#pragma unroll
    for (int j = 0; j < kEltsPerLoad; j++) {
      // TODO(draken): Use float square in "+" if accuracy issue ?
      // TODO(draken): Why use bf16x2 in v1 kernel
      inp_plus_res[j] = inp[j] + res[j];
      rms_input[j] = __bfloat162float(inp_plus_res[j]);
      thread_sum += rms_input[j] * rms_input[j];
    }

    store<__nv_bfloat16, kEltsPerLoad>(&output_pre_norm[offset], inp_plus_res);
  }

  __pipeline_wait_prior(0);

  float full_sum = block_reduce_sum_full<float, kNumWarps>(thread_sum);
  float rcp_rms = rsqrtf(full_sum * kInvHiddenSize + epsilon);

  {
    vec_t<__nv_bfloat16, kEltsPerLoad> rms_out;
    auto gamma_vec = to<float>(load<__nv_bfloat16, kEltsPerLoad>(&smem_gamma[thread_load_offset]));

#pragma unroll
    for (uint32_t j = 0; j < kEltsPerLoad; j++) {
      rms_out[j] = __float2bfloat16(gamma_vec[j] * rms_input[j] * rcp_rms);
    }

    store<__nv_bfloat16, kEltsPerLoad>(&output_norm[offset], rms_out);
  }

  constexpr int kEltsSize = sizeof(__nv_bfloat16);
  cudaGridDependencySynchronize();
  scheduler.wait_ctas_and_update(
      {static_cast<uint32_t>(round_up(num_tokens, world_size) * kHiddenSize * kEltsSize),
       static_cast<uint32_t>(num_tokens * kHiddenSize * kEltsSize), 0, 0});
}

}  // namespace kernels

template <uint32_t kHiddenSize>
static inline void launch_fuse_allreduce_rmsnorm_v2(
    const void* input_ptr, const void* mc_input_ptr, void** buffer_ptrs_dev, void* buffer_ptr_local,
    uint32_t* buffer_flags, const void* in_residual_ptr, const void* gamma_ptr, void* output_ptr,
    void* out_residual_ptr, int rank, int world_size, double rms_norm_eps, int num_tokens,
    bool launch_with_pdl, cudaStream_t stream) {
  constexpr uint32_t kEltsPerThread = sizeof(float4) / sizeof(__nv_bfloat16);
  constexpr uint32_t kAllreduceBlock = 128;
  constexpr uint32_t kBlocksPerToken = kHiddenSize / (kAllreduceBlock * kEltsPerThread);
  constexpr uint32_t kRmsBlock = kHiddenSize / kEltsPerThread;
  // smem layout: 3 buffers (input/residual/gamma) of kHiddenSize bf16 elements each.
  constexpr int kSmemSize = 3 * kHiddenSize * sizeof(__nv_bfloat16);

  static_assert(kBlocksPerToken * kAllreduceBlock * kEltsPerThread == kHiddenSize,
                "kHiddenSize must be a multiple of kAllreduceBlock * kEltsPerThread (1024)");
  static_assert(kRmsBlock * kEltsPerThread == kHiddenSize,
                "kHiddenSize must be a multiple of kEltsPerThread (8)");

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = launch_with_pdl ? 1 : 0;

  // 1. Launch two_shot_allreduce_kernel.
  {
    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = dim3(num_tokens, kBlocksPerToken);
    launch_config.blockDim = dim3(kAllreduceBlock);
    launch_config.dynamicSmemBytes = 0;
    launch_config.stream = stream;
    launch_config.attrs = attribute;
    launch_config.numAttrs = 1;

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE)                                                       \
  do {                                                                                            \
    auto allreduce_kernel = kernels::two_shot_allreduce_kernel<WORLD_SIZE, kHiddenSize>;          \
    cudaLaunchKernelEx(&launch_config, allreduce_kernel, static_cast<__nv_bfloat16*>(output_ptr), \
                       static_cast<const __nv_bfloat16*>(input_ptr),                              \
                       reinterpret_cast<__nv_bfloat16**>(buffer_ptrs_dev),                        \
                       static_cast<__nv_bfloat16*>(const_cast<void*>(mc_input_ptr)),              \
                       static_cast<uint32_t>(num_tokens), static_cast<uint32_t>(rank),            \
                       buffer_flags);                                                             \
  } while (0)
    switch (world_size) {
      case 2:
        LAUNCH_ALLREDUCE_KERNEL(2);
        break;
      case 4:
        LAUNCH_ALLREDUCE_KERNEL(4);
        break;
      case 8:
        LAUNCH_ALLREDUCE_KERNEL(8);
        break;
      default:
        return;
    }
#undef LAUNCH_ALLREDUCE_KERNEL
  }

  // TODO(draken): open cudaLaunchAttributeClusterDimension and use CUTA to replace cooprative
  // groups.

  // 2. Launch rmsnorm_kernel.
  {
    auto rmsnorm_kernel = kernels::rmsnorm_kernel<kHiddenSize>;
    cudaFuncSetAttribute(rmsnorm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);

    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = dim3(num_tokens, 1, 1);
    launch_config.blockDim = dim3(kRmsBlock);
    launch_config.dynamicSmemBytes = kSmemSize;
    launch_config.stream = stream;
    launch_config.attrs = attribute;
    launch_config.numAttrs = 1;

    cudaLaunchKernelEx(
        &launch_config, rmsnorm_kernel, static_cast<__nv_bfloat16*>(out_residual_ptr),
        static_cast<__nv_bfloat16*>(output_ptr), static_cast<__nv_bfloat16*>(buffer_ptr_local),
        static_cast<const __nv_bfloat16*>(gamma_ptr), static_cast<float>(rms_norm_eps),
        static_cast<const __nv_bfloat16*>(in_residual_ptr), static_cast<uint32_t>(num_tokens),
        static_cast<uint32_t>(world_size), buffer_flags);
  }
}

void fuse_allreduce_rmsnorm_v2_async(const void* input_ptr, const void* mc_input_ptr,
                                     void** buffer_ptrs_dev, void* buffer_ptr_local,
                                     uint32_t* buffer_flags, const void* in_residual_ptr,
                                     const void* gamma_ptr, void* output_ptr,
                                     void* out_residual_ptr, int rank, int world_size,
                                     double rms_norm_eps, int num_tokens, int hidden_size,
                                     bool launch_with_pdl, cudaStream_t stream) {
#define DISPATCH_FUSE_AR_RMS_V2(HIDDEN)                                                            \
  case HIDDEN:                                                                                     \
    launch_fuse_allreduce_rmsnorm_v2<HIDDEN>(                                                      \
        input_ptr, mc_input_ptr, buffer_ptrs_dev, buffer_ptr_local, buffer_flags, in_residual_ptr, \
        gamma_ptr, output_ptr, out_residual_ptr, rank, world_size, rms_norm_eps, num_tokens,       \
        launch_with_pdl, stream);                                                                  \
    break

  switch (hidden_size) {
    DISPATCH_FUSE_AR_RMS_V2(4096);
    DISPATCH_FUSE_AR_RMS_V2(5120);
    DISPATCH_FUSE_AR_RMS_V2(7168);
    default:
      return;
  }
#undef DISPATCH_FUSE_AR_RMS_V2
}

}  // namespace allreduce
}  // namespace hpc
