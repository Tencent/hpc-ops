// Copyright 2025 hpc-ops authors

#ifndef SRC_ALLREDUCE_FUSE_ALLREDUCE_RMSNORM_LOW_LATENCY_H_
#define SRC_ALLREDUCE_FUSE_ALLREDUCE_RMSNORM_LOW_LATENCY_H_

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>

namespace hpc {
namespace allreduce {

enum MNNVLTwoShotStage : uint8_t {
  SCATTER = 0,
  BROADCAST = 1,
  NUM_STAGES = 2,
};

struct AllReduceFusionParams {
  int nRanks;
  int rank;
  int numTokens;
  int tokenDim;
  void** bufferPtrsDev;
  void* bufferPtrLocal;
  void* multicastPtr;
  uint32_t* bufferFlags;
  bool rmsNormFusion;
  bool launchWithPdl;

  void const* input;
  void const* residualIn;
  void const* gamma;
  double epsilon;

  void* residualOut;
  void* output;
  cudaStream_t stream = nullptr;
};

template <typename T1, typename T2>
__forceinline__ __device__ __host__ constexpr T1 ceil_div(const T1 x, const T2 y) noexcept {
  return (x + y - 1) / y;
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ constexpr T1 round_up(const T1 x, const T2 y) noexcept {
  return ceil_div(x, y) * y;
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ constexpr T1 round_down(const T1 x, const T2 y) noexcept {
  return (x / y) * y;
}

// This function is thread-safe and cached the sm_count.
// But it will only check the current CUDA device, thus assuming each process handles single GPU.
inline int GetCudaMultiProcessorCount() {
  static std::atomic<int> sm_count{0};
  int cached = sm_count.load(std::memory_order_relaxed);
  if (cached == 0) {
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    cached = device_prop.multiProcessorCount;
    sm_count.store(cached, std::memory_order_relaxed);
  }
  return cached;
}

constexpr uint16_t kNEGZERO_FP16 = 0x8000U;

template <typename T>
union Fp16BitCast {
  T mFp;
  uint16_t mInt;

  constexpr Fp16BitCast() : mInt(0) {}

  constexpr Fp16BitCast(T val) : mFp(val) {}

  constexpr Fp16BitCast(uint16_t val) : mInt(val) {}
};

template <typename T>
inline __device__ float toFloat(T val) {
  return val;
}

template <>
inline __device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}
template <>
inline __device__ float toFloat<__nv_half>(__nv_half val) {
  return __half2float(val);
}

template <typename T>
inline __device__ T fromFloat(float val) {
  return val;
}

template <>
inline __device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}

template <>
inline __device__ __nv_half fromFloat<__nv_half>(float val) {
  return __float2half(val);
}

template <typename T>
static constexpr __device__ __host__ T negZero() {
  if constexpr (std::is_same_v<T, float>) {
    return -0.0F;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __nv_half>) {
    return Fp16BitCast<T>(kNEGZERO_FP16).mFp;
  } else {
    static_assert(sizeof(T) == 0, "negativeZero not specialized for this type");
  }
  return T{};  // Never reached, but needed for compilation
}

template <typename T>
static inline __device__ bool isNegZero(T val) {
  if constexpr (std::is_same_v<T, float>) {
    return val == 0.F && signbit(val);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __nv_half>) {
    return Fp16BitCast<T>(val).mInt == kNEGZERO_FP16;
  } else {
    static_assert(sizeof(T) == 0, "isNegZero not specialized for this type");
  }
  return false;  // Never reached, but needed for compilation
}

template <typename PackedType, typename T>
constexpr __device__ __host__ PackedType getPackedLamportInit() {
  static_assert(sizeof(PackedType) % sizeof(T) == 0, "PackedType size must be divisible by T size");
  constexpr int kNumElements = sizeof(PackedType) / sizeof(T);

  union PackedT {
    PackedType mPacked;
    std::array<T, kNumElements> mElements;

    constexpr PackedT() : mElements{} {
      for (int i = 0; i < kNumElements; i++) {
        mElements[i] = negZero<T>();
      }
    }
  };

  PackedT initValue{};
  return initValue.mPacked;
}

// A helper class to get the correct base pointer for a given layout
struct LamportBufferLayout {
  uint32_t numStages = 1;
  uint32_t bytesPerBuffer = 0;
  static constexpr uint32_t sNumLamportBuffers = 3;

  // Implicitly inlined
  [[nodiscard]] __device__ __host__ size_t getTotalBytes() const {
    return numStages * static_cast<size_t>(bytesPerBuffer / numStages) * sNumLamportBuffers;
  }

  // Implicitly inlined
  [[nodiscard]] __device__ __host__ void* getStagePtr(void* bufferBasePtr, uint32_t lamportIndex,
                                                      uint32_t stageIndex) const {
    // Typecast to avoid warnings
    return reinterpret_cast<void*>(
        reinterpret_cast<char*>(bufferBasePtr) +
        static_cast<size_t>((lamportIndex * numStages + stageIndex) *
                            static_cast<size_t>(bytesPerBuffer / numStages)));
  }
};
// Current Index
// Dirty Index
// bytes_per_buffer
// Dirty num_stages
// Dirty bytes_to_clear = {stage0, stage1, stage2, stage3}  # We fix this to 4 stages
// offset_access_ptr

// The following section contains device-only code (uses CUDA built-in
// variables like threadIdx/blockDim/blockIdx/gridDim and PTX inline asm).
// Guard it so that host-only translation units (e.g. *.cc compiled by g++)
// that include this header can still parse the non-device declarations
// above and the host helpers below.
#ifdef __CUDACC__

namespace cg = cooperative_groups;

// PackedType is the one used in kernel for Lamport buffer (LDG.128 or LDG.64)
template <typename PackedType = float4>
struct __attribute__((aligned(32))) LamportFlags {
 public:
  __device__ explicit LamportFlags(uint32_t* bufferFlags, uint32_t numStages = 1)
      : mBufferFlagsPtr(bufferFlags), mFlagAccessPtr(&bufferFlags[8]) {
    mCurBufferLayout.numStages = numStages;
    uint4 flag = reinterpret_cast<uint4*>(bufferFlags)[0];
    mCurrentIndex = flag.x;
    mDirtyIndex = flag.y;
    // Buffer size is unchanged as the flag should be coupled to each buffer
    mCurBufferLayout.bytesPerBuffer = flag.z;
    mDirtyBufferLayout.bytesPerBuffer = flag.z;
    mDirtyBufferLayout.numStages = flag.w;
    *reinterpret_cast<uint4*>(&mBytesToClear) = reinterpret_cast<uint4*>(bufferFlags)[1];
  }

  // Return the base pointer of the lamport buffer indexed by mCurrentIndex and the stageIdx
  [[nodiscard]] __device__ void* getCurLamportBuf(void* bufferBasePtr, int stageIdx = 0) const {
    return mCurBufferLayout.getStagePtr(bufferBasePtr, mCurrentIndex, stageIdx);
  }

  // Fill the dirty lamport buffer with the init value; Use stageIdx to select the stage to clear,
  // -1 to clear all
  // FIXME: Current kernel may use less stages than the dirty numStages; How to guarantee the
  // correctness? CAUTION: This function requires all threads in the grid to participate and ASSUME
  // 1D thread block layout!
  __device__ void clearDirtyLamportBuf(void* bufferBasePtr, int stageIdx = -1) {
    // Rasterize the threads to 1D for flexible clearing

    uint32_t globalCtaIdx = blockIdx.x * gridDim.y + blockIdx.y;
    uint32_t globalTid = globalCtaIdx * blockDim.x + threadIdx.x;
    uint32_t numThreads = gridDim.x * gridDim.y * blockDim.x;

    if (stageIdx == -1) {
      // Clear all stages
      for (uint32_t i = 0; i < mDirtyBufferLayout.numStages; i++) {
        clearPackedBuf(bufferBasePtr, globalTid, numThreads, mBytesToClear[i], mDirtyIndex, i);
      }
    } else if (stageIdx < mDirtyBufferLayout.numStages) {
      clearPackedBuf(bufferBasePtr, globalTid, numThreads, mBytesToClear[stageIdx], mDirtyIndex,
                     stageIdx);
    }
  }

  __device__ void ctaArrive() {
    cg::cluster_group cluster = cg::this_cluster();
    // We update the atomic counter per cluster
    int tid = cluster.thread_rank();
    cluster.sync();
    if (tid == 0) {
      // red.async does not accept the .release qualifier (rejected by ptxas on
      // sm_90a / CUDA 13). Use the plain release reduction instead, which is the
      // intended global atomic-add-with-release semantics for the flag counter.
      asm volatile("red.release.global.gpu.add.u32 [%0], %1;" ::"l"(mFlagAccessPtr), "r"(1)
                   : "memory");
    }
  }

  __device__ void waitAndUpdate(uint4 bytesToClearPerStage) {
    cg::grid_group grid = cg::this_grid();
    // Use the first thread instead of the last thread as the last thread may exit early
    bool isLastCtaT0 = grid.thread_rank() == 0;
    int targetCount = grid.num_clusters();
    if (isLastCtaT0) {
      uint4* flagPtr = reinterpret_cast<uint4*>(mBufferFlagsPtr);
      while (*reinterpret_cast<uint32_t volatile*>(mFlagAccessPtr) < targetCount) {
      }
      // 'Current' becomes 'Dirty'
      flagPtr[0] = {(mCurrentIndex + 1) % 3,          // Current index
                    mCurrentIndex,                    // Dirty index
                    mCurBufferLayout.bytesPerBuffer,  // Buffer size
                    mCurBufferLayout.numStages};      // Dirty - Number of stages
      flagPtr[1] = bytesToClearPerStage;
      *mFlagAccessPtr = 0;
    }
  }

 private:
  uint32_t* mBufferFlagsPtr;
  uint32_t* mFlagAccessPtr;

  uint32_t mCurrentIndex, mDirtyIndex;
  // So that we can access it with uint4
  alignas(16) std::array<uint32_t, 4> mBytesToClear;
  LamportBufferLayout mCurBufferLayout, mDirtyBufferLayout;

  inline __device__ void clearPackedBuf(void* bufferBasePtr, uint32_t globalTid,
                                        uint32_t numThreads, uint32_t bytesToClear,
                                        uint8_t dirtyIndex, uint8_t stageIdx) {
    // Round up to the float4 boundary
    uint32_t clearBoundary = ceil_div<uint32_t>(bytesToClear, sizeof(PackedType));
    for (uint32_t packedIdx = globalTid; packedIdx < clearBoundary; packedIdx += numThreads) {
      reinterpret_cast<PackedType*>(
          mDirtyBufferLayout.getStagePtr(bufferBasePtr, dirtyIndex, stageIdx))[packedIdx] =
          getPackedLamportInit<PackedType, float>();
    }
  }
};

template <typename PackedType, typename T>
union PackedVec {
  PackedType packed;
  T elements[sizeof(PackedType) / sizeof(T)];

  __device__ PackedVec& operator+=(PackedVec& other) {
#pragma unroll
    for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++) {
      elements[i] += other.elements[i];
    }
    return *this;
  }

  __device__ PackedVec operator+(PackedVec& other) {
    PackedVec result;
#pragma unroll
    for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++) {
      result.elements[i] = elements[i] + other.elements[i];
    }
    return result;
  }
};

template <typename PackedType, typename T>
inline __device__ PackedType loadPacked(T* ptr) {
  return *reinterpret_cast<PackedType*>(ptr);
}

template <typename PackedType, typename T>
inline __device__ const PackedType loadPacked(T const* ptr) {
  return *reinterpret_cast<PackedType const*>(ptr);
}

template <typename PackedType>
inline __device__ PackedType loadPackedVolatile(void const* ptr) {
  static_assert(sizeof(PackedType) == 0, "Not implemented");
  return PackedType{};
}

template <>
inline __device__ float4 loadPackedVolatile<float4>(void const* ptr) {
  float4 returnValue;
  asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(returnValue.x), "=f"(returnValue.y), "=f"(returnValue.z), "=f"(returnValue.w)
               : "l"(ptr));
  return returnValue;
}

template <>
inline __device__ float2 loadPackedVolatile<float2>(void const* ptr) {
  float2 returnValue;
  asm volatile("ld.volatile.global.v2.f32 {%0, %1}, [%2];\n"
               : "=f"(returnValue.x), "=f"(returnValue.y)
               : "l"(ptr));
  return returnValue;
}

template <typename T_IN>
inline __device__ void copyF4(T_IN* dst, T_IN const* src) {
  float4* dst4 = reinterpret_cast<float4*>(dst);
  float4 const* src4 = reinterpret_cast<float4 const*>(src);
  __pipeline_memcpy_async(dst4, src4, sizeof(float4));
}

uint32_t constexpr kWARP_SIZE = 32U;
uint32_t constexpr kLOG2_WARP_SIZE = 5U;
uint32_t constexpr kLANE_ID_MASK = 0x1f;
uint32_t constexpr kFINAL_MASK = 0xffffffff;

template <typename T>
inline __device__ T warpReduceSumFull(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(kFINAL_MASK, val, mask, kWARP_SIZE);
  }
  return val;
}

template <typename T>
inline __device__ T warpReduceSumPartial(T val) {
  int laneId = threadIdx.x & kLANE_ID_MASK;
  // We make sure only the last warp will call this function
  int warpSize = blockDim.x - (threadIdx.x & ~(kWARP_SIZE - 1));
  unsigned int active_mask = (1U << warpSize) - 1;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    int targetLane = laneId ^ mask;
    auto tmp = __shfl_xor_sync(active_mask, val, mask, kWARP_SIZE);
    val += targetLane < warpSize ? tmp : 0;
  }
  return val;
}

// SYNC:
//  - True: share the sum across all threads
//  - False: only thread 0 get the sum; Other thread's value is undefined.
template <typename T, bool SYNC = false>
inline __device__ T blockReduceSumPartial(T val) {
  __shared__ T smem[kWARP_SIZE];
  int laneId = threadIdx.x & kLANE_ID_MASK;
  int warpId = threadIdx.x >> kLOG2_WARP_SIZE;
  int warpNum = (blockDim.x + kWARP_SIZE - 1) >>
                kLOG2_WARP_SIZE;  // Ceiling division to include partial warps

  val = (warpId == warpNum - 1) ? warpReduceSumPartial(val) : warpReduceSumFull(val);
  if (laneId == 0) {
    smem[warpId] = val;
  }
  __syncthreads();

  if (warpId == 0) {
    val = (laneId < warpNum) ? smem[laneId] : (T)0.f;
    // Need to consider the corner case where we only have one warp and it is partial
    val = (warpNum == 1) ? warpReduceSumPartial(val) : warpReduceSumFull(val);

    if constexpr (SYNC) {
      if (laneId == 0) {
        smem[warpId] = val;
      }
    }
  }
  if constexpr (SYNC) {
    __syncthreads();
    val = smem[0];
  }
  return val;
}

template <typename T>
inline __device__ T blockReduceSumFull(T val) {
  __shared__ T smem[kWARP_SIZE];
  int lane_id = threadIdx.x & kLANE_ID_MASK;
  int warp_id = threadIdx.x >> kLOG2_WARP_SIZE;
  int warp_num = blockDim.x >> kLOG2_WARP_SIZE;

  val = warpReduceSumFull(val);
  if (lane_id == 0) {
    smem[warp_id] = val;
  }
  __syncthreads();

  val = (lane_id < warp_num) ? smem[lane_id] : (T)0.f;
  val = warpReduceSumFull(val);

  return val;
}

template <typename T, bool SYNC = false>
inline __device__ T blockReduceSum(T val) {
  bool hasPartialWarp = (blockDim.x & kLANE_ID_MASK) != 0;
  if (hasPartialWarp) {
    return blockReduceSumPartial<T, SYNC>(val);
  } else {
    return blockReduceSumFull<T>(val);
  }
}

#endif  // __CUDACC__

// A helper function to tune the grid configuration for fused oneshot and rmsnorm kernels
// Return (block_size, cluster_size, loads_per_thread)
// TODO(draken): I have modified this function. This function assumes the target is B200 (sm_100),
// which always supports CGA.
inline std::tuple<int, int, int> adjustGridConfig(int numTokens, int dim, int eltsPerThread) {
  // B200 always supports CGA, start with the preferred cluster size
  int clusterSize = 8;
  int blockSize = 128;
  // ========================== Adjust the grid configuration ==========================
  int threadsNeeded = ceil_div(dim, eltsPerThread);
  int loadsPerThread = 1;

  while (threadsNeeded % clusterSize != 0 && clusterSize > 1) {
    clusterSize /= 2;
  }
  blockSize = ceil_div(threadsNeeded, clusterSize);
  while (blockSize < 128 && clusterSize >= 2) {
    blockSize *= 2;
    clusterSize /= 2;
  }
  int smCount = GetCudaMultiProcessorCount();
  while (numTokens * clusterSize > smCount && clusterSize > 1 && blockSize <= 512) {
    blockSize *= 2;
    clusterSize /= 2;
  }
  // Trying to scale up use CGA
  while (blockSize > 1024) {
    if (clusterSize < 8) {
      clusterSize = clusterSize << 1;
    } else {
      break;
    }
    blockSize = ceil_div(threadsNeeded, clusterSize * loadsPerThread);
  }
  return {blockSize, clusterSize, loadsPerThread};
}

template <typename T>
cudaError_t fuse_allreduce_rmsnorm_low_latency_async(AllReduceFusionParams const& params);

}  // namespace allreduce
}  // namespace hpc

#endif  // SRC_ALLREDUCE_FUSE_ALLREDUCE_RMSNORM_LOW_LATENCY_H_
