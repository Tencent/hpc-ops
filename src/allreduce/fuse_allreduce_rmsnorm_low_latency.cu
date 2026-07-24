// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <algorithm>

#include "src/allreduce/fuse_allreduce_rmsnorm_low_latency.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace allreduce {
namespace kernels {

template <uint8_t WorldSize, typename T, typename PackedType = float4>
__global__ __launch_bounds__(128) void twoshotAllreduceKernel(
    T* outputPtr, T const* shardPtr, T** inputPtrs, T* mcastPtr, uint32_t const numTokens,
    uint32_t const tokenDim, uint32_t const rank, uint32_t* bufferFlags,
    bool const wait_for_results) {
  constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
  constexpr int kLAMPORT_ELTS_PER_PACKED = sizeof(PackedType) / sizeof(float);
  constexpr uint32_t kELT_SIZE = sizeof(T);

  int packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
  int token = blockIdx.x;
  // Offset w.r.t. the input shard
  int threadOffset = token * tokenDim + packedIdx * kELTS_PER_THREAD;

  int destRank = token % WorldSize;
  int destTokenOffset = token / WorldSize;
  cudaGridDependencySynchronize();
  LamportFlags<PackedType> flag(bufferFlags, MNNVLTwoShotStage::NUM_STAGES);

  T* scatterBufLocal =
      reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[rank], MNNVLTwoShotStage::SCATTER));
  T* scatterBufDest =
      reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[destRank], MNNVLTwoShotStage::SCATTER));
  T* broadcastBufW =
      reinterpret_cast<T*>(flag.getCurLamportBuf(mcastPtr, MNNVLTwoShotStage::BROADCAST));
  T* broadcastBufR =
      reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[rank], MNNVLTwoShotStage::BROADCAST));

  cudaTriggerProgrammaticLaunchCompletion();
  // Make sure the clear function is called before OOB thread exits
  if (packedIdx * kELTS_PER_THREAD >= tokenDim) {
    flag.clearDirtyLamportBuf(inputPtrs[rank], -1);
    return;
  }

  // =============================== Scatter ===============================

  // Load vectorized data
  PackedVec<PackedType, T> val;
  val.packed = loadPacked<PackedType>(&shardPtr[threadOffset]);
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    if (isNegZero(val.elements[i])) {
      val.elements[i] = fromFloat<T>(0.F);
    }
  }

  // Store vectorized data
  reinterpret_cast<PackedType*>(
      &scatterBufDest[destTokenOffset * tokenDim * WorldSize + rank * tokenDim])[packedIdx] =
      val.packed;

  flag.clearDirtyLamportBuf(inputPtrs[rank], MNNVLTwoShotStage::SCATTER);

  // =============================== Reduction and Broadcast ===============================

  if ((token % WorldSize) == rank) {
    int localToken = token / WorldSize;
    float accum[kELTS_PER_THREAD] = {0.F};

    // Use float as we only check each float value for validity
    PackedVec<PackedType, float> valuesLamport[WorldSize];
    while (1) {
      bool valid = true;
#pragma unroll
      for (int r = 0; r < WorldSize; r++) {
        valuesLamport[r].packed = loadPackedVolatile<PackedType>(
            &scatterBufLocal[localToken * tokenDim * WorldSize + r * tokenDim +
                             packedIdx * kELTS_PER_THREAD]);

        // Check validity across all elements
#pragma unroll
        for (int i = 0; i < kLAMPORT_ELTS_PER_PACKED; i++) {
          valid &= !isNegZero(valuesLamport[r].elements[i]);
        }
      }
      if (valid) {
        break;
      }
    }

    // Now we view it as the value for reduction
    auto values = reinterpret_cast<PackedVec<PackedType, T>*>(valuesLamport);
#pragma unroll
    for (int r = 0; r < WorldSize; r++) {
#pragma unroll
      for (int i = 0; i < kELTS_PER_THREAD; i++) {
        accum[i] += toFloat<T>(values[r].elements[i]);
      }
    }

    // Store vectorized result
    PackedVec<PackedType, T> packedAccum;
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      packedAccum.elements[i] = fromFloat<T>(accum[i]);
    }
    reinterpret_cast<PackedType*>(&broadcastBufW[token * tokenDim])[packedIdx] = packedAccum.packed;
  }

  flag.clearDirtyLamportBuf(inputPtrs[rank], MNNVLTwoShotStage::BROADCAST);

  // Optionally wait for results if the next layer isn't doing the Lamport check
  if (wait_for_results) {
    // Update the atomic counter to indicate the block has read the offsets
    flag.ctaArrive();

    PackedVec<PackedType, float> valLamport;
    valLamport.packed = loadPackedVolatile<PackedType>(&broadcastBufR[threadOffset]);
    while (isNegZero(valLamport.elements[0])) {
      valLamport.packed = loadPackedVolatile<PackedType>(&broadcastBufR[threadOffset]);
    }
    if (outputPtr) {
      reinterpret_cast<PackedType*>(&outputPtr[threadOffset])[0] = valLamport.packed;
    }

    // Update the buffer flags
    flag.waitAndUpdate(
        {static_cast<uint32_t>(round_up(numTokens, WorldSize) * tokenDim *
                               kELT_SIZE),                         // Clear Size for scatter stage
         static_cast<uint32_t>(numTokens * tokenDim * kELT_SIZE),  // Clear Size for broadcast stage
         0, 0});
    // If not wait for results, we will rely on the following kernel to update the buffer
  }
}

template <typename T_IN, typename T_OUT, int LoadsPerThread = 1>
__global__ __launch_bounds__(1024) void rmsNormLamport(T_IN* outputPreNorm, T_OUT* outputNorm,
                                                       T_IN* bufferInput, T_IN const* gamma,
                                                       float epsilon, T_IN const* residual,
                                                       uint32_t numTokens, uint32_t dim,
                                                       uint32_t worldSize, uint32_t* bufferFlags) {
  static_assert(std::is_same_v<T_IN, T_OUT>, "T_IN and T_OUT must be the same type");
  static int const kELTS_PER_LOAD = sizeof(float4) / sizeof(T_IN);

  uint32_t const token = blockIdx.x;
  uint32_t const blockSize = blockDim.x;
  uint32_t const threadOffset = threadIdx.x;

  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  uint32_t numThreads = cluster.num_threads();
  uint32_t clusterSize = cluster.num_blocks();
  uint32_t blockOffset = cluster.block_rank();
  uint32_t const dimPadded = round_up(dim, kELTS_PER_LOAD * numThreads);
  uint32_t const elemsPerThread = dimPadded / numThreads;
  uint32_t const loadStride = blockSize;

  extern __shared__ uint8_t smem[];
  float rInput[LoadsPerThread * kELTS_PER_LOAD];
  uint32_t offsets[LoadsPerThread * kELTS_PER_LOAD];

  uint32_t const smemBufferSize = blockSize * elemsPerThread * sizeof(T_IN);
  T_IN* smemInput = (T_IN*)&smem[0];
  T_IN* smemResidual = (T_IN*)&smem[smemBufferSize];
  T_IN* smemGamma = (T_IN*)&smem[2 * smemBufferSize];

  LamportFlags<float4> flag(bufferFlags, MNNVLTwoShotStage::NUM_STAGES);
  T_IN* input = reinterpret_cast<T_IN*>(
      flag.getCurLamportBuf(reinterpret_cast<void*>(bufferInput), MNNVLTwoShotStage::BROADCAST));

  cudaTriggerProgrammaticLaunchCompletion();
  // The offset that current thread should load from. Note that the hidden dimension is split by CGA
  // size and each block loads a contiguous chunk; The size of chunk that each block processes
  uint32_t const blockChunkSize = ceil_div(dim, clusterSize * kELTS_PER_LOAD) * kELTS_PER_LOAD;
  uint32_t const blockLoadOffset = token * dim + blockOffset * blockChunkSize;

#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    // Each block load a contiguous chunk of tokens
    uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    offsets[i] = blockLoadOffset + threadLoadOffset;
  }

#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      copyF4(&smemResidual[threadLoadOffset], &residual[blockLoadOffset + threadLoadOffset]);
    }
  }
  __pipeline_commit();
#pragma unroll
  for (uint32_t i = 0; i < LoadsPerThread; i++) {
    uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      copyF4(&smemGamma[threadLoadOffset], &gamma[blockOffset * blockChunkSize + threadLoadOffset]);
    }
  }
  __pipeline_commit();

  flag.ctaArrive();
  bool valid = false;
  // ACQBLK if not lamport
  while (!valid) {
    valid = true;
#pragma unroll
    for (uint32_t i = 0; i < LoadsPerThread; i++) {
      uint32_t threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;

      if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
        float4* dst4 = reinterpret_cast<float4*>(&smemInput[threadLoadOffset]);
        float4 const* src4 = reinterpret_cast<float4 const*>(&input[offsets[i]]);

        float4 value = loadPackedVolatile<float4>(src4);
        // Assume that the 16B were written atomically, so we only need to check one value
        valid &= !isNegZero(value.x);
        *dst4 = value;
      }
    }
  }

  __pipeline_wait_prior(1);
  __syncthreads();

  float threadSum = 0.f;
#pragma unroll
  for (int i = 0; i < LoadsPerThread; i++) {
    int threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      PackedVec<float4, T_IN> inp{.packed = loadPacked<float4>(&smemInput[threadLoadOffset])};
      PackedVec<float4, T_IN> res{.packed = loadPacked<float4>(&smemResidual[threadLoadOffset])};

      PackedVec<float4, T_IN> inp_plus_res = inp + res;
#pragma unroll
      for (int j = 0; j < kELTS_PER_LOAD; j++) {
        rInput[i * kELTS_PER_LOAD + j] = toFloat<T_IN>(inp_plus_res.elements[j]);
        threadSum += toFloat<T_IN>(inp_plus_res.elements[j] * inp_plus_res.elements[j]);
      }

      *reinterpret_cast<float4*>(&outputPreNorm[blockLoadOffset + threadLoadOffset]) =
          inp_plus_res.packed;
    }
  }

  __pipeline_wait_prior(0);

  float blockSum = blockReduceSum<float, true>(threadSum);

  float fullSum = blockSum;
  __shared__ float sharedVal[8];
  // B200 always supports CGA Reduction
  int const numBlocks = cluster.num_blocks();
  if (numBlocks > 1) {
    fullSum = 0.F;
    // Need to reduce over the entire cluster
    int const blockRank = cluster.block_rank();
    if (threadIdx.x < numBlocks) {
      cluster.map_shared_rank(&sharedVal[0], threadIdx.x)[blockRank] = blockSum;
    }
    cluster.barrier_wait(cluster.barrier_arrive());
    for (int i = 0; i < numBlocks; ++i) {
      fullSum += sharedVal[i];
    }
  }

  float rcpRms = rsqrtf(fullSum / dim + epsilon);

#pragma unroll
  for (int i = 0; i < LoadsPerThread; i++) {
    PackedVec<float4, T_OUT> r_out;
    uint32_t threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
    if (blockOffset * blockChunkSize + threadLoadOffset < dim) {
      PackedVec<float4, T_IN> gamma = {.packed = loadPacked<float4>(&smemGamma[threadLoadOffset])};

#pragma unroll
      for (uint32_t j = 0; j < kELTS_PER_LOAD; j++) {
        r_out.elements[j] = fromFloat<T_OUT>(toFloat<T_IN>(gamma.elements[j]) *
                                             rInput[i * kELTS_PER_LOAD + j] * rcpRms);
      }

      *reinterpret_cast<float4*>(&outputNorm[blockLoadOffset + threadLoadOffset]) = r_out.packed;
    }
  }
  constexpr int kELTS_SIZE = sizeof(T_IN);

  // Assume the previous kernel does not modify the buffer_flags.
  cudaGridDependencySynchronize();
  // Update the buffer pointers
  flag.waitAndUpdate({static_cast<uint32_t>(round_up(numTokens, worldSize) * dim * kELTS_SIZE),
                      static_cast<uint32_t>(numTokens * dim * kELTS_SIZE), 0, 0});
}

}  // namespace kernels

template <typename T>
cudaError_t fuse_allreduce_rmsnorm_low_latency_async(AllReduceFusionParams const& params) {
  int const numTokens = params.numTokens;
  int const tokenDim = params.tokenDim;
  int const numEltsPerThread = sizeof(float4) / sizeof(T);
  if (tokenDim % numEltsPerThread != 0) {
    return cudaErrorInvalidValue;
  }

  int const arNumThreads = ceil_div(tokenDim, numEltsPerThread);
  int const arNumBlocksPerToken = ceil_div(arNumThreads, 128);

  dim3 arGrid(numTokens, arNumBlocksPerToken);

  cudaLaunchAttribute arAttrs[1];
  arAttrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  arAttrs[0].val.programmaticStreamSerializationAllowed = params.launchWithPdl ? 1 : 0;

  cudaLaunchConfig_t arConfig{
      .gridDim = arGrid,
      .blockDim = 128,
      .dynamicSmemBytes = 0,
      .stream = params.stream,
      .attrs = arAttrs,
      .numAttrs = 1,
  };

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE)                                                       \
  do {                                                                                            \
    cudaError_t _err = cudaLaunchKernelEx(                                                        \
        &arConfig, &kernels::twoshotAllreduceKernel<WORLD_SIZE, T>, output, input, ucPtrs,        \
        mcastPtr, numTokens, tokenDim, params.rank, params.bufferFlags, (!params.rmsNormFusion)); \
    if (_err != cudaSuccess) {                                                                    \
      return _err;                                                                                \
    }                                                                                             \
  } while (0)
  T** ucPtrs = reinterpret_cast<T**>(params.bufferPtrsDev);
  T* mcastPtr = reinterpret_cast<T*>(params.multicastPtr);
  T* output = reinterpret_cast<T*>(params.output);
  T const* input = reinterpret_cast<T const*>(params.input);
  switch (params.nRanks) {
    case 2:
      LAUNCH_ALLREDUCE_KERNEL(2);
      break;
    case 4:
      LAUNCH_ALLREDUCE_KERNEL(4);
      break;
    case 8:
      LAUNCH_ALLREDUCE_KERNEL(8);
      break;
    case 16:
      LAUNCH_ALLREDUCE_KERNEL(16);
      break;
    case 32:
      LAUNCH_ALLREDUCE_KERNEL(32);
      break;
    case 64:
      LAUNCH_ALLREDUCE_KERNEL(64);
      break;
    default:
      return cudaErrorInvalidValue;
  }
#undef LAUNCH_ALLREDUCE_KERNEL

  // Launch the rmsnorm lamport kernel if fusion is enabled
  if (params.rmsNormFusion) {
    // B200 (sm_100) always supports CGA
    auto gridConfig = adjustGridConfig(numTokens, tokenDim, numEltsPerThread);
    int rnBlockSize = std::get<0>(gridConfig);
    int rnClusterSize = std::get<1>(gridConfig);
    int rnLoadsPerThread = std::get<2>(gridConfig);

    int rnNumThreads = rnClusterSize * rnBlockSize;
    dim3 rnGrid(numTokens, rnClusterSize, 1);
    cudaLaunchConfig_t rnConfig;
    cudaLaunchAttribute rnAttrs[2];
    rnConfig.stream = params.stream;
    rnConfig.gridDim = rnGrid;
    rnConfig.blockDim = rnBlockSize;
    rnConfig.attrs = rnAttrs;
    rnAttrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    rnAttrs[0].val.programmaticStreamSerializationAllowed = params.launchWithPdl ? 1 : 0;
    rnAttrs[1].id = cudaLaunchAttributeClusterDimension;
    rnAttrs[1].val.clusterDim.x = 1;
    rnAttrs[1].val.clusterDim.y = rnClusterSize;
    rnAttrs[1].val.clusterDim.z = 1;
    rnConfig.numAttrs = 2;

    bool const rnUseCGA = rnClusterSize > 1;
    int const dimPadded = round_up(tokenDim, numEltsPerThread * rnNumThreads);
    int const iters = dimPadded / rnNumThreads;

    size_t const smemSize = 3 * rnBlockSize * iters * sizeof(T);

#define RUN_RMSNORM_KERNEL(LOADS_PER_THREAD)                                                       \
  do {                                                                                             \
    cudaError_t _err =                                                                             \
        cudaFuncSetAttribute(&kernels::rmsNormLamport<T, T, LOADS_PER_THREAD>,                     \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize);               \
    if (_err != cudaSuccess) {                                                                     \
      return _err;                                                                                 \
    }                                                                                              \
    rnConfig.dynamicSmemBytes = smemSize;                                                          \
    _err = cudaLaunchKernelEx(&rnConfig, &kernels::rmsNormLamport<T, T, LOADS_PER_THREAD>,         \
                              residualOut, output, bufferInput, gamma,                             \
                              static_cast<float>(params.epsilon), residualIn, numTokens, tokenDim, \
                              params.nRanks, params.bufferFlags);                                  \
    if (_err != cudaSuccess) {                                                                     \
      return _err;                                                                                 \
    }                                                                                              \
  } while (0)

    T* residualOut = reinterpret_cast<T*>(params.residualOut);
    T* output = reinterpret_cast<T*>(params.output);
    T* bufferInput = reinterpret_cast<T*>(params.bufferPtrLocal);
    T const* gamma = reinterpret_cast<T const*>(params.gamma);
    T const* residualIn = reinterpret_cast<T const*>(params.residualIn);
    if (rnUseCGA) {
      RUN_RMSNORM_KERNEL(1);
    } else {
      switch (rnLoadsPerThread) {
        case 1:
          RUN_RMSNORM_KERNEL(1);
          break;
        case 2:
          RUN_RMSNORM_KERNEL(2);
          break;
        case 3:
          RUN_RMSNORM_KERNEL(3);
          break;
        case 4:
          RUN_RMSNORM_KERNEL(4);
          break;
        case 5:
          RUN_RMSNORM_KERNEL(5);
          break;
        case 6:
          RUN_RMSNORM_KERNEL(6);
          break;
        case 7:
          RUN_RMSNORM_KERNEL(7);
          break;
        case 8:
          RUN_RMSNORM_KERNEL(8);
          break;
        default:
          return cudaErrorInvalidValue;
      }  // switch (rnLoadsPerThread)
    }  // if (rnUseCGA)
#undef RUN_RMSNORM_KERNEL

  }  // if (params.rmsNormFusion)
  return cudaSuccess;
}

// Explicit template instantiations so that symbols are available to callers
// compiled in other translation units (e.g. entry.cc compiled by host compiler).
template cudaError_t fuse_allreduce_rmsnorm_low_latency_async<__nv_bfloat16>(
    AllReduceFusionParams const& params);

}  // namespace allreduce
}  // namespace hpc
