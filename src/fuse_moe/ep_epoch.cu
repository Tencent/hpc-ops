// Copyright (C) 2026 Tencent.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/library.h>

#include <cstdint>

namespace hpc {
namespace fuse_moe {

namespace {

__device__ __forceinline__ void store_release_system(uint32_t *address, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" : : "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ uint32_t load_acquire_system(const uint32_t *address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__global__ void publish_ready_kernel(const int64_t *__restrict__ signal_ptrs, int rank,
                                     int world_size, const int32_t *__restrict__ generation_ptr) {
  int target = blockIdx.x * blockDim.x + threadIdx.x;
  if (target >= world_size) {
    return;
  }
  auto *target_signal = reinterpret_cast<uint32_t *>(signal_ptrs[target]);
  store_release_system(target_signal + rank, static_cast<uint32_t>(generation_ptr[0]));
}

__global__ void wait_ready_kernel(const uint32_t *__restrict__ local_signal, int world_size,
                                  const int32_t *__restrict__ generation_ptr) {
  int producer = blockIdx.x * blockDim.x + threadIdx.x;
  if (producer >= world_size) {
    return;
  }
  uint32_t generation = static_cast<uint32_t>(generation_ptr[0]);
  while (static_cast<int32_t>(load_acquire_system(local_signal + producer) - generation) < 0) {
    __nanosleep(64);
  }
}

void check_generation(const at::Tensor &generation, int device) {
  TORCH_CHECK(generation.is_cuda(), "FusedMoE EP generation must be CUDA");
  TORCH_CHECK(generation.is_contiguous(), "FusedMoE EP generation must be contiguous");
  TORCH_CHECK(generation.scalar_type() == at::kInt, "FusedMoE EP generation must be int32");
  TORCH_CHECK(generation.numel() == 1, "FusedMoE EP generation must contain one value");
  TORCH_CHECK(generation.get_device() == device,
              "FusedMoE EP generation must be on the operation device");
}

}  // namespace

void fuse_moe_ep_publish(const at::Tensor &signal_ptrs, int64_t rank, int64_t world_size,
                         const at::Tensor &generation, int64_t signal_words) {
  TORCH_CHECK(world_size > 0 && world_size <= 72, "FusedMoE EP world_size must be in [1, 72]");
  TORCH_CHECK(signal_words >= world_size, "FusedMoE EP signal storage is too small");
  TORCH_CHECK(signal_ptrs.is_cuda(), "FusedMoE EP signal pointer table must be CUDA");
  TORCH_CHECK(signal_ptrs.is_contiguous(), "FusedMoE EP signal pointer table must be contiguous");
  TORCH_CHECK(signal_ptrs.dim() == 1 && signal_ptrs.numel() >= world_size,
              "FusedMoE EP signal pointer table must contain every rank");
  TORCH_CHECK(signal_ptrs.scalar_type() == at::kLong,
              "FusedMoE EP signal pointer table must be int64");
  TORCH_CHECK(rank >= 0 && rank < world_size, "FusedMoE EP rank is out of range");
  check_generation(generation, signal_ptrs.get_device());

  auto stream = at::cuda::getCurrentCUDAStream(signal_ptrs.get_device());
  constexpr int threads = 32;
  int blocks = (world_size + threads - 1) / threads;
  publish_ready_kernel<<<blocks, threads, 0, stream>>>(
      signal_ptrs.const_data_ptr<int64_t>(), static_cast<int>(rank), static_cast<int>(world_size),
      generation.const_data_ptr<int32_t>());
  auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "FusedMoE EP publish launch failed: ", cudaGetErrorString(error));
}

void fuse_moe_ep_wait(const at::Tensor &local_signal, int64_t world_size,
                      const at::Tensor &generation) {
  TORCH_CHECK(world_size > 0 && world_size <= 72, "FusedMoE EP world_size must be in [1, 72]");
  TORCH_CHECK(local_signal.is_cuda(), "FusedMoE EP local signal must be CUDA");
  TORCH_CHECK(local_signal.is_contiguous(), "FusedMoE EP local signal must be contiguous");
  TORCH_CHECK(local_signal.dim() == 1 && local_signal.numel() >= world_size,
              "FusedMoE EP local signal must contain every rank");
  TORCH_CHECK(local_signal.scalar_type() == at::kUInt32, "FusedMoE EP local signal must be uint32");
  check_generation(generation, local_signal.get_device());

  auto stream = at::cuda::getCurrentCUDAStream(local_signal.get_device());
  constexpr int threads = 32;
  int blocks = (world_size + threads - 1) / threads;
  wait_ready_kernel<<<blocks, threads, 0, stream>>>(local_signal.const_data_ptr<uint32_t>(),
                                                    static_cast<int>(world_size),
                                                    generation.const_data_ptr<int32_t>());
  auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "FusedMoE EP wait launch failed: ", cudaGetErrorString(error));
}

}  // namespace fuse_moe
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "fuse_moe_ep_publish(Tensor(a!) signal_ptrs, int rank, int world_size, Tensor generation, "
      "int "
      "signal_words) -> ()");
  m.impl("fuse_moe_ep_publish", c10::DispatchKey::CUDA, &hpc::fuse_moe::fuse_moe_ep_publish);
  m.def("fuse_moe_ep_wait(Tensor(a!) local_signal, int world_size, Tensor generation) -> ()");
  m.impl("fuse_moe_ep_wait", c10::DispatchKey::CUDA, &hpc::fuse_moe::fuse_moe_ep_wait);
}
