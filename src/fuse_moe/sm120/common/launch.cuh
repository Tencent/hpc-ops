// Copyright (C) 2026 Tencent.
//
// Shared kernel launches for the SM120 MoE: CUTLASS GEMMs and PDL host kernels.

#ifndef SRC_FUSE_MOE_SM120_COMMON_LAUNCH_CUH_
#define SRC_FUSE_MOE_SM120_COMMON_LAUNCH_CUH_

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cutlass/device_kernel.h>

namespace hpc::fuse_moe::sm120 {

template <typename Kernel>
__forceinline__ void launch_cutlass_kernel(typename Kernel::Params const& params, int num_sms,
                                           cudaStream_t stream) {
  cudaLaunchConfig_t launch_config;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;

  launch_config.gridDim = Kernel::get_grid_shape(num_sms);
  launch_config.blockDim = Kernel::get_block_shape();
  launch_config.dynamicSmemBytes = Kernel::kSmemSize;
  launch_config.stream = stream;
  launch_config.attrs = attrs;
  launch_config.numAttrs = 1;

  auto kernel_ptr = &cutlass::device_kernel<Kernel>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      Kernel::kSmemSize));
  C10_CUDA_CHECK(cudaLaunchKernelEx(&launch_config, kernel_ptr, params));
}

template <typename Kernel, typename... Args>
void launch_pdl(Kernel kernel, dim3 grid, dim3 block, cudaStream_t stream, Args... args) {
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}

}  // namespace hpc::fuse_moe::sm120

#endif  // SRC_FUSE_MOE_SM120_COMMON_LAUNCH_CUH_
