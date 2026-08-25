// Copyright (C) 2026 Tencent.

#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

#include "src/fuse_moe/fuse_moe.h"

namespace hpc {
namespace fuse_moe {
namespace kernels {

__global__ void pull_indexed_rows_kernel(
    const void *const *__restrict__ input_ptrs, const void *const *__restrict__ input_scale_ptrs,
    int rows_per_shard, int source_row_limit, int64_t input_row_stride,
    int64_t input_scale_row_stride, const int *__restrict__ row_indices,
    const int *__restrict__ cu_num_tokens_per_group, const int *__restrict__ cu_tiles,
    void *__restrict__ output, float *__restrict__ output_scale, int hidden_size, int scale_groups,
    int num_padded_tokens, int num_expert_local, int tile_m) {
  cudaGridDependencySynchronize();

  const int output_row = blockIdx.x;
  const int active_rows = cu_num_tokens_per_group[num_expert_local];
  if (output_row < active_rows) {
    const int source_row = row_indices[output_row];
    const bool source_row_valid = source_row >= 0 && source_row < source_row_limit;
    const int safe_source_row = source_row_valid ? source_row : 0;
    const int owner = safe_source_row / rows_per_shard;
    const int owner_row = safe_source_row - owner * rows_per_shard;

    const auto *source = static_cast<const uint8_t *>(input_ptrs[owner]) +
                         static_cast<uint64_t>(owner_row) * input_row_stride;
    auto *destination =
        static_cast<uint8_t *>(output) + static_cast<uint64_t>(output_row) * hidden_size;
    for (int column = threadIdx.x * 16; column < hidden_size; column += blockDim.x * 16) {
      *reinterpret_cast<uint4 *>(destination + column) =
          *reinterpret_cast<const uint4 *>(source + column);
    }

    const auto *source_scale = static_cast<const float *>(input_scale_ptrs[owner]) +
                               static_cast<uint64_t>(owner_row) * input_scale_row_stride;
    int expert = 0;
    while (output_row >= cu_num_tokens_per_group[expert + 1]) {
      ++expert;
    }
    const int scale_row = cu_tiles[expert] * tile_m + output_row - cu_num_tokens_per_group[expert];
    for (int column = threadIdx.x; column < scale_groups; column += blockDim.x) {
      // Native HPC grouped GEMM consumes activation scales as
      // [K / 128, padded M], not route-major rows.
      output_scale[static_cast<uint64_t>(column) * num_padded_tokens + scale_row] =
          source_row_valid ? source_scale[column] : 0.0f;
    }
  }

  cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace kernels

void pull_indexed_rows_async(const void *input_ptrs_dev, const void *input_scale_ptrs_dev,
                             int rows_per_shard, int num_input_ptrs, int64_t input_row_stride,
                             int64_t input_scale_row_stride, void *row_indices_ptr,
                             const void *cu_num_tokens_per_group_ptr, const void *cu_tiles_ptr,
                             void *output_ptr, void *output_scale_ptr, int max_rows,
                             int hidden_size, int scale_groups, int num_padded_tokens,
                             int num_expert_local, int tile_m, cudaStream_t stream) {
  if (max_rows == 0) {
    return;
  }
  dim3 block(256);
  dim3 grid(max_rows);
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = attribute;
  config.numAttrs = 1;
  auto error = cudaLaunchKernelEx(
      &config, kernels::pull_indexed_rows_kernel, static_cast<const void *const *>(input_ptrs_dev),
      static_cast<const void *const *>(input_scale_ptrs_dev), rows_per_shard,
      rows_per_shard * num_input_ptrs, input_row_stride, input_scale_row_stride,
      static_cast<const int *>(row_indices_ptr),
      static_cast<const int *>(cu_num_tokens_per_group_ptr), static_cast<const int *>(cu_tiles_ptr),
      output_ptr, static_cast<float *>(output_scale_ptr), hidden_size, scale_groups,
      num_padded_tokens, num_expert_local, tile_m);
  TORCH_CHECK(error == cudaSuccess, "indexed row pull launch failed: ", cudaGetErrorString(error));
}

}  // namespace fuse_moe
}  // namespace hpc
