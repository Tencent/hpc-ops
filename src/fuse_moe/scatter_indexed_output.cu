// Copyright (C) 2026 Tencent.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <torch/library.h>

#include <cstdint>

namespace hpc {
namespace fuse_moe {

namespace {

__global__ void scatter_indexed_output_kernel(const uint4 *__restrict__ rows,
                                              const int64_t *__restrict__ output_ptrs,
                                              const int32_t *__restrict__ destination_rows,
                                              int vectors_per_row, int owner_capacity,
                                              int producer_slot, int producer_slots,
                                              int world_size) {
  int input_row = blockIdx.x;
  int logical_row = destination_rows[input_row];
  int logical_capacity = owner_capacity * world_size;
  if (logical_row < 0 || logical_row >= logical_capacity) {
    return;
  }

  int owner = logical_row / owner_capacity;
  int owner_row = logical_row - owner * owner_capacity;
  auto *output = reinterpret_cast<uint4 *>(output_ptrs[owner]);
  int64_t destination_vector =
      (static_cast<int64_t>(owner_row) * producer_slots + producer_slot) * vectors_per_row;
  int64_t source_vector = static_cast<int64_t>(input_row) * vectors_per_row;

  for (int vector_index = threadIdx.x; vector_index < vectors_per_row; vector_index += blockDim.x) {
    output[destination_vector + vector_index] = rows[source_vector + vector_index];
  }
}

}  // namespace

void scatter_indexed_output(const at::Tensor &rows, const at::Tensor &output_ptrs,
                            const at::Tensor &destination_rows, int64_t owner_capacity,
                            int64_t producer_slot, int64_t producer_slots) {
  TORCH_CHECK(rows.is_cuda(), "scatter rows must be CUDA");
  TORCH_CHECK(rows.is_contiguous(), "scatter rows must be contiguous");
  TORCH_CHECK(rows.dim() == 2, "scatter rows must be two-dimensional");
  TORCH_CHECK(rows.scalar_type() == at::kBFloat16, "scatter rows must be bfloat16");
  TORCH_CHECK(rows.size(1) > 0 && rows.size(1) % 8 == 0,
              "scatter hidden size must be a positive multiple of 8");

  TORCH_CHECK(output_ptrs.is_cuda(), "output pointer table must be CUDA");
  TORCH_CHECK(output_ptrs.is_contiguous(), "output pointer table must be contiguous");
  TORCH_CHECK(output_ptrs.dim() == 1 && output_ptrs.numel() > 0,
              "output pointer table must be a non-empty vector");
  TORCH_CHECK(output_ptrs.scalar_type() == at::kLong, "output pointer table must be int64");
  TORCH_CHECK(output_ptrs.get_device() == rows.get_device(),
              "output pointer table and rows must be on the same CUDA device");

  TORCH_CHECK(destination_rows.is_cuda(), "destination_rows must be CUDA");
  TORCH_CHECK(destination_rows.is_contiguous(), "destination_rows must be contiguous");
  TORCH_CHECK(destination_rows.dim() == 1, "destination_rows must be one-dimensional");
  TORCH_CHECK(destination_rows.scalar_type() == at::kInt, "destination_rows must be int32");
  TORCH_CHECK(destination_rows.get_device() == rows.get_device(),
              "destination_rows and rows must be on the same CUDA device");
  TORCH_CHECK(destination_rows.numel() == rows.size(0),
              "destination_rows and rows must contain the same number of rows");

  TORCH_CHECK(owner_capacity > 0, "owner_capacity must be positive");
  TORCH_CHECK(producer_slots > 0, "producer_slots must be positive");
  TORCH_CHECK(producer_slot >= 0 && producer_slot < producer_slots,
              "producer_slot must be in [0, producer_slots)");

  if (rows.size(0) == 0) {
    return;
  }

  constexpr int kBytesPerVector = sizeof(uint4);
  constexpr int kBf16Bytes = sizeof(__nv_bfloat16);
  int vectors_per_row = rows.size(1) * kBf16Bytes / kBytesPerVector;
  int threads = vectors_per_row < 256 ? 128 : 256;
  auto stream = at::cuda::getCurrentCUDAStream(rows.get_device());
  scatter_indexed_output_kernel<<<rows.size(0), threads, 0, stream>>>(
      reinterpret_cast<const uint4 *>(rows.const_data_ptr()), output_ptrs.const_data_ptr<int64_t>(),
      destination_rows.const_data_ptr<int32_t>(), vectors_per_row, static_cast<int>(owner_capacity),
      static_cast<int>(producer_slot), static_cast<int>(producer_slots),
      static_cast<int>(output_ptrs.numel()));
  auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "scatter indexed output launch failed: ", cudaGetErrorString(error));
}

}  // namespace fuse_moe
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "scatter_indexed_output(Tensor rows, Tensor(a!) output_ptrs, Tensor destination_rows, int "
      "owner_capacity, int producer_slot, int producer_slots) -> ()");
  m.impl("scatter_indexed_output", c10::DispatchKey::CUDA, &hpc::fuse_moe::scatter_indexed_output);
}
