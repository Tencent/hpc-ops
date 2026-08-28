// Copyright (C) 2026 Tencent.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <tuple>

#include "src/topk/topk.h"

namespace hpc {
namespace topk {

torch::Tensor topk_filtered_entry(const torch::Tensor &logits, const torch::Tensor &ke,
                                  torch::Tensor &output, int64_t topk,
                                  const torch::Tensor &num_valid_rows,
                                  std::optional<torch::Tensor> counters,
                                  std::optional<torch::Tensor> workspace) {
  TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits must be float32");
  TORCH_CHECK(logits.stride(1) == 1, "logits inner (column) stride must be 1");
  TORCH_CHECK(logits.stride(0) >= logits.size(1), "logits rows must not overlap");
  TORCH_CHECK(ke.is_cuda(), "ke must be on CUDA");
  TORCH_CHECK(ke.dim() == 1, "ke must be 1D");
  TORCH_CHECK(ke.is_contiguous(), "ke tensor must be contiguous");
  TORCH_CHECK(ke.scalar_type() == torch::kInt32, "ke must be int32");
  TORCH_CHECK(ke.device() == logits.device(), "ke must be on the same device as logits");
  TORCH_CHECK(num_valid_rows.scalar_type() == torch::kInt32, "num_valid_rows must be int32");
  TORCH_CHECK(num_valid_rows.is_cuda(), "num_valid_rows must be on CUDA");
  TORCH_CHECK(num_valid_rows.numel() == 1, "num_valid_rows must have exactly one element");
  TORCH_CHECK(num_valid_rows.is_contiguous(), "num_valid_rows must be contiguous");
  TORCH_CHECK(num_valid_rows.device() == logits.device(),
              "num_valid_rows must be on the same device as logits");
  TORCH_CHECK(output.is_cuda(), "output must be on CUDA");
  TORCH_CHECK(output.scalar_type() == torch::kInt32, "output must be int32");
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.stride(1) == 1, "output inner (column) stride must be 1");
  TORCH_CHECK(output.device() == logits.device(), "output must be on the same device as logits");
  TORCH_CHECK(topk == 2048 || topk == 512, "topk must be 2048 or 512");
  TORCH_CHECK(ke.numel() <= logits.size(0), "ke rows exceed logits rows");
  TORCH_CHECK(ke.numel() <= std::numeric_limits<int>::max(), "ke rows exceed the supported range");
  TORCH_CHECK(logits.size(1) <= std::numeric_limits<int>::max(),
              "logits columns exceed the supported range");
  TORCH_CHECK(logits.stride(0) <= std::numeric_limits<uint32_t>::max(),
              "logits row stride exceeds the supported range");
  TORCH_CHECK(output.stride(0) <= std::numeric_limits<uint32_t>::max(),
              "output row stride exceeds the supported range");

  int n = logits.size(1);
  int row_stride = logits.stride(0);
  int row_capacity = static_cast<int>(ke.numel());
  int out_stride = output.stride(0);

  TORCH_CHECK(static_cast<int64_t>(output.size(0)) >= row_capacity, "output rows < row capacity");
  TORCH_CHECK(static_cast<int64_t>(output.size(1)) >= topk, "output cols < topk");
  TORCH_CHECK(output.stride(0) >= topk, "output rows must not overlap");

  torch::Tensor cnt;
  if (counters.has_value()) {
    cnt = counters.value();
    TORCH_CHECK(cnt.dtype() == torch::kUInt8, "counters must be uint8");
    TORCH_CHECK(cnt.is_cuda(), "counters must be on CUDA");
    TORCH_CHECK(cnt.is_contiguous(), "counters must be contiguous");
    TORCH_CHECK(cnt.device() == logits.device(), "counters must be on the same device as logits");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(cnt.mutable_data_ptr()) % alignof(int32_t) == 0,
                "counters must be int32-aligned");
    TORCH_CHECK(static_cast<size_t>(cnt.numel()) >= topk_filtered_counters_bytes(row_capacity),
                "counters is too small for topk_filtered");
  } else {
    cnt = torch::zeros({static_cast<int64_t>(topk_filtered_counters_bytes(row_capacity))},
                       torch::dtype(torch::kUInt8).device(logits.device()));
  }

  torch::Tensor ws;
  if (workspace.has_value()) {
    ws = workspace.value();
    TORCH_CHECK(ws.dtype() == torch::kUInt8, "workspace must be uint8");
    TORCH_CHECK(ws.is_cuda(), "workspace must be on CUDA");
    TORCH_CHECK(ws.is_contiguous(), "workspace must be contiguous");
    TORCH_CHECK(ws.device() == logits.device(), "workspace must be on the same device as logits");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(ws.mutable_data_ptr()) % alignof(int32_t) == 0,
                "workspace must be int32-aligned");
    TORCH_CHECK(static_cast<size_t>(ws.numel()) >= topk_filtered_min_workspace_bytes(n),
                "workspace is too small for topk_filtered");
  } else {
    ws = torch::empty({static_cast<int64_t>(topk_filtered_workspace_bytes(row_capacity, n))},
                      torch::dtype(torch::kUInt8).device(logits.device()));
  }

  auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());
  bool ok = topk_filtered_async(
      output.mutable_data_ptr<int>(), logits.const_data_ptr<float>(), ke.const_data_ptr<int>(),
      topk, num_valid_rows.const_data_ptr<int>(), row_capacity, n, row_stride, out_stride,
      cnt.mutable_data_ptr(), cnt.numel(), ws.mutable_data_ptr(), ws.numel(), stream);
  TORCH_CHECK(ok, "launch topk_filtered kernel failed!");

  return output;
}

std::tuple<int64_t, int64_t> topk_filtered_workspace_size_entry(int64_t num_rows,
                                                                int64_t max_kv_len) {
  TORCH_CHECK(num_rows >= 0, "num_rows must be non-negative");
  TORCH_CHECK(max_kv_len >= 0, "max_kv_len must be non-negative");
  TORCH_CHECK(num_rows <= std::numeric_limits<int>::max(), "num_rows exceeds the supported range");
  TORCH_CHECK(max_kv_len <= std::numeric_limits<int>::max(),
              "max_kv_len exceeds the supported range");
  return std::make_tuple(
      static_cast<int64_t>(topk_filtered_counters_bytes(static_cast<int>(num_rows))),
      static_cast<int64_t>(
          topk_filtered_workspace_bytes(static_cast<int>(num_rows), static_cast<int>(max_kv_len))));
}

int64_t topk_filtered_min_workspace_size_entry(int64_t max_kv_len) {
  TORCH_CHECK(max_kv_len >= 0, "max_kv_len must be non-negative");
  TORCH_CHECK(max_kv_len <= std::numeric_limits<int>::max(),
              "max_kv_len exceeds the supported range");
  return static_cast<int64_t>(topk_filtered_min_workspace_bytes(static_cast<int>(max_kv_len)));
}

int64_t topk_filtered_peak_workspace_size_entry(int64_t max_kv_len) {
  TORCH_CHECK(max_kv_len >= 0, "max_kv_len must be non-negative");
  TORCH_CHECK(max_kv_len <= std::numeric_limits<int>::max(),
              "max_kv_len exceeds the supported range");
  return static_cast<int64_t>(topk_filtered_peak_workspace_bytes(static_cast<int>(max_kv_len)));
}

}  // namespace topk
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "topk_filtered(Tensor logits, Tensor ke, Tensor! output, int topk, "
      "Tensor num_valid_rows, Tensor? counters, Tensor? workspace) -> (Tensor)");
  m.impl("topk_filtered", torch::kCUDA, &hpc::topk::topk_filtered_entry);

  m.def("topk_filtered_workspace_size(int num_rows, int max_kv_len) -> (int, int)",
        &hpc::topk::topk_filtered_workspace_size_entry);

  m.def("topk_filtered_min_workspace_size(int max_kv_len) -> int",
        &hpc::topk::topk_filtered_min_workspace_size_entry);

  m.def("topk_filtered_peak_workspace_size(int max_kv_len) -> int",
        &hpc::topk::topk_filtered_peak_workspace_size_entry);
}
