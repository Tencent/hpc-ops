// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/topk/topk.h"

namespace hpc {
namespace topk {

torch::Tensor entry(const torch::Tensor &logits, int64_t num_sp_tokens, int64_t topk,
                    const torch::Tensor &seqlens, std::optional<torch::Tensor> topk_indices) {
  auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());

  TORCH_CHECK(seqlens.is_contiguous(), "seqlens tensor must be contiguous");
  TORCH_CHECK(topk == 2048, "topk must be 2048");

  int num_rows = logits.size(0);
  int row_stride = logits.stride(0);

  torch::Tensor output;
  if (topk_indices.has_value()) {
    output = topk_indices.value();
  } else {
    output = torch::empty({num_rows, topk}, torch::dtype(torch::kInt32).device(logits.device()));
  }

  const auto *logits_ptr = logits.const_data_ptr<float>();
  const auto *seqlens_ptr = seqlens.const_data_ptr<int>();
  auto *output_ptr = output.mutable_data_ptr<int>();

  bool running = topk_per_row_async(output_ptr, logits_ptr, seqlens_ptr, topk, num_sp_tokens,
                                    num_rows, row_stride, stream);

  TORCH_CHECK(running, "launch topk_per_row_smallm_async failed!");

  return output;
}

}  // namespace topk
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "topk_per_row(Tensor logits, int num_sp_tokens, int topk, Tensor seqlens, Tensor? "
      "topk_indices) -> "
      "(Tensor)");
  m.impl("topk_per_row", torch::kCUDA, &hpc::topk::entry);
}
