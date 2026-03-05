// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

#include "src/topk/topk.h"

namespace hpc {
namespace topk {

torch::Tensor topk_per_row_entry(const torch::Tensor &logits, int64_t num_sp_tokens, int64_t topk,
                                 const torch::Tensor &seqlens,
                                 std::optional<torch::Tensor> topk_indices) {
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

torch::Tensor topk_per_row_varlen_entry(const torch::Tensor &logits,
                                        const torch::Tensor &cu_seqlens_q,
                                        const torch::Tensor &seqlens_kv, int64_t topk,
                                        int64_t compress_ratio, bool deterministic,
                                        std::optional<torch::Tensor> topk_indices) {
  auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());

  TORCH_CHECK(cu_seqlens_q.is_contiguous(), "cu_seqlens_q tensor must be contiguous");
  TORCH_CHECK(seqlens_kv.is_contiguous(), "seqlens_kv tensor must be contiguous");

  TORCH_CHECK(topk == 512, "topk must be 512");

  int num_rows = logits.size(0);
  int row_stride = logits.stride(0);
  int num_batch = cu_seqlens_q.size(0) - 1;

  TORCH_CHECK(row_stride % 4 == 0,
              "topk_per_row_varlen logits input stride(0) must to be divisible by 4.");

  torch::Tensor output;
  if (topk_indices.has_value()) {
    output = topk_indices.value();
  } else {
    output = torch::empty({num_rows, topk}, torch::dtype(torch::kInt32).device(logits.device()));
  }

  const auto *logits_ptr = logits.const_data_ptr<float>();
  const auto *cu_seqlens_q_ptr = cu_seqlens_q.const_data_ptr<int>();
  const auto *seqlens_kv_ptr = seqlens_kv.const_data_ptr<int>();

  auto *output_ptr = output.mutable_data_ptr<int>();

  bool running = topk_per_row_varlen_async(output_ptr, logits_ptr, cu_seqlens_q_ptr, seqlens_kv_ptr,
                                           topk, compress_ratio, num_batch, num_rows, row_stride,
                                           deterministic, stream);

  TORCH_CHECK(running, "launch topk_per_row_varlen_async failed!");

  return output;
}

std::tuple<torch::Tensor, torch::Tensor> grouped_topk_entry(const torch::Tensor &scores,
                                                            int64_t topk, int64_t num_expert_group,
                                                            int64_t topk_group, double scale,
                                                            bool renormalize,
                                                            std::optional<torch::Tensor> bias) {
  auto stream = at::cuda::getCurrentCUDAStream(scores.get_device());

  TORCH_CHECK(scores.is_contiguous(), "scores tensor must be contiguous");

  int num_tokens = scores.size(0);
  int num_experts = scores.size(1);

  TORCH_CHECK(num_expert_group == 1, "we only support num_expert_group == 1 now!");

  TORCH_CHECK(num_experts <= 256, "num_experts must <= 256.");

  torch::Tensor topk_weights =
      torch::empty({num_tokens, topk}, torch::dtype(torch::kFloat32).device(scores.device()));
  torch::Tensor topk_ids =
      torch::empty({num_tokens, topk}, torch::dtype(torch::kInt32).device(scores.device()));

  const float *bias_ptr = nullptr;
  if (bias.has_value()) {
    bias_ptr = bias.value().const_data_ptr<float>();
  }

  const auto *scores_ptr = scores.const_data_ptr<float>();
  auto *topk_weights_ptr = topk_weights.mutable_data_ptr<float>();
  auto *topk_ids_ptr = topk_ids.mutable_data_ptr<int>();

  bool running =
      grouped_topk_async(topk_weights_ptr, topk_ids_ptr, scores_ptr, bias_ptr, scale, num_tokens,
                         topk, topk_group, num_experts, num_expert_group, renormalize, stream);

  TORCH_CHECK(running, "launch grouped_topk_async failed!");

  return std::make_tuple(topk_weights, topk_ids);
}

}  // namespace topk
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "topk_per_row(Tensor logits, int num_sp_tokens, int topk, Tensor seqlens, Tensor? "
      "topk_indices) -> "
      "(Tensor)");
  m.impl("topk_per_row", torch::kCUDA, &hpc::topk::topk_per_row_entry);

  m.def(
      "topk_per_row_varlen(Tensor logits, Tensor cu_seqlens_q, Tensor seqlens_kv, int topk, int "
      "compress_ratio, bool deterministic,"
      "Tensor? "
      "topk_indices) -> "
      "(Tensor)");
  m.impl("topk_per_row_varlen", torch::kCUDA, &hpc::topk::topk_per_row_varlen_entry);

  m.def(
      "grouped_topk(Tensor scores, int topk, int num_expert_group, int topk_group, float scale, "
      "bool renormalize, Tensor? bias) ->"
      "(Tensor, Tensor)");
  m.impl("grouped_topk", torch::kCUDA, &hpc::topk::grouped_topk_entry);
}
