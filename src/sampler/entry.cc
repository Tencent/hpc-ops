// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/sampler/sampler.h"

namespace hpc {
namespace sampler {

torch::Tensor fused_repetition_penalties_softmax_entry(
    const torch::Tensor& logits, std::optional<torch::Tensor> penalties_masks_ptrs,
    std::optional<torch::Tensor> repetition_penalties, double repetition_penalties_val,
    std::optional<torch::Tensor> temperature, double temperature_val) {
  TORCH_CHECK(logits.is_contiguous(), "logits tensor must be contiguous");
  TORCH_CHECK(logits.dim() == 2, "logits tensor must be dim == 2");
  int num_batch = logits.size(0);
  int vocab_size = logits.size(1);

  if (penalties_masks_ptrs.has_value()) {
    TORCH_CHECK(penalties_masks_ptrs->is_contiguous(),
                "penalties_masks_ptrs tensor must be contiguous");
    TORCH_CHECK(penalties_masks_ptrs->dim() == 1,
                "penalties_masks_ptrs tensor must be dim == 1, but get ",
                penalties_masks_ptrs->dim());
    TORCH_CHECK(penalties_masks_ptrs->size(0) == num_batch,
                "penalties_masks_ptrs tensor must be shape [num_batch(", num_batch,
                "),], but get [", penalties_masks_ptrs->size(0), ",]");
  }

  if (repetition_penalties.has_value()) {
    TORCH_CHECK(repetition_penalties->is_contiguous(),
                "repetition_penalties tensor must be contiguous");
    TORCH_CHECK(repetition_penalties->dim() == 1,
                "repetition_penalties tensor must be dim == 1, but get ",
                repetition_penalties->dim());
    TORCH_CHECK(repetition_penalties->size(0) == num_batch,
                "repetition_penalties tensor must be shape [num_batch(", num_batch,
                "),], but get [", repetition_penalties->size(0), ",]");
  }

  if (temperature.has_value()) {
    TORCH_CHECK(temperature->is_contiguous(), "temperature tensor must be contiguous");
    TORCH_CHECK(temperature->dim() == 1, "temperature tensor must be dim == 1, but get ",
                temperature->dim());
    TORCH_CHECK(temperature->size(0) == num_batch, "temperature tensor must be shape [num_batch(",
                num_batch, "),], but get [", temperature->size(0), ",]");
  }

  torch::Tensor out = torch::empty({num_batch, vocab_size}, logits.options());

  auto* out_ptr = out.mutable_data_ptr<float>();
  auto* logits_ptr = logits.data_ptr<float>();
  const uint8_t** penalties_masks_ptrs_ptr = nullptr;
  if (penalties_masks_ptrs.has_value()) {
    penalties_masks_ptrs_ptr =
        reinterpret_cast<const uint8_t**>(penalties_masks_ptrs->data_ptr<uint64_t>());
  }
  float* repetition_penalties_ptr = nullptr;
  if (repetition_penalties.has_value()) {
    repetition_penalties_ptr = repetition_penalties->data_ptr<float>();
  }
  float* temperature_ptr = nullptr;
  if (temperature.has_value()) {
    temperature_ptr = temperature->data_ptr<float>();
  }

  auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());

  fused_repetition_penalties_softmax_async(
      out_ptr, logits_ptr, penalties_masks_ptrs_ptr, repetition_penalties_ptr,
      repetition_penalties_val, temperature_ptr, temperature_val, num_batch, vocab_size, stream);

  return out;
}

torch::Tensor topk_topp_mask_logits_entry(const torch::Tensor& logits,
                                          std::optional<torch::Tensor> topk, int64_t topk_val,
                                          std::optional<torch::Tensor> topp, double topp_val,
                                          std::optional<torch::Tensor> reject_threshold,
                                          double reject_threshold_val, int64_t max_topk_val) {
  auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());
  TORCH_CHECK(logits.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(logits.dtype() == torch::kFloat32,
              "logits must be float32 for current implementation");

  int int_bytes = 0;
  void* topk_ptr = nullptr;
  if (topk.has_value()) {
    topk_ptr = topk->data_ptr();
    TORCH_CHECK(topk->is_contiguous(), "topk tensor must be contiguous");
    if (topk->scalar_type() == torch::kInt32) {
      int_bytes = sizeof(int);
    } else if (topk->scalar_type() == torch::kInt64) {
      int_bytes = sizeof(int64_t);
    } else {
      TORCH_CHECK(false, "topk dtype must be int32 or int64");
    }
  }

  void* topp_ptr = nullptr;
  if (topp.has_value()) {
    topp_ptr = topp->data_ptr();
    TORCH_CHECK(topp->is_contiguous(), "topp tensor must be contiguous");
    TORCH_CHECK(topp->scalar_type() == torch::kFloat32, "topp dtype must be float32");
  }

  if (reject_threshold.has_value()) {
    TORCH_CHECK(reject_threshold->is_contiguous(), "reject_threshold tensor must be contiguous");
  }

  int batch_size = logits.size(0);
  int vocab_size = logits.size(1);
  int vocab_size_padded = logits.stride(0);

  // Used for temp storage of topk tokens and logits of first stage topk
  constexpr int kBlockPerBatch = 8;
  TORCH_CHECK(max_topk_val > 0, "max_topk_val must be positive");

  torch::Tensor middle_logits = torch::empty({batch_size, max_topk_val * kBlockPerBatch},
                                             torch::dtype(torch::kFloat32).device(logits.device()));
  torch::Tensor middle_tokens = torch::empty({batch_size, max_topk_val * kBlockPerBatch},
                                             torch::dtype(torch::kInt32).device(logits.device()));
  torch::Tensor sample_tokens =
      torch::empty({batch_size, 1}, torch::dtype(torch::kInt32).device(logits.device()));

  torch::Tensor output_logits =
      torch::empty({batch_size, vocab_size}, torch::dtype(torch::kFloat32).device(logits.device()));

  void* output_logits_ptr = output_logits.mutable_data_ptr();
  void* sample_tokens_ptr = sample_tokens.mutable_data_ptr();
  void* middle_logits_ptr = middle_logits.mutable_data_ptr();
  void* middle_tokens_ptr = middle_tokens.mutable_data_ptr();
  void* logits_ptr = logits.data_ptr();

  void* reject_threshold_ptr = nullptr;
  if (reject_threshold.has_value()) {
    reject_threshold_ptr = reject_threshold->data_ptr();
  }

  topk_topp_mask_logits_async(output_logits_ptr, sample_tokens_ptr, middle_logits_ptr,
                              middle_tokens_ptr, logits_ptr, topk_ptr, topk_val, topp_ptr, topp_val,
                              reject_threshold_ptr, reject_threshold_val, batch_size, vocab_size,
                              vocab_size_padded, int_bytes, max_topk_val, stream);

  return output_logits;
}
}  // namespace sampler
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "fused_repetition_penalties_softmax(Tensor logits, Tensor? penalties_masks_ptrs, Tensor? "
      "repetition_penalties, float repetition_penalties_val, Tensor? temperature, float "
      "temperature_val) -> Tensor");
  m.impl("fused_repetition_penalties_softmax", torch::kCUDA,
         &hpc::sampler::fused_repetition_penalties_softmax_entry);
  m.def(
      "topk_topp_mask_logits(Tensor logits, Tensor? topk, int topk_val,Tensor? topp, float "
      "topp_val, Tensor? reject_threshold, "
      "float reject_threshold_val, int max_topk_val) -> Tensor");
  m.impl("topk_topp_mask_logits", torch::kCUDA, &hpc::sampler::topk_topp_mask_logits_entry);
}
