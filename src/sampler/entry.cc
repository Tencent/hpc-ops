// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/sampler/fused_repetition_penalties_softmax.h"

namespace hpc {
namespace sampler {
namespace fused_repetition_penalties_softmax {

torch::Tensor entry(const torch::Tensor& logits, std::optional<torch::Tensor> penalties_masks_ptrs,
                    std::optional<torch::Tensor> repetition_penalties,
                    double repetition_penalties_val, std::optional<torch::Tensor> temperature,
                    double temperature_val) {
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

}  // namespace fused_repetition_penalties_softmax
}  // namespace sampler
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def("fused_repetition_penalties_softmax",
        &hpc::sampler::fused_repetition_penalties_softmax::entry);
}
