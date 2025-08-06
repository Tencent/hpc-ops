#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/sampler/fused_repetition_penalties_softmax.h"

namespace hpc {
namespace sampler {
namespace fused_repetition_penalties_softmax {

torch::Tensor entry(const torch::Tensor& logits, const torch::Tensor& penalties_masks_ptrs,
                    double repetition_penalties, double temperature) {
  auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());

  TORCH_CHECK(logits.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(penalties_masks_ptrs.is_contiguous(), "input tensor must be contiguous");

  int num_batch = logits.size(0);
  int vocab_size = logits.size(1);

  TORCH_CHECK((vocab_size == 129024 || vocab_size == 128512), "we only support vocab_size == 129024 and 128512");

  torch::Tensor out = torch::empty({num_batch, vocab_size}, logits.options());

  const auto* logits_ptr = logits.data_ptr<float>();
  auto* out_ptr = out.mutable_data_ptr<float>();
  const uint8_t** repetition_penalties_ptr =
      reinterpret_cast<const uint8_t**>(penalties_masks_ptrs.data_ptr<uint64_t>());

  if (temperature <= 0) {
    temperature = 1;
  }

  fused_repetition_penalties_softmax_async(out_ptr, logits_ptr, repetition_penalties_ptr,
                                           repetition_penalties, temperature, num_batch, vocab_size,
                                           stream);

  return out;
}

}  // namespace fused_repetition_penalties_softmax
}  // namespace sampler
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def("fused_repetition_penalties_softmax",
        &hpc::sampler::fused_repetition_penalties_softmax::entry);
}
