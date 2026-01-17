// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

#include "src/routing_method/routing_method.h"

namespace hpc {
namespace routing_method {

std::tuple<torch::Tensor, torch::Tensor> deepseekv4_routing_method_entry(
    torch::Tensor &score, std::optional<torch::Tensor> bias, std::optional<torch::Tensor> input_ids,
    std::optional<torch::Tensor> tid2eid, int64_t topk, double route_scale, bool is_hash,
    std::optional<torch::Tensor> out_weights, std::optional<torch::Tensor> out_indices) {
  auto stream = at::cuda::getCurrentCUDAStream(score.get_device());

  TORCH_CHECK(score.is_contiguous(), "score tensor must be contiguous");
  TORCH_CHECK(score.device().is_cuda(), "score tensor's device must be cuda");
  TORCH_CHECK(score.scalar_type() == torch::kBFloat16, "score tensor's data type must be bfloat16");
  TORCH_CHECK(score.dim() == 2, "score tensor's dim must be 2");

  int batch_size = score.size(0);
  int num_expert = score.size(1);
  TORCH_CHECK(num_expert == 256, "num expert must be 256");
  TORCH_CHECK(topk == 6, "topk must be 6");

  torch::Tensor out_weights_tensor;
  if (out_weights.has_value()) {
    out_weights_tensor = out_weights.value();
    TORCH_CHECK(out_weights_tensor.is_contiguous(), "out_weights tensor must be contiguous");
    TORCH_CHECK(out_weights_tensor.device().is_cuda(), "out_weights tensor's device must be cuda");
    TORCH_CHECK(out_weights_tensor.scalar_type() == torch::kFloat32,
                "out_weights tensor's data type must be float32");
    TORCH_CHECK(out_weights_tensor.dim() == 2, "out_weights tensor's dim must be 2");
    TORCH_CHECK(out_weights_tensor.size(0) == batch_size,
                "out_weights tensor first dim must be batch_size");
    TORCH_CHECK(out_weights_tensor.size(1) == topk, "out_weights tensor second dim must be topk");
  } else {
    out_weights_tensor = torch::empty({batch_size, topk}, score.options().dtype(torch::kFloat32));
  }

  torch::Tensor out_indices_tensor;
  if (out_indices.has_value()) {
    out_indices_tensor = out_indices.value();
    TORCH_CHECK(out_indices_tensor.is_contiguous(), "out_indices tensor must be contiguous");
    TORCH_CHECK(out_indices_tensor.device().is_cuda(), "out_indices tensor's device must be cuda");
    TORCH_CHECK(out_indices_tensor.scalar_type() == torch::kInt32,
                "out_indices tensor's data type must be int32");
    TORCH_CHECK(out_indices_tensor.dim() == 2, "out_indices tensor's dim must be 2");
    TORCH_CHECK(out_indices_tensor.size(0) == batch_size,
                "out_indices tensor first dim must be batch_size");
    TORCH_CHECK(out_indices_tensor.size(1) == topk, "out_indices tensor second dim must be topk");
  } else {
    out_indices_tensor = torch::empty({batch_size, topk}, score.options().dtype(torch::kInt32));
  }

  torch::Tensor bias_tensor;
  torch::Tensor input_ids_tensor;
  torch::Tensor tid2eid_tensor;
  if (is_hash) {
    TORCH_CHECK(!bias.has_value(), "bias must be None when in hash mode");
    TORCH_CHECK(input_ids.has_value(), "input_ids must be not None when in hash mode");
    TORCH_CHECK(tid2eid.has_value(), "tid2eid must be not None when in hash mode");

    input_ids_tensor = input_ids.value();
    TORCH_CHECK(input_ids_tensor.is_contiguous(), "input_ids tensor must be contiguous");
    TORCH_CHECK(input_ids_tensor.device().is_cuda(), "input_ids tensor's device must be cuda");
    TORCH_CHECK(input_ids_tensor.scalar_type() == torch::kInt32,
                "input_ids tensor's data type must be int32");
    TORCH_CHECK(input_ids_tensor.dim() == 1, "input_ids tensor's dim must be 1");
    TORCH_CHECK(input_ids_tensor.size(-1) == batch_size,
                "input_ids tensor last dim must be batch_size");

    tid2eid_tensor = tid2eid.value();
    TORCH_CHECK(tid2eid_tensor.is_contiguous(), "tid2eid tensor must be contiguous");
    TORCH_CHECK(tid2eid_tensor.device().is_cuda(), "tid2eid tensor's device must be cuda");
    TORCH_CHECK(tid2eid_tensor.scalar_type() == torch::kInt32,
                "tid2eid tensor's data type must be int32");
    TORCH_CHECK(tid2eid_tensor.dim() == 2, "tid2eid tensor's dim must be 2");
    TORCH_CHECK(tid2eid_tensor.size(-1) == topk, "tid2eid tensor last dim must be topk");
  } else {
    TORCH_CHECK(bias.has_value(), "bias must be not None when in non-hash mode");
    TORCH_CHECK(!input_ids.has_value(), "input_ids must be None when in non-hash mode");
    TORCH_CHECK(!tid2eid.has_value(), "tid2eid must be None when in non-hash mode");

    bias_tensor = bias.value();
    TORCH_CHECK(bias_tensor.is_contiguous(), "bias tensor must be contiguous");
    TORCH_CHECK(bias_tensor.device().is_cuda(), "bias tensor's device must be cuda");
    TORCH_CHECK(bias_tensor.scalar_type() == torch::kFloat32,
                "bias tensor's data type must be float32");
    TORCH_CHECK(bias_tensor.dim() == 1, "bias tensor's dim must be 1");
    TORCH_CHECK(bias_tensor.size(-1) == num_expert, "bias tensor last dim must be num_expert");
  }

  const auto *score_ptr = reinterpret_cast<const __nv_bfloat16 *>(score.const_data_ptr());
  auto *out_weights_ptr = reinterpret_cast<float *>(out_weights_tensor.mutable_data_ptr());
  auto *out_indices_ptr = reinterpret_cast<int32_t *>(out_indices_tensor.mutable_data_ptr());

  const float *bias_ptr = nullptr;
  const int32_t *input_ids_ptr = nullptr;
  const int32_t *tid2eid_ptr = nullptr;

  if (is_hash) {
    input_ids_ptr = reinterpret_cast<const int32_t *>(input_ids_tensor.const_data_ptr());
    tid2eid_ptr = reinterpret_cast<const int32_t *>(tid2eid_tensor.const_data_ptr());
  } else {
    bias_ptr = reinterpret_cast<const float *>(bias_tensor.const_data_ptr());
  }

  deepseekv4_routing_method_async(out_weights_ptr, out_indices_ptr, score_ptr, bias_ptr,
                                  input_ids_ptr, tid2eid_ptr, batch_size, num_expert, topk,
                                  route_scale, is_hash, stream);

  return std::make_tuple(out_weights_tensor, out_indices_tensor);
}

}  // namespace routing_method
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "deepseekv4_routing_method(Tensor score, Tensor? bias, Tensor? input_ids,"
      "Tensor? tid2eid, int topk, float route_scale, bool is_hash, Tensor? out_weights, Tensor? "
      "out_indices) -> "
      "(Tensor, Tensor)");
  m.impl("deepseekv4_routing_method", torch::kCUDA,
         &hpc::routing_method::deepseekv4_routing_method_entry);
}
