// Copyright (C) 2026 Tencent.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

#include "src/fuse_moe/fuse_moe.h"

namespace hpc {
namespace fuse_moe {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
count_and_gather_entry(const torch::Tensor &x, const torch::Tensor &topk_ids,
                       const int64_t num_expert, const int64_t rank_ep,
                       const int64_t intermediate_size, const int64_t num_seq_per_group_avg) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda(), "x tensor must be cuda");
  TORCH_CHECK(topk_ids.device().is_cuda(), "topk_ids tensor must be cuda");
  TORCH_CHECK(x.is_contiguous(), "x tensor a must be contiguous");
  TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids tensor a must be contiguous");
  TORCH_CHECK(x.size(0) == topk_ids.size(0), "x and topk_ids must share the same k");

  int num_seq = x.size(0);
  int hidden_size = x.size(1);
  int num_topk = topk_ids.size(1);

  auto options = x.options();
  torch::Tensor gate_up_input = torch::empty({num_seq * num_topk, hidden_size}, options);
  torch::Tensor gate_up_output =
      torch::empty({num_seq * num_topk, intermediate_size}, options.dtype(torch::kBFloat16));
  torch::Tensor down_input = torch::empty({num_seq * num_topk, intermediate_size / 2}, options);
  torch::Tensor down_output =
      torch::empty({num_seq * num_topk, hidden_size}, options.dtype(torch::kBFloat16));

  torch::Tensor topk_pos = torch::empty({num_seq, num_topk}, options.dtype(torch::kInt32));
  torch::Tensor seqlens = torch::zeros({num_expert}, options.dtype(torch::kInt32));
  torch::Tensor cu_seqlens = torch::empty({num_expert + 1}, options.dtype(torch::kInt32));
  torch::Tensor tiles = torch::empty({num_expert}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_expert + 1}, options.dtype(torch::kInt32));
  torch::Tensor gate_up_tmas = torch::empty({num_expert * 2, 128}, options.dtype(torch::kInt8));
  torch::Tensor dowm_tmas = torch::empty({num_expert * 2, 128}, options.dtype(torch::kInt8));

  const auto *x_ptr = x.const_data_ptr();
  const auto *topk_ids_ptr = topk_ids.const_data_ptr();

  auto *gate_up_input_ptr = gate_up_input.mutable_data_ptr();
  auto *gate_up_output_ptr = gate_up_output.mutable_data_ptr();
  auto *down_input_ptr = down_input.mutable_data_ptr();
  auto *down_output_ptr = down_output.mutable_data_ptr();
  auto *topk_pos_ptr = topk_pos.mutable_data_ptr();
  auto *seqlens_ptr = seqlens.mutable_data_ptr();
  auto *cu_seqlens_ptr = cu_seqlens.mutable_data_ptr();
  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();
  auto *gate_up_tmas_ptr = gate_up_tmas.mutable_data_ptr();
  auto *dowm_tmas_ptr = dowm_tmas.mutable_data_ptr();

  count_and_gather_async(gate_up_input_ptr, gate_up_output_ptr, down_input_ptr, down_output_ptr,
                         x_ptr, topk_ids_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr,
                         gate_up_tmas_ptr, dowm_tmas_ptr, tiles_ptr, cu_tiles_ptr, num_seq,
                         hidden_size, intermediate_size, num_topk, num_expert, rank_ep,
                         num_seq_per_group_avg, stream);

  return std::make_tuple(gate_up_input, gate_up_output, topk_pos, seqlens, cu_seqlens, tiles,
                         cu_tiles, gate_up_tmas, dowm_tmas);
}

torch::Tensor reduce_entry(const torch::Tensor &x, const torch::Tensor &topk_pos,
                           const torch::Tensor &topk_scale,
                           const std::optional<torch::Tensor> &shared_output) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda(), "x tensor must be cuda");
  TORCH_CHECK(topk_pos.device().is_cuda(), "topk_pos tensor must be cuda");
  TORCH_CHECK(topk_scale.device().is_cuda(), "topk_scale tensor must be cuda");
  TORCH_CHECK(x.is_contiguous(), "x tensor must be contiguous");
  TORCH_CHECK(topk_pos.is_contiguous(), "topk_pos tensor must be contiguous");
  TORCH_CHECK(topk_scale.is_contiguous(), "topk_scale tensor must be contiguous");
  TORCH_CHECK(topk_pos.size(0) == topk_scale.size(0),
              "topk_pos and topk_scale must share the same num_seq");
  TORCH_CHECK(topk_pos.size(1) == topk_scale.size(1),
              "topk_pos and topk_scale must share the same num_topk");

  const void *shared_output_ptr = nullptr;
  if (shared_output.has_value()) {
    const auto shared_output_tensor = shared_output.value();
    TORCH_CHECK(shared_output_tensor.device().is_cuda(), "shared_output tensor must be cuda");
    TORCH_CHECK(shared_output_tensor.is_contiguous(), "shared_output tensor must be contiguous");
    TORCH_CHECK(shared_output_tensor.dtype() == torch::kBFloat16,
                "shared_output tensor dtype must be bfloat16");
    TORCH_CHECK(
        shared_output_tensor.size(0) == x.size(0) && shared_output_tensor.size(1) == x.size(1),
        "shared_output tensor shape must be same as x tensor");
    shared_output_ptr = shared_output_tensor.const_data_ptr();
  }

  int total_num_seq = x.size(0);
  int hidden_size = x.size(1);
  int num_seq = topk_pos.size(0);
  int num_topk = topk_pos.size(1);
  TORCH_CHECK(num_topk <= 128, "num_topk must less than or equal to 128");

  auto options = x.options();
  torch::Tensor y = torch::empty({num_seq, hidden_size}, options.dtype(torch::kBFloat16));

  const auto *x_ptr = x.const_data_ptr();
  const auto *topk_pos_ptr = topk_pos.const_data_ptr();
  const auto *topk_scale_ptr = topk_scale.const_data_ptr();

  auto *y_ptr = y.mutable_data_ptr();

  reduce_async(y_ptr, x_ptr, topk_pos_ptr, topk_scale_ptr, shared_output_ptr, total_num_seq,
               num_seq, hidden_size, num_topk, stream);

  return y;
}

torch::Tensor fuse_moe_pertensor_fp8_entry(
    const torch::Tensor &x, const torch::Tensor &gate_up_weight, const torch::Tensor &down_weight,
    const torch::Tensor &gate_up_scale, const torch::Tensor &down_scale,
    const torch::Tensor &act_and_mul_scale, const torch::Tensor &topk_ids,
    const torch::Tensor &topk_scale, const std::optional<torch::Tensor> &shared_output,
    int64_t rank_ep, int64_t num_expert_total, bool use_bf16_mul) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  TORCH_CHECK(x.device().is_cuda(), "x tensor must be cuda");
  TORCH_CHECK(gate_up_weight.device().is_cuda(), "gate_up_weight tensor must be cuda");
  TORCH_CHECK(gate_up_scale.device().is_cuda(), "gate_up_scale tensor must be cuda");
  TORCH_CHECK(down_scale.device().is_cuda(), "down_scale tensor must be cuda");
  TORCH_CHECK(act_and_mul_scale.device().is_cuda(), "act_and_mul_scale tensor must be cuda");
  TORCH_CHECK(topk_ids.device().is_cuda(), "topk_ids tensor must be cuda");
  TORCH_CHECK(topk_scale.device().is_cuda(), "topk_scale tensor must be cuda");

  TORCH_CHECK(x.is_contiguous(), "x tensor must be contiguous");
  TORCH_CHECK(gate_up_weight.is_contiguous(), "gate_up_weight tensor must be contiguous");
  TORCH_CHECK(gate_up_scale.is_contiguous(), "gate_up_scale tensor must be contiguous");
  TORCH_CHECK(down_weight.is_contiguous(), "down_weight tensor must be contiguous");
  TORCH_CHECK(down_scale.is_contiguous(), "down_scale tensor must be contiguous");
  TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids tensor must be contiguous");
  TORCH_CHECK(topk_scale.is_contiguous(), "topk_scale tensor must be contiguous");

  TORCH_CHECK(x.size(0) == topk_ids.size(0), "x and topk_ids must share the same num_seq");
  TORCH_CHECK(topk_ids.size(0) == topk_scale.size(0),
              "topk_ids and topk_scale must share the same num_seq");
  TORCH_CHECK(topk_ids.size(1) == topk_scale.size(1),
              "topk_ids and topk_scale must share the same num_topk");
  TORCH_CHECK(x.size(1) == gate_up_weight.size(2), "x and weight must share the same k");
  TORCH_CHECK(gate_up_weight.size(0) == down_weight.size(0),
              "gate_up_weight and down_weight must share the same num_expert");

  const void *shared_output_ptr = nullptr;
  if (shared_output.has_value()) {
    const auto shared_output_tensor = shared_output.value();
    TORCH_CHECK(shared_output_tensor.device().is_cuda(), "shared_output tensor must be cuda");
    TORCH_CHECK(shared_output_tensor.is_contiguous(), "shared_output tensor must be contiguous");
    TORCH_CHECK(shared_output_tensor.dtype() == torch::kBFloat16,
                "shared_output tensor dtype must be bfloat16");
    TORCH_CHECK(
        shared_output_tensor.size(0) == x.size(0) && shared_output_tensor.size(1) == x.size(1),
        "shared_output tensor shape must be same as x tensor");
    shared_output_ptr = shared_output_tensor.const_data_ptr();
  }

  int num_seq = x.size(0);
  int hidden_size = x.size(1);
  int num_expert = gate_up_weight.size(0);
  int intermediate_size = gate_up_weight.size(1);
  int num_topk = topk_ids.size(1);
  TORCH_CHECK(num_topk <= 128, "num_topk must less than or equal to 128");

  auto options = x.options();
  torch::Tensor y = torch::empty({num_seq, hidden_size}, options.dtype(torch::kBFloat16));

  torch::Tensor gate_up_input = torch::empty({num_seq * num_topk, hidden_size}, options);
  torch::Tensor gate_up_output =
      torch::empty({num_seq * num_topk, intermediate_size}, options.dtype(torch::kBFloat16));
  torch::Tensor gate_up_tmas = torch::empty({num_expert * 2, 128}, options.dtype(torch::kInt8));
  torch::Tensor down_input = torch::empty({num_seq * num_topk, intermediate_size / 2}, options);
  torch::Tensor down_output =
      torch::empty({num_seq * num_topk, hidden_size}, options.dtype(torch::kBFloat16));
  torch::Tensor down_tmas = torch::empty({num_expert * 2, 128}, options.dtype(torch::kInt8));

  torch::Tensor topk_pos = torch::empty({num_seq, num_topk}, options.dtype(torch::kInt32));
  torch::Tensor seqlens = torch::zeros({num_expert}, options.dtype(torch::kInt32));
  torch::Tensor cu_seqlens = torch::empty({num_expert + 1}, options.dtype(torch::kInt32));
  torch::Tensor tiles = torch::empty({num_expert}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_expert + 1}, options.dtype(torch::kInt32));

  const auto *x_ptr = x.const_data_ptr();
  const auto *topk_ids_ptr = topk_ids.const_data_ptr();
  const auto *topk_scale_ptr = topk_scale.const_data_ptr();
  const auto *gate_up_weight_ptr = gate_up_weight.const_data_ptr();
  const auto *gate_up_scale_ptr = gate_up_scale.const_data_ptr();
  const auto *act_and_mul_scale_ptr = act_and_mul_scale.const_data_ptr();
  const auto *down_weight_ptr = down_weight.const_data_ptr();
  const auto *down_scale_ptr = down_scale.const_data_ptr();

  auto *y_ptr = y.mutable_data_ptr();
  auto *topk_pos_ptr = topk_pos.mutable_data_ptr();
  auto *seqlens_ptr = seqlens.mutable_data_ptr();
  auto *cu_seqlens_ptr = cu_seqlens.mutable_data_ptr();
  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();
  auto *gate_up_input_ptr = gate_up_input.mutable_data_ptr();
  auto *gate_up_output_ptr = gate_up_output.mutable_data_ptr();
  auto *gate_up_tmas_ptr = gate_up_tmas.mutable_data_ptr();
  auto *down_input_ptr = down_input.mutable_data_ptr();
  auto *down_output_ptr = down_output.mutable_data_ptr();
  auto *down_tmas_ptr = down_tmas.mutable_data_ptr();

  fuse_moe_pertensor_fp8_async(
      y_ptr, x_ptr, gate_up_input_ptr, gate_up_output_ptr, gate_up_weight_ptr, gate_up_scale_ptr,
      gate_up_tmas_ptr, act_and_mul_scale_ptr, down_input_ptr, down_output_ptr, down_weight_ptr,
      down_scale_ptr, down_tmas_ptr, topk_ids_ptr, topk_scale_ptr, topk_pos_ptr, seqlens_ptr,
      cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr, shared_output_ptr, num_seq, hidden_size,
      intermediate_size, num_topk, num_expert_total, num_expert, rank_ep, use_bf16_mul, stream);

  return y;
}

torch::Tensor fuse_moe_blockwise_fp8_entry(
    const torch::Tensor &x, const torch::Tensor &x_scale, const torch::Tensor &gate_up_weight,
    const torch::Tensor &gate_up_weight_scale, const torch::Tensor &down_weight,
    const torch::Tensor &down_weight_scale, const torch::Tensor &topk_ids,
    const torch::Tensor &topk_scale, const std::optional<torch::Tensor> &shared_output,
    int64_t rank_ep, int64_t num_expert_total) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  TORCH_CHECK(x.device().is_cuda(), "x tensor must be cuda");
  TORCH_CHECK(x_scale.device().is_cuda(), "x_scale tensor must be cuda");
  TORCH_CHECK(gate_up_weight.device().is_cuda(), "gate_up_weight tensor must be cuda");
  TORCH_CHECK(gate_up_weight_scale.device().is_cuda(), "gate_up_weight_scale tensor must be cuda");
  TORCH_CHECK(down_weight.device().is_cuda(), "down_weight tensor must be cuda");
  TORCH_CHECK(down_weight_scale.device().is_cuda(), "down_weight_scale tensor must be cuda");
  TORCH_CHECK(topk_ids.device().is_cuda(), "topk_ids tensor must be cuda");
  TORCH_CHECK(topk_scale.device().is_cuda(), "topk_scale tensor must be cuda");

  TORCH_CHECK(x.is_contiguous(), "x tensor must be contiguous");
  TORCH_CHECK(x_scale.is_contiguous(), "x_scale tensor must be contiguous");
  TORCH_CHECK(gate_up_weight.is_contiguous(), "gate_up_weight tensor must be contiguous");
  TORCH_CHECK(gate_up_weight_scale.is_contiguous(),
              "gate_up_weight_scale tensor must be contiguous");
  TORCH_CHECK(down_weight.is_contiguous(), "down_weight tensor must be contiguous");
  TORCH_CHECK(down_weight_scale.is_contiguous(), "down_weight_scale tensor must be contiguous");
  TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids tensor must be contiguous");
  TORCH_CHECK(topk_scale.is_contiguous(), "topk_scale tensor must be contiguous");

  TORCH_CHECK(x.size(0) == topk_ids.size(0), "x and topk_ids must share the same num_tokens");
  TORCH_CHECK(topk_ids.size(0) == topk_scale.size(0),
              "topk_ids and topk_scale must share the same num_tokens");
  TORCH_CHECK(topk_ids.size(1) == topk_scale.size(1),
              "topk_ids and topk_scale must share the same num_topk");
  TORCH_CHECK(x.size(1) == gate_up_weight.size(2), "x and weight must share the same k");
  TORCH_CHECK(gate_up_weight.size(0) == down_weight.size(0),
              "gate_up_weight and down_weight must share the same num_expert");
  TORCH_CHECK(x_scale.size(0) == x.size(0), "x_scale and x must share the same nun_tokens");
  TORCH_CHECK(x_scale.size(1) == x.size(1) / 128, "x_scale must be per 128 blockwise quant");
  TORCH_CHECK(gate_up_weight_scale.size(1) == gate_up_weight.size(1) / 128,
              "gate_up_weight must be per 128 blockwise quant");
  TORCH_CHECK(gate_up_weight_scale.size(2) == (gate_up_weight.size(2) / 128 + 3) / 4 * 4,
              "gate_up_weight must be per 128 blockwise quant and must be aligned to 4");
  TORCH_CHECK(down_weight_scale.size(1) == down_weight.size(1) / 128,
              "down_weight must be per 128 blockwise quant");
  TORCH_CHECK(down_weight_scale.size(2) == (down_weight.size(2) / 128 + 3) / 4 * 4,
              "down_weight must be per 128 blockwise quant and must be aligned to 4");

  const void *shared_output_ptr = nullptr;
  if (shared_output.has_value()) {
    const auto shared_output_tensor = shared_output.value();
    TORCH_CHECK(shared_output_tensor.device().is_cuda(), "shared_output tensor must be cuda");
    TORCH_CHECK(shared_output_tensor.is_contiguous(), "shared_output tensor must be contiguous");
    TORCH_CHECK(shared_output_tensor.dtype() == torch::kBFloat16,
                "shared_output tensor dtype must be bfloat16");
    TORCH_CHECK(
        shared_output_tensor.size(0) == x.size(0) && shared_output_tensor.size(1) == x.size(1),
        "shared_output tensor shape must be same as x tensor");
    shared_output_ptr = shared_output_tensor.const_data_ptr();
  }

  int num_tokens = x.size(0);
  int hidden_size = x.size(1);
  int num_experts = gate_up_weight.size(0);
  int intermediate_size = gate_up_weight.size(1);
  int num_topk = topk_ids.size(1);
  int num_tokens_per_group_avg = num_tokens * num_topk / num_expert_total;
  int aligned_size = 0;
  int gate_up_weight_scale_lastdim_pad4 = gate_up_weight_scale.size(-1);
  int down_weight_scale_lastdim_pad4 = down_weight_scale.size(-1);

  TORCH_CHECK(num_topk <= 128, "num_topk must less than or equal to 128");

  if (num_tokens_per_group_avg <= 16) {
    aligned_size = 16;
  } else if (num_tokens_per_group_avg <= 32) {
    aligned_size = 32;
  } else if (num_tokens_per_group_avg <= 48) {
    aligned_size = 48;
  } else {
    aligned_size = 64;
  }
  int num_padded_tokens =
      (num_tokens * num_topk + num_expert_total * aligned_size + aligned_size - 1) / aligned_size *
      aligned_size;

  auto options = x.options();
  torch::Tensor y = torch::empty({num_tokens, hidden_size}, options.dtype(torch::kBFloat16));
  torch::Tensor gate_up_input =
      torch::empty({num_tokens * num_topk, hidden_size}, options.dtype(torch::kFloat8_e4m3fn));
  torch::Tensor gate_up_input_scale =
      torch::empty({x_scale.size(1), num_padded_tokens}, options.dtype(torch::kFloat32));
  torch::Tensor gate_up_output =
      torch::empty({num_tokens * num_topk, intermediate_size}, options.dtype(torch::kBFloat16));
  torch::Tensor gate_up_tmas = torch::empty({num_experts * 2, 128}, options.dtype(torch::kInt8));
  torch::Tensor down_input = torch::empty({num_tokens * num_topk, intermediate_size / 2},
                                          options.dtype(torch::kFloat8_e4m3fn));
  torch::Tensor down_input_scale = torch::empty({intermediate_size / 2 / 128, num_padded_tokens},
                                                options.dtype(torch::kFloat32));
  torch::Tensor down_output =
      torch::empty({num_tokens * num_topk, hidden_size}, options.dtype(torch::kBFloat16));
  torch::Tensor down_tmas = torch::empty({num_experts * 2, 128}, options.dtype(torch::kInt8));
  torch::Tensor topk_pos = torch::empty({num_tokens, num_topk}, options.dtype(torch::kInt32));
  torch::Tensor num_tokens_per_group = torch::zeros({num_experts}, options.dtype(torch::kInt32));
  torch::Tensor cu_num_tokens_per_group =
      torch::empty({num_experts + 1}, options.dtype(torch::kInt32));
  torch::Tensor tiles = torch::empty({num_experts}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_experts + 1}, options.dtype(torch::kInt32));

  const auto *x_ptr = x.const_data_ptr();
  const auto *x_scale_ptr = x_scale.const_data_ptr();
  const auto *topk_ids_ptr = topk_ids.const_data_ptr();
  const auto *topk_scale_ptr = topk_scale.const_data_ptr();
  const auto *gate_up_weight_ptr = gate_up_weight.const_data_ptr();
  const auto *gate_up_weight_scale_ptr = gate_up_weight_scale.const_data_ptr();
  const auto *down_weight_ptr = down_weight.const_data_ptr();
  const auto *down_weight_scale_ptr = down_weight_scale.const_data_ptr();

  auto *y_ptr = y.mutable_data_ptr();
  auto *topk_pos_ptr = topk_pos.mutable_data_ptr();
  auto *num_tokens_per_group_ptr = num_tokens_per_group.mutable_data_ptr();
  auto *cu_num_tokens_per_group_ptr = cu_num_tokens_per_group.mutable_data_ptr();
  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();
  auto *gate_up_input_ptr = gate_up_input.mutable_data_ptr();
  auto *gate_up_input_scale_ptr = gate_up_input_scale.mutable_data_ptr();
  auto *gate_up_output_ptr = gate_up_output.mutable_data_ptr();
  auto *gate_up_tmas_ptr = gate_up_tmas.mutable_data_ptr();
  auto *down_input_ptr = down_input.mutable_data_ptr();
  auto *down_input_scale_ptr = down_input_scale.mutable_data_ptr();
  auto *down_output_ptr = down_output.mutable_data_ptr();
  auto *down_tmas_ptr = down_tmas.mutable_data_ptr();

  fuse_moe_blockwise_fp8_async(
      y_ptr, x_ptr, x_scale_ptr, gate_up_input_ptr, gate_up_input_scale_ptr, gate_up_output_ptr,
      gate_up_weight_ptr, gate_up_weight_scale_ptr, gate_up_tmas_ptr, down_input_ptr,
      down_input_scale_ptr, down_output_ptr, down_weight_ptr, down_weight_scale_ptr, down_tmas_ptr,
      topk_ids_ptr, topk_scale_ptr, topk_pos_ptr, num_tokens_per_group_ptr,
      cu_num_tokens_per_group_ptr, tiles_ptr, cu_tiles_ptr, shared_output_ptr, num_tokens,
      num_padded_tokens, hidden_size, intermediate_size, num_topk, num_expert_total, num_experts,
      gate_up_weight_scale_lastdim_pad4, down_weight_scale_lastdim_pad4, rank_ep, stream);
  return y;
}

}  // namespace fuse_moe
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "count_and_gather(Tensor x, Tensor topk_ids, int num_expert, int rank_ep, int "
      "intermediate_size, int num_seq_per_group_avg"
      ") -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.impl("count_and_gather", torch::kCUDA, &hpc::fuse_moe::count_and_gather_entry);

  m.def(
      "reduce(Tensor x, Tensor topk_pos, Tensor topk_scale, Tensor ? shared_output"
      ") -> (Tensor)");
  m.impl("reduce", torch::kCUDA, &hpc::fuse_moe::reduce_entry);

  m.def(
      "fuse_moe_pertensor_fp8(Tensor x, Tensor gate_up_weight, Tensor down_weight, Tensor "
      "gate_up_scale, "
      "Tensor down_scale, Tensor act_and_mul_scale, Tensor topk_ids, Tensor topk_scale, Tensor ? "
      "shared_output, "
      "int rank_ep, int num_expert_total, bool use_bf16_mul) -> (Tensor)");
  m.impl("fuse_moe_pertensor_fp8", torch::kCUDA, &hpc::fuse_moe::fuse_moe_pertensor_fp8_entry);

  m.def(
      "fuse_moe_blockwise_fp8(Tensor x, Tensor x_scale, Tensor gate_up_weight, Tensor "
      "gate_up_weight_scale, "
      "Tensor down_weight, Tensor down_weight_scale, Tensor topk_ids, Tensor topk_scale, Tensor ? "
      "shared_output, "
      "int rank_ep, int num_expert_total) -> (Tensor)");
  m.impl("fuse_moe_blockwise_fp8", torch::kCUDA, &hpc::fuse_moe::fuse_moe_blockwise_fp8_entry);
}
