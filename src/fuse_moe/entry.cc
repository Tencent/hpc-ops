// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

#include "src/fuse_moe/fuse_moe.h"

namespace hpc {
namespace fuse_moe {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
count_and_gather_entry(const torch::Tensor &x, const torch::Tensor &topk_ids,
                       const int64_t num_expert, const int64_t eprank,
                       const int64_t intermediate_size) {
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
  torch::Tensor y = torch::empty({num_seq * num_topk, hidden_size}, options);
  torch::Tensor yg =
      torch::empty({num_seq * num_topk, intermediate_size}, options.dtype(torch::kBFloat16));

  torch::Tensor topk_pos = torch::empty({num_seq, num_topk}, options.dtype(torch::kInt32));
  torch::Tensor seqlens = torch::empty({num_expert}, options.dtype(torch::kInt32));
  torch::Tensor cu_seqlens = torch::empty({num_expert + 1}, options.dtype(torch::kInt32));
  torch::Tensor tiles = torch::empty({num_expert}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_expert + 1}, options.dtype(torch::kInt32));
  torch::Tensor tmas = torch::empty({num_expert * 2, 128}, options.dtype(torch::kInt8));

  const auto *x_ptr = x.const_data_ptr();
  const auto *topk_ids_ptr = topk_ids.const_data_ptr();

  auto *y_ptr = y.mutable_data_ptr();
  auto *yg_ptr = yg.mutable_data_ptr();
  auto *topk_pos_ptr = topk_pos.mutable_data_ptr();
  auto *seqlens_ptr = seqlens.mutable_data_ptr();
  auto *cu_seqlens_ptr = cu_seqlens.mutable_data_ptr();
  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();
  auto *tmas_ptr = tmas.mutable_data_ptr();

  count_and_gather_async(y_ptr, yg_ptr, x_ptr, topk_ids_ptr, topk_pos_ptr, seqlens_ptr,
                         cu_seqlens_ptr, tmas_ptr, tiles_ptr, cu_tiles_ptr, num_seq, hidden_size,
                         intermediate_size, num_topk, num_expert, eprank, stream);

  return std::make_tuple(y, yg, topk_pos, seqlens, cu_seqlens, tiles, cu_tiles, tmas);
}

torch::Tensor reduce_entry(const torch::Tensor &x, const torch::Tensor &topk_pos,
                           const torch::Tensor &topk_scale) {
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

  int total_num_seq = x.size(0);
  int hidden_size = x.size(1);
  int num_seq = topk_pos.size(0);
  int num_topk = topk_pos.size(1);

  auto options = x.options();
  torch::Tensor y = torch::empty({num_seq, hidden_size}, options.dtype(torch::kBFloat16));

  const auto *x_ptr = x.const_data_ptr();
  const auto *topk_pos_ptr = topk_pos.const_data_ptr();
  const auto *topk_scale_ptr = topk_scale.const_data_ptr();

  auto *y_ptr = y.mutable_data_ptr();

  reduce_async(y_ptr, x_ptr, topk_pos_ptr, topk_scale_ptr, total_num_seq, num_seq, hidden_size,
               num_topk, stream);

  return y;
}

torch::Tensor fuse_moe_entry(const torch::Tensor &x, const torch::Tensor &gate_up_weight,
                             const torch::Tensor &down_weight, const torch::Tensor &gate_up_scale,
                             const torch::Tensor &down_scale,
                             const torch::Tensor &act_and_mul_scale, const torch::Tensor &topk_ids,
                             const torch::Tensor &topk_scale, int64_t eprank) {
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

  int num_seq = x.size(0);
  int hidden_size = x.size(1);
  int num_expert = gate_up_weight.size(0);
  int intermediate_size = gate_up_weight.size(1);
  int num_topk = topk_ids.size(1);

  auto options = x.options();
  torch::Tensor y = torch::empty({num_seq, hidden_size}, options.dtype(torch::kBFloat16));

  torch::Tensor gate_up_input =
      torch::empty({num_seq * num_topk, hidden_size}, options.dtype(torch::kBFloat16));
  torch::Tensor gate_up_output =
      torch::empty({num_seq * num_topk, intermediate_size}, options.dtype(torch::kBFloat16));
  torch::Tensor gate_up_tmas = torch::empty({num_expert * 2, 128}, options.dtype(torch::kInt8));
  torch::Tensor down_input =
      torch::empty({num_seq * num_topk, intermediate_size / 2}, options.dtype(torch::kBFloat16));
  torch::Tensor down_output =
      torch::empty({num_seq * num_topk, hidden_size}, options.dtype(torch::kBFloat16));
  torch::Tensor down_tmas = torch::empty({num_expert * 2, 128}, options.dtype(torch::kInt8));

  torch::Tensor topk_pos = torch::empty({num_seq, num_topk}, options.dtype(torch::kInt32));
  torch::Tensor seqlens = torch::empty({num_expert}, options.dtype(torch::kInt32));
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

  fuse_moe_async(y_ptr, x_ptr, gate_up_input_ptr, gate_up_output_ptr, gate_up_weight_ptr,
                 gate_up_scale_ptr, gate_up_tmas_ptr, act_and_mul_scale_ptr, down_input_ptr,
                 down_output_ptr, down_weight_ptr, down_scale_ptr, down_tmas_ptr, topk_ids_ptr,
                 topk_scale_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr,
                 num_seq, hidden_size, intermediate_size, num_topk, num_expert, eprank, stream);

  return y;
}

}  // namespace fuse_moe
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "count_and_gather(Tensor x, Tensor topk_ids, int num_expert, int eprank, int "
      "intermediate_size"
      ") -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.impl("count_and_gather", torch::kCUDA, &hpc::fuse_moe::count_and_gather_entry);

  m.def(
      "reduce(Tensor x, Tensor topk_pos, Tensor topk_scale"
      ") -> (Tensor)");
  m.impl("reduce", torch::kCUDA, &hpc::fuse_moe::reduce_entry);

  m.def(
      "fuse_moe(Tensor x, Tensor gate_up_weight, Tensor down_weight, Tensor gate_up_scale, "
      "Tensor down_scale, Tensor act_and_mul_scale, Tensor topk_ids, Tensor topk_scale,"
      "int eprank) -> (Tensor)");
  m.impl("fuse_moe", torch::kCUDA, &hpc::fuse_moe::fuse_moe_entry);
}
