// Copyright (C) 2026 Tencent.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

#include "src/fuse_moe/fuse_moe.h"
#include "src/fuse_moe/fuse_moe_cp_async.h"
#include "src/utils/utils.h"

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

  HPC_ARCH_DISPATCH(
      "count_and_gather", 90,
      count_and_gather_async(gate_up_input_ptr, gate_up_output_ptr, down_input_ptr, down_output_ptr,
                             x_ptr, topk_ids_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr,
                             gate_up_tmas_ptr, dowm_tmas_ptr, tiles_ptr, cu_tiles_ptr, nullptr,
                             nullptr, num_seq, hidden_size, intermediate_size, num_topk, num_expert,
                             rank_ep, num_seq_per_group_avg, stream));

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

  HPC_ARCH_DISPATCH("reduce", 90,
                    reduce_async(y_ptr, x_ptr, topk_pos_ptr, topk_scale_ptr, shared_output_ptr,
                                 total_num_seq, num_seq, hidden_size, num_topk, false, stream));

  return y;
}

torch::Tensor fuse_moe_cp_async_entry(
    const torch::Tensor &x, const torch::Tensor &gate_up_weight, const torch::Tensor &down_weight,
    const torch::Tensor &gate_up_scale, const torch::Tensor &down_scale,
    const torch::Tensor &act_and_mul_scale, const torch::Tensor &topk_ids,
    const torch::Tensor &topk_scale, std::optional<torch::Tensor> shared_output, int64_t rank_ep,
    int64_t num_expert_total, bool use_bf16_mul, std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  TORCH_CHECK(x.dtype() == torch::kFloat8_e4m3fn &&
                  gate_up_weight.dtype() == torch::kFloat8_e4m3fn &&
                  down_weight.dtype() == torch::kFloat8_e4m3fn,
              "x, gate_up_weight and down_weight dtype must be fp8_e4m3");
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32, "topk_ids dtype must be int32");
  TORCH_CHECK(gate_up_scale.dtype() == torch::kFloat32 && down_scale.dtype() == torch::kFloat32 &&
                  act_and_mul_scale.dtype() == torch::kFloat32 &&
                  topk_scale.dtype() == torch::kFloat32,
              "gate_up_scale, down_scale, act_and_mul_scale and topk_scale dtype must be float32");

  TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(gate_up_weight.device().is_cuda(), "gate_up_weight must be a CUDA tensor");
  TORCH_CHECK(down_weight.device().is_cuda(), "down_weight must be a CUDA tensor");
  TORCH_CHECK(topk_ids.device().is_cuda(), "topk_ids must be a CUDA tensor");
  TORCH_CHECK(topk_scale.device().is_cuda(), "topk_scale must be a CUDA tensor");

  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(gate_up_weight.is_contiguous(), "gate_up_weight must be contiguous");
  TORCH_CHECK(down_weight.is_contiguous(), "down_weight must be contiguous");
  TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
  TORCH_CHECK(topk_scale.is_contiguous(), "topk_scale must be contiguous");

  TORCH_CHECK(x.size(0) == topk_ids.size(0), "x and topk_ids must share the same num_seq");
  TORCH_CHECK(topk_ids.size(0) == topk_scale.size(0),
              "topk_ids and topk_scale must share the same num_seq");
  TORCH_CHECK(topk_ids.size(1) == topk_scale.size(1),
              "topk_ids and topk_scale must share the same num_topk");
  TORCH_CHECK(x.size(1) == gate_up_weight.size(2), "x and gate_up_weight must share the same k");
  TORCH_CHECK(gate_up_weight.size(0) == down_weight.size(0),
              "gate_up_weight and down_weight must share the same num_expert");
  TORCH_CHECK(num_expert_total > 0, "num_expert_total must be positive");

  const void *shared_output_ptr = nullptr;
  if (shared_output.has_value()) {
    const auto &so = shared_output.value();
    TORCH_CHECK(so.device().is_cuda(), "shared_output must be a CUDA tensor");
    TORCH_CHECK(so.is_contiguous(), "shared_output must be contiguous");
    TORCH_CHECK(so.dtype() == torch::kBFloat16, "shared_output dtype must be bfloat16");
    TORCH_CHECK(so.size(0) == x.size(0) && so.size(1) == x.size(1),
                "shared_output shape must match x");
    shared_output_ptr = so.const_data_ptr();
  }

  int num_seq = x.size(0);
  int hidden_size = x.size(1);
  int num_expert_local = gate_up_weight.size(0);
  int intermediate_size = gate_up_weight.size(1);  // gate+up fused: 2 * actual_intermediate
  int num_topk = topk_ids.size(1);
  int total_num_seq = num_seq * num_topk;

  TORCH_CHECK(num_topk <= 128, "num_topk must be <= 128");
  TORCH_CHECK(num_expert_local <= 512, "num_expert_local must be <= 512, got ", num_expert_local);
  TORCH_CHECK(intermediate_size % 128 == 0,
              "gate_up_weight.size(1) (= 2 * intermediate_size) must be a multiple of 128; "
              "Down GEMM requires intermediate_size %% 64 == 0, got ",
              intermediate_size / 2);
  TORCH_CHECK(hidden_size % 64 == 0, "hidden_size must be a multiple of 64, got ", hidden_size);

  auto options = x.options();

  torch::Tensor y;
  void *y_ptr = nullptr;
  if (output.has_value()) {
    TORCH_CHECK(output.value().size(0) == num_seq && output.value().size(1) == hidden_size,
                "output shape must be [num_seq, hidden_size]");
    TORCH_CHECK(output.value().dtype() == torch::kBFloat16, "output dtype must be bfloat16");
    TORCH_CHECK(output.value().device().is_cuda(), "output must be a CUDA tensor");
    y_ptr = output.value().mutable_data_ptr();
  } else {
    y = torch::empty({num_seq, hidden_size}, options.dtype(torch::kBFloat16));
    y_ptr = y.mutable_data_ptr();
  }

  torch::Tensor gate_up_output =
      torch::empty({total_num_seq, intermediate_size}, options.dtype(torch::kBFloat16));

  torch::Tensor down_input =
      torch::empty({total_num_seq, intermediate_size / 2}, options.dtype(torch::kFloat8_e4m3fn));

  torch::Tensor down_output =
      torch::empty({total_num_seq, hidden_size}, options.dtype(torch::kBFloat16));

  torch::Tensor topk_pos = torch::empty({num_seq, num_topk}, options.dtype(torch::kInt32));

  torch::Tensor row_indices = torch::empty({total_num_seq}, options.dtype(torch::kInt32));

  torch::Tensor seqlens = torch::zeros({num_expert_local}, options.dtype(torch::kInt32));
  torch::Tensor cu_seqlens = torch::empty({num_expert_local + 1}, options.dtype(torch::kInt32));
  torch::Tensor tiles = torch::empty({num_expert_local}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_expert_local + 1}, options.dtype(torch::kInt32));

  constexpr int kMinTileM = 8;
  constexpr int kGemmTileN = 64;
  int gate_up_num_tile_n = (intermediate_size + kGemmTileN - 1) / kGemmTileN;
  int down_num_tile_n = (hidden_size + kGemmTileN - 1) / kGemmTileN;
  int max_tile_m = total_num_seq / kMinTileM + num_expert_local;
  int gateup_task_map_len = max_tile_m * gate_up_num_tile_n;
  int down_task_map_len = max_tile_m * down_num_tile_n;

  torch::Tensor gateup_task_map =
      torch::empty({gateup_task_map_len, 4}, options.dtype(torch::kInt32));
  torch::Tensor down_task_map = torch::empty({down_task_map_len, 4}, options.dtype(torch::kInt32));
  auto *gateup_task_map_ptr = gateup_task_map.mutable_data_ptr();
  auto *down_task_map_ptr = down_task_map.mutable_data_ptr();

  const auto *x_ptr = x.const_data_ptr();
  const auto *topk_ids_ptr = topk_ids.const_data_ptr();
  const auto *topk_scale_ptr = topk_scale.const_data_ptr();
  const auto *gate_up_weight_ptr = gate_up_weight.const_data_ptr();
  const auto *gate_up_scale_ptr = gate_up_scale.const_data_ptr();
  const auto *act_and_mul_scale_ptr = act_and_mul_scale.const_data_ptr();
  const auto *down_weight_ptr = down_weight.const_data_ptr();
  const auto *down_scale_ptr = down_scale.const_data_ptr();

  auto *gate_up_output_ptr = gate_up_output.mutable_data_ptr();
  auto *down_input_ptr = down_input.mutable_data_ptr();
  auto *down_output_ptr = down_output.mutable_data_ptr();
  auto *topk_pos_ptr = topk_pos.mutable_data_ptr();
  auto *row_indices_ptr = row_indices.mutable_data_ptr();
  auto *seqlens_ptr = seqlens.mutable_data_ptr();
  auto *cu_seqlens_ptr = cu_seqlens.mutable_data_ptr();
  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();

  HPC_ARCH_DISPATCH(
      "fuse_moe_cp_async", 90,
      fuse_moe_cp_async::fuse_moe_cp_async(
          y_ptr, x_ptr, gate_up_output_ptr, gate_up_weight_ptr, gate_up_scale_ptr,
          act_and_mul_scale_ptr, down_input_ptr, down_output_ptr, down_weight_ptr, down_scale_ptr,
          topk_ids_ptr, topk_scale_ptr, topk_pos_ptr, row_indices_ptr, seqlens_ptr, cu_seqlens_ptr,
          tiles_ptr, cu_tiles_ptr, gateup_task_map_ptr, down_task_map_ptr, gateup_task_map_len,
          down_task_map_len, shared_output_ptr, num_seq, hidden_size, intermediate_size, num_topk,
          num_expert_total, num_expert_local, static_cast<int>(rank_ep), use_bf16_mul, stream));

  if (output.has_value()) {
    return output.value();
  }
  return y;
}

torch::Tensor fuse_moe_entry(const torch::Tensor &x, const torch::Tensor &gate_up_weight,
                             const torch::Tensor &down_weight, const torch::Tensor &gate_up_scale,
                             const torch::Tensor &down_scale,
                             const torch::Tensor &act_and_mul_scale, const torch::Tensor &topk_ids,
                             const torch::Tensor &topk_scale,
                             std::optional<torch::Tensor> shared_output, int64_t rank_ep,
                             int64_t num_expert_total, bool use_bf16_mul,
                             std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  TORCH_CHECK(x.dtype() == torch::kFloat8_e4m3fn &&
                  gate_up_weight.dtype() == torch::kFloat8_e4m3fn &&
                  down_weight.dtype() == torch::kFloat8_e4m3fn,
              "x, gate_up_weight and down_weight dtype must be fp8_e4m3");
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32, "topk_ids dtype must be int32");
  TORCH_CHECK(gate_up_scale.dtype() == torch::kFloat32 && down_scale.dtype() == torch::kFloat32 &&
                  act_and_mul_scale.dtype() == torch::kFloat32 &&
                  topk_scale.dtype() == torch::kFloat32,
              "gate_up_scale, down_scale, act_and_mul_scale and topk_scale dtype must be float32");
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
  int aligned_size = 0;
  int num_tokens_per_group_avg = num_seq * num_topk / num_expert_total;
  TORCH_CHECK(num_topk <= 128, "num_topk must less than or equal to 128");

  if (intermediate_size / 2 <= 512 && (intermediate_size / 2) % 64 == 0 && hidden_size % 64 == 0) {
    return fuse_moe_cp_async_entry(x, gate_up_weight, down_weight, gate_up_scale, down_scale,
                                   act_and_mul_scale, topk_ids, topk_scale, shared_output, rank_ep,
                                   num_expert_total, use_bf16_mul, output);
  }

  auto options = x.options();
  torch::Tensor y;
  void *y_ptr = nullptr;
  if (output.has_value()) {
    TORCH_CHECK(output.value().size(0) == num_seq && output.value().size(1) == hidden_size,
                "output shape must be [num_tokens, hidden_size]");
    TORCH_CHECK(output.value().dtype() == torch::kBFloat16, "output dtype must be bfloat16");
    TORCH_CHECK(output.value().device().is_cuda(), "output must be cuda tensor");
    y_ptr = output.value().mutable_data_ptr();
  } else {
    y = torch::empty({num_seq, hidden_size}, options.dtype(torch::kBFloat16));
    y_ptr = y.mutable_data_ptr();
  }

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

  if (num_tokens_per_group_avg <= 8) {
    aligned_size = 8;
  } else if (num_tokens_per_group_avg <= 16) {
    aligned_size = 16;
  } else if (num_tokens_per_group_avg <= 32) {
    aligned_size = 32;
  } else if (num_tokens_per_group_avg <= 48) {
    aligned_size = 48;
  } else if (num_tokens_per_group_avg <= 64) {
    aligned_size = 64;
  } else if (num_tokens_per_group_avg <= 96) {
    aligned_size = 48;
  } else if (num_tokens_per_group_avg <= 128) {
    aligned_size = 32;
  } else if (num_tokens_per_group_avg <= 144) {
    aligned_size = 48;
  } else {
    aligned_size = 64;
  }

  int num_sm = get_sm_count();
  constexpr int kTileN = 128;
  int num_gateup_tiles = ((num_seq + aligned_size - 1) / aligned_size) *
                         ((intermediate_size + kTileN - 1) / kTileN) * num_expert;
  int num_down_tiles = ((num_seq + aligned_size - 1) / aligned_size) *
                       ((hidden_size + kTileN - 1) / kTileN) * num_expert;
  int num_gateup_waves = (num_gateup_tiles + num_sm - 1) / num_sm + 1;
  int num_down_waves = (num_down_tiles + num_sm - 1) / num_sm + 1;
  torch::Tensor gateup_task_map;
  torch::Tensor down_task_map;
  void *gateup_task_map_ptr = nullptr;
  void *down_task_map_ptr = nullptr;

  if (num_tokens_per_group_avg <= 8) {
    gateup_task_map = torch::empty({num_gateup_waves, num_sm, 4}, options.dtype(torch::kInt32));
    gateup_task_map_ptr = gateup_task_map.mutable_data_ptr();
    down_task_map = torch::empty({num_down_waves, num_sm, 4}, options.dtype(torch::kInt32));
    down_task_map_ptr = down_task_map.mutable_data_ptr();
  }

  const auto *x_ptr = x.const_data_ptr();
  const auto *topk_ids_ptr = topk_ids.const_data_ptr();
  const auto *topk_scale_ptr = topk_scale.const_data_ptr();
  const auto *gate_up_weight_ptr = gate_up_weight.const_data_ptr();
  const auto *gate_up_scale_ptr = gate_up_scale.const_data_ptr();
  const auto *act_and_mul_scale_ptr = act_and_mul_scale.const_data_ptr();
  const auto *down_weight_ptr = down_weight.const_data_ptr();
  const auto *down_scale_ptr = down_scale.const_data_ptr();

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

  HPC_ARCH_DISPATCH(
      "fuse_moe", 90,
      fuse_moe_async(y_ptr, x_ptr, gate_up_input_ptr, gate_up_output_ptr, gate_up_weight_ptr,
                     gate_up_scale_ptr, gate_up_tmas_ptr, act_and_mul_scale_ptr, down_input_ptr,
                     down_output_ptr, down_weight_ptr, down_scale_ptr, down_tmas_ptr, topk_ids_ptr,
                     topk_scale_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, tiles_ptr,
                     cu_tiles_ptr, shared_output_ptr, gateup_task_map_ptr, down_task_map_ptr,
                     num_gateup_waves, num_down_waves, num_seq, hidden_size, intermediate_size,
                     num_topk, num_expert_total, num_expert, rank_ep, use_bf16_mul, stream));
  if (output.has_value()) {
    return output.value();
  } else {
    return y;
  }
}

torch::Tensor fuse_moe_blockwise_entry(
    const torch::Tensor &x, const torch::Tensor &x_scale, const torch::Tensor &gate_up_weight,
    const torch::Tensor &gate_up_weight_scale, const torch::Tensor &down_weight,
    const torch::Tensor &down_weight_scale, const torch::Tensor &topk_ids,
    const torch::Tensor &topk_scale, std::optional<torch::Tensor> shared_output, int64_t rank_ep,
    int64_t num_expert_total, std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  TORCH_CHECK(x.dtype() == torch::kFloat8_e4m3fn &&
                  gate_up_weight.dtype() == torch::kFloat8_e4m3fn &&
                  down_weight.dtype() == torch::kFloat8_e4m3fn,
              "x, gate_up_weight and down_weight dtype must be fp8_e4m3");
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32, "topk_ids dtype must be int32");
  TORCH_CHECK(gate_up_weight_scale.dtype() == torch::kFloat32 &&
                  down_weight_scale.dtype() == torch::kFloat32 &&
                  topk_scale.dtype() == torch::kFloat32,
              "gate_up_scale, down_scale, act_and_mul_scale and topk_scale dtype must be float32");
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

  if (num_tokens_per_group_avg <= 8) {
    aligned_size = 8;
  } else if (num_tokens_per_group_avg <= 16) {
    aligned_size = 16;
  } else if (num_tokens_per_group_avg <= 32) {
    aligned_size = 32;
  } else if (num_tokens_per_group_avg <= 48) {
    aligned_size = 48;
  } else if (num_tokens_per_group_avg <= 64) {
    aligned_size = 64;
  } else if (num_tokens_per_group_avg <= 96) {
    aligned_size = 48;
  } else if (num_tokens_per_group_avg <= 128) {
    aligned_size = 32;
  } else if (num_tokens_per_group_avg <= 144) {
    aligned_size = 48;
  } else {
    aligned_size = 64;
  }
  int num_padded_tokens =
      (num_tokens * num_topk + num_expert_total * aligned_size + aligned_size - 1) / aligned_size *
      aligned_size;

  auto options = x.options();
  torch::Tensor y;
  void *y_ptr = nullptr;
  if (output.has_value()) {
    TORCH_CHECK(output.value().size(0) == num_tokens && output.value().size(1) == hidden_size,
                "output shape must be [num_tokens, hidden_size]");
    TORCH_CHECK(output.value().dtype() == torch::kBFloat16, "output dtype must be bfloat16");
    TORCH_CHECK(output.value().device().is_cuda(), "output must be cuda tensor");
    y_ptr = output.value().mutable_data_ptr();
  } else {
    y = torch::empty({num_tokens, hidden_size}, options.dtype(torch::kBFloat16));
    y_ptr = y.mutable_data_ptr();
  }
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

  int num_sm = get_sm_count();
  constexpr int kTileN = 128;
  int num_gateup_tiles = ((num_tokens + aligned_size - 1) / aligned_size) *
                         ((intermediate_size + kTileN - 1) / kTileN) * num_experts;
  int num_down_tiles = ((num_tokens + aligned_size - 1) / aligned_size) *
                       ((hidden_size + kTileN - 1) / kTileN) * num_experts;
  int num_gateup_waves = (num_gateup_tiles + num_sm - 1) / num_sm + 1;
  int num_down_waves = (num_down_tiles + num_sm - 1) / num_sm + 1;
  torch::Tensor gateup_task_map;
  torch::Tensor down_task_map;
  void *gateup_task_map_ptr = nullptr;
  void *down_task_map_ptr = nullptr;

  if (num_tokens_per_group_avg <= 8) {
    gateup_task_map = torch::empty({num_gateup_waves, num_sm, 4}, options.dtype(torch::kInt32));
    gateup_task_map_ptr = gateup_task_map.mutable_data_ptr();
    down_task_map = torch::empty({num_down_waves, num_sm, 4}, options.dtype(torch::kInt32));
    down_task_map_ptr = down_task_map.mutable_data_ptr();
  }

  const auto *x_ptr = x.const_data_ptr();
  const auto *x_scale_ptr = x_scale.const_data_ptr();
  const auto *topk_ids_ptr = topk_ids.const_data_ptr();
  const auto *topk_scale_ptr = topk_scale.const_data_ptr();
  const auto *gate_up_weight_ptr = gate_up_weight.const_data_ptr();
  const auto *gate_up_weight_scale_ptr = gate_up_weight_scale.const_data_ptr();
  const auto *down_weight_ptr = down_weight.const_data_ptr();
  const auto *down_weight_scale_ptr = down_weight_scale.const_data_ptr();

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

  HPC_ARCH_DISPATCH(
      "fuse_moe_blockwise", 90,
      fuse_moe_blockwise_async(
          y_ptr, x_ptr, x_scale_ptr, gate_up_input_ptr, gate_up_input_scale_ptr, gate_up_output_ptr,
          gate_up_weight_ptr, gate_up_weight_scale_ptr, gate_up_tmas_ptr, down_input_ptr,
          down_input_scale_ptr, down_output_ptr, down_weight_ptr, down_weight_scale_ptr,
          down_tmas_ptr, topk_ids_ptr, topk_scale_ptr, topk_pos_ptr, num_tokens_per_group_ptr,
          cu_num_tokens_per_group_ptr, tiles_ptr, cu_tiles_ptr, shared_output_ptr,
          gateup_task_map_ptr, down_task_map_ptr, num_gateup_waves, num_down_waves, num_tokens,
          num_padded_tokens, hidden_size, intermediate_size, num_topk, num_expert_total,
          num_experts, gate_up_weight_scale_lastdim_pad4, down_weight_scale_lastdim_pad4, rank_ep,
          stream));
  if (output.has_value()) {
    return output.value();
  } else {
    return y;
  }
}

static bool is_aligned16(const torch::Tensor &tensor) {
  return reinterpret_cast<uintptr_t>(tensor.const_data_ptr()) % 16 == 0;
}

// Elements of a packed int32 weight scale: four UE8M0 scales per word along K,
// rows padded to four.
static int64_t packed_weight_scale_numel(int64_t num_expert, int64_t n, int64_t k) {
  return num_expert * ((n + 3) / 4 * 4) * ((k + 127) / 128);
}

// Writes the MoE result into output, which the callers have shaped like x.
static void fuse_moe_mx_run(torch::Tensor &output, const torch::Tensor &x,
                            const std::optional<torch::Tensor> &x_scale,
                            const torch::Tensor &gate_up_weight,
                            const torch::Tensor &gate_up_weight_scale,
                            const torch::Tensor &down_weight,
                            const torch::Tensor &down_weight_scale, const torch::Tensor &topk_ids,
                            const torch::Tensor &topk_scale,
                            const std::optional<torch::Tensor> &shared_output, int64_t rank_ep,
                            int64_t num_expert_total, double swiglu_limit) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  const bool fp8_input = x.dtype() == torch::kFloat8_e4m3fn;
  const bool mxfp4_weight = gate_up_weight.dtype() == torch::kUInt8;
  TORCH_CHECK(x.dtype() == torch::kBFloat16 || fp8_input, "x dtype must be bfloat16 or fp8_e4m3");
  TORCH_CHECK(fp8_input == x_scale.has_value(),
              "x_scale is required for fp8_e4m3 x and must be None for bfloat16 x");
  TORCH_CHECK(mxfp4_weight || gate_up_weight.dtype() == torch::kFloat8_e4m3fn,
              "gate_up_weight dtype must be uint8 (packed MXFP4) or fp8_e4m3");
  TORCH_CHECK(down_weight.dtype() == gate_up_weight.dtype(),
              "gate_up_weight and down_weight must share the same dtype");
  TORCH_CHECK(
      gate_up_weight_scale.dtype() == torch::kInt32 && down_weight_scale.dtype() == torch::kInt32,
      "weight scales must use the packed int32 layout");
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32, "topk_ids dtype must be int32");
  TORCH_CHECK(topk_scale.dtype() == torch::kFloat32, "topk_scale dtype must be float32");
  TORCH_CHECK(x.is_cuda() && gate_up_weight.is_cuda() && gate_up_weight_scale.is_cuda() &&
                  down_weight.is_cuda() && down_weight_scale.is_cuda() && topk_ids.is_cuda() &&
                  topk_scale.is_cuda(),
              "all inputs must be cuda tensors");
  TORCH_CHECK(x.is_contiguous() && gate_up_weight.is_contiguous() &&
                  gate_up_weight_scale.is_contiguous() && down_weight.is_contiguous() &&
                  down_weight_scale.is_contiguous() && topk_ids.is_contiguous() &&
                  topk_scale.is_contiguous(),
              "all inputs must be contiguous");
  TORCH_CHECK(x.dim() == 2 && gate_up_weight.dim() == 3 && down_weight.dim() == 3,
              "x and the weights must have rank 2, 3 and 3");
  TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.sizes() == topk_scale.sizes() &&
                  topk_ids.size(0) == x.size(0),
              "topk_ids and topk_scale must be [num_tokens, num_topk]");

  const int num_tokens = x.size(0);
  const int hidden_size = x.size(1);
  const int num_topk = topk_ids.size(1);
  const int num_experts = gate_up_weight.size(0);
  // The gate-up GEMM's N is 2I (gate rows then up rows); SwiGLU halves it to the
  // down GEMM's K, I.
  const int gate_up_size = gate_up_weight.size(1);
  const int down_input_size = gate_up_size / 2;
  // Packed MXFP4 holds two elements per byte along K.
  const int k_div = mxfp4_weight ? 2 : 1;
  TORCH_CHECK(num_tokens >= 1, "num_tokens must be positive");
  TORCH_CHECK(num_topk >= 1, "num_topk must be positive");
  TORCH_CHECK(hidden_size % 128 == 0 && down_input_size % 128 == 0,
              "hidden_size and down_input_size must be multiples of 128");
  TORCH_CHECK(gate_up_size == 2 * down_input_size && gate_up_weight.size(2) == hidden_size / k_div,
              "gate_up_weight must be [num_expert, gate_up_size, hidden_size / ", k_div, "]");
  TORCH_CHECK(down_weight.size(0) == num_experts && down_weight.size(1) == hidden_size &&
                  down_weight.size(2) == down_input_size / k_div,
              "down_weight must be [num_expert, hidden_size, down_input_size / ", k_div, "]");
  TORCH_CHECK(gate_up_weight_scale.numel() ==
                  packed_weight_scale_numel(num_experts, gate_up_size, hidden_size),
              "gate_up_weight_scale size does not match the packed int32 layout");
  TORCH_CHECK(down_weight_scale.numel() ==
                  packed_weight_scale_numel(num_experts, hidden_size, down_input_size),
              "down_weight_scale size does not match the packed int32 layout");
  TORCH_CHECK(num_expert_total == 0 || num_expert_total >= num_experts,
              "num_expert_total must be zero or at least num_expert");
  // The GEMMs load the weights and their packed scales with TMA, which needs
  // 16-byte aligned base addresses. These are model weights prepared once at
  // load time, so a misaligned one is rejected instead of copied every call.
  TORCH_CHECK(is_aligned16(gate_up_weight) && is_aligned16(down_weight),
              "gate_up_weight and down_weight must be 16-byte aligned (TMA)");
  TORCH_CHECK(is_aligned16(gate_up_weight_scale) && is_aligned16(down_weight_scale),
              "gate_up_weight_scale and down_weight_scale must be 16-byte aligned (TMA)");

  // E4M3 x takes uint8 UE8M0 scales per 32 columns, or float32 powers of two per
  // 128 columns.
  bool block128_scale = false;
  if (fp8_input) {
    const auto &scale = x_scale.value();
    block128_scale = scale.dtype() == torch::kFloat32;
    TORCH_CHECK(block128_scale || scale.dtype() == torch::kUInt8,
                "x_scale dtype must be uint8 or float32");
    const int block = block128_scale ? 128 : 32;
    TORCH_CHECK(scale.is_cuda() && scale.is_contiguous() && scale.dim() == 2 &&
                    scale.size(0) == num_tokens && scale.size(1) * block == hidden_size,
                "x_scale must be contiguous cuda [num_tokens, hidden_size / ", block, "]");
  }
  if (shared_output.has_value()) {
    const auto &shared = shared_output.value();
    TORCH_CHECK(shared.is_cuda() && shared.is_contiguous() && shared.dtype() == torch::kBFloat16 &&
                    shared.sizes() == x.sizes(),
                "shared_output must be contiguous cuda bfloat16 with the shape of x");
  }
  TORCH_CHECK(output.is_cuda() && output.is_contiguous() && output.dtype() == torch::kBFloat16 &&
                  output.sizes() == x.sizes(),
              "output must be contiguous cuda bfloat16 with the shape of x");

  FuseMoeMxPlan plan{};
  HPC_ARCH_DISPATCH(
      "fuse_moe_mx", 120,
      plan = fuse_moe_mx_plan(num_tokens, num_topk, num_experts, num_expert_total, hidden_size,
                              down_input_size, mxfp4_weight, shared_output.has_value()));

  // The kernels read x in 16-byte vectors and copy x_scale a 4-byte word at a
  // time; the down GEMM adds into the output with 16-byte vector atomics.
  const torch::Tensor input = is_aligned16(x) ? x : x.clone();
  const void *input_scale_ptr = nullptr;
  torch::Tensor input_scale;
  if (fp8_input) {
    input_scale = x_scale.value();
    if (reinterpret_cast<uintptr_t>(input_scale.const_data_ptr()) % 4 != 0) {
      input_scale = input_scale.clone();
    }
    input_scale_ptr = input_scale.const_data_ptr();
  }
  // A misaligned output is computed in a temporary and copied back at the end.
  const bool direct_output = is_aligned16(output);
  torch::Tensor y =
      direct_output ? output
                    : torch::empty({num_tokens, hidden_size}, x.options().dtype(torch::kBFloat16));

  // The routing sets the output up as a copy of shared_output, or as zeros,
  // unless the gate_up GEMM zeroes it; a misaligned shared_output is copied here.
  const bool init_in_routing = !plan.zero_output_in_gate_up_gemm &&
                               (!shared_output.has_value() || is_aligned16(*shared_output));
  if (shared_output.has_value() && !init_in_routing) {
    y.copy_(shared_output.value());
  }
  const void *shared_output_ptr =
      init_in_routing && shared_output.has_value() ? shared_output->const_data_ptr() : nullptr;

  auto int_options = topk_ids.options();
  auto fp8_options = x.options().dtype(torch::kFloat8_e4m3fn);
  torch::Tensor topk_pos = torch::empty_like(topk_ids);
  torch::Tensor row_to_route = torch::empty({plan.max_padded_rows}, int_options);
  torch::Tensor num_tokens_per_group = torch::empty({num_experts}, int_options);
  torch::Tensor cu_padded_tokens_per_group = torch::empty({num_experts + 1}, int_options);
  torch::Tensor cu_tiles = torch::empty({num_experts + 1}, int_options);
  torch::Tensor route_counts = plan.zero_route_counts ? torch::zeros({num_experts}, int_options)
                                                      : torch::empty({num_experts}, int_options);
  torch::Tensor gate_up_input =
      fp8_input ? torch::Tensor() : torch::empty({num_tokens, hidden_size}, fp8_options);
  torch::Tensor gate_up_input_scale =
      torch::empty({plan.scale_m_padded * (hidden_size / 128)}, int_options);
  torch::Tensor down_input = torch::empty({plan.max_padded_rows, down_input_size}, fp8_options);
  torch::Tensor down_input_scale =
      torch::empty({plan.scale_m_padded * (down_input_size / 128)}, int_options);

  HPC_ARCH_DISPATCH(
      "fuse_moe_mx", 120,
      fuse_moe_mx_async(
          y.mutable_data_ptr(), input.const_data_ptr(), input_scale_ptr, block128_scale,
          gate_up_weight.const_data_ptr(), gate_up_weight_scale.const_data_ptr(),
          down_weight.const_data_ptr(), down_weight_scale.const_data_ptr(),
          topk_ids.const_data_ptr(), topk_scale.const_data_ptr(), shared_output_ptr,
          init_in_routing, topk_pos.mutable_data_ptr(), row_to_route.mutable_data_ptr(),
          num_tokens_per_group.mutable_data_ptr(), cu_padded_tokens_per_group.mutable_data_ptr(),
          cu_tiles.mutable_data_ptr(), route_counts.mutable_data_ptr(),
          fp8_input ? nullptr : gate_up_input.mutable_data_ptr(),
          gate_up_input_scale.mutable_data_ptr(), down_input.mutable_data_ptr(),
          down_input_scale.mutable_data_ptr(), num_tokens, num_topk, hidden_size, down_input_size,
          num_experts, rank_ep, mxfp4_weight, static_cast<float>(swiglu_limit), plan, stream));

  if (!direct_output) {
    output.copy_(y);
  }
}

torch::Tensor fuse_moe_mx_entry(const torch::Tensor &x, std::optional<torch::Tensor> x_scale,
                                const torch::Tensor &gate_up_weight,
                                const torch::Tensor &gate_up_weight_scale,
                                const torch::Tensor &down_weight,
                                const torch::Tensor &down_weight_scale,
                                const torch::Tensor &topk_ids, const torch::Tensor &topk_scale,
                                std::optional<torch::Tensor> shared_output, int64_t rank_ep,
                                int64_t num_expert_total, double swiglu_limit) {
  torch::Tensor output = torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
  fuse_moe_mx_run(output, x, x_scale, gate_up_weight, gate_up_weight_scale, down_weight,
                  down_weight_scale, topk_ids, topk_scale, shared_output, rank_ep, num_expert_total,
                  swiglu_limit);
  return output;
}

void fuse_moe_mx_out_entry(torch::Tensor output, const torch::Tensor &x,
                           std::optional<torch::Tensor> x_scale,
                           const torch::Tensor &gate_up_weight,
                           const torch::Tensor &gate_up_weight_scale,
                           const torch::Tensor &down_weight, const torch::Tensor &down_weight_scale,
                           const torch::Tensor &topk_ids, const torch::Tensor &topk_scale,
                           std::optional<torch::Tensor> shared_output, int64_t rank_ep,
                           int64_t num_expert_total, double swiglu_limit) {
  fuse_moe_mx_run(output, x, x_scale, gate_up_weight, gate_up_weight_scale, down_weight,
                  down_weight_scale, topk_ids, topk_scale, shared_output, rank_ep, num_expert_total,
                  swiglu_limit);
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
      "fuse_moe(Tensor x, Tensor gate_up_weight, Tensor down_weight, Tensor gate_up_scale, "
      "Tensor down_scale, Tensor act_and_mul_scale, Tensor topk_ids, Tensor topk_scale, Tensor ? "
      "shared_output, "
      "int rank_ep, int num_expert_total, bool use_bf16_mul, Tensor ? output) -> (Tensor)");
  m.impl("fuse_moe", torch::kCUDA, &hpc::fuse_moe::fuse_moe_entry);

  m.def(
      "fuse_moe_pertensor_fp8(Tensor x, Tensor gate_up_weight, Tensor down_weight, Tensor "
      "gate_up_scale, Tensor down_scale, Tensor act_and_mul_scale, Tensor topk_ids, Tensor "
      "topk_scale, Tensor ? shared_output, int rank_ep, int num_expert_total, bool use_bf16_mul, "
      "Tensor ? output) -> (Tensor)");
  m.impl("fuse_moe_pertensor_fp8", torch::kCUDA, &hpc::fuse_moe::fuse_moe_entry);

  m.def(
      "fuse_moe_blockwise(Tensor x, Tensor x_scale, Tensor gate_up_weight, Tensor "
      "gate_up_weight_scale, "
      "Tensor down_weight, Tensor down_weight_scale, Tensor topk_ids, Tensor topk_scale, Tensor ? "
      "shared_output, "
      "int rank_ep, int num_expert_total, Tensor ? output) -> (Tensor)");
  m.impl("fuse_moe_blockwise", torch::kCUDA, &hpc::fuse_moe::fuse_moe_blockwise_entry);

  m.def(
      "fuse_moe_blockwise_fp8(Tensor x, Tensor x_scale, Tensor gate_up_weight, Tensor "
      "gate_up_weight_scale, Tensor down_weight, Tensor down_weight_scale, Tensor topk_ids, "
      "Tensor topk_scale, Tensor ? shared_output, int rank_ep, int num_expert_total, Tensor ? "
      "output) -> (Tensor)");
  m.impl("fuse_moe_blockwise_fp8", torch::kCUDA, &hpc::fuse_moe::fuse_moe_blockwise_entry);

  m.def(
      "fuse_moe_mx(Tensor x, Tensor ? x_scale, Tensor gate_up_weight, Tensor "
      "gate_up_weight_scale, Tensor down_weight, Tensor down_weight_scale, Tensor topk_ids, "
      "Tensor topk_scale, Tensor ? shared_output, int rank_ep, int num_expert_total, float "
      "swiglu_limit) -> (Tensor)");
  m.impl("fuse_moe_mx", torch::kCUDA, &hpc::fuse_moe::fuse_moe_mx_entry);

  m.def(
      "fuse_moe_mx_out(Tensor(a!) output, Tensor x, Tensor ? x_scale, Tensor gate_up_weight, "
      "Tensor gate_up_weight_scale, Tensor down_weight, Tensor down_weight_scale, Tensor "
      "topk_ids, Tensor topk_scale, Tensor ? shared_output, int rank_ep, int num_expert_total, "
      "float swiglu_limit) -> ()");
  m.impl("fuse_moe_mx_out", torch::kCUDA, &hpc::fuse_moe::fuse_moe_mx_out_entry);
}
