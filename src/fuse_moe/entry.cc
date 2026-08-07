// Copyright (C) 2026 Tencent.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm>
#include <tuple>

#include "src/fuse_moe/cp_async/fuse_moe_cp_async.h"
#include "src/fuse_moe/fuse_moe.h"
#include "src/fuse_moe/small_batch_route_mma.h"
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

  count_and_gather_async(gate_up_input_ptr, gate_up_output_ptr, down_input_ptr, down_output_ptr,
                         x_ptr, topk_ids_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr,
                         gate_up_tmas_ptr, dowm_tmas_ptr, tiles_ptr, cu_tiles_ptr, nullptr, nullptr,
                         num_seq, hidden_size, intermediate_size, num_topk, num_expert, rank_ep,
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
               num_seq, hidden_size, num_topk, false, stream);

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
  // If A groups are non-empty, sum(ceil(c_i / kMinTileM)) is at most
  // A + floor((total_routes - A) / kMinTileM). The expression increases with
  // A, whose upper bound is min(total_routes, local_experts).
  const int max_nonempty_groups = std::min(total_num_seq, num_expert_local);
  const int max_tile_m = max_nonempty_groups +
                         (total_num_seq - max_nonempty_groups) / kMinTileM;
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

  fuse_moe_cp_async::fuse_moe_cp_async(
      y_ptr, x_ptr, gate_up_output_ptr, gate_up_weight_ptr, gate_up_scale_ptr,
      act_and_mul_scale_ptr, down_input_ptr, down_output_ptr, down_weight_ptr, down_scale_ptr,
      topk_ids_ptr, topk_scale_ptr, topk_pos_ptr, row_indices_ptr, seqlens_ptr, cu_seqlens_ptr,
      tiles_ptr, cu_tiles_ptr, gateup_task_map_ptr, down_task_map_ptr, gateup_task_map_len,
      down_task_map_len, shared_output_ptr, num_seq, hidden_size, intermediate_size, num_topk,
      num_expert_total, num_expert_local, static_cast<int>(rank_ep), use_bf16_mul, stream);

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
  TORCH_CHECK(down_weight.device().is_cuda(), "down_weight tensor must be cuda");
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

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
    TORCH_CHECK(y.size(0) == num_seq && y.size(1) == hidden_size,
                "output shape must be [num_tokens, hidden_size]");
    TORCH_CHECK(y.dtype() == torch::kBFloat16, "output dtype must be bfloat16");
    TORCH_CHECK(y.device().is_cuda(), "output must be cuda tensor");
    TORCH_CHECK(y.is_contiguous(), "output must be contiguous");
  } else {
    y = torch::empty({num_seq, hidden_size}, options.dtype(torch::kBFloat16));
  }
  void *y_ptr = y.mutable_data_ptr();

  TORCH_CHECK(intermediate_size % 2 == 0,
              "gate_up_weight output dimension must be even");
  const int actual_intermediate_size = intermediate_size / 2;
  TORCH_CHECK(down_weight.size(1) == hidden_size &&
                  down_weight.size(2) == actual_intermediate_size,
              "down_weight shape must be [num_expert, hidden_size, intermediate_size]");
  const bool use_small_batch_route_mma =
      num_seq > 0 && num_seq <= 4 && num_topk == 8 &&
      actual_intermediate_size > 0 && actual_intermediate_size <= 512 &&
      actual_intermediate_size % 64 == 0 && hidden_size % 64 == 0;
  if (use_small_batch_route_mma) {
    const int64_t num_routes = static_cast<int64_t>(num_seq) * num_topk;
    int num_splits = 1;
    if (hidden_size > 2048) {
      const int gate_tile_k = hidden_size % 128 == 0 ? 128 : 64;
      const int num_k_tiles = hidden_size / gate_tile_k;
      const int gate_tasks_without_split =
          static_cast<int>(num_routes) * (intermediate_size / 64);
      const int target_gate_tasks = get_sm_count() * 4;
      const int desired_splits =
          (target_gate_tasks + gate_tasks_without_split - 1) /
          gate_tasks_without_split;
      for (int candidate : {1, 2, 4, 8}) {
        if (candidate >= desired_splits && num_k_tiles % candidate == 0) {
          num_splits = candidate;
          break;
        }
        if (num_k_tiles % candidate == 0) {
          num_splits = candidate;
        }
      }
    }
    const int64_t workspace_bytes =
        num_routes * num_splits * intermediate_size * sizeof(at::BFloat16) +
        num_routes * actual_intermediate_size +
        num_routes * hidden_size * sizeof(at::BFloat16);
    torch::Tensor workspace =
        torch::empty({workspace_bytes}, options.dtype(torch::kUInt8));
    fuse_moe_small_batch_route_mma_async(
        y_ptr, x.const_data_ptr(), workspace.mutable_data_ptr(),
        gate_up_weight.const_data_ptr(), gate_up_scale.const_data_ptr(),
        act_and_mul_scale.const_data_ptr(), down_weight.const_data_ptr(),
        down_scale.const_data_ptr(), topk_ids.const_data_ptr(), topk_scale.const_data_ptr(),
        shared_output_ptr, num_seq, hidden_size, actual_intermediate_size, num_topk,
        num_splits, num_expert, rank_ep, use_bf16_mul, stream);
    return y;
  }

  if (actual_intermediate_size <= 512 && actual_intermediate_size % 64 == 0 &&
      hidden_size % 64 == 0) {
    return fuse_moe_cp_async_entry(x, gate_up_weight, down_weight, gate_up_scale, down_scale,
                                   act_and_mul_scale, topk_ids, topk_scale, shared_output, rank_ep,
                                   num_expert_total, use_bf16_mul, y);
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

  fuse_moe_async(y_ptr, x_ptr, gate_up_input_ptr, gate_up_output_ptr, gate_up_weight_ptr,
                 gate_up_scale_ptr, gate_up_tmas_ptr, act_and_mul_scale_ptr, down_input_ptr,
                 down_output_ptr, down_weight_ptr, down_scale_ptr, down_tmas_ptr, topk_ids_ptr,
                 topk_scale_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr,
                 shared_output_ptr, gateup_task_map_ptr, down_task_map_ptr, num_gateup_waves,
                 num_down_waves, num_seq, hidden_size, intermediate_size, num_topk,
                 num_expert_total, num_expert, rank_ep, use_bf16_mul, stream);
  return y;
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
  TORCH_CHECK(x_scale.dtype() == torch::kFloat32 &&
                  gate_up_weight_scale.dtype() == torch::kFloat32 &&
                  down_weight_scale.dtype() == torch::kFloat32 &&
                  topk_scale.dtype() == torch::kFloat32,
              "x_scale, gate_up_scale, down_scale and topk_scale dtype must be float32");
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

  const int actual_intermediate_size = intermediate_size / 2;
  const bool use_small_batch_route_mma =
      num_tokens > 0 && num_tokens <= 4 && num_topk == 8 && hidden_size <= 4096 &&
      actual_intermediate_size >= 128 && actual_intermediate_size <= 768 &&
      hidden_size % 128 == 0 &&
      actual_intermediate_size % 64 == 0;
  const int down_weight_scale_num_blocks =
      use_small_batch_route_mma ? (actual_intermediate_size + 127) / 128
                                : actual_intermediate_size / 128;
  TORCH_CHECK(down_weight_scale.size(2) ==
                  (down_weight_scale_num_blocks + 3) / 4 * 4,
              "down_weight must be per 128 blockwise quant and must be aligned to 4");

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
    TORCH_CHECK(y.size(0) == num_tokens && y.size(1) == hidden_size,
                "output shape must be [num_tokens, hidden_size]");
    TORCH_CHECK(y.dtype() == torch::kBFloat16, "output dtype must be bfloat16");
    TORCH_CHECK(y.device().is_cuda(), "output must be cuda tensor");
    TORCH_CHECK(y.is_contiguous(), "output must be contiguous");
  } else {
    y = torch::empty({num_tokens, hidden_size}, options.dtype(torch::kBFloat16));
  }
  void *y_ptr = y.mutable_data_ptr();

  if (use_small_batch_route_mma) {
    const int64_t num_routes = static_cast<int64_t>(num_tokens) * num_topk;
    int num_splits = 1;
    if (hidden_size > 2048) {
      const int num_k_tiles = hidden_size / 128;
      const int gate_tasks_without_split =
          static_cast<int>(num_routes) * (intermediate_size / 64);
      const int target_gate_tasks = get_sm_count() * 4;
      const int desired_splits =
          (target_gate_tasks + gate_tasks_without_split - 1) /
          gate_tasks_without_split;
      for (int candidate : {1, 2, 4, 8}) {
        if (candidate >= desired_splits && num_k_tiles % candidate == 0) {
          num_splits = candidate;
          break;
        }
        if (num_k_tiles % candidate == 0) {
          num_splits = candidate;
        }
      }
    }
    const int64_t num_intermediate_blocks =
        (actual_intermediate_size + 127) / 128;
    const int64_t workspace_bytes =
        num_routes * num_splits * 2 * actual_intermediate_size *
            sizeof(at::BFloat16) +
        num_routes * actual_intermediate_size +
        num_routes * num_intermediate_blocks * sizeof(float) +
        num_routes * hidden_size * sizeof(at::BFloat16);
    torch::Tensor workspace =
        torch::empty({workspace_bytes}, options.dtype(torch::kUInt8));
    fuse_moe_blockwise_small_batch_route_mma_async(
        y_ptr, x.const_data_ptr(), x_scale.const_data_ptr(),
        workspace.mutable_data_ptr(), gate_up_weight.const_data_ptr(),
        gate_up_weight_scale.const_data_ptr(), down_weight.const_data_ptr(),
        down_weight_scale.const_data_ptr(), topk_ids.const_data_ptr(),
        topk_scale.const_data_ptr(), shared_output_ptr, num_tokens, hidden_size,
        actual_intermediate_size, num_topk, num_splits, num_experts,
        gate_up_weight_scale_lastdim_pad4, down_weight_scale_lastdim_pad4, rank_ep,
        stream);
    return y;
  }

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

  fuse_moe_blockwise_async(
      y_ptr, x_ptr, x_scale_ptr, gate_up_input_ptr, gate_up_input_scale_ptr, gate_up_output_ptr,
      gate_up_weight_ptr, gate_up_weight_scale_ptr, gate_up_tmas_ptr, down_input_ptr,
      down_input_scale_ptr, down_output_ptr, down_weight_ptr, down_weight_scale_ptr, down_tmas_ptr,
      topk_ids_ptr, topk_scale_ptr, topk_pos_ptr, num_tokens_per_group_ptr,
      cu_num_tokens_per_group_ptr, tiles_ptr, cu_tiles_ptr, shared_output_ptr, gateup_task_map_ptr,
      down_task_map_ptr, num_gateup_waves, num_down_waves, num_tokens, num_padded_tokens,
      hidden_size, intermediate_size, num_topk, num_expert_total, num_experts,
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
}
