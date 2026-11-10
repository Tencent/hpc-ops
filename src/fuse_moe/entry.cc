// Copyright (C) 2026 Tencent.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

#include "src/fuse_moe/fuse_moe.h"
#include "src/fuse_moe/fuse_moe_cp_async.h"
#include "src/fuse_moe/fuse_moe_warp_decode.h"
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

namespace {

constexpr int kWarpDecodeScaleGroup = 128;

bool is_warp_decode_shape(int64_t hidden, int64_t inter, int64_t experts, int64_t topk) {
  if (experts == 256 && topk == 8) {
    return (hidden == 2048 || hidden == 3072) && (inter == 256 || inter == 512);
  }
  if (experts == 512 && topk == 10) {
    return (hidden == 2048 && (inter == 256 || inter == 512)) ||
           (hidden == 4096 && (inter == 128 || inter == 256));
  }
  return false;
}

// Common tensor checks run in the public entries before decode is selected.
torch::Tensor fuse_moe_pertensor_warp_decode_entry(
    const torch::Tensor &x, const torch::Tensor &gate_up_weight, const torch::Tensor &down_weight,
    const torch::Tensor &gate_up_scale, const torch::Tensor &down_scale,
    const torch::Tensor &act_and_mul_scale, const torch::Tensor &topk_ids,
    const torch::Tensor &topk_scale, std::optional<torch::Tensor> shared_output, int64_t rank_ep,
    int64_t num_expert_total, bool use_bf16_mul, std::optional<torch::Tensor> output) {
  const auto device = x.device();
  constexpr int64_t hidden = 4096;
  constexpr int64_t inter = 192;
  constexpr int64_t topk = 8;
  const int64_t batch_size = x.size(0);
  const int64_t local_experts = gate_up_weight.size(0);

  TORCH_CHECK(down_weight.dim() == 3 && topk_scale.dim() == 2,
              "down_weight must be 3D and topk_scale must be 2D");
  TORCH_CHECK(gate_up_weight.device() == device && down_weight.device() == device &&
                  gate_up_scale.device() == device && down_scale.device() == device &&
                  act_and_mul_scale.device() == device && topk_ids.device() == device &&
                  topk_scale.device() == device,
              "all inputs must be on the same CUDA device as x");
  TORCH_CHECK(act_and_mul_scale.is_contiguous(), "act_and_mul_scale must be contiguous");
  TORCH_CHECK(local_experts > 0 && num_expert_total % local_experts == 0 && rank_ep >= 0 &&
                  rank_ep < num_expert_total / local_experts,
              "rank_ep and local experts must form a valid partition of num_expert_total");
  TORCH_CHECK(down_weight.size(1) == hidden && down_weight.size(2) == inter,
              "down_weight shape must be [local_experts, hidden, intermediate]");
  TORCH_CHECK(gate_up_scale.numel() >= local_experts && down_scale.numel() >= local_experts &&
                  act_and_mul_scale.numel() >= 1,
              "projection scales must contain one value per local expert and the activation "
              "scale must be nonempty");
  TORCH_CHECK(reinterpret_cast<size_t>(x.const_data_ptr()) % 16 == 0 &&
                  reinterpret_cast<size_t>(gate_up_weight.const_data_ptr()) % 16 == 0 &&
                  reinterpret_cast<size_t>(down_weight.const_data_ptr()) % 16 == 0,
              "x, gate_up_weight and down_weight must be 16-byte aligned");

  const void *shared_ptr = nullptr;
  if (shared_output.has_value()) {
    const auto &so = shared_output.value();
    TORCH_CHECK(so.device() == device, "shared_output must be on the same CUDA device as x");
    TORCH_CHECK(so.sizes() == x.sizes(), "shared_output shape must match x");
    shared_ptr = so.const_data_ptr();
  }
  if (output.has_value()) {
    const auto &out = output.value();
    TORCH_CHECK(out.device() == device, "output must be on the same CUDA device as x");
    TORCH_CHECK(out.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(out.dtype() == torch::kBFloat16, "output dtype must be bfloat16");
    TORCH_CHECK(out.sizes() == x.sizes(), "output shape must match x");
    for (const auto &input : {x, gate_up_weight, down_weight, gate_up_scale, down_scale,
                              act_and_mul_scale, topk_ids, topk_scale}) {
      at::assert_no_overlap(out, input);
    }
    if (shared_output.has_value()) {
      at::assert_no_partial_overlap(out, *shared_output);
    }
  }

  const c10::DeviceGuard device_guard(device);
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  auto options = x.options();
  auto y = output.has_value() ? *output
                              : torch::empty({batch_size, hidden}, options.dtype(at::kBFloat16));
  if (batch_size == 0) return y;

  const int expert_start = rank_ep * local_experts;
  auto buffer = torch::empty({batch_size * topk, inter}, options.dtype(at::kFloat8_e4m3fn));
  // GEMM scales are per-expert multipliers. The activation scale is instead
  // multiplied before FP8 conversion; no blockwise amax quantization is used.
  HPC_ARCH_DISPATCH(
      "fuse_moe.warp_decode", 90,
      warp_decode::gate_up_pertensor_async(
          x.const_data_ptr(), gate_up_weight.const_data_ptr(),
          gate_up_scale.const_data_ptr<float>(), act_and_mul_scale.const_data_ptr<float>(),
          topk_ids.const_data_ptr<int>(), buffer.mutable_data_ptr(), batch_size, local_experts,
          expert_start, use_bf16_mul, stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  HPC_ARCH_DISPATCH(
      "fuse_moe.warp_decode", 90,
      warp_decode::down_pertensor_async(
          buffer.const_data_ptr(), down_weight.const_data_ptr(), down_scale.const_data_ptr<float>(),
          topk_ids.const_data_ptr<int>(), topk_scale.const_data_ptr<float>(), shared_ptr,
          y.mutable_data_ptr(), batch_size, local_experts, expert_start, stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor fuse_moe_warp_decode_entry(
    const torch::Tensor &x, const torch::Tensor &x_scale, const torch::Tensor &gate_up_weight,
    const torch::Tensor &gate_up_weight_scale, const torch::Tensor &down_weight,
    const torch::Tensor &down_weight_scale, const torch::Tensor &topk_ids,
    const torch::Tensor &topk_scale, std::optional<torch::Tensor> shared_output, int64_t rank_ep,
    int64_t num_expert_total, std::optional<torch::Tensor> output) {
  const auto device = x.device();
  const int64_t batch_size = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t inter = gate_up_weight.size(1) / 2;
  const int64_t local_experts = gate_up_weight.size(0);
  const int64_t topk = topk_ids.size(1);
  constexpr int q = kWarpDecodeScaleGroup;

  TORCH_CHECK(down_weight.dim() == 3 && topk_scale.dim() == 2,
              "down_weight must be 3D and topk_scale must be 2D");
  TORCH_CHECK(x_scale.dtype() == torch::kFloat32, "x_scale dtype must be float32");
  TORCH_CHECK(x_scale.device() == device && gate_up_weight.device() == device &&
                  gate_up_weight_scale.device() == device && down_weight.device() == device &&
                  down_weight_scale.device() == device && topk_ids.device() == device &&
                  topk_scale.device() == device,
              "all inputs must be on the same CUDA device as x");
  TORCH_CHECK(local_experts > 0 && num_expert_total % local_experts == 0 && rank_ep >= 0 &&
                  rank_ep < num_expert_total / local_experts,
              "rank_ep and local experts must form a valid partition of num_expert_total");
  TORCH_CHECK(gate_up_weight.size(1) == 2 * inter && down_weight.size(1) == hidden &&
                  down_weight.size(2) == inter,
              "gate_up_weight and down_weight must have shapes [E,2*I,H] and [E,H,I]");
  TORCH_CHECK(
      gate_up_weight_scale.size(0) >= local_experts && down_weight_scale.size(0) >= local_experts,
      "weight scales must contain one block-scale matrix per local expert");
  TORCH_CHECK(reinterpret_cast<size_t>(x.const_data_ptr()) % 16 == 0 &&
                  reinterpret_cast<size_t>(gate_up_weight.const_data_ptr()) % 16 == 0 &&
                  reinterpret_cast<size_t>(down_weight.const_data_ptr()) % 16 == 0,
              "x, gate_up_weight and down_weight must be 16-byte aligned");

  const void *shared_ptr = nullptr;
  if (shared_output.has_value()) {
    const auto &so = shared_output.value();
    TORCH_CHECK(so.device() == device, "shared_output must be on the same CUDA device as x");
    TORCH_CHECK(so.sizes() == x.sizes(), "shared_output shape must match x");
    shared_ptr = so.const_data_ptr();
  }
  if (output.has_value()) {
    const auto &out = output.value();
    TORCH_CHECK(out.device() == device, "output must be on the same CUDA device as x");
    TORCH_CHECK(out.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(out.dtype() == torch::kBFloat16, "output dtype must be bfloat16");
    TORCH_CHECK(out.sizes() == x.sizes(), "output shape must match x");
    for (const auto &input : {x, x_scale, gate_up_weight, gate_up_weight_scale, down_weight,
                              down_weight_scale, topk_ids, topk_scale}) {
      at::assert_no_overlap(out, input);
    }
    if (shared_output.has_value()) {
      at::assert_no_partial_overlap(out, *shared_output);
    }
  }

  const c10::DeviceGuard device_guard(device);
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  auto options = x.options();
  auto y = output.has_value() ? *output
                              : torch::empty({batch_size, hidden}, options.dtype(at::kBFloat16));
  if (batch_size == 0) return y;

  // Keep the standalone warp_decode rounding boundaries and measured policy.
  // HPC's TMA backend has different activation and per-expert BF16 boundaries.
  // The narrow 397B TP8 path also benefits from inline quantization at B=2.
  const bool fuse_quant = (batch_size == 1 && inter == 256 && (hidden == 2048 || hidden == 3072)) ||
                          (batch_size <= 2 && inter == 128 && hidden == 4096 && topk == 10);
  const int expert_start = rank_ep * local_experts;
  const int num_tokens = batch_size;
  const int hidden_size = hidden;
  const int intermediate_size = inter;
  const int num_experts = local_experts;
  const int num_topk = topk;
  auto buffer = torch::empty({batch_size * topk, inter}, options.dtype(at::kBFloat16));

  HPC_ARCH_DISPATCH(
      "fuse_moe_blockwise.warp_decode", 90,
      warp_decode::gate_up_async(
          x.const_data_ptr(), gate_up_weight.const_data_ptr(), x_scale.const_data_ptr<float>(),
          gate_up_weight_scale.const_data_ptr<float>(), topk_ids.const_data_ptr<int>(),
          buffer.mutable_data_ptr(), num_tokens, hidden_size, intermediate_size, num_experts,
          num_topk, expert_start, stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (fuse_quant) {
    HPC_ARCH_DISPATCH(
        "fuse_moe_blockwise.warp_decode", 90,
        warp_decode::down_bf16_async(
            buffer.const_data_ptr(), down_weight.const_data_ptr(),
            down_weight_scale.const_data_ptr<float>(), topk_ids.const_data_ptr<int>(),
            topk_scale.const_data_ptr<float>(), shared_ptr, y.mutable_data_ptr(), num_tokens,
            hidden_size, intermediate_size, num_experts, num_topk, expert_start, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto quantized = torch::empty({batch_size * topk, inter}, options.dtype(at::kFloat8_e4m3fn));
    auto buffer_scale = torch::empty({batch_size * topk, inter / q}, options.dtype(at::kFloat));
    HPC_ARCH_DISPATCH(
        "fuse_moe_blockwise.warp_decode", 90,
        warp_decode::quantize_async(buffer.const_data_ptr(), quantized.mutable_data_ptr(),
                                    buffer_scale.mutable_data_ptr<float>(), num_tokens * num_topk,
                                    intermediate_size, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    HPC_ARCH_DISPATCH(
        "fuse_moe_blockwise.warp_decode", 90,
        warp_decode::down_async(quantized.const_data_ptr(), down_weight.const_data_ptr(),
                                buffer_scale.const_data_ptr<float>(),
                                down_weight_scale.const_data_ptr<float>(),
                                topk_ids.const_data_ptr<int>(), topk_scale.const_data_ptr<float>(),
                                shared_ptr, y.mutable_data_ptr(), num_tokens, hidden_size,
                                intermediate_size, num_experts, num_topk, expert_start, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return y;
}

}  // namespace

torch::Tensor fuse_moe_entry(const torch::Tensor &x, const torch::Tensor &gate_up_weight,
                             const torch::Tensor &down_weight, const torch::Tensor &gate_up_scale,
                             const torch::Tensor &down_scale,
                             const torch::Tensor &act_and_mul_scale, const torch::Tensor &topk_ids,
                             const torch::Tensor &topk_scale,
                             std::optional<torch::Tensor> shared_output, int64_t rank_ep,
                             int64_t num_expert_total, bool use_bf16_mul,
                             std::optional<torch::Tensor> output) {
  TORCH_CHECK(num_expert_total > 0, "num_expert_total must be positive");

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

  // SM90 per-tensor FP8 decode: Hunyuan-v3 TP8, batch 0-8.
  // H=4096, intermediate/rank=192, E=192, top-k=8.
  if (x.is_cuda() && x.dim() == 2 && gate_up_weight.dim() == 3 && topk_ids.dim() == 2 &&
      get_sm_arch() == 90 && x.size(0) <= 8 && x.size(1) == 4096 && gate_up_weight.size(1) == 384 &&
      num_expert_total == 192 && topk_ids.size(1) == 8 && gate_up_scale.dim() == 1 &&
      down_scale.dim() == 1 && act_and_mul_scale.dim() == 1) {
    return fuse_moe_pertensor_warp_decode_entry(
        x, gate_up_weight, down_weight, gate_up_scale, down_scale, act_and_mul_scale, topk_ids,
        topk_scale, shared_output, rank_ep, num_expert_total, use_bf16_mul, output);
  }

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
  TORCH_CHECK(num_expert_total > 0, "num_expert_total must be positive");

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

  // SM90 blockwise FP8 decode, batch 0-8:
  // Qwen3.5/3.6-35B-A3B TP1/TP2; Qwen3.5-122B-A10B TP2/TP4;
  // Qwen3.5-397B-A17B TP4/TP8; Qwen3-Next-80B-A3B TP1/TP2.
  if (x.is_cuda() && x.dim() == 2 && gate_up_weight.dim() == 3 && topk_ids.dim() == 2 &&
      get_sm_arch() == 90 && x.size(0) <= 8 &&
      is_warp_decode_shape(x.size(1), gate_up_weight.size(1) / 2, num_expert_total,
                           topk_ids.size(1)) &&
      x_scale.dim() == 2 && gate_up_weight_scale.dim() == 3 && down_weight_scale.dim() == 3) {
    return fuse_moe_warp_decode_entry(x, x_scale, gate_up_weight, gate_up_weight_scale, down_weight,
                                      down_weight_scale, topk_ids, topk_scale, shared_output,
                                      rank_ep, num_expert_total, output);
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
