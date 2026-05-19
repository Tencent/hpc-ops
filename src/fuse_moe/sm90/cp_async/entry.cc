// Copyright 2026 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/fuse_moe/sm90/cp_async/fuse_moe_cp_async.h"
#include "src/utils/utils.h"

namespace hpc {
namespace fuse_moe_cp_async {

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

  // Output tensor
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

  // Intermediate buffers
  // gate_up_output: gate+up projection output [total_num_seq, intermediate_size] bf16
  torch::Tensor gate_up_output =
      torch::empty({total_num_seq, intermediate_size}, options.dtype(torch::kBFloat16));

  // down_input: activation+quant output [total_num_seq, intermediate_size/2] fp8
  torch::Tensor down_input =
      torch::empty({total_num_seq, intermediate_size / 2}, options.dtype(torch::kFloat8_e4m3fn));

  // down_output: down projection output [total_num_seq, hidden_size] bf16
  torch::Tensor down_output =
      torch::empty({total_num_seq, hidden_size}, options.dtype(torch::kBFloat16));

  // topk_pos: sorted position for each (token, topk_j) pair [num_seq, num_topk] int32
  torch::Tensor topk_pos = torch::empty({num_seq, num_topk}, options.dtype(torch::kInt32));

  // row_indices: original token row for each sorted slot [total_num_seq] int32
  torch::Tensor row_indices = torch::empty({total_num_seq}, options.dtype(torch::kInt32));

  // seqlens / cu_seqlens / tiles / cu_tiles
  torch::Tensor seqlens = torch::zeros({num_expert_local}, options.dtype(torch::kInt32));
  torch::Tensor cu_seqlens = torch::empty({num_expert_local + 1}, options.dtype(torch::kInt32));
  torch::Tensor tiles = torch::empty({num_expert_local}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_expert_local + 1}, options.dtype(torch::kInt32));

  // task_maps for gate_up and down gemm (int4 per iblock).  Upper bound the
  // number of iblocks using the smallest possible kTileM (8) so the buffer
  // is valid across the runtime dispatch.
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
  // Task-map tail entries are initialized by the routing/build kernels, so
  // no separate cudaMemsetAsync is needed before the fused MoE pipeline.

  // Pointers
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

  fuse_moe_cp_async(y_ptr, x_ptr, gate_up_output_ptr, gate_up_weight_ptr, gate_up_scale_ptr,
                    act_and_mul_scale_ptr, down_input_ptr, down_output_ptr, down_weight_ptr,
                    down_scale_ptr, topk_ids_ptr, topk_scale_ptr, topk_pos_ptr, row_indices_ptr,
                    seqlens_ptr, cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr, gateup_task_map_ptr,
                    down_task_map_ptr, gateup_task_map_len, down_task_map_len, shared_output_ptr,
                    num_seq, hidden_size, intermediate_size, num_topk, num_expert_total,
                    num_expert_local, static_cast<int>(rank_ep), use_bf16_mul, stream);

  if (output.has_value()) {
    return output.value();
  }
  return y;
}

}  // namespace fuse_moe_cp_async
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "fuse_moe_cp_async(Tensor x, Tensor gate_up_weight, Tensor down_weight, "
      "Tensor gate_up_scale, Tensor down_scale, Tensor act_and_mul_scale, "
      "Tensor topk_ids, Tensor topk_scale, Tensor ? shared_output, "
      "int rank_ep, int num_expert_total, bool use_bf16_mul, Tensor ? output) -> (Tensor)");
  m.impl("fuse_moe_cp_async", torch::kCUDA, &hpc::fuse_moe_cp_async::fuse_moe_cp_async_entry);
}
