// Copyright 2026 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

#include "src/fuse_moe/sm100/mxfp8/fuse_moe_mxfp8.h"
#include "src/group_gemm/sm100/group_gemm.h"

namespace hpc {
namespace fuse_moe {

torch::Tensor fuse_moe_mxfp8_entry(const torch::Tensor &x, const torch::Tensor &x_scale,
                                   const torch::Tensor &gate_up_weight,
                                   const torch::Tensor &gate_up_weight_scale_packed,
                                   const torch::Tensor &down_weight,
                                   const torch::Tensor &down_weight_scale_packed,
                                   const torch::Tensor &topk_ids, const torch::Tensor &topk_scale,
                                   std::optional<torch::Tensor> shared_output, int64_t rank_ep,
                                   int64_t num_expert_total, std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  TORCH_CHECK(x.dtype() == torch::kFloat8_e4m3fn, "x must be fp8_e4m3");
  TORCH_CHECK((gate_up_weight.dtype() == torch::kFloat8_e4m3fn &&
               down_weight.dtype() == torch::kFloat8_e4m3fn) ||
                  (gate_up_weight.dtype() == torch::kUInt8 && down_weight.dtype() == torch::kUInt8),
              "gate_up_weight and down_weight must both be fp8_e4m3 (mxfp8) "
              "or both be uint8 (mxfp4 packed)");
  bool is_fp4 = (gate_up_weight.dtype() == torch::kUInt8);
  TORCH_CHECK(x_scale.dtype() == torch::kUInt8 &&
                  gate_up_weight_scale_packed.dtype() == torch::kUInt8 &&
                  down_weight_scale_packed.dtype() == torch::kUInt8,
              "x_scale and weight_scale_packed must be uint8 (UE8M0 raw bits)");
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32, "topk_ids dtype must be int32");
  TORCH_CHECK(topk_scale.dtype() == torch::kFloat32, "topk_scale dtype must be float32");
  TORCH_CHECK(x.is_cuda() && gate_up_weight.is_cuda() && down_weight.is_cuda(),
              "x / weights must be cuda");
  TORCH_CHECK(x.is_contiguous() && gate_up_weight.is_contiguous() && down_weight.is_contiguous(),
              "x / weights must be contiguous");
  TORCH_CHECK(topk_ids.size(0) == x.size(0), "topk_ids and x must share num_seq");
  TORCH_CHECK(topk_ids.size(0) == topk_scale.size(0) && topk_ids.size(1) == topk_scale.size(1),
              "topk_ids and topk_scale must share shape");

  int num_seq = static_cast<int>(x.size(0));
  int hidden = static_cast<int>(x.size(1));
  int num_topk = static_cast<int>(topk_ids.size(1));
  int num_expert_local = static_cast<int>(gate_up_weight.size(0));
  int intermediate = static_cast<int>(gate_up_weight.size(1)) / 2;

  // Expected weight K (size(2)): fp8 → full K, fp4 → K/2 (2 x e2m1 per byte).
  int gate_up_k_expected = is_fp4 ? hidden / 2 : hidden;
  int down_k_expected = is_fp4 ? intermediate / 2 : intermediate;
  TORCH_CHECK(gate_up_weight.size(2) == gate_up_k_expected,
              "gate_up_weight last dim must equal hidden (mxfp8) or hidden/2 (mxfp4)");
  TORCH_CHECK(down_weight.size(0) == num_expert_local, "down_weight num_expert_local mismatch");
  TORCH_CHECK(down_weight.size(1) == hidden, "down_weight first dim must equal hidden");
  TORCH_CHECK(down_weight.size(2) == down_k_expected,
              "down_weight second dim must equal intermediate (mxfp8) or intermediate/2 (mxfp4)");
  TORCH_CHECK(gate_up_weight.size(1) % 2 == 0, "gate_up_weight dim1 must be even");
  // mxfp4 weight (sub-byte e2m1) needs K%128==0 for the TMA descriptor byte alignment.
  if (is_fp4) {
    TORCH_CHECK(hidden % 128 == 0, "mxfp4 requires hidden to be a multiple of 128");
    TORCH_CHECK(intermediate % 128 == 0, "mxfp4 requires intermediate to be a multiple of 128");
  } else {
    TORCH_CHECK(hidden % 32 == 0, "hidden must be multiple of 32");
    TORCH_CHECK(intermediate % 32 == 0, "intermediate must be multiple of 32");
  }
  TORCH_CHECK(x_scale.size(0) == num_seq && x_scale.size(1) == hidden / 32,
              "x_scale shape must be (num_seq, hidden/32)");
  TORCH_CHECK(num_topk <= 128, "num_topk must be <= 128");

  int total_num_seq = num_seq * num_topk;

  // shared_output validation
  const void *shared_output_ptr = nullptr;
  if (shared_output.has_value()) {
    auto t = shared_output.value();
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "shared_output must be cuda + contiguous");
    TORCH_CHECK(t.dtype() == torch::kBFloat16, "shared_output must be bf16");
    TORCH_CHECK(t.size(0) == num_seq && t.size(1) == hidden,
                "shared_output shape must equal (num_seq, hidden)");
    shared_output_ptr = t.const_data_ptr();
  }

  // output: when shared_output is provided, accumulate directly into it (same as fp8 path).
  auto opt = x.options();
  torch::Tensor y;
  if (shared_output.has_value()) {
    y = shared_output.value();
  } else if (output.has_value()) {
    y = output.value();
    TORCH_CHECK(y.size(0) == num_seq && y.size(1) == hidden,
                "output shape must equal (num_seq, hidden)");
    TORCH_CHECK(y.dtype() == torch::kBFloat16, "output dtype must be bf16");
  } else {
    y = torch::zeros({num_seq, hidden}, opt.dtype(torch::kBFloat16));
  }

  // workspace buffers
  auto u8 = opt.dtype(torch::kUInt8);
  auto i32 = opt.dtype(torch::kInt32);
  auto bf16 = opt.dtype(torch::kBFloat16);

  // gate_up input/output
  torch::Tensor gate_up_input = torch::empty({total_num_seq, hidden}, opt);
  torch::Tensor gate_up_output = torch::empty({total_num_seq, intermediate * 2}, bf16);
  torch::Tensor gate_up_tmas = torch::empty({num_expert_local * 2, 128}, opt.dtype(torch::kInt8));

  // down input/output
  torch::Tensor down_input = torch::zeros({total_num_seq, intermediate}, opt);
  // down_input_scale: row-major (total_num_seq, K_sf), inline prepacked by down GEMM kernel.
  int inter_k_sf = intermediate / 32;
  torch::Tensor down_input_scale = torch::zeros({total_num_seq, inter_k_sf}, u8);
  torch::Tensor down_output = torch::empty({total_num_seq, hidden}, bf16);
  torch::Tensor down_tmas = torch::empty({num_expert_local * 2, 128}, opt.dtype(torch::kInt8));

  // routing scratch
  torch::Tensor topk_pos = torch::empty({num_seq, num_topk}, i32);
  // gateup cp_async row indirection: post-permutation row → source token row.
  torch::Tensor gateup_x_row_map = torch::empty({2 * total_num_seq + num_expert_local}, i32);
  torch::Tensor seqlens = torch::empty({num_expert_local}, i32);
  torch::Tensor cu_seqlens = torch::empty({num_expert_local + 1}, i32);
  torch::Tensor tiles = torch::empty({2 * num_expert_local}, i32);
  torch::Tensor cu_tiles = torch::empty({2 * (num_expert_local + 1)}, i32);

  fuse_moe_mxfp8_async(
      y.mutable_data_ptr(), x.const_data_ptr(), x_scale.const_data_ptr(),
      gate_up_input.mutable_data_ptr(), gate_up_output.mutable_data_ptr(),
      gate_up_weight.const_data_ptr(), gate_up_weight_scale_packed.const_data_ptr(),
      gate_up_tmas.mutable_data_ptr(), down_input.mutable_data_ptr(),
      down_input_scale.mutable_data_ptr(), down_output.mutable_data_ptr(),
      down_weight.const_data_ptr(), down_weight_scale_packed.const_data_ptr(),
      down_tmas.mutable_data_ptr(), topk_ids.const_data_ptr(), topk_scale.const_data_ptr(),
      topk_pos.mutable_data_ptr(), gateup_x_row_map.mutable_data_ptr(), seqlens.mutable_data_ptr(),
      cu_seqlens.mutable_data_ptr(), tiles.mutable_data_ptr(), cu_tiles.mutable_data_ptr(),
      shared_output_ptr, num_seq, hidden, intermediate, num_topk,
      static_cast<int>(num_expert_total), num_expert_local, static_cast<int>(rank_ep), is_fp4,
      stream);

  return y;
}

std::tuple<torch::Tensor, torch::Tensor> act_mul_and_mxfp8_quant_entry(const torch::Tensor &gate_up,
                                                                       int64_t num_valid_rows) {
  auto stream = at::cuda::getCurrentCUDAStream(gate_up.get_device());
  TORCH_CHECK(gate_up.is_cuda() && gate_up.is_contiguous(), "gate_up must be cuda+contiguous");
  TORCH_CHECK(gate_up.dtype() == torch::kBFloat16, "gate_up must be bf16");
  TORCH_CHECK(gate_up.dim() == 2, "gate_up must be 2D (total_rows, 2*intermediate)");

  int total_rows = static_cast<int>(gate_up.size(0));
  int intermediate_size = static_cast<int>(gate_up.size(1)) / 2;

  auto options = gate_up.options();
  torch::Tensor out_fp8 =
      torch::empty({total_rows, intermediate_size}, options.dtype(torch::kFloat8_e4m3fn));
  torch::Tensor out_scale =
      torch::empty({total_rows, intermediate_size / 32}, options.dtype(torch::kUInt8));

  // Use cu_seqlens last element as valid_row_range (single int on device)
  torch::Tensor valid_row_range =
      torch::tensor({static_cast<int>(num_valid_rows)}, options.dtype(torch::kInt32));

  act_mul_and_mxfp8_quant_async(out_fp8.mutable_data_ptr(), out_scale.mutable_data_ptr(),
                                gate_up.const_data_ptr(), valid_row_range.const_data_ptr(),
                                total_rows, intermediate_size, stream, false);

  return std::make_tuple(out_fp8, out_scale);
}

}  // namespace fuse_moe
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "fuse_moe_mxfp8(Tensor x, Tensor x_scale, Tensor gate_up_weight, "
      "Tensor gate_up_weight_scale_packed, Tensor down_weight, "
      "Tensor down_weight_scale_packed, Tensor topk_ids, Tensor topk_scale, "
      "Tensor? shared_output, int rank_ep, int num_expert_total, "
      "Tensor? output) -> (Tensor)");
  m.impl("fuse_moe_mxfp8", torch::kCUDA, &hpc::fuse_moe::fuse_moe_mxfp8_entry);

  m.def("act_mul_and_mxfp8_quant(Tensor gate_up, int num_valid_rows) -> (Tensor, Tensor)");
  m.impl("act_mul_and_mxfp8_quant", torch::kCUDA, &hpc::fuse_moe::act_mul_and_mxfp8_quant_entry);
}
