// Copyright (C) 2026 Tencent.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>
#include <iostream>
#include "src/group_gemm/group_gemm.h"
#include "src/utils/utils.h"

namespace hpc {
namespace group_gemm {

torch::Tensor group_gemm_fp8_entry(const torch::Tensor &x, const torch::Tensor &weight,
                                   const torch::Tensor &seqlens, const torch::Tensor &cu_seqlens,
                                   const torch::Tensor &y_scale,
                                   const int64_t num_seq_per_group_avg,
                                   std::optional<torch::Tensor> output,
                                   std::optional<torch::Tensor> tma_desc,
                                   std::optional<torch::Tensor> task_map_workspace) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda(), "x tensor must be cuda");
  TORCH_CHECK(weight.device().is_cuda(), "weight tensor must be cuda");
  TORCH_CHECK(seqlens.device().is_cuda(), "seqlens tensor must be cuda");
  TORCH_CHECK(cu_seqlens.device().is_cuda(), "cu_seqlens tensor must be cuda");
  TORCH_CHECK(x.dtype() == torch::kFloat8_e4m3fn && weight.dtype() == torch::kFloat8_e4m3fn,
              "x and weight dtype must be fp8_e4m3");
  TORCH_CHECK(seqlens.dtype() == torch::kInt32 && cu_seqlens.dtype() == torch::kInt32,
              "seqlens and cu_seqlens dtype must be int32");
  TORCH_CHECK(y_scale.dtype() == torch::kFloat32, "y_scale dtype must be float32");
  TORCH_CHECK(x.is_contiguous(), "x tensor a must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor a must be contiguous");
  TORCH_CHECK(seqlens.size(0) == weight.size(0),
              "seqlens and weight must share the same num_group");
  TORCH_CHECK(x.size(1) == weight.size(2), "x and weight must share the same k");

  int m = x.size(0);
  int k = x.size(1);
  int n = weight.size(1);
  int num_group = seqlens.size(0);

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
  }

  torch::Tensor tmas;
  bool update_tma = true;
  if (tma_desc.has_value()) {
    tmas = tma_desc.value();
    update_tma = false;
  } else {
    tmas = torch::empty({num_group * 2, 128}, options);
  }

  int num_waves = 0;
  torch::Tensor task_map;
  void *task_map_ptr = nullptr;

  if (num_seq_per_group_avg <= 8 && update_tma && task_map_workspace.has_value()) {
    num_waves = task_map_workspace.value().size(0);
    task_map_ptr = task_map_workspace.value().mutable_data_ptr();
  }

  torch::Tensor tiles = torch::empty({num_group}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_group + 1}, options.dtype(torch::kInt32));

  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *seqlens_ptr = seqlens.const_data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr();
  const auto *yscale_ptr = y_scale.const_data_ptr();
  auto *tmas_ptr = tmas.mutable_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();

  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();

  group_gemm_fp8_async(y_ptr, x_ptr, weight_ptr, seqlens_ptr, cu_seqlens_ptr, yscale_ptr, tmas_ptr,
                       tiles_ptr, cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k,
                       num_seq_per_group_avg, update_tma, false, stream);

  return y;
}

torch::Tensor group_gemm_blockwise_fp8_entry(
    const torch::Tensor &x, const torch::Tensor &weight, const torch::Tensor &seqlens,
    const torch::Tensor &cu_seqlens, const torch::Tensor &x_scale, const torch::Tensor &w_scale,
    const int64_t num_seq_per_group_avg, std::optional<torch::Tensor> output,
    std::optional<torch::Tensor> tma_desc, std::optional<torch::Tensor> task_map_workspace) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda(), "x tensor must be cuda");
  TORCH_CHECK(weight.device().is_cuda(), "weight tensor must be cuda");
  TORCH_CHECK(seqlens.device().is_cuda(), "seqlens tensor must be cuda");
  TORCH_CHECK(cu_seqlens.device().is_cuda(), "cu_seqlens tensor must be cuda");
  TORCH_CHECK(x.is_contiguous(), "x tensor a must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor a must be contiguous");
  TORCH_CHECK(x.dtype() == torch::kFloat8_e4m3fn && weight.dtype() == torch::kFloat8_e4m3fn,
              "x and weight dtype must be fp8_e4m3");
  TORCH_CHECK(seqlens.dtype() == torch::kInt32 && cu_seqlens.dtype() == torch::kInt32,
              "seqlens and cu_seqlens dtype must be int32");
  TORCH_CHECK(x_scale.dtype() == torch::kFloat32 && w_scale.dtype() == torch::kFloat32,
              "x_scale and w_scale dtype must be float32");
  TORCH_CHECK(seqlens.size(0) == weight.size(0),
              "seqlens and weight must share the same num_group");
  TORCH_CHECK(x.size(1) == weight.size(2), "x and weight must share the same k");
  TORCH_CHECK(w_scale.size(2) % 4 == 0, "w_scale must be multiple of 4");

  int m = x.size(0);
  int k = x.size(1);
  int n = weight.size(1);
  int m_pad = x_scale.size(1);
  int num_block_k_pad4 = w_scale.size(2);
  int num_group = seqlens.size(0);

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
  }

  torch::Tensor tmas;
  bool update_tma = true;
  if (tma_desc.has_value()) {
    tmas = tma_desc.value();
    update_tma = false;
  } else {
    tmas = torch::empty({num_group * 2, 128}, options);
  }

  int num_waves = 0;
  torch::Tensor task_map;
  void *task_map_ptr = nullptr;

  if (num_seq_per_group_avg <= 8 && update_tma && task_map_workspace.has_value()) {
    num_waves = task_map_workspace.value().size(0);
    task_map_ptr = task_map_workspace.value().mutable_data_ptr();
  }

  torch::Tensor tiles = torch::empty({num_group}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_group + 1}, options.dtype(torch::kInt32));

  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *seqlens_ptr = seqlens.const_data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr();
  const auto *xscale_ptr = x_scale.const_data_ptr();
  const auto *wscale_ptr = w_scale.const_data_ptr();
  auto *tmas_ptr = tmas.mutable_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();

  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();

  group_gemm_blockwise_fp8_async(y_ptr, x_ptr, weight_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr,
                                 wscale_ptr, tmas_ptr, tiles_ptr, cu_tiles_ptr, task_map_ptr,
                                 num_waves, num_group, m, n, k, m_pad, num_block_k_pad4,
                                 num_seq_per_group_avg, update_tma, false, stream);

  return y;
}

torch::Tensor group_gemm_bf16_entry(const torch::Tensor &x, const torch::Tensor &weight,
                                             const torch::Tensor &seqlens,
                                             const torch::Tensor &cu_seqlens,
                                             const int64_t num_seq_per_group_avg,
                                             std::optional<torch::Tensor> output,
                                             std::optional<torch::Tensor> tma_desc) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda(), "x tensor must be cuda");
  TORCH_CHECK(weight.device().is_cuda(), "weight tensor must be cuda");
  TORCH_CHECK(seqlens.device().is_cuda(), "seqlens tensor must be cuda");
  TORCH_CHECK(cu_seqlens.device().is_cuda(), "cu_seqlens tensor must be cuda");
  TORCH_CHECK(x.is_contiguous(), "x tensor a must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor a must be contiguous");
  TORCH_CHECK(seqlens.size(0) == weight.size(0),
              "seqlens and weight must share the same num_group");
  TORCH_CHECK(x.size(1) == weight.size(2), "x and weight must share the same k");

  int m = x.size(0);
  int k = x.size(1);
  int n = weight.size(1);
  int num_group = seqlens.size(0);

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
  }

  torch::Tensor tmas;
  bool update_tma = true;
  if (tma_desc.has_value()) {
    tmas = tma_desc.value();
    update_tma = false;
  } else {
    tmas = torch::empty({num_group * 2, 128}, options);
  }

  torch::Tensor tiles = torch::empty({num_group}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_group + 1}, options.dtype(torch::kInt32));

  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *seqlens_ptr = seqlens.const_data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr();
  auto *tmas_ptr = tmas.mutable_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();

  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();

  group_gemm_bf16_async(y_ptr, x_ptr, weight_ptr, seqlens_ptr, cu_seqlens_ptr,
                                 tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k,
                                 num_seq_per_group_avg, update_tma, false, stream);

  return y;
}

torch::Tensor reformat_x_scale_entry(const torch::Tensor &x_scale, const torch::Tensor &seqlens,
                                     const torch::Tensor &cu_seqlens,
                                     std::optional<torch::Tensor> out_x_scale,
                                     const int64_t num_seq_per_group_avg) {
  auto stream = at::cuda::getCurrentCUDAStream(x_scale.get_device());
  TORCH_CHECK(x_scale.device().is_cuda(), "x_scale tensor must be cuda");
  TORCH_CHECK(seqlens.device().is_cuda(), "seqlens tensor must be cuda");
  TORCH_CHECK(cu_seqlens.device().is_cuda(), "cu_seqlens tensor must be cuda");
  TORCH_CHECK(x_scale.is_contiguous(), "x_scale tensor a must be contiguous");
  TORCH_CHECK(seqlens.is_contiguous(), "seqlens tensor a must be contiguous");
  TORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens tensor a must be contiguous");

  int m = x_scale.size(0);
  int n = x_scale.size(1);
  TORCH_CHECK(n == 16 || n == 32 || n == 56,
              "n must be 16, 32 or 56(for group gemm k=2048, k=4096 or k=7168)");

  int num_group = seqlens.size(0);
  int tilem = 0;
  if (num_seq_per_group_avg <= 8) {
    tilem = 8;
  } else if (num_seq_per_group_avg <= 16) {
    tilem = 16;
  } else if (num_seq_per_group_avg <= 32) {
    tilem = 32;
  } else if (num_seq_per_group_avg <= 48) {
    tilem = 48;
  } else {
    tilem = 64;
  }
  int num_seq_pad_per_group = m / num_group;
  TORCH_CHECK(num_seq_pad_per_group % tilem == 0,
              "The sparse pad length of x_scale for each group must be aligned to multiple of "
              "8/16/32/48/64 according to num_seq_per_group_avg");

  torch::Tensor output;
  if (out_x_scale.has_value()) {
    output = out_x_scale.value();
  } else {
    output = torch::empty({n, m}, x_scale.options());
  }

  const auto *xscale_ptr = x_scale.const_data_ptr();
  const auto *seqlens_ptr = seqlens.const_data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr();
  auto *output_ptr = output.mutable_data_ptr();

  reformat_x_scale_async(output_ptr, xscale_ptr, seqlens_ptr, cu_seqlens_ptr, num_group, m, n,
                         tilem, stream);

  return output;
}

}  // namespace group_gemm
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "group_gemm_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, Tensor y_scale, "
      "int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc, Tensor? task_map_workspace) -> "
      "(Tensor)");
  m.impl("group_gemm_fp8", torch::kCUDA, &hpc::group_gemm::group_gemm_fp8_entry);

  m.def(
      "group_gemm_pertensor_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, Tensor "
      "y_scale, int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc, Tensor? "
      "task_map_workspace) -> (Tensor)");
  m.impl("group_gemm_pertensor_fp8", torch::kCUDA, &hpc::group_gemm::group_gemm_fp8_entry);

  m.def(
      "group_gemm_bf16(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, "
      "int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc) -> (Tensor)");
  m.impl("group_gemm_bf16", torch::kCUDA,
         &hpc::group_gemm::group_gemm_bf16_entry);

  m.def(
      "group_gemm_blockwise_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, Tensor "
      "xscale, Tensor wscale,"
      "int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc, Tensor? task_map_workspace) -> "
      "(Tensor)");
  m.impl("group_gemm_blockwise_fp8", torch::kCUDA,
         &hpc::group_gemm::group_gemm_blockwise_fp8_entry);

  m.def(
      "reformat_x_scale(Tensor x_scale, Tensor seqlens, Tensor cu_seqlens, "
      "Tensor? out_x_scale, int num_seq_per_group_avg) -> (Tensor)");
  m.impl("reformat_x_scale", torch::kCUDA, &hpc::group_gemm::reformat_x_scale_entry);
}
