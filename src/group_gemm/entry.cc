// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/group_gemm/group_gemm.h"

namespace hpc {
namespace group_gemm {

torch::Tensor group_gemm_fp8(const torch::Tensor &x, const torch::Tensor &weight,
                             const torch::Tensor &seqlens, const torch::Tensor &cu_seqlens,
                             const torch::Tensor &y_scale, std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda(), "x tensor must be cuda");
  TORCH_CHECK(weight.device().is_cuda(), "weight tensor must be cuda");
  TORCH_CHECK(seqlens.device().is_cuda(), "seqlens tensor must be cuda");
  TORCH_CHECK(cu_seqlens.device().is_cuda(), "cu_seqlens tensor must be cuda");
  TORCH_CHECK(x.is_contiguous(), "x tensor a must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor a must be contiguous");
  TORCH_CHECK(seqlens.size(0) == weight.size(0), "x and weight must share the same k");
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
  torch::Tensor tmas = torch::empty({num_group * 2, 128}, options);

  torch::Tensor tiles = torch::empty({num_group}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_group + 1}, options.dtype(torch::kInt32));

  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *seqlens_ptr = seqlens.const_data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr();
  const auto *y_scale_ptr = y_scale.const_data_ptr();
  auto *tmas_ptr = tmas.mutable_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();

  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();

  group_gemm_fp8_async(y_ptr, x_ptr, weight_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale_ptr, tmas_ptr,
                       tiles_ptr, cu_tiles_ptr, num_group, m, n, k, stream);

  return y;
}

}  // namespace group_gemm
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "group_gemm_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, Tensor y_scale, "
      "Tensor? output) -> (Tensor)");
  m.impl("group_gemm_fp8", torch::kCUDA, &hpc::group_gemm::group_gemm_fp8);
}
