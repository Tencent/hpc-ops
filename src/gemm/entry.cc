// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/gemm/gemm.h"

namespace hpc {
namespace gemm {

torch::Tensor pad_and_transpose_entry(const torch::Tensor &x) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  TORCH_CHECK(x.dtype() == torch::kFloat32, "input dtype must be float32");
  int m = x.size(0);
  int n = x.size(1);
  auto options = x.options();

  int m_pad = (m + 3) / 4 * 4;
  torch::Tensor y = torch::empty({n, m_pad}, options);
  const auto *x_ptr = x.const_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();

  bool running = pad_and_transpose_async(y_ptr, x_ptr, m, n, m_pad, stream);

  TORCH_CHECK(running, "pad_and_transpose_async launch failed!");

  return y;
}

torch::Tensor gemm_blockwise_entry(const torch::Tensor &x, const torch::Tensor &weight,
                                   const torch::Tensor &x_scale, const torch::Tensor &weight_scale,
                                   bool trans_xscale, std::optional<torch::Tensor> bias) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.is_contiguous(), "x tensor must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor must be contiguous");
  TORCH_CHECK(x_scale.is_contiguous(), "x_scale tensor must be contiguous");
  TORCH_CHECK(weight_scale.is_contiguous(), "weight_scale tensor must be contiguous");

  TORCH_CHECK(x.dtype() == torch::kFloat8_e4m3fn, "x dtype must be float8_e4m3");
  TORCH_CHECK(weight.dtype() == torch::kFloat8_e4m3fn, "weight dtype must be float8_e4m3");
  TORCH_CHECK(x_scale.dtype() == torch::kFloat32, "x_scale dtype must be float32");
  TORCH_CHECK(weight_scale.dtype() == torch::kFloat32, "weight_scale dtype must be float32");

  int m = x.size(0);
  int k = x.size(1);
  int n = weight.size(0);
  int num_block_n = weight_scale.size(0);
  int num_block_k = weight_scale.size(1);

  TORCH_CHECK(k % 128 == 0, "k % 128 == 0 must be true");
  TORCH_CHECK(n % 8 == 0, "n % 8 == 0 must be true");
  TORCH_CHECK(num_block_n == (n + 127) / 128, "num_block_n == (n + 127) / 128 must be true");
  TORCH_CHECK(num_block_k == ((k + 127) / 128 + 3) / 4 * 4,
              "num_block_k == ((k + 127) / 128 + 3) / 4 * 4 must be true");
  TORCH_CHECK(num_block_k % 4 == 0, "weight_scale.size(1) % 4 must be 0");
  TORCH_CHECK(k <= 16384, "only support k <= 16384");

  TORCH_CHECK(x.size(1) == weight.size(1), "x and weight must have same k");

  const void *bias_ptr = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "bias dtype must be float32");
    TORCH_CHECK(bias.value().is_contiguous(), "bias tensor must be contiguous");
    TORCH_CHECK(bias.value().size(0) == n, "bias shape must be n");
    bias_ptr = bias.value().const_data_ptr();
  }

  auto options = x.options();
  torch::Tensor y = torch::empty({m, n}, options.dtype(torch::kBFloat16));

  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *x_scale_ptr = x_scale.const_data_ptr();
  const auto *weight_scale_ptr = weight_scale.const_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();

  int splitk = 1;
  if (m < 32 && k >= 2048) {
    if (n <= 512) {
      splitk = 8;
    } else if (n <= 1024) {
      splitk = 4;
    } else if (n <= 4096) {
      splitk = 2;
    }
  }

  torch::Tensor split_y;
  torch::Tensor split_flag;
  void *split_y_ptr = nullptr;
  void *split_flag_ptr = nullptr;

  if (splitk != 1) {
    split_y = torch::empty({splitk, m, n}, options.dtype(torch::kFloat32));
    split_y_ptr = split_y.mutable_data_ptr();
    split_flag = torch::zeros({(m + 7) / 8, (n + 127) / 128}, options.dtype(torch::kInt32));
    split_flag_ptr = split_flag.mutable_data_ptr();
  }

  bool running = false;

  if (trans_xscale) {
    int m_pad = (m + 3) / 4 * 4;
    auto scale_options = x_scale.options();
    torch::Tensor new_x_scale = torch::empty({k / 128, m_pad}, scale_options);
    auto *new_x_scale_ptr = new_x_scale.mutable_data_ptr();
    pad_and_transpose_async(new_x_scale_ptr, x_scale_ptr, m, x_scale.size(1), m_pad, stream);
    running = gemm_blockwise_fp8_async(y_ptr, split_y_ptr, split_flag_ptr, x_ptr, weight_ptr,
                                       new_x_scale_ptr, weight_scale_ptr, bias_ptr, m, n, k, m_pad,
                                       num_block_k, num_block_n, splitk, stream);
  } else {
    int m_pad = x_scale.size(1);
    TORCH_CHECK(x_scale.size(0) == k / 128, "x_scale dim 0 must be k / 128");
    TORCH_CHECK(m_pad == (m + 3) / 4 * 4, "x_scale dim 1 must aligned to 4");
    running = gemm_blockwise_fp8_async(y_ptr, split_y_ptr, split_flag_ptr, x_ptr, weight_ptr,
                                       x_scale_ptr, weight_scale_ptr, bias_ptr, m, n, k, m_pad,
                                       num_block_k, num_block_n, splitk, stream);
  }

  TORCH_CHECK(running, "gemm_blockwise_fp8_async launch failed!");

  return y;
}

torch::Tensor gemm_entry(const torch::Tensor &x, const torch::Tensor &weight) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.is_contiguous(), "x tensor a must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor a must be contiguous");
  TORCH_CHECK(x.size(1) == weight.size(1), "x and weight must share the same k");

  int m = x.size(0);
  int k = x.size(1);
  int n = weight.size(0);

  auto options = x.options();
  torch::Tensor y = torch::empty({m, n}, options.dtype(torch::kBFloat16));

  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();

  if (x.dtype() == torch::kBFloat16) {
    gemm_bf16_async(y_ptr, x_ptr, weight_ptr, m, n, k, stream);
  } else {
    gemm_fp8_async(y_ptr, x_ptr, weight_ptr, m, n, k, stream);
  }

  return y;
}

}  // namespace gemm
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "gemm_blockwise(Tensor x, Tensor weight, Tensor x_scale, Tensor weight_scale, bool "
      "trans_xscale, Tensor? bias) "
      "-> (Tensor)");
  m.impl("gemm_blockwise", torch::kCUDA, &hpc::gemm::gemm_blockwise_entry);

  m.def("pad_and_transpose(Tensor x) -> (Tensor)");
  m.impl("pad_and_transpose", torch::kCUDA, &hpc::gemm::pad_and_transpose_entry);

  m.def("gemm(Tensor x, Tensor w) -> (Tensor)");
  m.impl("gemm", torch::kCUDA, &hpc::gemm::gemm_entry);
}
