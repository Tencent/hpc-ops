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

  int m = x.size(0);
  int n = x.size(1);
  auto options = x.options();

  int m_pad = (m + 3) / 4 * 4;
  torch::Tensor y = torch::empty({n, m_pad}, options);
  const auto *x_ptr = x.const_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();
  pad_and_transpose_async(y_ptr, x_ptr, m, n, m_pad, stream);
  return y;
}

torch::Tensor gemm_blockwise_entry(const torch::Tensor &x, const torch::Tensor &weight,
                                   const torch::Tensor &x_scale, const torch::Tensor &weight_scale,
                                   const torch::Tensor &bias) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.is_contiguous(), "x tensor must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor must be contiguous");
  TORCH_CHECK(x_scale.is_contiguous(), "x_scale tensor must be contiguous");
  TORCH_CHECK(weight_scale.is_contiguous(), "weight_scale tensor must be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "bias tensor must be contiguous");

  TORCH_CHECK(x.dtype() == torch::kFloat8_e4m3fn, "x dtype must be float8_e4m3");
  TORCH_CHECK(weight.dtype() == torch::kFloat8_e4m3fn, "weight dtype must be float8_e4m3");
  TORCH_CHECK(x_scale.dtype() == torch::kFloat32, "x_scale dtype must be float32");
  TORCH_CHECK(weight_scale.dtype() == torch::kFloat32, "weight_scale dtype must be float32");
  TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias dtype must be float32");

  int m = x.size(0);
  int k = x.size(1);
  int n = weight.size(0);

  TORCH_CHECK(k % 128 == 0, "k % 128 must be 0");
  TORCH_CHECK(n % 128 == 0, "n % 128 must be 0");
  TORCH_CHECK(k <= 16384, "only support k <= 16384");

  TORCH_CHECK(x.size(1) == weight.size(1), "x and weight must have same k");
  TORCH_CHECK((weight_scale.size(0) == n / 128),
              "weight_scale must be blockwise quant, and blocksize must be 128");
  TORCH_CHECK(weight_scale.size(1) == k / 128, "weight_scale dim -1 must be equal to k / 128");
  TORCH_CHECK(weight_scale.size(1) % 4 == 0, "The last dim of weigt_scale must be divisible by 4.")
  TORCH_CHECK(bias.size(0) == n, "bias shape must be n");

  auto options = x.options();
  torch::Tensor y = torch::empty({m, n}, options.dtype(torch::kBFloat16));

  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *x_scale_ptr = x_scale.const_data_ptr();
  const auto *weight_scale_ptr = weight_scale.const_data_ptr();
  const auto *bias_ptr = bias.const_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();

  if (x_scale.size(0) == m && x_scale.size(1) == k / 128) {
    int m_pad = (m + 3) / 4 * 4;
    auto options = x_scale.options();
    torch::Tensor new_x_scale = torch::empty({k / 128, m_pad}, options);
    auto *new_x_scale_ptr = new_x_scale.mutable_data_ptr();
    pad_and_transpose_async(new_x_scale_ptr, x_scale_ptr, m, x_scale.size(1), m_pad, stream);
    gemm_blockwise_async(y_ptr, x_ptr, weight_ptr, new_x_scale_ptr, weight_scale_ptr, bias_ptr, m,
                         n, k, m_pad, stream);
  } else {
    int m_pad = x_scale.size(1);
    TORCH_CHECK(x_scale.size(0) == k / 128, "x_scale dim 0 must be k / 128");
    TORCH_CHECK(m_pad == (m + 3) / 4 * 4, "x_scale dim 1 must aligned to 4");
    gemm_blockwise_async(y_ptr, x_ptr, weight_ptr, x_scale_ptr, weight_scale_ptr, bias_ptr, m, n, k,
                         m_pad, stream);
  }
  return y;
}

torch::Tensor gemm_bf16xfp32_entry(const torch::Tensor &x, const torch::Tensor &w_high,
                                   const torch::Tensor &w_low, double scale, bool use_fp32_output,
                                   std::optional<torch::Tensor> split_flag) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.is_contiguous(), "x tensor must be contiguous");
  TORCH_CHECK(w_high.is_contiguous(), "w_high tensor must be contiguous");
  TORCH_CHECK(w_low.is_contiguous(), "w_low tensor must be contiguous");

  TORCH_CHECK(x.dtype() == torch::kBFloat16, "x dtype must be bfloat16");
  TORCH_CHECK(w_high.dtype() == torch::kBFloat16, "w_high dtype must be bfloat16");
  TORCH_CHECK(w_low.dtype() == torch::kBFloat16, "w_low dtype must be bfloat16");

  int m = x.size(0);
  int k = x.size(1);
  int n = w_high.size(0);

  TORCH_CHECK(n % 128 == 0, "n must to be divided by 128.");

  auto options = x.options();

  auto out_dtype = torch::kBFloat16;
  if (use_fp32_output) {
    out_dtype = torch::kFloat32;
  }

  int split_k = 1;
  torch::Tensor split_y;
  torch::Tensor split_flag_tensor;
  void *split_y_ptr = nullptr;
  void *split_flag_ptr = nullptr;

  if (m <= 32) {
    // use wgmma 64x16x16 instruction and splitk.
    if (n == 512) {
      split_k = 8;
    } else if (n == 1024) {
      split_k = 4;
    } else if (n == 2048) {
      split_k = 2;
    }
  }

  if (split_k != 1) {
    split_y = torch::empty({split_k, m, n}, options.dtype(torch::kFloat32));
    if (split_flag.has_value()) {
      split_flag_tensor = split_flag.value();
    } else {
      split_flag_tensor = torch::zeros({(m + 15) / 16, n / 128}, options.dtype(torch::kInt32));
    }
    split_y_ptr = split_y.mutable_data_ptr();
    split_flag_ptr = split_flag_tensor.mutable_data_ptr();
  }

  torch::Tensor y = torch::empty({m, n}, options.dtype(out_dtype));

  const auto *x_ptr = x.const_data_ptr();
  const auto *w_high_ptr = w_high.const_data_ptr();
  const auto *w_low_ptr = w_low.const_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();

  bool running = gemm_bf16xfp32_async(y_ptr, split_y_ptr, split_flag_ptr, x_ptr, w_high_ptr,
                                      w_low_ptr, m, n, k, scale, use_fp32_output, split_k, stream);

  TORCH_CHECK(running, "gemm_bf16xfp32 launch failed!");

  return y;
}

}  // namespace gemm
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "gemm_blockwise(Tensor x, Tensor weight, Tensor x_scale, Tensor weight_scale, Tensor bias) "
      "-> (Tensor)");
  m.impl("gemm_blockwise", torch::kCUDA, &hpc::gemm::gemm_blockwise_entry);

  m.def("pad_and_transpose(Tensor x) -> (Tensor)");
  m.impl("pad_and_transpose", torch::kCUDA, &hpc::gemm::pad_and_transpose_entry);

  m.def(
      "gemm_bf16xfp32(Tensor x, Tensor w_high, Tensor w_low, "
      "float scale, bool use_fp32_output, Tensor? split_flag) -> (Tensor)");
  m.impl("gemm_bf16xfp32", torch::kCUDA, &hpc::gemm::gemm_bf16xfp32_entry);
}
