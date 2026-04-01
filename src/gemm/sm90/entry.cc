// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/gemm/gemm.h"

namespace hpc {
namespace gemm {

torch::Tensor gemm_bf16xfp32_entry(const torch::Tensor &x, const torch::Tensor &w_high,
                                   const torch::Tensor &w_low, double scale, bool use_fp32_output,
                                   bool use_splitk, std::optional<torch::Tensor> split_flag) {
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

  TORCH_CHECK(n % 64 == 0, "n must to be divided by 64.");

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

  if (use_splitk) {
    if (m <= 32) {
      // use wgmma 64x16x16 instruction and splitk.
      if (n == 512 || n == 192) {
        split_k = 8;
      } else if (n == 1024) {
        split_k = 4;
      } else if (n == 2048) {
        split_k = 2;
      }
    }
  }

  if (split_k != 1) {
    split_y = torch::empty({split_k, m, n}, options.dtype(torch::kFloat32));
    if (split_flag.has_value()) {
      split_flag_tensor = split_flag.value();
    } else {
      split_flag_tensor =
          torch::zeros({(m + 15) / 16, (n + 127) / 128}, options.dtype(torch::kInt32));
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
      "gemm_bf16xfp32(Tensor x, Tensor w_high, Tensor w_low, "
      "float scale, bool use_fp32_output, bool use_splitk, Tensor? split_flag) -> (Tensor)");
  m.impl("gemm_bf16xfp32", torch::kCUDA, &hpc::gemm::gemm_bf16xfp32_entry);
}
