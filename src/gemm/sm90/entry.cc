// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/gemm/gemm.h"

namespace hpc {
namespace gemm {

struct KernelConfig {
  int split_k;
  int k_warpgroup_n;
  int kTileM;  // 16 for small-m MMA, 64 for large-m MMA
};

struct Segment {
  int m_max;
  KernelConfig cfg;
};

static constexpr std::array<Segment, 13> kN192Segments = {{
    {32, {8, 2, 16}},
    {48, {8, 1, 16}},
    {64, {8, 2, 16}},
    {96, {4, 1, 16}},
    {144, {4, 2, 16}},
    {208, {2, 1, 16}},
    {304, {2, 2, 16}},
    {416, {1, 1, 16}},
    {624, {1, 2, 16}},
    {832, {2, 1, 64}},
    {1024, {1, 2, 16}},
    {2048, {4, 1, 64}},
    {0x7fffffff, {1, 1, 64}},  // sentinel: catches anything larger
}};

static inline KernelConfig select_config(int m, int n, bool use_splitk) {
  constexpr int kN192 = 192;
  constexpr int kN512 = 512;
  constexpr int kN1024 = 1024;
  constexpr int kN2048 = 2048;
  constexpr int kMThreshold128 = 128;
  constexpr int kDefaultKtm64 = 64;
  constexpr int kDefaultKtm16 = 16;
  constexpr int kDefaultSk8 = 8;
  constexpr int kDefaultSk4 = 4;
  constexpr int kDefaultSk2 = 2;
  constexpr int kDefaultSk1 = 1;
  constexpr int kDefaultWgn = 2;
  constexpr int kDefaultWg1 = 1;
  constexpr int kDefaultKtmForLargeM = 64;

  if (n == kN192) {
    for (const auto &seg : kN192Segments) {
      if (m <= seg.m_max) return seg.cfg;
    }
    return {kDefaultSk1, kDefaultWg1, kDefaultKtm64};
  }

  // Fallback for n != 192: preserve the original heuristic.
  int sk = 1;
  if (use_splitk && m <= kMThreshold128) {
    if (n == kN512) {
      sk = kDefaultSk8;
    } else if (n == kN1024) {
      sk = kDefaultSk4;
    } else if (n == kN2048) {
      sk = kDefaultSk2;
    }
  }

  int wgn = kDefaultWgn;
  int ktm = (m > kMThreshold128) ? kDefaultKtmForLargeM : kDefaultKtm16;

  return {sk, wgn, ktm};
}

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

  KernelConfig cfg = select_config(m, n, use_splitk);

  torch::Tensor split_y;
  torch::Tensor split_flag_tensor;
  void *split_y_ptr = nullptr;
  void *split_flag_ptr = nullptr;

  if (cfg.split_k != 1) {
    split_y = torch::empty({cfg.split_k, m, n}, options.dtype(torch::kFloat32));
    if (split_flag.has_value()) {
      split_flag_tensor = split_flag.value();
    } else {
      const int tile_m = cfg.kTileM;
      const int tile_n = 64 * cfg.k_warpgroup_n;
      split_flag_tensor = torch::zeros({(m + tile_m - 1) / tile_m, (n + tile_n - 1) / tile_n},
                                       options.dtype(torch::kInt32));
    }
    split_y_ptr = split_y.mutable_data_ptr();
    split_flag_ptr = split_flag_tensor.mutable_data_ptr();
  }

  torch::Tensor y = torch::empty({m, n}, options.dtype(out_dtype));

  const auto *x_ptr = x.const_data_ptr();
  const auto *w_high_ptr = w_high.const_data_ptr();
  const auto *w_low_ptr = w_low.const_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();

  bool running =
      gemm_bf16xfp32_async(y_ptr, split_y_ptr, split_flag_ptr, x_ptr, w_high_ptr, w_low_ptr, m, n,
                           k, scale, use_fp32_output, cfg.split_k, stream);

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
