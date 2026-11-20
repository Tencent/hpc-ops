// Copyright (C) 2026 Tencent.
#include <torch/library.h>

#include <optional>

#include "src/fuse_moe/amd/reduce.h"
#include "src/utils/amd/runtime.h"

namespace hpc::fuse_moe::amd {

at::Tensor reduce_entry(const at::Tensor& x, const at::Tensor& positions, const at::Tensor& scales,
                        const std::optional<at::Tensor>& shared_output) {
  hpc::amd::check_tensor(x, x, at::kBFloat16, "x");
  hpc::amd::check_tensor(positions, x, at::kInt, "topk_pos");
  hpc::amd::check_tensor(scales, x, at::kFloat, "topk_scale");
  TORCH_CHECK(x.dim() == 2 && positions.dim() == 2 && scales.sizes() == positions.sizes(),
              "expected x [tokens,D], topk_pos and topk_scale [N,K]");
  TORCH_CHECK(positions.size(1) <= 128, "num_topk must be <= 128");
  const int64_t rows = positions.size(0);
  const int64_t hidden = x.size(1);
  if (shared_output) {
    hpc::amd::check_tensor(*shared_output, x, at::kBFloat16, "shared_output");
    TORCH_CHECK(shared_output->dim() == 2 && shared_output->size(0) == rows &&
                    shared_output->size(1) == hidden,
                "shared_output must have shape [N,D]");
  }
  const c10::DeviceGuard guard(x.device());
  auto y = at::empty({rows, hidden}, x.options());
  reduce_async(hpc::amd::bf16_data(y), hpc::amd::bf16_data(x), positions.data_ptr<int>(),
               scales.data_ptr<float>(),
               shared_output ? hpc::amd::bf16_data(*shared_output) : nullptr, rows, x.size(0),
               hidden, positions.size(1), hpc::amd::current_stream(x));
  C10_HIP_CHECK(hipGetLastError());
  return y;
}

}  // namespace hpc::fuse_moe::amd

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def("reduce(Tensor x, Tensor topk_pos, Tensor topk_scale, Tensor? shared_output) -> Tensor");
  m.impl("reduce", c10::DispatchKey::CUDA, &hpc::fuse_moe::amd::reduce_entry);
  // TODO(amd): count_and_gather's FP8/TMA entry and the FP8 matrix MoE entries.
  // Their schemas remain available with explicit TODO errors in src/C/amd/unsupported.cc.
}
