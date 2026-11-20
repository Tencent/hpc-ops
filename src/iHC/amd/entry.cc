// Copyright (C) 2026 Tencent.
#include <torch/library.h>

#include <optional>
#include <tuple>

#include "src/iHC/amd/iHC.h"
#include "src/utils/amd/runtime.h"

namespace hpc::iHC::amd {

std::tuple<at::Tensor, at::Tensor> pre_head_entry(const at::Tensor& x, const at::Tensor& w,
                                                  const at::Tensor& scale, const at::Tensor& base,
                                                  double norm_eps, double hc_eps, double magnitude,
                                                  const std::optional<at::Tensor>& rms_weight,
                                                  double rms_eps, bool cast_bfloat_for_norm,
                                                  bool has_post) {
  hpc::amd::check_tensor(x, x, at::kBFloat16, "x");
  hpc::amd::check_tensor(w, x, at::kFloat, "w");
  hpc::amd::check_tensor(scale, x, at::kFloat, "hc_scale");
  hpc::amd::check_tensor(base, x, at::kFloat, "hc_base");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 4, "x must have shape [N, 4, D]");
  const int hidden = x.size(2);
  TORCH_CHECK(hidden == 4096 || hidden == 6144, "hidden dim must be 4096 or 6144");
  const int proj = has_post ? 8 : 4;
  TORCH_CHECK(w.dim() == 2 && w.size(0) == proj && w.size(1) == 4 * hidden,
              "w has an invalid projection shape");
  TORCH_CHECK(scale.dim() == 1 && scale.size(0) == (has_post ? 2 : 1),
              "hc_scale has an invalid shape");
  TORCH_CHECK(base.dim() == 1 && base.size(0) == proj, "hc_base has an invalid shape");
  if (rms_weight) {
    hpc::amd::check_tensor(*rms_weight, x, at::kBFloat16, "rms_weight");
    TORCH_CHECK(rms_weight->dim() == 1 && rms_weight->size(0) == hidden,
                "rms_weight must have shape [D]");
  }
  const c10::DeviceGuard guard(x.device());
  auto y = at::empty({x.size(0), hidden}, x.options());
  auto post = at::empty({x.size(0), has_post ? 4 : 0}, x.options().dtype(at::kFloat));
  pre_head_async(hpc::amd::bf16_data(y), has_post ? post.data_ptr<float>() : nullptr,
                 hpc::amd::bf16_data(x), w.data_ptr<float>(), scale.data_ptr<float>(),
                 base.data_ptr<float>(), x.size(0), hidden, norm_eps, hc_eps, magnitude,
                 rms_weight ? hpc::amd::bf16_data(*rms_weight) : nullptr, rms_eps,
                 cast_bfloat_for_norm, hpc::amd::current_stream(x));
  C10_HIP_CHECK(hipGetLastError());
  return {y, post};
}

std::tuple<at::Tensor, at::Tensor> pre_entry(const at::Tensor& x, const at::Tensor& w,
                                             const at::Tensor& scale, const at::Tensor& base,
                                             double norm_eps, double hc_eps, double magnitude,
                                             std::optional<at::Tensor> rms_weight, double rms_eps,
                                             bool cast_bfloat_for_norm) {
  return pre_head_entry(x, w, scale, base, norm_eps, hc_eps, magnitude, rms_weight, rms_eps,
                        cast_bfloat_for_norm, true);
}

at::Tensor head_entry(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scale,
                      const at::Tensor& base, double norm_eps, double hc_eps,
                      std::optional<at::Tensor> rms_weight, double rms_eps,
                      bool cast_bfloat_for_norm) {
  return std::get<0>(pre_head_entry(x, w, scale, base, norm_eps, hc_eps, 0.0, rms_weight, rms_eps,
                                    cast_bfloat_for_norm, false));
}

at::Tensor post_entry(const at::Tensor& x, const at::Tensor& residual, const at::Tensor& post) {
  hpc::amd::check_tensor(x, x, at::kBFloat16, "x");
  hpc::amd::check_tensor(residual, x, at::kBFloat16, "residual");
  hpc::amd::check_tensor(post, x, at::kFloat, "H_post");
  TORCH_CHECK(x.dim() == 2 && residual.dim() == 3 && post.dim() == 2,
              "expected x [N,D], residual [N,4,D], H_post [N,4]");
  const int hidden = x.size(1);
  TORCH_CHECK(hidden == 4096 || hidden == 6144, "hidden dim must be 4096 or 6144");
  TORCH_CHECK(residual.size(0) == x.size(0) && residual.size(1) == 4 &&
                  residual.size(2) == hidden && post.size(0) == x.size(0) && post.size(1) == 4,
              "residual or H_post shape does not match x");
  const c10::DeviceGuard guard(x.device());
  auto y = at::empty_like(residual);
  post_async(hpc::amd::bf16_data(y), hpc::amd::bf16_data(x), hpc::amd::bf16_data(residual),
             post.data_ptr<float>(), x.size(0), hidden, hpc::amd::current_stream(x));
  C10_HIP_CHECK(hipGetLastError());
  return y;
}

}  // namespace hpc::iHC::amd

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "fuse_ihc_pre(Tensor x, Tensor w, Tensor hc_scale, Tensor hc_base, float norm_eps, "
      "float hc_eps, float magnitude, Tensor? rms_weight=None, float rms_eps=0.0, "
      "bool cast_bfloat_for_norm=False) -> (Tensor output_y, Tensor output_H_post)");
  m.impl("fuse_ihc_pre", c10::DispatchKey::CUDA, &hpc::iHC::amd::pre_entry);
  m.def(
      "fuse_ihc_head(Tensor x, Tensor w, Tensor hc_scale, Tensor hc_base, float norm_eps, "
      "float hc_eps, Tensor? rms_weight=None, float rms_eps=0.0, "
      "bool cast_bfloat_for_norm=False) -> Tensor output");
  m.impl("fuse_ihc_head", c10::DispatchKey::CUDA, &hpc::iHC::amd::head_entry);
  m.def("fuse_ihc_post(Tensor x, Tensor residual, Tensor H_post) -> Tensor output");
  m.impl("fuse_ihc_post", c10::DispatchKey::CUDA, &hpc::iHC::amd::post_entry);
  // TODO(amd): fuse_ihc_post_pre and its stage1/stage2 fusion, registered in unsupported.cc.
}
