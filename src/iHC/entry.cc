// Copyright 2026 Tencent

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

#include "src/iHC/iHC.h"
// For HPC_TARGET_ARCH, which gates the sm90-only decode path below. The header
// also #errors if the macro is missing, rather than letting the gate silently
// evaluate false.
#include "src/utils/utils.h"

namespace hpc {
namespace iHC {

std::tuple<torch::Tensor, torch::Tensor> fuse_ihc_pre_entry(
    const torch::Tensor &x, const torch::Tensor &w, const torch::Tensor &hc_scale,
    const torch::Tensor &hc_base, double norm_eps, double hc_eps, double magnitude,
    std::optional<torch::Tensor> rms_weight, double rms_eps, bool cast_bfloat_for_norm) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  TORCH_CHECK(x.is_contiguous(), "x tensor must be contiguous");
  TORCH_CHECK(w.is_contiguous(), "w tensor must be contiguous");
  TORCH_CHECK(hc_scale.is_contiguous(), "hc_scale tensor must be contiguous");
  TORCH_CHECK(hc_base.is_contiguous(), "hc_base tensor must be contiguous");

  TORCH_CHECK(x.device().is_cuda(), "x tensor's device must be cuda");
  TORCH_CHECK(w.device().is_cuda(), "w tensor's device must be cuda");
  TORCH_CHECK(hc_scale.device().is_cuda(), "hc_scale tensor's device must be cuda");
  TORCH_CHECK(hc_base.device().is_cuda(), "hc_base tensor's device must be cuda");

  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x tensor data type must be bfloat16");
  TORCH_CHECK(w.scalar_type() == torch::kFloat32, "w tensor data type must be float32");
  TORCH_CHECK(hc_scale.scalar_type() == torch::kFloat32,
              "hc_scale tensor data type must be float32");
  TORCH_CHECK(hc_base.scalar_type() == torch::kFloat32, "hc_base tensor data type must be float32");

  TORCH_CHECK(x.dim() == 3, "x tensor's dim must be 3");
  TORCH_CHECK(w.dim() == 2, "w tensor's dim must be 2");
  TORCH_CHECK(hc_scale.dim() == 1, "hc_scale tensor's dim must be 1");
  TORCH_CHECK(hc_base.dim() == 1, "hc_base tensor's dim must be 1");

  int num_batch = x.size(0);
  int hc_mult = x.size(1);
  int hidden_dim = x.size(2);

  TORCH_CHECK(hc_mult == 4, "hc_mult must be 4");
  TORCH_CHECK(hidden_dim == 6144 || hidden_dim == 4096,
              "hidden dim must be 6144 (HYV4 production) or 4096");
  TORCH_CHECK(w.size(0) == 2 * hc_mult, "w tensor's first dim must be 2 * hc_mult");
  TORCH_CHECK(w.size(1) == hc_mult * hidden_dim,
              "w tensor's second dim must be hc_mult * hidden_dim");
  TORCH_CHECK(hc_scale.size(-1) == 2, "hc_scale tensor's last dim must be 2");
  TORCH_CHECK(hc_base.size(-1) == 2 * hc_mult, "hc_base tensor's last dim must be 2 * hc_mult");

  const __nv_bfloat16 *rms_weight_ptr = nullptr;
  if (rms_weight.has_value()) {
    const auto &g = rms_weight.value();
    TORCH_CHECK(g.is_contiguous(), "rms_weight tensor must be contiguous");
    TORCH_CHECK(g.device().is_cuda(), "rms_weight tensor's device must be cuda");
    TORCH_CHECK(g.scalar_type() == torch::kBFloat16,
                "rms_weight tensor data type must be bfloat16");
    TORCH_CHECK(g.dim() == 1 && g.size(0) == hidden_dim,
                "rms_weight must be 1-D with hidden_dim elements");
    rms_weight_ptr = reinterpret_cast<const __nv_bfloat16 *>(g.const_data_ptr());
  }

  auto options = x.options();
  torch::Tensor output_y = torch::empty({num_batch, hidden_dim}, options);
  torch::Tensor output_H_post = torch::empty({num_batch, hc_mult}, options.dtype(torch::kFloat32));

  const auto *x_ptr = reinterpret_cast<const __nv_bfloat16 *>(x.const_data_ptr());
  const auto *w_ptr = reinterpret_cast<const float *>(w.const_data_ptr());
  const auto *hc_scale_ptr = reinterpret_cast<const float *>(hc_scale.const_data_ptr());
  const auto *hc_base_ptr = reinterpret_cast<const float *>(hc_base.const_data_ptr());
  auto *output_y_ptr = reinterpret_cast<__nv_bfloat16 *>(output_y.mutable_data_ptr());
  auto *output_H_post_ptr = reinterpret_cast<float *>(output_H_post.mutable_data_ptr());

  // sm90 routes decode-shaped batches to a K-split path: the kernel below has a
  // grid of exactly num_batch, which cannot fill an sm90 device at small token
  // counts (latency-flat ~12-13us from 1 to 64 tokens on H20, against 3.8us for
  // the K-split path at one token). The RMSNorm fold is not implemented there,
  // so a request carrying rms_weight stays on the shared kernel.
  bool handled = false;
#if HPC_TARGET_ARCH == 90
  if (rms_weight_ptr == nullptr && sm90::ihc_pre_decode_supported(num_batch, hc_mult, hidden_dim)) {
    const size_t scratch_floats =
        sm90::ihc_pre_decode_scratch_floats(num_batch, hc_mult, hidden_dim);
    torch::Tensor scratch =
        torch::empty({static_cast<int64_t>(scratch_floats)}, options.dtype(torch::kFloat32));
    sm90::fuse_ihc_pre_decode_async(output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr,
                                    hc_base_ptr, num_batch, hc_mult, hidden_dim, norm_eps, hc_eps,
                                    magnitude,
                                    reinterpret_cast<float *>(scratch.mutable_data_ptr()), stream);
    handled = true;
  }
#endif

  if (!handled) {
    fuse_ihc_pre_async(output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr,
                       num_batch, hc_mult, hidden_dim, norm_eps, hc_eps, magnitude, stream,
                       rms_weight_ptr, rms_eps, cast_bfloat_for_norm);
  }

  return std::make_tuple(output_y, output_H_post);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fuse_ihc_post_pre_entry(
    const torch::Tensor &xa, const torch::Tensor &residual, const torch::Tensor &H_post_in,
    const torch::Tensor &w, const torch::Tensor &hc_scale, const torch::Tensor &hc_base,
    double norm_eps, double hc_eps, double magnitude, std::optional<torch::Tensor> rms_weight,
    double rms_eps, bool cast_bfloat_for_norm) {
  auto stream = at::cuda::getCurrentCUDAStream(xa.get_device());

  for (const auto *t : {&xa, &residual, &H_post_in, &w, &hc_scale, &hc_base}) {
    TORCH_CHECK(t->is_contiguous(), "all input tensors must be contiguous");
    TORCH_CHECK(t->device().is_cuda(), "all input tensors' device must be cuda");
  }
  TORCH_CHECK(xa.scalar_type() == torch::kBFloat16, "xa tensor data type must be bfloat16");
  TORCH_CHECK(residual.scalar_type() == torch::kBFloat16,
              "residual tensor data type must be bfloat16");
  TORCH_CHECK(H_post_in.scalar_type() == torch::kFloat32,
              "H_post_in tensor data type must be float32");
  TORCH_CHECK(w.scalar_type() == torch::kFloat32, "w tensor data type must be float32");
  TORCH_CHECK(hc_scale.scalar_type() == torch::kFloat32,
              "hc_scale tensor data type must be float32");
  TORCH_CHECK(hc_base.scalar_type() == torch::kFloat32, "hc_base tensor data type must be float32");

  TORCH_CHECK(xa.dim() == 2, "xa tensor's dim must be 2 (num_batch, hidden_dim)");
  TORCH_CHECK(residual.dim() == 3, "residual tensor's dim must be 3 (num_batch, hc, hidden_dim)");
  TORCH_CHECK(H_post_in.dim() == 2, "H_post_in tensor's dim must be 2");
  TORCH_CHECK(w.dim() == 2, "w tensor's dim must be 2");

  const int num_batch = xa.size(0);
  const int hidden_dim = xa.size(1);
  const int hc_mult = residual.size(1);

  TORCH_CHECK(hc_mult == 4, "hc_mult must be 4");
  TORCH_CHECK(hidden_dim == 6144 || hidden_dim == 4096,
              "hidden dim must be 6144 (HYV4 production) or 4096");
  TORCH_CHECK(residual.size(0) == num_batch && residual.size(2) == hidden_dim,
              "residual must be (num_batch, hc_mult, hidden_dim)");
  TORCH_CHECK(H_post_in.size(0) == num_batch && H_post_in.size(1) == hc_mult,
              "H_post_in must be (num_batch, hc_mult)");
  TORCH_CHECK(w.size(0) == 2 * hc_mult, "w tensor's first dim must be 2 * hc_mult");
  TORCH_CHECK(w.size(1) == hc_mult * hidden_dim,
              "w tensor's second dim must be hc_mult * hidden_dim");
  TORCH_CHECK(hc_scale.size(-1) == 2, "hc_scale tensor's last dim must be 2");
  TORCH_CHECK(hc_base.size(-1) == 2 * hc_mult, "hc_base tensor's last dim must be 2 * hc_mult");

  const __nv_bfloat16 *rms_weight_ptr = nullptr;
  if (rms_weight.has_value()) {
    const auto &g = rms_weight.value();
    TORCH_CHECK(g.is_contiguous(), "rms_weight tensor must be contiguous");
    TORCH_CHECK(g.device().is_cuda(), "rms_weight tensor's device must be cuda");
    TORCH_CHECK(g.scalar_type() == torch::kBFloat16,
                "rms_weight tensor data type must be bfloat16");
    TORCH_CHECK(g.dim() == 1 && g.size(0) == hidden_dim,
                "rms_weight must be 1-D with hidden_dim elements");
    rms_weight_ptr = reinterpret_cast<const __nv_bfloat16 *>(g.const_data_ptr());
  }

  auto options = xa.options();
  torch::Tensor y = torch::empty({num_batch, hc_mult, hidden_dim}, options);
  torch::Tensor z = torch::empty({num_batch, hidden_dim}, options);
  torch::Tensor H_post_out = torch::empty({num_batch, hc_mult}, options.dtype(torch::kFloat32));

  const size_t scratch_floats = ihc_post_pre_scratch_floats(num_batch, hc_mult, hidden_dim);
  torch::Tensor scratch =
      torch::empty({static_cast<int64_t>(scratch_floats)}, options.dtype(torch::kFloat32));
  float *scratch_ptr =
      scratch_floats > 0 ? reinterpret_cast<float *>(scratch.mutable_data_ptr()) : nullptr;

  fuse_ihc_post_pre_async(reinterpret_cast<__nv_bfloat16 *>(y.mutable_data_ptr()),
                          reinterpret_cast<__nv_bfloat16 *>(z.mutable_data_ptr()),
                          reinterpret_cast<float *>(H_post_out.mutable_data_ptr()),
                          reinterpret_cast<const __nv_bfloat16 *>(xa.const_data_ptr()),
                          reinterpret_cast<const __nv_bfloat16 *>(residual.const_data_ptr()),
                          reinterpret_cast<const float *>(H_post_in.const_data_ptr()),
                          reinterpret_cast<const float *>(w.const_data_ptr()),
                          reinterpret_cast<const float *>(hc_scale.const_data_ptr()),
                          reinterpret_cast<const float *>(hc_base.const_data_ptr()), num_batch,
                          hc_mult, hidden_dim, norm_eps, hc_eps, magnitude, scratch_ptr, stream,
                          rms_weight_ptr, rms_eps, cast_bfloat_for_norm);

  return std::make_tuple(y, z, H_post_out);
}

torch::Tensor fuse_ihc_post_entry(const torch::Tensor &x, const torch::Tensor &residual,
                                  const torch::Tensor &H_post) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  TORCH_CHECK(x.is_contiguous(), "x tensor must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual tensor must be contiguous");
  TORCH_CHECK(H_post.is_contiguous(), "H_post tensor must be contiguous");

  TORCH_CHECK(x.device().is_cuda(), "x tensor's device must be cuda");
  TORCH_CHECK(residual.device().is_cuda(), "residual tensor's device must be cuda");
  TORCH_CHECK(H_post.device().is_cuda(), "H_post tensor's device must be cuda");

  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x tensor data type must be bfloat16");
  TORCH_CHECK(residual.scalar_type() == torch::kBFloat16,
              "residual tensor data type must be bfloat16");
  TORCH_CHECK(H_post.scalar_type() == torch::kFloat32, "H_post tensor data type must be float32");

  TORCH_CHECK(x.dim() == 2, "x tensor's dim must be 2");
  TORCH_CHECK(residual.dim() == 3, "residual tensor's dim must be 3");
  TORCH_CHECK(H_post.dim() == 2, "H_post tensor's dim must be 2");

  int num_batch = residual.size(0);
  int hc_mult = residual.size(1);
  int hidden_dim = residual.size(2);

  TORCH_CHECK(hc_mult == 4, "hc_mult must be 4");
  TORCH_CHECK(hidden_dim == 6144 || hidden_dim == 4096,
              "hidden dim must be 6144 (HYV4 production) or 4096");
  TORCH_CHECK(x.size(0) == num_batch, "x tensor's first dim must be num_batch");
  TORCH_CHECK(x.size(1) == hidden_dim, "x tensor's second dim must be hidden_dim");
  TORCH_CHECK(H_post.size(0) == num_batch, "H_post tensor's first dim must be num_batch");
  TORCH_CHECK(H_post.size(1) == hc_mult, "H_post tensor's second dim must be hc_mult");

  auto options = x.options();
  torch::Tensor output = torch::empty({num_batch, hc_mult, hidden_dim}, options);

  const auto *x_ptr = reinterpret_cast<const __nv_bfloat16 *>(x.const_data_ptr());
  const auto *residual_ptr = reinterpret_cast<const __nv_bfloat16 *>(residual.const_data_ptr());
  const auto *H_post_ptr = reinterpret_cast<const float *>(H_post.const_data_ptr());
  auto *output_ptr = reinterpret_cast<__nv_bfloat16 *>(output.mutable_data_ptr());

  fuse_ihc_post_async(output_ptr, x_ptr, residual_ptr, H_post_ptr, num_batch, hc_mult, hidden_dim,
                      stream);

  return output;
}

torch::Tensor fuse_ihc_head_entry(const torch::Tensor &x, const torch::Tensor &w,
                                  const torch::Tensor &hc_scale, const torch::Tensor &hc_base,
                                  double norm_eps, double hc_eps,
                                  std::optional<torch::Tensor> rms_weight, double rms_eps,
                                  bool cast_bfloat_for_norm) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  TORCH_CHECK(x.is_contiguous(), "x tensor must be contiguous");
  TORCH_CHECK(w.is_contiguous(), "w tensor must be contiguous");
  TORCH_CHECK(hc_scale.is_contiguous(), "hc_scale tensor must be contiguous");
  TORCH_CHECK(hc_base.is_contiguous(), "hc_base tensor must be contiguous");

  TORCH_CHECK(x.device().is_cuda(), "x tensor's device must be cuda");
  TORCH_CHECK(w.device().is_cuda(), "w tensor's device must be cuda");
  TORCH_CHECK(hc_scale.device().is_cuda(), "hc_scale tensor's device must be cuda");
  TORCH_CHECK(hc_base.device().is_cuda(), "hc_base tensor's device must be cuda");

  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x tensor data type must be bfloat16");
  TORCH_CHECK(w.scalar_type() == torch::kFloat32, "w tensor data type must be float32");
  TORCH_CHECK(hc_scale.scalar_type() == torch::kFloat32,
              "hc_scale tensor data type must be float32");
  TORCH_CHECK(hc_base.scalar_type() == torch::kFloat32, "hc_base tensor data type must be float32");

  TORCH_CHECK(x.dim() == 3, "x tensor's dim must be 3");
  TORCH_CHECK(w.dim() == 2, "w tensor's dim must be 2");
  TORCH_CHECK(hc_scale.dim() == 1, "hc_scale tensor's dim must be 1");
  TORCH_CHECK(hc_base.dim() == 1, "hc_base tensor's dim must be 1");

  int num_batch = x.size(0);
  int hc_mult = x.size(1);
  int hidden_dim = x.size(2);

  TORCH_CHECK(hc_mult == 4, "hc_mult must be 4");
  TORCH_CHECK(hidden_dim == 6144 || hidden_dim == 4096,
              "hidden dim must be 6144 (HYV4 production) or 4096");
  TORCH_CHECK(w.size(0) == hc_mult, "w tensor's first dim must be hc_mult");
  TORCH_CHECK(w.size(1) == hc_mult * hidden_dim,
              "w tensor's second dim must be hc_mult * hidden_dim");
  TORCH_CHECK(hc_scale.size(-1) == 1, "hc_scale tensor's last dim must be 1");
  TORCH_CHECK(hc_base.size(-1) == hc_mult, "hc_base tensor's last dim must be hc_mult");

  const __nv_bfloat16 *rms_weight_ptr = nullptr;
  if (rms_weight.has_value()) {
    const auto &g = rms_weight.value();
    TORCH_CHECK(g.is_contiguous(), "rms_weight tensor must be contiguous");
    TORCH_CHECK(g.device().is_cuda(), "rms_weight tensor's device must be cuda");
    TORCH_CHECK(g.scalar_type() == torch::kBFloat16,
                "rms_weight tensor data type must be bfloat16");
    TORCH_CHECK(g.dim() == 1 && g.size(0) == hidden_dim,
                "rms_weight must be 1-D with hidden_dim elements");
    rms_weight_ptr = reinterpret_cast<const __nv_bfloat16 *>(g.const_data_ptr());
  }

  auto options = x.options();
  torch::Tensor output = torch::empty({num_batch, hidden_dim}, options);

  const auto *x_ptr = reinterpret_cast<const __nv_bfloat16 *>(x.const_data_ptr());
  const auto *w_ptr = reinterpret_cast<const float *>(w.const_data_ptr());
  const auto *hc_scale_ptr = reinterpret_cast<const float *>(hc_scale.const_data_ptr());
  const auto *hc_base_ptr = reinterpret_cast<const float *>(hc_base.const_data_ptr());
  auto *output_ptr = reinterpret_cast<__nv_bfloat16 *>(output.mutable_data_ptr());

  fuse_ihc_head_async(output_ptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch, hc_mult,
                      hidden_dim, norm_eps, hc_eps, stream, rms_weight_ptr, rms_eps,
                      cast_bfloat_for_norm);

  return output;
}

}  // namespace iHC
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "fuse_ihc_pre(Tensor x, Tensor w, Tensor hc_scale, Tensor hc_base, float norm_eps, float "
      "hc_eps, float magnitude, Tensor? rms_weight=None, float rms_eps=0.0, "
      "bool cast_bfloat_for_norm=False) "
      "-> (Tensor output_y, Tensor output_H_post)");
  m.impl("fuse_ihc_pre", torch::kCUDA, &hpc::iHC::fuse_ihc_pre_entry);

  m.def(
      "fuse_ihc_post_pre(Tensor xa, Tensor residual, Tensor H_post_in, Tensor w, "
      "Tensor hc_scale, Tensor hc_base, float norm_eps, float hc_eps, float magnitude, "
      "Tensor? rms_weight=None, float rms_eps=0.0, bool cast_bfloat_for_norm=False) "
      "-> (Tensor y, Tensor z, Tensor H_post_out)");
  m.impl("fuse_ihc_post_pre", torch::kCUDA, &hpc::iHC::fuse_ihc_post_pre_entry);

  m.def("fuse_ihc_post(Tensor x, Tensor residual, Tensor H_post) -> Tensor output");
  m.impl("fuse_ihc_post", torch::kCUDA, &hpc::iHC::fuse_ihc_post_entry);

  m.def(
      "fuse_ihc_head(Tensor x, Tensor w, Tensor hc_scale, Tensor hc_base, float norm_eps, float "
      "hc_eps, Tensor? rms_weight=None, float rms_eps=0.0, bool cast_bfloat_for_norm=False) "
      "-> Tensor output");
  m.impl("fuse_ihc_head", torch::kCUDA, &hpc::iHC::fuse_ihc_head_entry);
}
