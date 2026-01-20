// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

#include "src/mHC/mHC.h"

namespace hpc {
namespace mHC {

torch::Tensor fuse_cal_mixes_hat_hat_H_and_r_entry(const torch::Tensor &input,
                                                   const torch::Tensor &weight_a,
                                                   const torch::Tensor &weight_b, double norm_eps) {
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(weight_a.is_contiguous(), "weight_a tensor must be contiguous");
  TORCH_CHECK(weight_b.is_contiguous(), "weight_b tensor must be contiguous");

  TORCH_CHECK(input.device().is_cuda(), "input tensor's device must be cuda");
  TORCH_CHECK(weight_a.device().is_cuda(), "weight_a tensor's device must be cuda");
  TORCH_CHECK(weight_b.device().is_cuda(), "weight_b tensor's device must be cuda");

  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input tensor data type must be bfloat16");
  TORCH_CHECK(weight_a.scalar_type() == torch::kBFloat16,
              "weight_a tensor data type must be bfloat16");
  TORCH_CHECK(weight_b.scalar_type() == torch::kBFloat16,
              "weight_b tensor data type must be bfloat16");

  TORCH_CHECK(input.dim() == 2, "input tensor's dim must be 2");
  TORCH_CHECK(weight_a.dim() == 2, "weight_a tensor's dim must be 2");
  TORCH_CHECK(weight_b.dim() == 2, "weight_b tensor's dim must be 2");

  TORCH_CHECK(input.size(1) == weight_a.size(1), "input and weight_a must be same in last dim");
  TORCH_CHECK(input.size(1) == weight_b.size(1), "input and weight_b must be same in last dim");
  TORCH_CHECK(input.size(1) == 16384, "only support hc dim is 16384");

  int num_batch = input.size(0);
  int hidden_dim = input.size(1);

  auto options = input.options();
  torch::Tensor norm_x = torch::empty({num_batch, hidden_dim}, options);

  const auto *input_ptr = reinterpret_cast<const __nv_bfloat16 *>(input.const_data_ptr());
  auto *output_ptr = reinterpret_cast<__nv_bfloat16 *>(norm_x.mutable_data_ptr());

  reciprocal_mean_square_root_norm_async(output_ptr, input_ptr, num_batch, hidden_dim, norm_eps,
                                         stream);

  // TODO(lando): impl fuse two bf16 gemm here

  return norm_x;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fuse_cal_three_H_entry(
    const torch::Tensor &mixes_hat_hat_H, const torch::Tensor &hc_scale,
    const torch::Tensor &hc_base, int64_t hc_mult, int64_t hc_sinkhorn_iters, double hc_eps) {
  auto stream = at::cuda::getCurrentCUDAStream(mixes_hat_hat_H.get_device());

  TORCH_CHECK(mixes_hat_hat_H.is_contiguous(), "mixes_hat_hat_H tensor must be contiguous");
  TORCH_CHECK(hc_scale.is_contiguous(), "hc_scale tensor must be contiguous");
  TORCH_CHECK(hc_base.is_contiguous(), "hc_base tensor must be contiguous");

  TORCH_CHECK(mixes_hat_hat_H.device().is_cuda(), "mixes_hat_hat_H tensor's device must be cuda");
  TORCH_CHECK(hc_scale.device().is_cuda(), "hc_scale tensor's device must be cuda");
  TORCH_CHECK(hc_base.device().is_cuda(), "hc_base tensor's device must be cuda");

  TORCH_CHECK(mixes_hat_hat_H.scalar_type() == torch::kFloat32,
              "mixes_hat_hat_H tensor data type must be float32");
  TORCH_CHECK(hc_scale.scalar_type() == torch::kFloat32,
              "hc_scale tensor data type must be float32");
  TORCH_CHECK(hc_base.scalar_type() == torch::kFloat32, "hc_base tensor data type must be float32");

  TORCH_CHECK(mixes_hat_hat_H.dim() == 2, "mixes_hat_hat_H tensor's dim must be 2");
  TORCH_CHECK(hc_scale.dim() == 1, "hc_scale tensor's dim must be 1");
  TORCH_CHECK(hc_base.dim() == 1, "hc_base tensor's dim must be 1");

  int num_batch = mixes_hat_hat_H.size(0);
  int hc_dim = 2 * hc_mult + hc_mult * hc_mult;
  TORCH_CHECK(hc_mult == 4, "hc_mult must be 4");
  TORCH_CHECK(mixes_hat_hat_H.size(-1) == hc_dim,
              "mixes_hat_hat_H tensor's last dim must be hc_dim");
  TORCH_CHECK(hc_scale.size(-1) == 3, "hc_scale tensor's last dim must be 3");
  TORCH_CHECK(hc_base.size(-1) == hc_dim, "hc_base tensor's last dim must be hc_dim");

  auto options = mixes_hat_hat_H.options();
  torch::Tensor output_H_pre = torch::empty({num_batch, hc_mult}, options);
  torch::Tensor output_H_post = torch::empty({num_batch, hc_mult}, options);
  torch::Tensor output_H_res = torch::empty({num_batch, hc_mult, hc_mult}, options);

  const auto *mixes_hat_hat_H_ptr =
      reinterpret_cast<const float *>(mixes_hat_hat_H.const_data_ptr());
  const auto *hc_scale_ptr = reinterpret_cast<const float *>(hc_scale.const_data_ptr());
  const auto *hc_base_ptr = reinterpret_cast<const float *>(hc_base.const_data_ptr());
  auto *output_H_pre_ptr = reinterpret_cast<float *>(output_H_pre.mutable_data_ptr());
  auto *output_H_post_ptr = reinterpret_cast<float *>(output_H_post.mutable_data_ptr());
  auto *output_H_res_ptr = reinterpret_cast<float *>(output_H_res.mutable_data_ptr());

  fuse_cal_three_H_async(output_H_pre_ptr, output_H_post_ptr, output_H_res_ptr, mixes_hat_hat_H_ptr,
                         hc_scale_ptr, hc_base_ptr, num_batch, hc_mult, hc_sinkhorn_iters, hc_eps,
                         stream);

  return std::make_tuple(output_H_pre, output_H_post, output_H_res);
}

torch::Tensor fuse_hc_pre_mapping_entry(const torch::Tensor &x, const torch::Tensor &H_pre) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  TORCH_CHECK(x.is_contiguous(), "x tensor must be contiguous");
  TORCH_CHECK(H_pre.is_contiguous(), "H_pre tensor must be contiguous");

  TORCH_CHECK(x.device().is_cuda(), "x tensor's device must be cuda");
  TORCH_CHECK(H_pre.device().is_cuda(), "H_pre tensor's device must be cuda");

  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x tensor data type must be bfloat16");
  TORCH_CHECK(H_pre.scalar_type() == torch::kFloat32, "H_pre tensor data type must be float32");

  TORCH_CHECK(x.dim() == 3, "x tensor's dim must be 3");
  TORCH_CHECK(H_pre.dim() == 2, "H_pre tensor's dim must be 2");

  int num_batch = x.size(0);
  int hc_mult = x.size(1);
  int hidden_dim = x.size(2);

  TORCH_CHECK(hc_mult == 4, "hc_mult must be 4");
  TORCH_CHECK(hidden_dim == 4096, "hidden dim must be 4096");
  TORCH_CHECK(H_pre.size(0) == num_batch, "H_pre tensor's first dim must be num_batch");
  TORCH_CHECK(H_pre.size(1) == hc_mult, "H_pre tensor's second dim must be hc_mult");

  auto options = x.options();
  torch::Tensor output = torch::empty({num_batch, hidden_dim}, options);

  const auto *x_ptr = reinterpret_cast<const __nv_bfloat16 *>(x.const_data_ptr());
  const auto *H_pre_ptr = reinterpret_cast<const float *>(H_pre.const_data_ptr());
  auto *output_ptr = reinterpret_cast<__nv_bfloat16 *>(output.mutable_data_ptr());

  fuse_hc_pre_mapping_async(output_ptr, x_ptr, H_pre_ptr, num_batch, hc_mult, hidden_dim, stream);

  return output;
}

torch::Tensor fuse_H_post_mapping_H_res_mapping_and_residual_add_entry(
    const torch::Tensor &x, const torch::Tensor &residual, const torch::Tensor &H_post,
    const torch::Tensor &H_res) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  TORCH_CHECK(x.is_contiguous(), "x tensor must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual tensor must be contiguous");
  TORCH_CHECK(H_post.is_contiguous(), "H_post tensor must be contiguous");
  TORCH_CHECK(H_res.is_contiguous(), "H_res tensor must be contiguous");

  TORCH_CHECK(x.device().is_cuda(), "x tensor's device must be cuda");
  TORCH_CHECK(residual.device().is_cuda(), "residual tensor's device must be cuda");
  TORCH_CHECK(H_post.device().is_cuda(), "H_post tensor's device must be cuda");
  TORCH_CHECK(H_res.device().is_cuda(), "H_res tensor's device must be cuda");

  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x tensor data type must be bfloat16");
  TORCH_CHECK(residual.scalar_type() == torch::kBFloat16,
              "residual tensor data type must be bfloat16");
  TORCH_CHECK(H_post.scalar_type() == torch::kFloat32, "H_post tensor data type must be float32");
  TORCH_CHECK(H_res.scalar_type() == torch::kFloat32, "H_res tensor data type must be float32");

  TORCH_CHECK(x.dim() == 2, "x tensor's dim must be 2");
  TORCH_CHECK(residual.dim() == 3, "residual tensor's dim must be 3");
  TORCH_CHECK(H_post.dim() == 2, "H_post tensor's dim must be 2");
  TORCH_CHECK(H_res.dim() == 3, "H_res tensor's dim must be 3");

  int num_batch = residual.size(0);
  int hc_mult = residual.size(1);
  int hidden_dim = residual.size(2);

  TORCH_CHECK(hc_mult == 4, "hc_mult must be 4");
  TORCH_CHECK(hidden_dim == 4096, "hidden dim must be 4096");
  TORCH_CHECK(x.size(0) == num_batch, "x tensor's first dim must be num_batch");
  TORCH_CHECK(x.size(1) == hidden_dim, "x tensor's second dim must be hidden_dim");
  TORCH_CHECK(H_post.size(0) == num_batch, "H_post tensor's first dim must be num_batch");
  TORCH_CHECK(H_post.size(1) == hc_mult, "H_post tensor's second dim must be hc_mult");
  TORCH_CHECK(H_res.size(0) == num_batch, "H_res tensor's first dim must be num_batch");
  TORCH_CHECK(H_res.size(1) == hc_mult, "H_res tensor's second dim must be hc_mult");
  TORCH_CHECK(H_res.size(2) == hc_mult, "H_res tensor's third dim must be hc_mult");

  auto options = x.options();
  torch::Tensor output = torch::empty({num_batch, hc_mult, hidden_dim}, options);

  const auto *x_ptr = reinterpret_cast<const __nv_bfloat16 *>(x.const_data_ptr());
  const auto *residual_ptr = reinterpret_cast<const __nv_bfloat16 *>(residual.const_data_ptr());
  const auto *H_post_ptr = reinterpret_cast<const float *>(H_post.const_data_ptr());
  const auto *H_res_ptr = reinterpret_cast<const float *>(H_res.const_data_ptr());
  auto *output_ptr = reinterpret_cast<__nv_bfloat16 *>(output.mutable_data_ptr());

  fuse_H_post_mapping_H_res_mapping_and_residual_add_async(output_ptr, x_ptr, residual_ptr,
                                                           H_post_ptr, H_res_ptr, num_batch,
                                                           hc_mult, hidden_dim, stream);

  return output;
}

}  // namespace mHC
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "fuse_cal_mixes_hat_hat_H_and_r(Tensor input, Tensor weight_a, Tensor weight_b, float "
      "norm_eps) -> "
      "Tensor output_mixes");
  m.impl("fuse_cal_mixes_hat_hat_H_and_r", torch::kCUDA,
         &hpc::mHC::fuse_cal_mixes_hat_hat_H_and_r_entry);

  m.def(
      "fuse_cal_three_H(Tensor mixes_hat_hat_H, Tensor hc_scale, Tensor hc_base, int hc_mult, int "
      "hc_sinkhorn_iters, float hc_eps) -> "
      "(Tensor output_pre, Tensor output_post, Tensor output_res)");
  m.impl("fuse_cal_three_H", torch::kCUDA, &hpc::mHC::fuse_cal_three_H_entry);

  m.def("fuse_hc_pre_mapping(Tensor x, Tensor H_pre) -> Tensor output");
  m.impl("fuse_hc_pre_mapping", torch::kCUDA, &hpc::mHC::fuse_hc_pre_mapping_entry);

  m.def(
      "fuse_H_post_mapping_H_res_mapping_and_residual_add(Tensor x, Tensor residual, Tensor "
      "H_post, Tensor H_res) -> Tensor output");
  m.impl("fuse_H_post_mapping_H_res_mapping_and_residual_add", torch::kCUDA,
         &hpc::mHC::fuse_H_post_mapping_H_res_mapping_and_residual_add_entry);
}
