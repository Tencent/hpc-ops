#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/mamba/conv1d_state.h"
#include "src/mamba/selective_state.h"

namespace hpc {
namespace mamba {

torch::Tensor selective_state_update(torch::Tensor& ssm_states,
                                     const torch::Tensor& zxbcdt,
                                     const torch::Tensor& AD,
                                     const torch::Tensor& bias,
                                     const torch::Tensor& indices,
                                     int64_t num_group) {
  auto stream = at::cuda::getCurrentCUDAStream(ssm_states.get_device());

  TORCH_CHECK(ssm_states.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(zxbcdt.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(indices.is_contiguous(), "input tensor must be contiguous");

  [[maybe_unused]] int num_max_batch = ssm_states.size(0);
  int num_head = ssm_states.size(1);
  int head_dim = ssm_states.size(2);
  int state_dim = ssm_states.size(3);
  int num_batch = zxbcdt.size(0);

  using T = __nv_bfloat16;
  torch::Tensor out =
      torch::empty({num_batch, num_head * head_dim}, zxbcdt.options());

  auto* ssm_states_ptr = ssm_states.mutable_data_ptr<float>();
  auto* out_ptr = (T*)out.mutable_data_ptr();

  const auto* zxbcdt_ptr = (T*)zxbcdt.const_data_ptr();

  const auto* AD_ptr = AD.const_data_ptr<float>();
  const auto* bias_ptr = bias.const_data_ptr<float>();
  const auto* indices_ptr = indices.const_data_ptr<int>();

  TORCH_CHECK(state_dim == 128, "we only support state_dim == 128");
  // we hard code the 4 in the kernel
  TORCH_CHECK(num_head == num_group * 4,
              "we only support num_head == num_group * 4");

  selective_state_update_async(out_ptr, ssm_states_ptr, indices_ptr, zxbcdt_ptr,
                               AD_ptr, bias_ptr, num_batch, num_head, head_dim,
                               num_group, state_dim, stream);

  return out;
}

void causal_conv1d_update_entry(torch::Tensor& zxbcdt,
                                torch::Tensor& conv_states,
                                const torch::Tensor& weight,
                                const torch::Tensor& bias,
                                const torch::Tensor& indices, int64_t d_inner,
                                int64_t num_head) {
  auto stream = at::cuda::getCurrentCUDAStream(conv_states.get_device());
  TORCH_CHECK(zxbcdt.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(conv_states.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(indices.is_contiguous(), "input tensor must be contiguous");

  using T = __nv_bfloat16;

  auto* zxbcdt_ptr = (T*)zxbcdt.const_data_ptr();
  auto* conv_states_ptr = (T*)conv_states.mutable_data_ptr();

  const auto* weight_ptr = (T*)weight.const_data_ptr();
  const auto* bias_ptr = (T*)bias.const_data_ptr();
  const auto* indices_ptr = indices.const_data_ptr<int>();

  int num_batch = zxbcdt.size(0);
  TORCH_CHECK(num_batch == indices.size(0),
              "indice.size(0) should be equal to num_batch");

  int conv_dim = conv_states.size(2);
  int state_len = conv_states.size(1);
  TORCH_CHECK(conv_dim == weight.size(1),
              "weight.size(1) should be equal to conv_dim");

  int d_conv = weight.size(0);
  causal_conv1d_update_async(zxbcdt_ptr, conv_states_ptr, weight_ptr, bias_ptr,
                             indices_ptr, num_batch, state_len, d_conv,
                             conv_dim, d_inner, num_head, stream);
}

}  // namespace mamba
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def("selective_state_update", &hpc::mamba::selective_state_update);
}

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def("causal_conv1d_update", &hpc::mamba::causal_conv1d_update_entry);
}
