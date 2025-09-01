// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/gem3/gem3.h"

namespace hpc {
namespace gem3 {

torch::Tensor entry(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v,
                    const torch::Tensor &qscale, const torch::Tensor &kscale) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  TORCH_CHECK(q.is_contiguous(), "q tensor a must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k tensor b must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v tensor b must be contiguous");
  TORCH_CHECK(qscale.is_contiguous(), "q scale tensor b must be contiguous");
  TORCH_CHECK(kscale.is_contiguous(), "k scale tensor b must be contiguous");

  int num_batch = q.size(0);
  int num_seq = q.size(1);
  int num_qk_dim = q.size(2);
  int num_v_dim = v.size(2);

  // TODO(reed): change to empty_like
  // torch::Tensor y = torch::empty_like(v);
  torch::Tensor y = torch::zeros_like(v);

  auto options = q.options().dtype(torch::kUInt64);
  torch::Tensor tmas = torch::empty({6, 16}, options);

  const auto *q_ptr = q.const_data_ptr();
  const auto *k_ptr = k.const_data_ptr();
  const auto *v_ptr = v.const_data_ptr();
  const auto *qscale_ptr = qscale.const_data_ptr();
  const auto *kscale_ptr = kscale.const_data_ptr();

  void *tmas_ptr = tmas.mutable_data_ptr();

  using T = __nv_bfloat16;
  auto *y_ptr = reinterpret_cast<T *>(y.mutable_data_ptr());

  gem3_async(y_ptr, q_ptr, k_ptr, v_ptr, qscale_ptr, kscale_ptr, tmas_ptr, num_batch, num_seq,
             num_qk_dim, num_v_dim, stream);

  return y;
}

}  // namespace gem3
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) { m.def("gem3", &hpc::gem3::entry); }
