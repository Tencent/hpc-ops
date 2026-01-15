// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/kv_compressor_decode/kv_compressor_decode.h"

namespace hpc {
namespace kv_compressor_decode {

torch::Tensor kv_compressor_decode(const torch::Tensor& kv, const torch::Tensor& score,
                                   const torch::Tensor& ape, torch::Tensor& kv_states,
                                   torch::Tensor& score_states, const torch::Tensor& state_idx,
                                   const torch::Tensor& start_pos,
                                   const torch::Tensor& cu_compress_seqlens, const int64_t head_dim,
                                   const int64_t ratio, bool overlap,
                                   std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(kv.get_device());

  TORCH_CHECK(kv.dtype() == torch::kFloat32, "kv dtype must be float32");
  TORCH_CHECK(score.dtype() == torch::kFloat32, "score dtype must be float32");
  TORCH_CHECK(ape.dtype() == torch::kFloat32, "ape dtype must be float32");
  TORCH_CHECK(kv_states.dtype() == torch::kFloat32, "kv_states dtype must be float32");
  TORCH_CHECK(score_states.dtype() == torch::kFloat32, "score_states dtype must be float32");
  TORCH_CHECK(state_idx.dtype() == torch::kInt32, "score_states dtype must be int32");
  TORCH_CHECK(start_pos.dtype() == torch::kInt32, "start_pos dtype must be int32");
  TORCH_CHECK(cu_compress_seqlens.dtype() == torch::kInt32,
              "cu_compress_seqlens dtype must be int32");

  TORCH_CHECK(kv.dim() == 2, "kv dim must be 2");
  TORCH_CHECK(score.dim() == kv.dim(), "score dim must be same as kv dim");
  TORCH_CHECK(kv.size(0) == score.size(0) && kv.size(1) == score.size(1),
              "score must have same shape with kv")

  TORCH_CHECK(kv_states.dim() == score_states.dim() && kv_states.dim() == 3,
              "kv_states and score_states dim must be  3");
  TORCH_CHECK(score_states.size(0) == kv_states.size(0) &&
                  score_states.size(1) == kv_states.size(1) &&
                  score_states.size(2) == kv_states.size(2),
              "score_states shape must be equal to kv_states");

  TORCH_CHECK(kv_states.size(-1) == kv.size(-1), "kv_states and kv must have same dim");

  // mtp
  int batch_size = cu_compress_seqlens.size(0) - 1;
  TORCH_CHECK(kv.size(0) % batch_size == 0, "kv.size(0) % batch_size == 0 must be true");
  int mtp = kv.size(0) / batch_size - 1;
  TORCH_CHECK(mtp == 1 || mtp == 0, "when speculative decoding is open, only support mtp = 1 now");

  TORCH_CHECK(state_idx.numel() == batch_size, "state_idx len must be batch_size");
  TORCH_CHECK(start_pos.numel() == batch_size, "start_pos len must be batch_size");
  TORCH_CHECK(cu_compress_seqlens.numel() == batch_size + 1, "ratio len must be batch_size + 1");

  int coff = 1;
  if (overlap) {
    TORCH_CHECK(ratio == 4, "ratio must be 4 when overlap is true");
    TORCH_CHECK(head_dim == 128 || head_dim == 512, "head_dim must be 128 or 512 when ratio is 4");
    coff = 2;
  } else {
    TORCH_CHECK(ratio == 128, "ratio must be 128 when overlap is false");
    TORCH_CHECK(head_dim == 512, "head_dim must be 512 when ratio is 128");
  }
  TORCH_CHECK(head_dim == 512 || head_dim == 128,
              "now only suport head_dim == 512 or head_dim == 128");
  TORCH_CHECK(kv.size(-1) == coff * head_dim, "kv.size(-1) must be (overlap + 1) * head_dim");
  TORCH_CHECK(kv_states.size(0) >= batch_size, "kv_states.size(0) must be greater than batch_size");
  TORCH_CHECK(kv_states.size(1) == coff * ratio, "kv_states.size(1) must be (overlap + 1) * ratio");
  TORCH_CHECK(kv_states.size(2) == coff * head_dim,
              "kv_states.size(2) must be (overlap + 1) * head_dim");
  TORCH_CHECK(ape.size(0) == ratio, "ape.size(0) must be equal to ratio");
  TORCH_CHECK(ape.size(1) == coff * head_dim,
              "ape.size(1) must be equal to (overlap + 1) * head_dim");

  const auto* kv_ptr = kv.const_data_ptr();
  const auto* score_ptr = score.const_data_ptr();
  const auto* ape_ptr = ape.const_data_ptr();
  auto* kv_states_ptr = kv_states.mutable_data_ptr();
  auto* score_states_ptr = score_states.mutable_data_ptr();
  const auto* state_idx_ptr = state_idx.const_data_ptr();
  const auto* start_pos_ptr = start_pos.const_data_ptr();
  const auto* cu_compress_seqlens_ptr = cu_compress_seqlens.const_data_ptr();

  if (output.has_value()) {
    auto* y_ptr = output.value().mutable_data_ptr();
    kv_compressor_decode_async(y_ptr, kv_ptr, score_ptr, ape_ptr, kv_states_ptr, score_states_ptr,
                               state_idx_ptr, start_pos_ptr, cu_compress_seqlens_ptr, batch_size,
                               head_dim, ratio, mtp, stream);
    return output.value();
  } else {
    auto options = kv.options();
    torch::Tensor y = torch::empty({batch_size, head_dim}, options);
    y.fill_(0);
    auto* y_ptr = y.mutable_data_ptr();
    kv_compressor_decode_async(y_ptr, kv_ptr, score_ptr, ape_ptr, kv_states_ptr, score_states_ptr,
                               state_idx_ptr, start_pos_ptr, cu_compress_seqlens_ptr, batch_size,
                               head_dim, ratio, mtp, stream);
    return y;
  }
}
}  // namespace kv_compressor_decode
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "kv_compressor_decode(Tensor kv, Tensor score, Tensor ape, Tensor kv_states,"
      "Tensor score_states, Tensor state_idx, Tensor start_pos, Tensor cu_compress_seqlens,"
      "int head_dim, int ration, bool overlap, Tensor ? output) -> (Tensor)");
  m.impl("kv_compressor_decode", torch::kCUDA, &hpc::kv_compressor_decode::kv_compressor_decode);
}
