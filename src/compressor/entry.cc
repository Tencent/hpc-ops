// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <optional>
#include <tuple>

#include "src/compressor/compressor.h"

namespace hpc {
namespace compressor {

torch::Tensor kv_compressor_entry(const torch::Tensor &kv, const torch::Tensor &score,
                                  const torch::Tensor &cu_seqlens,
                                  const torch::Tensor &cu_compressed_seqlens,
                                  int64_t total_compressed_seqlen, torch::Tensor &kv_states,
                                  torch::Tensor &score_states, const torch::Tensor &state_index,
                                  const torch::Tensor &start_pos, const torch::Tensor &ape,
                                  int64_t ratio, bool overlap, int64_t head_dim, bool is_prefill) {
  auto stream = at::cuda::getCurrentCUDAStream(kv.get_device());
  // kcache and vcache maybe not contiguous, we access them by stride
  TORCH_CHECK(kv.is_contiguous(), "kv tensor must be contiguous");
  TORCH_CHECK(score.is_contiguous(), "score tensor must be contiguous");
  TORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens tensor must be contiguous");
  TORCH_CHECK(cu_compressed_seqlens.is_contiguous(),
              "cu_compressed_seqlens tensor must be contiguous");
  TORCH_CHECK(ape.is_contiguous(), "ape tensor must be contiguous");

  // type check
  using DType = float;
  TORCH_CHECK(kv.scalar_type() == torch::kFloat32, "kv tensor data type must be float32");
  TORCH_CHECK(score.scalar_type() == torch::kFloat32, "score tensor data type must be float32");
  TORCH_CHECK(cu_seqlens.scalar_type() == torch::kInt32,
              "cu_seqlens tensor data type must be int32");
  TORCH_CHECK(cu_compressed_seqlens.scalar_type() == torch::kInt32,
              "cu_compressed_seqlens tensor data type must be int32");
  TORCH_CHECK(kv_states.scalar_type() == torch::kFloat32,
              "kv_states tensor data type must be float32");
  TORCH_CHECK(score_states.scalar_type() == torch::kFloat32,
              "score_states tensor data type must be float32");
  TORCH_CHECK(ape.scalar_type() == torch::kFloat32, "ape tensor data type must be float32");
  TORCH_CHECK(state_index.scalar_type() == torch::kInt32,
              "state_index tensor data type must be int32");
  TORCH_CHECK(start_pos.scalar_type() == torch::kInt32, "start_pos tensor data type must be int32");

  // Get dimensions from input tensors
  int num_batch = cu_seqlens.size(0) - 1;
  int total_seqlen = kv.size(0);
  int hidden_size = kv.size(1);

  if (overlap) {
    TORCH_CHECK(hidden_size == head_dim * 2, "if overlap, kv.size(1) must be head_dim*2");
    TORCH_CHECK(ratio == 4, "if overlap, ratio must be 4");
  }

  // Create output tensors or use provided ones
  torch::Tensor compressed_kv = torch::empty({total_compressed_seqlen, head_dim},
                                             torch::dtype(kv.dtype()).device(kv.device()));

  // Prepare pointers
  auto *compressed_kv_ptr = compressed_kv.mutable_data_ptr<DType>();
  const auto *kv_ptr = kv.const_data_ptr<DType>();
  const auto *score_ptr = score.const_data_ptr<DType>();
  const auto *ape_ptr = ape.const_data_ptr<DType>();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr<int>();
  const auto *cu_compressed_seqlens_ptr = cu_compressed_seqlens.const_data_ptr<int>();
  const auto *state_index_ptr = state_index.const_data_ptr<int>();
  const auto *start_pos_ptr = start_pos.const_data_ptr<int>();
  auto *kv_states_ptr = kv_states.mutable_data_ptr<DType>();
  auto *score_states_ptr = score_states.mutable_data_ptr<DType>();

  // Launch kernel
  bool running = kv_compressor_fp32_async(
      compressed_kv_ptr, kv_ptr, score_ptr, cu_seqlens_ptr, cu_compressed_seqlens_ptr,
      kv_states_ptr, score_states_ptr, state_index_ptr, start_pos_ptr, ape_ptr, num_batch,
      total_seqlen, ratio, overlap, head_dim, is_prefill, stream);
  TORCH_CHECK(running, "kv compressor fp32 launch failed!");
  return compressed_kv;
}

torch::Tensor kv_compressor_decode_entry(const torch::Tensor &kv, const torch::Tensor &score,
                                         const torch::Tensor &ape, torch::Tensor &kv_states,
                                         torch::Tensor &score_states,
                                         const torch::Tensor &state_idx,
                                         const torch::Tensor &start_pos,
                                         const torch::Tensor &cu_compress_seqlens,
                                         const int64_t head_dim, const int64_t ratio, bool overlap,
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

  const auto *kv_ptr = kv.const_data_ptr();
  const auto *score_ptr = score.const_data_ptr();
  const auto *ape_ptr = ape.const_data_ptr();
  auto *kv_states_ptr = kv_states.mutable_data_ptr();
  auto *score_states_ptr = score_states.mutable_data_ptr();
  const auto *state_idx_ptr = state_idx.const_data_ptr();
  const auto *start_pos_ptr = start_pos.const_data_ptr();
  const auto *cu_compress_seqlens_ptr = cu_compress_seqlens.const_data_ptr();

  if (output.has_value()) {
    auto *y_ptr = output.value().mutable_data_ptr();
    kv_compressor_decode_async(y_ptr, kv_ptr, score_ptr, ape_ptr, kv_states_ptr, score_states_ptr,
                               state_idx_ptr, start_pos_ptr, cu_compress_seqlens_ptr, batch_size,
                               head_dim, ratio, mtp, stream);
    return output.value();
  } else {
    auto options = kv.options();
    torch::Tensor y = torch::empty({batch_size, head_dim}, options);
    y.fill_(0);
    auto *y_ptr = y.mutable_data_ptr();
    kv_compressor_decode_async(y_ptr, kv_ptr, score_ptr, ape_ptr, kv_states_ptr, score_states_ptr,
                               state_idx_ptr, start_pos_ptr, cu_compress_seqlens_ptr, batch_size,
                               head_dim, ratio, mtp, stream);
    return y;
  }
}

}  // namespace compressor
}  // namespace hpc

// Register the function with optional output tensors
TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "kv_compressor(Tensor kv, Tensor score, Tensor cu_seqlens, Tensor cu_compressed_seqlens, int "
      "total_compressed_seqlen, "
      "Tensor! kv_states, Tensor! score_states, Tensor state_index, Tensor start_pos, Tensor ape, "
      "int ratio, bool overlap, int head_dim, bool is_prefill) -> (Tensor)");
  m.impl("kv_compressor", torch::kCUDA, &hpc::compressor::kv_compressor_entry);
}
TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "kv_compressor_decode(Tensor kv, Tensor score, Tensor ape, Tensor kv_states,"
      "Tensor score_states, Tensor state_idx, Tensor start_pos, Tensor cu_compress_seqlens,"
      "int head_dim, int ration, bool overlap, Tensor ? output) -> (Tensor)");
  m.impl("kv_compressor_decode", torch::kCUDA, &hpc::compressor::kv_compressor_decode_entry);
}
