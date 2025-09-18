// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm>
#include <tuple>

#include "src/mamba/conv1d_state.h"
#include "src/mamba/selective_state.h"
#include "src/mamba/selective_state_scan.h"

namespace hpc {
namespace mamba {

constexpr int kTmaDescCount = 11;

torch::Tensor selective_state_update(torch::Tensor &ssm_states, const torch::Tensor &zxbcdt,
                                     const torch::Tensor &AD, const torch::Tensor &bias,
                                     const torch::Tensor &indices, int64_t num_group,
                                     int64_t num_sp_tokens,
                                     c10::optional<torch::Tensor> num_accepted_tokens) {
  auto stream = at::cuda::getCurrentCUDAStream(ssm_states.get_device());

  TORCH_CHECK(ssm_states.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(zxbcdt.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(indices.is_contiguous(), "input tensor must be contiguous");

  [[maybe_unused]] int num_max_batch = ssm_states.size(0);
  int num_batch = indices.size(0);

  int num_head = ssm_states.size(1);
  int head_dim = ssm_states.size(2);
  int state_dim = ssm_states.size(3);

  const int *num_accepted_tokens_ptr = nullptr;
  if (num_sp_tokens > 1) {
    TORCH_CHECK(ssm_states.size(1) == num_sp_tokens,
                "in speculative sampling mode, ssm_states.size(1) == num_sp_tokens");
    TORCH_CHECK(zxbcdt.size(0) == num_batch * num_sp_tokens,
                "in speculative sampling mode, zxbcdt.size(0) == num_batch * "
                "num_sp_tokens");
    TORCH_CHECK(num_accepted_tokens.has_value(),
                "input tensor num_accepted_tokens must be provided");
    TORCH_CHECK(num_accepted_tokens.value().is_contiguous(),
                "input tensor num_accepted_tokens must be contiguous");

    num_head = ssm_states.size(2);
    head_dim = ssm_states.size(3);
    state_dim = ssm_states.size(4);

    num_accepted_tokens_ptr = num_accepted_tokens.value().const_data_ptr<int>();
  }

  using T = __nv_bfloat16;
  torch::Tensor out =
      torch::empty({num_batch * num_sp_tokens, num_head * head_dim}, zxbcdt.options());

  auto *ssm_states_ptr = ssm_states.mutable_data_ptr<float>();
  auto *out_ptr = reinterpret_cast<T *>(out.mutable_data_ptr());

  const auto *zxbcdt_ptr = reinterpret_cast<const T *>(zxbcdt.const_data_ptr());

  const auto *AD_ptr = AD.const_data_ptr<float>();
  const auto *bias_ptr = bias.const_data_ptr<float>();
  const auto *indices_ptr = indices.const_data_ptr<int>();

  TORCH_CHECK(state_dim == 128, "we only support state_dim == 128");
  // we hard code the 4 in the kernel
  TORCH_CHECK(num_head == num_group * 4, "we only support num_head == num_group * 4");

  selective_state_update_async(out_ptr, ssm_states_ptr, indices_ptr, zxbcdt_ptr, AD_ptr, bias_ptr,
                               num_batch, num_head, head_dim, num_group, state_dim, num_sp_tokens,
                               num_accepted_tokens_ptr, stream);

  return out;
}

void causal_conv1d_update_entry(torch::Tensor &zxbcdt, torch::Tensor &conv_states,
                                const torch::Tensor &weight, const torch::Tensor &bias,
                                const torch::Tensor &indices, int64_t d_inner, int64_t num_head,
                                int64_t num_spec_tokens,
                                c10::optional<torch::Tensor> num_accept_tokens) {
  auto stream = at::cuda::getCurrentCUDAStream(conv_states.get_device());
  TORCH_CHECK(zxbcdt.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(conv_states.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(indices.is_contiguous(), "input tensor must be contiguous");

  using T = __nv_bfloat16;

  auto *zxbcdt_ptr = reinterpret_cast<T *>(zxbcdt.mutable_data_ptr());
  auto *conv_states_ptr = reinterpret_cast<T *>(conv_states.mutable_data_ptr());

  const auto *weight_ptr = reinterpret_cast<const T *>(weight.const_data_ptr());
  const auto *bias_ptr = reinterpret_cast<const T *>(bias.const_data_ptr());
  const auto *indices_ptr = indices.const_data_ptr<int>();

  int num_batch = indices.size(0);
  int num_tokens = zxbcdt.size(0);
  int conv_dim = conv_states.size(2);
  int state_len = conv_states.size(1);

  TORCH_CHECK(conv_dim == weight.size(1), "weight.size(1) should be equal to conv_dim");
  TORCH_CHECK(num_head == 8 || num_head == 4, "num_head must be 8 or 4");

  int d_conv = weight.size(0);

  const int *num_accept_tokens_ptr = nullptr;
  if (num_accept_tokens.has_value()) {
    TORCH_CHECK(num_spec_tokens == 2, "we only support spec_total_tokens == 2");
    TORCH_CHECK(state_len == d_conv);
    TORCH_CHECK(num_accept_tokens.value().size(0) == num_batch);
    TORCH_CHECK(num_batch * num_spec_tokens == num_tokens);
    num_accept_tokens_ptr = (const int *)(num_accept_tokens.value().const_data_ptr());
  } else {
    TORCH_CHECK(state_len == d_conv - 1);
    TORCH_CHECK(num_batch == num_tokens);
  }

  causal_conv1d_update_async(zxbcdt_ptr, conv_states_ptr, weight_ptr, bias_ptr, indices_ptr,
                             num_batch, state_len, d_conv, conv_dim, d_inner, num_head,
                             num_spec_tokens, num_accept_tokens_ptr, stream);
}

void causal_conv1d_prefill_entry(torch::Tensor &y, torch::Tensor &zxbcdt,
                                 torch::Tensor &conv_states, const torch::Tensor &weight,
                                 const torch::Tensor &bias, const torch::Tensor &indices,
                                 const torch::Tensor &split_metadata, const torch::Tensor &x_scale,
                                 const torch::Tensor &y_scale, int64_t chunk_size,
                                 int64_t total_chunks, int64_t d_inner, int64_t num_head) {
  auto stream = at::cuda::getCurrentCUDAStream(zxbcdt.get_device());
  constexpr int kDConv = 4;
  constexpr int kChunkSize = 256;
  constexpr int kTileL = 32;
  TORCH_CHECK(zxbcdt.is_contiguous(), "zxbcdt tensor must be contiguous");
  TORCH_CHECK(conv_states.is_contiguous(), "conv_states tensor must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor must be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "bias tensor must be contiguous");
  TORCH_CHECK(indices.is_contiguous(), "indices tensor must be contiguous");
  TORCH_CHECK(chunk_size == kChunkSize, "chunk_size must be ", kChunkSize);

  using T = __nv_bfloat16;

  int num_batch = indices.size(0);
  int conv_dim = conv_states.size(2);
  int state_len = conv_states.size(1);
  int total_padded_seqlen = x_scale.size(1);

  TORCH_CHECK(state_len + 1 == kDConv, "kDConv must be", kDConv);

  torch::Tensor middle_y =
      torch::empty({total_chunks * (kChunkSize / kTileL), kDConv - 1, conv_dim}, zxbcdt.options());

  auto *y_ptr = reinterpret_cast<T *>(y.mutable_data_ptr());
  auto *zxbcdt_ptr = reinterpret_cast<T *>(zxbcdt.mutable_data_ptr());
  auto *conv_states_ptr = reinterpret_cast<T *>(conv_states.mutable_data_ptr());

  const auto *weight_ptr = reinterpret_cast<const T *>(weight.const_data_ptr());
  const auto *bias_ptr = reinterpret_cast<const T *>(bias.const_data_ptr());
  const auto *indices_ptr = indices.const_data_ptr<int>();
  auto *middle_y_ptr = reinterpret_cast<T *>(middle_y.mutable_data_ptr());
  const float *x_scale_ptr = x_scale.const_data_ptr<float>();
  const float *y_scale_ptr = y_scale.const_data_ptr<float>();
  const int *split_metadata_ptr = split_metadata.const_data_ptr<int>();

  auto running = causal_conv1d_prefill_async(
      middle_y_ptr, y_ptr, zxbcdt_ptr, conv_states_ptr, weight_ptr, bias_ptr, indices_ptr,
      split_metadata_ptr, x_scale_ptr, y_scale_ptr, num_batch, total_chunks, total_padded_seqlen,
      state_len, conv_dim, d_inner, num_head, kDConv, kChunkSize, kTileL, stream);
  TORCH_CHECK(running, "launch causal_conv1d_prefill_async failed!");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t, torch::Tensor, torch::Tensor>
exp_dA_chunked_cumsum_entry(const torch::Tensor &zxbcdt, const torch::Tensor &A,
                            const torch::Tensor &dt_bias, const torch::Tensor &host_seqlens,
                            torch::Tensor &host_split_metadata, const int64_t &chunk_size,
                            const int64_t &head_dim, const int64_t &ngroups,
                            const int64_t &dstate) {
  auto stream = at::cuda::getCurrentCUDAStream(zxbcdt.get_device());

  constexpr int kChunkSize = 256;
  TORCH_CHECK(zxbcdt.is_contiguous(), "zxbcdt tensor must be contiguous");
  TORCH_CHECK(A.is_contiguous(), "A tensor must be contiguous");
  TORCH_CHECK(dt_bias.is_contiguous(), "dt_bias tensor must be contiguous");
  TORCH_CHECK(host_split_metadata.is_contiguous(), "host_split_metadata tensor must be contiguous");
  TORCH_CHECK(host_split_metadata.is_pinned(), "host_split_metadata tensor must be pinned");
  TORCH_CHECK(chunk_size == kChunkSize, "chunk_size must be ", kChunkSize);

  const int batch_size = host_seqlens.size(0);
  const int nheads = A.size(0);
  const int total_seqlen = zxbcdt.size(0);
  const int zxbcdt_row_stride = zxbcdt.size(1);

  TORCH_CHECK(nheads == 8, "nheads must be ", 8);

  torch::Tensor split_metadata =
      torch::empty({6, batch_size + 1}, torch::dtype(torch::kInt32).device(zxbcdt.device()));

  torch::Tensor tma_desc = torch::empty({batch_size * kTmaDescCount, sizeof(CUtensorMap)},
                                        torch::dtype(torch::kInt8).device(zxbcdt.device()));

  torch::Tensor y = torch::empty({total_seqlen, nheads * head_dim}, zxbcdt.options());

  constexpr int kCumsumSeqlenIdx = 0;
  constexpr int kCumsumPaddedSeqlenIdx = 1;
  constexpr int kCumsumChunksIdx = 2;
  constexpr int kSeqlenIdx = 3;
  constexpr int kPaddedSeqlenIdx = 4;
  constexpr int kNumChunksIdx = 5;

  int cusum_seqlens = 0;
  int cumsum_padded_seqlens = 0;
  int cusum_chunks = 0;

  host_split_metadata[kCumsumSeqlenIdx][0] = 0;
  host_split_metadata[kCumsumPaddedSeqlenIdx][0] = 0;
  host_split_metadata[kCumsumChunksIdx][0] = 0;
  host_split_metadata[kSeqlenIdx][batch_size] = 0;
  host_split_metadata[kPaddedSeqlenIdx][batch_size] = 0;
  host_split_metadata[kNumChunksIdx][batch_size] = 0;

  int max_chunks = 0;
  const auto *host_seqlens_ptr = host_seqlens.const_data_ptr<int>();
  for (int i = 0; i < batch_size; i++) {
    int seqlen = host_seqlens_ptr[i];
    int padded_seqlen = ((seqlen + 4 - 1) / 4) * 4;
    int nchunk = (seqlen + kChunkSize - 1) / kChunkSize;

    cusum_seqlens += seqlen;
    cumsum_padded_seqlens += padded_seqlen;
    cusum_chunks += nchunk;

    host_split_metadata[kCumsumSeqlenIdx][i + 1] = cusum_seqlens;
    host_split_metadata[kCumsumPaddedSeqlenIdx][i + 1] = cumsum_padded_seqlens;
    host_split_metadata[kCumsumChunksIdx][i + 1] = cusum_chunks;
    host_split_metadata[kSeqlenIdx][i] = seqlen;
    host_split_metadata[kPaddedSeqlenIdx][i] = padded_seqlen;
    host_split_metadata[kNumChunksIdx][i] = nchunk;

    max_chunks = std::max(max_chunks, nchunk);
  }

  torch::Tensor yscale = torch::empty({nheads, cumsum_padded_seqlens},
                                      torch::dtype(torch::kFloat32).device(zxbcdt.device()));

  torch::Tensor xscale = torch::empty({nheads, cumsum_padded_seqlens},
                                      torch::dtype(torch::kFloat32).device(zxbcdt.device()));

  auto *split_metadata_ptr = split_metadata.mutable_data_ptr<int>();
  const auto *host_split_metadata_ptr = host_split_metadata.mutable_data_ptr();
  auto *yscale_ptr = yscale.mutable_data_ptr<float>();
  auto *xscale_ptr = xscale.mutable_data_ptr<float>();
  auto *tma_desc_ptr = tma_desc.mutable_data_ptr();
  const auto *y_ptr = y.const_data_ptr();
  const auto *zxbcdt_ptr = zxbcdt.const_data_ptr();
  const auto *A_ptr = A.const_data_ptr<float>();
  const auto *dt_bias_ptr = dt_bias.const_data_ptr<float>();

  cudaMemcpyAsync(split_metadata_ptr, host_split_metadata_ptr, sizeof(int) * 6 * (batch_size + 1),
                  cudaMemcpyHostToDevice, stream);

  auto running = exp_dA_chunked_cumsum_async(
      yscale_ptr, xscale_ptr, tma_desc_ptr, y_ptr, zxbcdt_ptr, A_ptr, dt_bias_ptr,
      split_metadata_ptr, chunk_size, max_chunks, batch_size, nheads, head_dim, ngroups, dstate,
      zxbcdt_row_stride, zxbcdt_row_stride - nheads, cumsum_padded_seqlens, kTmaDescCount, stream);

  TORCH_CHECK(running, "launch exp_dA_chunked_cumsum_async failed!");

  return std::make_tuple(yscale, xscale, split_metadata, cusum_chunks, tma_desc, y);
}

torch::Tensor chunk_states_bmm_entry(const torch::Tensor &zxbcdt, const torch::Tensor &scaled_x,
                                     const torch::Tensor &split_metadata,
                                     const torch::Tensor &tma_desc, int64_t total_chunks,
                                     int64_t nheads, int64_t ngroups, int64_t head_dim,
                                     int64_t dstate) {
  auto stream = at::cuda::getCurrentCUDAStream(zxbcdt.get_device());

  constexpr int kChunkSize = 256;
  TORCH_CHECK(zxbcdt.is_contiguous(), "zxbcdt tensor must be contiguous");
  TORCH_CHECK(split_metadata.is_contiguous(), "split_metadata tensor must be contiguous");
  TORCH_CHECK(tma_desc.is_contiguous(), "tma_desc tensor must be contiguous");
  TORCH_CHECK(head_dim == 80, "only support head_dim == 80");
  TORCH_CHECK(dstate == 128, "only support dstate == 128");

  const int zxbcdt_row_stride = zxbcdt.size(1);
  const int batch_size = split_metadata.size(1) - 1;

  torch::Tensor chunked_states =
      torch::empty({nheads, total_chunks, head_dim, dstate},
                   torch::dtype(torch::kFloat32).device(zxbcdt.device()));

  auto *chunked_states_ptr = chunked_states.mutable_data_ptr<float>();
  const auto *zxbcdt_ptr = zxbcdt.const_data_ptr();
  const auto *scaled_x_ptr = scaled_x.const_data_ptr();
  const auto *tma_desc_ptr = tma_desc.const_data_ptr();
  const auto *split_metadata_ptr = split_metadata.const_data_ptr<int>();

  auto running = chunk_states_bmm_async(chunked_states_ptr, zxbcdt_ptr, scaled_x_ptr, tma_desc_ptr,
                                        split_metadata_ptr, batch_size, total_chunks, nheads,
                                        ngroups, head_dim, dstate, zxbcdt_row_stride, kChunkSize,
                                        kTmaDescCount, stream);

  TORCH_CHECK(running, "launch chunk_states_bmm_async failed!");
  return chunked_states;
}

torch::Tensor chunk_states_passing_entry(const torch::Tensor &chunked_states,
                                         const torch::Tensor &yscale,
                                         const torch::Tensor &split_metadata,
                                         torch::Tensor &ssm_states, const torch::Tensor &indices,
                                         const int64_t &chunk_size) {
  auto stream = at::cuda::getCurrentCUDAStream(chunked_states.get_device());

  constexpr int kChunkSize = 256;
  TORCH_CHECK(chunked_states.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(chunked_states.scalar_type() == torch::kFloat32, "chunked_states dtype must be fp32");
  TORCH_CHECK(yscale.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(split_metadata.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(chunk_size == kChunkSize, "chunk_size must be ", kChunkSize);

  const int batch_size = indices.size(0);
  const int nheads = chunked_states.size(0);
  const int total_chunks = chunked_states.size(1);
  const int head_dim = chunked_states.size(2);
  const int dstate = chunked_states.size(3);
  const int total_padded_seqlen = yscale.size(1);

  torch::Tensor chunked_states_cumsum =
      torch::empty({nheads, total_chunks - batch_size, head_dim, dstate},
                   torch::dtype(torch::kBFloat16).device(chunked_states.device()));

  auto *chunked_states_cumsum_ptr = chunked_states_cumsum.mutable_data_ptr();
  auto *ssm_states_ptr = ssm_states.mutable_data_ptr<float>();
  const auto *indices_ptr = indices.const_data_ptr<int>();
  const auto *chunked_states_ptr = chunked_states.const_data_ptr<float>();
  const auto *yscale_ptr = yscale.const_data_ptr<float>();
  const auto *split_metadata_ptr = split_metadata.const_data_ptr<int>();

  auto running = chunk_states_passing_async(chunked_states_cumsum_ptr, ssm_states_ptr, indices_ptr,
                                            chunked_states_ptr, yscale_ptr, split_metadata_ptr,
                                            chunk_size, total_padded_seqlen, total_chunks,
                                            batch_size, nheads, head_dim, dstate, stream);
  TORCH_CHECK(running, "launch chunk_states_passing_async failed!");
  return chunked_states_cumsum;
}

torch::Tensor pre_y_bmm_entry(const torch::Tensor &chunked_states_cumsum,
                              const torch::Tensor &zxbcdt, const torch::Tensor &split_metadata,
                              const torch::Tensor &tma_desc, const int64_t &ngroups,
                              const int64_t &chunk_size) {
  auto stream = at::cuda::getCurrentCUDAStream(zxbcdt.get_device());

  constexpr int kChunkSize = 256;
  TORCH_CHECK(chunked_states_cumsum.is_contiguous(),
              "chunked_states_cumsum tensor must be contiguous");
  TORCH_CHECK(zxbcdt.is_contiguous(), "zxbcdt tensor must be contiguous");
  TORCH_CHECK(split_metadata.is_contiguous(), "split_metadata tensor must be contiguous");
  TORCH_CHECK(tma_desc.is_contiguous(), "tma_desc tensor must be contiguous");
  TORCH_CHECK(chunk_size == kChunkSize, "chunk_size must be ", kChunkSize);

  const int batch_size = split_metadata.size(1) - 1;
  const int nheads = chunked_states_cumsum.size(0);
  const int total_chunks = chunked_states_cumsum.size(1);
  const int head_dim = chunked_states_cumsum.size(2);
  const int dstate = chunked_states_cumsum.size(3);
  const int zxbcdt_row_stride = zxbcdt.size(1);

  TORCH_CHECK(head_dim == 80, "only support head_dim == 80");
  TORCH_CHECK(dstate == 128, "only support dstate == 128");

  torch::Tensor pre_y = torch::empty({nheads, total_chunks, kChunkSize, head_dim},
                                     torch::dtype(torch::kFloat32).device(zxbcdt.device()));

  if (total_chunks > 0) {
    auto *pre_y_ptr = pre_y.mutable_data_ptr<float>();
    const auto *zxbcdt_ptr = zxbcdt.data_ptr();
    const auto *chunked_states_cumsum_ptr = chunked_states_cumsum.data_ptr();
    const auto *tma_desc_ptr = tma_desc.data_ptr();
    const auto *split_metadata_ptr = split_metadata.data_ptr<int>();
    auto running =
        pre_y_bmm_async(pre_y_ptr, zxbcdt_ptr, chunked_states_cumsum_ptr, tma_desc_ptr,
                        split_metadata_ptr, batch_size, total_chunks, nheads, ngroups, head_dim,
                        dstate, zxbcdt_row_stride, kChunkSize, kTmaDescCount, stream);
    TORCH_CHECK(running, "launch pre_y_bmm_async failed!");
  }
  return pre_y;
}

torch::Tensor chunk_scan_gem3_entry(torch::Tensor &y, const torch::Tensor &zxbcdt,
                                    const torch::Tensor &pre_y, const torch::Tensor &xscale,
                                    const torch::Tensor &yscale, const torch::Tensor &D,
                                    const torch::Tensor &split_metadata,
                                    const torch::Tensor &tma_desc, const int64_t &ngroups,
                                    const int64_t &dstate, const int64_t &chunk_size) {
  auto stream = at::cuda::getCurrentCUDAStream(zxbcdt.get_device());

  constexpr int kChunkSize = 256;
  TORCH_CHECK(y.is_contiguous(), "y tensor must be contiguous");
  TORCH_CHECK(zxbcdt.is_contiguous(), "zxbcdt tensor must be contiguous");
  TORCH_CHECK(pre_y.is_contiguous(), "pre_y tensor must be contiguous");
  TORCH_CHECK(split_metadata.is_contiguous(), "split_metadata tensor must be contiguous");
  TORCH_CHECK(tma_desc.is_contiguous(), "tma_desc tensor must be contiguous");
  TORCH_CHECK(chunk_size == kChunkSize, "chunk_size must be ", kChunkSize);

  const int batch_size = split_metadata.size(1) - 1;
  const int nheads = pre_y.size(0);
  const int total_chunks = pre_y.size(1) + batch_size;
  const int head_dim = pre_y.size(3);
  const int zxbcdt_row_stride = zxbcdt.size(1);
  const int total_padded_seqlen = xscale.size(1);

  TORCH_CHECK(head_dim == 80, "only support head_dim == 80");
  TORCH_CHECK(dstate == 128, "only support dstate == 128");

  auto *y_ptr = y.mutable_data_ptr();
  const auto *zxbcdt_ptr = zxbcdt.data_ptr();
  const auto *pre_y_ptr = pre_y.data_ptr();
  const auto *xscale_ptr = xscale.data_ptr<float>();
  const auto *yscale_ptr = yscale.data_ptr<float>();
  const auto *D_ptr = D.data_ptr<float>();
  const auto *tma_desc_ptr = tma_desc.data_ptr();
  const auto *split_metadata_ptr = split_metadata.data_ptr<int>();

  auto running = chunk_scan_gem3_async(y_ptr, zxbcdt_ptr, pre_y_ptr, xscale_ptr, yscale_ptr, D_ptr,
                                       tma_desc_ptr, split_metadata_ptr, batch_size, total_chunks,
                                       total_padded_seqlen, nheads, ngroups, head_dim, dstate,
                                       zxbcdt_row_stride, kChunkSize, kTmaDescCount, stream);

  TORCH_CHECK(running, "launch chunk_scan_gem3_async failed!");

  return y;
}

torch::Tensor mamba_prefill_entry(torch::Tensor &zxbcdt, torch::Tensor &conv_states,
                                  torch::Tensor &ssm_states, const torch::Tensor &indices,
                                  const torch::Tensor &host_seqlens,
                                  const torch::Tensor &host_split_metadata,
                                  const torch::Tensor &conv_weight, const torch::Tensor &conv_bias,
                                  const torch::Tensor &A, const torch::Tensor &D,
                                  const torch::Tensor &dt_bias, const int64_t &ngroups,
                                  const int64_t &chunk_size) {
  auto stream = at::cuda::getCurrentCUDAStream(zxbcdt.get_device());
  constexpr int kChunkSize = 256;
  constexpr int kTileL = 32;
  constexpr int kDConv = 4;

  TORCH_CHECK(zxbcdt.is_contiguous(), "zxbcdt tensor must be contiguous");
  TORCH_CHECK(conv_states.is_contiguous(), "conv_states tensor must be contiguous");
  TORCH_CHECK(ssm_states.is_contiguous(), "ssm_states tensor must be contiguous");
  TORCH_CHECK(indices.is_contiguous(), "indices tensor must be contiguous");
  TORCH_CHECK(host_split_metadata.is_contiguous(), "host_split_metadata tensor must be contiguous");
  TORCH_CHECK(host_split_metadata.is_pinned(), "host_split_metadata tensor must be pinned");
  TORCH_CHECK(conv_weight.is_contiguous(), "conv_weight tensor must be contiguous");
  TORCH_CHECK(conv_bias.is_contiguous(), "conv_bias tensor must be contiguous");
  TORCH_CHECK(A.is_contiguous(), "A tensor must be contiguous");
  TORCH_CHECK(D.is_contiguous(), "D tensor must be contiguous");
  TORCH_CHECK(dt_bias.is_contiguous(), "dt_bias tensor must be contiguous");

  using T = __nv_bfloat16;

  const int batch_size = indices.size(0);
  const int nheads = A.size(0);
  const int total_seqlen = zxbcdt.size(0);
  const int head_dim = ssm_states.size(2);
  const int dstate = ssm_states.size(3);
  const int zxbcdt_row_stride = zxbcdt.size(1);
  const int conv_dim = conv_states.size(2);
  const int state_len = conv_states.size(1);

  TORCH_CHECK(state_len + 1 == kDConv, "kDConv must be", kDConv);
  TORCH_CHECK(chunk_size == kChunkSize, "chunk_size must be ", kChunkSize);
  TORCH_CHECK(head_dim == 80, "only support head_dim == 80");
  TORCH_CHECK(dstate == 128, "only support dstate == 128");

  // 0. compute split metadata on cpu
  constexpr int kCumsumSeqlenIdx = 0;
  constexpr int kCumsumPaddedSeqlenIdx = 1;
  constexpr int kCumsumChunksIdx = 2;
  constexpr int kSeqlenIdx = 3;
  constexpr int kPaddedSeqlenIdx = 4;
  constexpr int kNumChunksIdx = 5;

  int cusum_seqlens = 0;
  int cumsum_padded_seqlens = 0;
  int cusum_chunks = 0;

  host_split_metadata[kCumsumSeqlenIdx][0] = 0;
  host_split_metadata[kCumsumPaddedSeqlenIdx][0] = 0;
  host_split_metadata[kCumsumChunksIdx][0] = 0;
  host_split_metadata[kSeqlenIdx][batch_size] = 0;
  host_split_metadata[kPaddedSeqlenIdx][batch_size] = 0;
  host_split_metadata[kNumChunksIdx][batch_size] = 0;

  int max_chunks = 0;
  const auto *host_seqlens_ptr = host_seqlens.const_data_ptr<int>();
  for (int i = 0; i < batch_size; i++) {
    int seqlen = host_seqlens_ptr[i];
    int padded_seqlen = ((seqlen + 4 - 1) / 4) * 4;
    int nchunk = (seqlen + kChunkSize - 1) / kChunkSize;

    cusum_seqlens += seqlen;
    cumsum_padded_seqlens += padded_seqlen;
    cusum_chunks += nchunk;

    host_split_metadata[kCumsumSeqlenIdx][i + 1] = cusum_seqlens;
    host_split_metadata[kCumsumPaddedSeqlenIdx][i + 1] = cumsum_padded_seqlens;
    host_split_metadata[kCumsumChunksIdx][i + 1] = cusum_chunks;
    host_split_metadata[kSeqlenIdx][i] = seqlen;
    host_split_metadata[kPaddedSeqlenIdx][i] = padded_seqlen;
    host_split_metadata[kNumChunksIdx][i] = nchunk;

    max_chunks = std::max(max_chunks, nchunk);
  }
  torch::Tensor split_metadata =
      torch::empty({6, batch_size + 1}, torch::dtype(torch::kInt32).device(zxbcdt.device()));
  torch::Tensor tma_desc = torch::empty({batch_size * kTmaDescCount, sizeof(CUtensorMap)},
                                        torch::dtype(torch::kInt8).device(zxbcdt.device()));
  torch::Tensor yscale = torch::empty({nheads, cumsum_padded_seqlens},
                                      torch::dtype(torch::kFloat32).device(zxbcdt.device()));
  torch::Tensor xscale = torch::empty({nheads, cumsum_padded_seqlens},
                                      torch::dtype(torch::kFloat32).device(zxbcdt.device()));
  torch::Tensor middle_y =
      torch::empty({cusum_chunks * (kChunkSize / kTileL), kDConv - 1, conv_dim}, zxbcdt.options());
  torch::Tensor chunked_states =
      torch::empty({nheads, cusum_chunks, head_dim, dstate},
                   torch::dtype(torch::kFloat32).device(zxbcdt.device()));
  torch::Tensor chunked_states_cumsum =
      torch::empty({nheads, cusum_chunks - batch_size, head_dim, dstate},
                   torch::dtype(torch::kBFloat16).device(zxbcdt.device()));
  torch::Tensor y0 = torch::empty({nheads, cusum_chunks - batch_size, kChunkSize, head_dim},
                                  torch::dtype(torch::kFloat32).device(zxbcdt.device()));
  torch::Tensor y = torch::empty({total_seqlen, nheads * head_dim}, zxbcdt.options());

  auto *zxbcdt_ptr = reinterpret_cast<T *>(zxbcdt.mutable_data_ptr());
  const auto *A_ptr = A.const_data_ptr<float>();
  const auto *D_ptr = D.const_data_ptr<float>();
  const auto *dt_bias_ptr = dt_bias.const_data_ptr<float>();
  const auto *conv_weight_ptr = reinterpret_cast<const T *>(conv_weight.const_data_ptr());
  const auto *conv_bias_ptr = reinterpret_cast<const T *>(conv_bias.const_data_ptr());
  auto *conv_states_ptr = reinterpret_cast<T *>(conv_states.mutable_data_ptr());

  auto *ssm_states_ptr = reinterpret_cast<float *>(ssm_states.mutable_data_ptr());
  const auto *indices_ptr = indices.const_data_ptr<int>();
  auto *split_metadata_ptr = split_metadata.mutable_data_ptr<int>();
  auto *yscale_ptr = reinterpret_cast<float *>(yscale.mutable_data_ptr());
  auto *xscale_ptr = reinterpret_cast<float *>(xscale.mutable_data_ptr());
  auto *tma_desc_ptr = tma_desc.mutable_data_ptr();
  auto *middle_y_ptr = reinterpret_cast<T *>(middle_y.mutable_data_ptr());
  auto *chunked_states_ptr = chunked_states.mutable_data_ptr<float>();
  auto *chunked_states_cumsum_ptr = reinterpret_cast<T *>(chunked_states_cumsum.mutable_data_ptr());
  auto *y0_ptr = y0.mutable_data_ptr<float>();
  auto *y_ptr = y.mutable_data_ptr();

  // 1. transfer split metadata to gpu.
  cudaMemcpyAsync(split_metadata_ptr, host_split_metadata.data_ptr(),
                  sizeof(int) * 6 * (batch_size + 1), cudaMemcpyHostToDevice, stream);

  // 2. compute dA cumsum, x/y scale and update tma desc.
  auto running = exp_dA_chunked_cumsum_async(
      yscale_ptr, xscale_ptr, tma_desc_ptr, y_ptr, zxbcdt_ptr, A_ptr, dt_bias_ptr,
      split_metadata_ptr, chunk_size, max_chunks, batch_size, nheads, head_dim, ngroups, dstate,
      zxbcdt_row_stride, zxbcdt_row_stride - nheads, cumsum_padded_seqlens, kTmaDescCount, stream);
  TORCH_CHECK(running, "launch exp_dA_chunked_cumsum_async failed!");

  // 3. compute conv1d, update conv_states and xbc in place, and output scalex.
  running = causal_conv1d_prefill_async(
      middle_y_ptr, reinterpret_cast<T *>(y_ptr), zxbcdt_ptr, conv_states_ptr, conv_weight_ptr,
      conv_bias_ptr, indices_ptr, split_metadata_ptr, xscale_ptr, yscale_ptr, batch_size,
      cusum_chunks, cumsum_padded_seqlens, state_len, conv_dim, nheads * head_dim, nheads, kDConv,
      kChunkSize, kTileL, stream);
  TORCH_CHECK(running, "launch causal_conv1d_prefill_async failed!");

  // 4. compute chunk states
  running = chunk_states_bmm_async(chunked_states_ptr, zxbcdt_ptr, y_ptr, tma_desc_ptr,
                                   split_metadata_ptr, batch_size, cusum_chunks, nheads, ngroups,
                                   head_dim, dstate, zxbcdt_row_stride, kChunkSize, kTmaDescCount,
                                   stream);
  TORCH_CHECK(running, "launch chunk_states_bmm_async failed!");

  // 5. chunk states passing and compute final ssm_states
  running = chunk_states_passing_async(chunked_states_cumsum_ptr, ssm_states_ptr, indices_ptr,
                                       chunked_states_ptr, yscale_ptr, split_metadata_ptr,
                                       chunk_size, cumsum_padded_seqlens, cusum_chunks, batch_size,
                                       nheads, head_dim, dstate, stream);
  TORCH_CHECK(running, "launch chunk_states_passing_async failed!");

  // 6. compute y0
  if (cusum_chunks - batch_size > 0) {
    running =
        pre_y_bmm_async(y0_ptr, zxbcdt_ptr, chunked_states_cumsum_ptr, tma_desc_ptr,
                        split_metadata_ptr, batch_size, cusum_chunks - batch_size, nheads, ngroups,
                        head_dim, dstate, zxbcdt_row_stride, kChunkSize, kTmaDescCount, stream);
    TORCH_CHECK(running, "launch pre_y_bmm_async failed!");
  }

  // 7. compute chunk scan gem3 get final output
  running = chunk_scan_gem3_async(y_ptr, zxbcdt_ptr, y0_ptr, xscale_ptr, yscale_ptr, D_ptr,
                                  tma_desc_ptr, split_metadata_ptr, batch_size, cusum_chunks,
                                  cumsum_padded_seqlens, nheads, ngroups, head_dim, dstate,
                                  zxbcdt_row_stride, kChunkSize, kTmaDescCount, stream);
  TORCH_CHECK(running, "launch chunk_scan_gem3_async failed!");

  return y;
}

}  // namespace mamba
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def("selective_state_update", &hpc::mamba::selective_state_update);
}

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def("causal_conv1d_update", &hpc::mamba::causal_conv1d_update_entry);
}

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def("exp_dA_chunked_cumsum", &hpc::mamba::exp_dA_chunked_cumsum_entry);
  m.def("causal_conv1d_prefill", &hpc::mamba::causal_conv1d_prefill_entry);
  m.def("chunk_states_bmm", &hpc::mamba::chunk_states_bmm_entry);
  m.def("chunk_states_passing", &hpc::mamba::chunk_states_passing_entry);
  m.def("pre_y_bmm", &hpc::mamba::pre_y_bmm_entry);
  m.def("chunk_scan_gem3", &hpc::mamba::chunk_scan_gem3_entry);
  m.def("mamba_prefill", &hpc::mamba::mamba_prefill_entry);
}
