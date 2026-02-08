// Copyright (C) 2026 Tencent.

#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include "tvm_ffi_utils.h"

#include "src/attention/decode/decode.h"
#include "src/attention/prefill/prefill.h"

namespace hpc {
namespace attention {

tvm::ffi::Tensor attention_prefill_bf16_entry(const tvm::ffi::TensorView &q,
                                               const tvm::ffi::TensorView &k,
                                               const tvm::ffi::TensorView &v,
                                               const tvm::ffi::TensorView &seqlens_q,
                                               const tvm::ffi::TensorView &cu_seqlens_q,
                                               int64_t max_seqlens_q,
                                               tvm::ffi::Optional<tvm::ffi::Tensor> output) {
  auto stream = TVM_FFI_GET_CUDA_STREAM(q);
  TVM_FFI_CHECK_CUDA(q);
  TVM_FFI_CHECK_CUDA(k);
  TVM_FFI_CHECK_CUDA(v);
  TVM_FFI_CHECK_CUDA(seqlens_q);
  TVM_FFI_CHECK_CUDA(cu_seqlens_q);

  int total_seq_q = q.shape().at(0);
  int num_head_q = q.shape().at(1);
  int num_dim_qk = q.shape().at(2);

  int num_head_kv = v.shape().at(1);
  int num_dim_v = v.shape().at(2);

  int num_batch = seqlens_q.shape().at(0);

  auto device = q.device();
  tvm::ffi::Tensor y;
  if (output.has_value()) {
    y = tvm::ffi::Tensor(output.value());
  } else {
    y = tvm_ffi_empty({total_seq_q, num_head_q, num_dim_v}, dl_bfloat16, device);
  }

  int num_tmas = 4 * num_batch;
  tvm::ffi::Tensor tmas = tvm_ffi_empty({num_tmas, 64}, dl_bfloat16, device);

  const auto *q_ptr = q.data_ptr();
  const auto *k_ptr = k.data_ptr();
  const auto *v_ptr = v.data_ptr();
  const auto *seqlens_q_ptr = seqlens_q.data_ptr();
  const auto *cu_seqlens_q_ptr = cu_seqlens_q.data_ptr();
  void *tmas_ptr = tmas.data_ptr();

  using T = __nv_bfloat16;
  auto *y_ptr = reinterpret_cast<T *>(y.data_ptr());

  int ldQ = q.stride(0);  // num_head_q * num_dim_qk;
  int ldK = k.stride(0);  // num_head_kv * num_dim_qk;
  int ldV = v.stride(0);  // num_head_kv * num_dim_v;
  int ldY = y.stride(0);  // num_head_q * num_dim_v;

  attention_prefill_bf16_async(y_ptr, q_ptr, k_ptr, v_ptr, seqlens_q_ptr, cu_seqlens_q_ptr,
                               tmas_ptr, num_batch, total_seq_q, max_seqlens_q, num_dim_qk,
                               num_dim_v, num_head_q, num_head_kv, ldY, ldQ, ldK, ldV, stream);

  return y;
}

tvm::ffi::Tensor attention_with_kvcache_prefill_bf16_entry(
    const tvm::ffi::TensorView &q, const tvm::ffi::TensorView &kcache,
    const tvm::ffi::TensorView &vcache, const tvm::ffi::TensorView &cu_seqlens_q,
    const tvm::ffi::TensorView &block_ids, const tvm::ffi::TensorView &seqlens_kvcache,
    int64_t max_seqlens_q, tvm::ffi::Optional<tvm::ffi::Tensor> output) {
  auto stream = TVM_FFI_GET_CUDA_STREAM(q);
  TVM_FFI_CHECK_CUDA(q);
  TVM_FFI_CHECK_CUDA(kcache);
  TVM_FFI_CHECK_CUDA(vcache);
  TVM_FFI_CHECK_CUDA(cu_seqlens_q);
  TVM_FFI_CHECK_CUDA(block_ids);
  TVM_FFI_CHECK_CUDA(seqlens_kvcache);

  int total_seq_q = q.shape().at(0);
  int num_head_q = q.shape().at(1);
  int num_dim_qk = q.shape().at(2);

  int num_batch = cu_seqlens_q.shape().at(0) - 1;

  int num_kvcache_blocks = kcache.shape().at(0);
  int block_size = kcache.shape().at(1);

  int num_head_kv = kcache.shape().at(2);
  int num_dim_v = vcache.shape().at(3);

  int num_seq_max_blocks = block_ids.shape().at(1);

  auto device = q.device();
  tvm::ffi::Tensor y;
  if (output.has_value()) {
    y = tvm::ffi::Tensor(output.value());
  } else {
    y = tvm_ffi_empty({total_seq_q, num_head_q, num_dim_v}, dl_bfloat16, device);
  }

  int num_tmas = 2 * num_batch;
  tvm::ffi::Tensor tmas = tvm_ffi_empty({num_tmas, 64}, dl_bfloat16, device);

  const auto *q_ptr = q.data_ptr();
  const auto *kcache_ptr = kcache.data_ptr();
  const auto *vcache_ptr = vcache.data_ptr();
  const auto *cu_seqlens_q_ptr = cu_seqlens_q.data_ptr();
  const auto *block_ids_ptr = block_ids.data_ptr();
  const auto *seqlens_kvcache_ptr = seqlens_kvcache.data_ptr();
  void *tmas_ptr = tmas.data_ptr();

  using T = __nv_bfloat16;
  auto *y_ptr = reinterpret_cast<T *>(y.data_ptr());

  int ldQ = q.stride(0);       // num_head_q * num_dim_qk;
  int ldK = kcache.stride(0);  // num_head_kv * num_dim_qk;
  int ldV = vcache.stride(0);  // num_head_kv * num_dim_v;
  int ldY = y.stride(0);       // num_head_q * num_dim_v;

  attention_with_kvcache_prefill_bf16_async(
      y_ptr, q_ptr, kcache_ptr, vcache_ptr, cu_seqlens_q_ptr, block_ids_ptr, seqlens_kvcache_ptr,
      tmas_ptr, num_batch, total_seq_q, max_seqlens_q, num_dim_qk, num_dim_v, num_head_q,
      num_head_kv, num_kvcache_blocks, block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldV, stream);

  return y;
}

tvm::ffi::Tensor attention_with_kvcache_prefill_fp8_entry(
    const tvm::ffi::TensorView &q, const tvm::ffi::TensorView &kcache,
    const tvm::ffi::TensorView &vcache, const tvm::ffi::TensorView &qkscale,
    const tvm::ffi::TensorView &vscale, const tvm::ffi::TensorView &cu_seqlens_q,
    const tvm::ffi::TensorView &block_ids, const tvm::ffi::TensorView &seqlens_kvcache,
    int64_t max_seqlens_q, tvm::ffi::Optional<tvm::ffi::Tensor> output) {
  auto stream = TVM_FFI_GET_CUDA_STREAM(q);
  TVM_FFI_CHECK_CUDA(q);
  TVM_FFI_CHECK_CUDA(kcache);
  TVM_FFI_CHECK_CUDA(vcache);
  TVM_FFI_CHECK_CUDA(qkscale);
  TVM_FFI_CHECK_CUDA(vscale);
  TVM_FFI_CHECK_CUDA(cu_seqlens_q);
  TVM_FFI_CHECK_CUDA(block_ids);
  TVM_FFI_CHECK_CUDA(seqlens_kvcache);

  int total_seq_q = q.shape().at(0);
  int num_head_q = q.shape().at(1);
  int num_dim_qk = q.shape().at(2);

  int num_batch = cu_seqlens_q.shape().at(0) - 1;

  int num_kvcache_blocks = kcache.shape().at(0);
  int block_size = kcache.shape().at(1);

  int num_head_kv = kcache.shape().at(2);
  int num_dim_v = vcache.shape().at(3);

  int num_seq_max_blocks = block_ids.shape().at(1);

  int max_seqlens_q_pad = qkscale.shape().at(2);

  auto device = q.device();
  tvm::ffi::Tensor y;
  if (output.has_value()) {
    y = tvm::ffi::Tensor(output.value());
  } else {
    y = tvm_ffi_empty({total_seq_q, num_head_q, num_dim_v}, dl_bfloat16, device);
  }

  int num_tmas = 2 * num_batch;
  tvm::ffi::Tensor tmas = tvm_ffi_empty({num_tmas, 64}, dl_bfloat16, device);

  const auto *q_ptr = q.data_ptr();
  const auto *kcache_ptr = kcache.data_ptr();
  const auto *vcache_ptr = vcache.data_ptr();
  const auto *qkscale_ptr = qkscale.data_ptr();
  const auto *vscale_ptr = vscale.data_ptr();
  const auto *cu_seqlens_q_ptr = cu_seqlens_q.data_ptr();
  const auto *block_ids_ptr = block_ids.data_ptr();
  const auto *seqlens_kvcache_ptr = seqlens_kvcache.data_ptr();
  void *tmas_ptr = tmas.data_ptr();

  using T = __nv_bfloat16;
  auto *y_ptr = reinterpret_cast<T *>(y.data_ptr());

  int ldQ = q.stride(0);       // num_head_q * num_dim_qk;
  int ldK = kcache.stride(0);  // num_head_kv * num_dim_qk;
  int ldV = vcache.stride(0);  // num_head_kv * num_dim_v;
  int ldY = y.stride(0);       // num_head_q * num_dim_v;

  attention_with_kvcache_prefill_fp8_async(
      y_ptr, q_ptr, kcache_ptr, vcache_ptr, qkscale_ptr, vscale_ptr, cu_seqlens_q_ptr,
      block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q, max_seqlens_q,
      max_seqlens_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv, num_kvcache_blocks,
      block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldV, stream);

  return y;
}

tvm::ffi::Tensor attention_decode_bf16_entry(const tvm::ffi::TensorView &q,
                                              const tvm::ffi::TensorView &kcache,
                                              const tvm::ffi::TensorView &vcache,
                                              const tvm::ffi::TensorView &block_ids,
                                              const tvm::ffi::TensorView &num_seq_kvcache,
                                              bool new_kv_included, bool use_splitk,
                                              tvm::ffi::Optional<tvm::ffi::Tensor> output) {
  auto stream = TVM_FFI_GET_CUDA_STREAM(q);

  TVM_FFI_CHECK_CUDA(q);
  TVM_FFI_CHECK_CUDA(kcache);
  TVM_FFI_CHECK_CUDA(vcache);
  TVM_FFI_CHECK_CUDA(block_ids);
  TVM_FFI_CHECK_CONTIGUOUS(block_ids);
  TVM_FFI_CHECK_CONTIGUOUS(num_seq_kvcache);

  int num_batch = num_seq_kvcache.shape().at(0);
  int num_seq_q = q.shape().at(0) / num_batch;
  TVM_FFI_ICHECK(num_seq_q == 1) << "num_seq_q must be 1";
  int num_head_q = q.shape().at(1);
  int num_dim_qk = q.shape().at(2);

  int num_kvcache_blocks = kcache.shape().at(0);
  int block_size = kcache.shape().at(1);

  int num_head_k = kcache.shape().at(2);
  int num_head_v = vcache.shape().at(2);
  int num_dim_v = vcache.shape().at(3);

  int num_seq_max_blocks = block_ids.shape().at(1);

  const auto *q_ptr = q.data_ptr();
  auto *kcache_ptr = const_cast<void *>(kcache.data_ptr());
  auto *vcache_ptr = const_cast<void *>(vcache.data_ptr());
  const int *block_ids_ptr = reinterpret_cast<const int *>(block_ids.data_ptr());
  const int *num_seq_kvcache_ptr = reinterpret_cast<const int *>(num_seq_kvcache.data_ptr());

  auto device = q.device();
  tvm::ffi::Tensor y;
  if (output.has_value()) {
    y = tvm::ffi::Tensor(output.value());
  } else {
    y = tvm_ffi_empty({num_batch * num_seq_q, num_head_q, num_dim_v}, dl_bfloat16, device);
  }

  tvm::ffi::Tensor lse;
  tvm::ffi::Tensor split_out;

  // small batch increase splitk number to maximize sm usage.
  // 1. batch <= 32. split one request seqlenk to 16 parts.
  // 2. batch > 32. split one request seqlenk to 4 parts.
  int splitk = 0;
  if (use_splitk) {
    if (num_batch <= 32) {
      splitk = 16;
    } else {
      splitk = 4;
    }
  }

  void *lse_ptr = nullptr;
  void *split_out_ptr = nullptr;

  if (splitk > 0) {
    lse = tvm_ffi_empty({num_batch, splitk, num_head_q}, dl_float32, device);
    split_out =
        tvm_ffi_empty({num_batch, splitk, num_head_q, num_dim_v}, dl_float32, device);
    lse_ptr = lse.data_ptr();
    split_out_ptr = split_out.data_ptr();
  }

  auto *y_ptr = y.data_ptr();

  int ldQ = q.stride(0);  // num_head_q * num_dim_qk;
  int ldK = kcache.stride(0);
  int ldV = vcache.stride(0);
  int ldY = y.stride(0);  // num_head_q * num_dim_v;

  bool running = attention_decode_bf16_async(
      y_ptr, lse_ptr, split_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
      num_seq_kvcache_ptr, new_kv_included, splitk, num_batch, num_head_q, num_head_k, num_head_v,
      num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks, ldY, ldQ, ldK,
      ldV, stream);

  TVM_FFI_ICHECK(running) << "attn decode kernel launch failed!";

  return y;
}

tvm::ffi::Tensor attention_decode_fp8_entry(
    const tvm::ffi::TensorView &q, const tvm::ffi::TensorView &kcache,
    const tvm::ffi::TensorView &vcache, const tvm::ffi::TensorView &block_ids,
    const tvm::ffi::TensorView &num_seq_kvcache, const tvm::ffi::TensorView &qscale,
    const tvm::ffi::TensorView &kscale, const tvm::ffi::TensorView &vscale, bool new_kv_included,
    bool use_splitk, tvm::ffi::Optional<tvm::ffi::Tensor> split_flag,
    tvm::ffi::Optional<tvm::ffi::Tensor> output) {
  auto stream = TVM_FFI_GET_CUDA_STREAM(q);

  TVM_FFI_CHECK_CUDA(q);
  TVM_FFI_CHECK_CUDA(kcache);
  TVM_FFI_CHECK_CUDA(vcache);
  TVM_FFI_CHECK_CUDA(block_ids);
  TVM_FFI_CHECK_CONTIGUOUS(block_ids);
  TVM_FFI_CHECK_CONTIGUOUS(num_seq_kvcache);

  int num_batch = num_seq_kvcache.shape().at(0);
  int num_seq_q = q.shape().at(0) / num_batch;
  TVM_FFI_ICHECK(num_seq_q == 1) << "num_seq_q must be 1";
  int num_head_q = q.shape().at(1);
  int num_dim_qk = q.shape().at(2);

  int num_kvcache_blocks = kcache.shape().at(0);
  int block_size = kcache.shape().at(1);

  int num_head_k = kcache.shape().at(2);
  int num_head_v = vcache.shape().at(2);
  int num_dim_v = vcache.shape().at(3);

  int num_seq_max_blocks = block_ids.shape().at(1);
  int qscale_pad_stride = qscale.stride(0);

  const auto *q_ptr = q.data_ptr();
  auto *kcache_ptr = const_cast<void *>(kcache.data_ptr());
  auto *vcache_ptr = const_cast<void *>(vcache.data_ptr());
  const int *block_ids_ptr = reinterpret_cast<const int *>(block_ids.data_ptr());
  const int *num_seq_kvcache_ptr = reinterpret_cast<const int *>(num_seq_kvcache.data_ptr());
  const float *qscale_ptr = reinterpret_cast<const float *>(qscale.data_ptr());
  const float *kscale_ptr = reinterpret_cast<const float *>(kscale.data_ptr());
  const float *vscale_ptr = reinterpret_cast<const float *>(vscale.data_ptr());

  auto device = q.device();
  tvm::ffi::Tensor y;
  if (output.has_value()) {
    y = tvm::ffi::Tensor(output.value());
  } else {
    y = tvm_ffi_empty({num_batch * num_seq_q, num_head_q, num_dim_v}, dl_bfloat16, device);
  }

  tvm::ffi::Tensor lse;
  tvm::ffi::Tensor split_out_tensor;

  // small batch increase splitk number to maximize sm usage.
  int splitk = 0;
  int splitk_min_len = 0;

  if (use_splitk) {
    if (num_batch <= 32) {
      splitk = 4;
      splitk_min_len = 512;
    } else {
      splitk = 4;
      splitk_min_len = 4096;
    }
  }

  int consumers = 2;
  if (num_batch <= 156) {
    consumers = 2;
  } else if (num_batch <= 234) {
    consumers = 1;
  } else {
    consumers = 2;
  }

  tvm::ffi::Tensor split_flag_tensor;
  if (split_flag.has_value()) {
    split_flag_tensor = tvm::ffi::Tensor(split_flag.value());
  } else {
    split_flag_tensor = tvm_ffi_zeros({num_batch, num_head_k}, dl_int32, device);
  }

  void *lse_ptr = nullptr;
  void *split_out_ptr = nullptr;

  if (splitk > 0) {
    lse = tvm_ffi_empty({num_batch, splitk * consumers, num_head_q}, dl_float32, device);
    split_out_tensor =
        tvm_ffi_empty({num_batch, splitk * consumers, num_head_q, num_dim_v}, dl_float32, device);
    lse_ptr = lse.data_ptr();
    split_out_ptr = split_out_tensor.data_ptr();
  }

  auto *split_flag_ptr = reinterpret_cast<int *>(split_flag_tensor.data_ptr());

  auto *y_ptr = y.data_ptr();

  int ldQ = q.stride(0);  // num_head_q * num_dim_qk;
  int ldK = kcache.stride(0);
  int ldV = vcache.stride(0);
  int ldY = y.stride(0);  // num_head_q * num_dim_v;

  bool running = attention_decode_fp8_async(
      y_ptr, lse_ptr, split_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
      num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr, new_kv_included,
      splitk, splitk_min_len, consumers, num_batch, num_head_q, num_head_k, num_head_v, num_dim_qk,
      num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks, qscale_pad_stride, ldY, ldQ,
      ldK, ldV, stream);

  TVM_FFI_ICHECK(running) << "attn decode kernel launch failed!";

  return y;
}

}  // namespace attention
}  // namespace hpc

TVM_FFI_DLL_EXPORT_TYPED_FUNC(attention_prefill_bf16,
                               hpc::attention::attention_prefill_bf16_entry);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(attention_with_kvcache_prefill_bf16,
                               hpc::attention::attention_with_kvcache_prefill_bf16_entry);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(attention_with_kvcache_prefill_fp8,
                               hpc::attention::attention_with_kvcache_prefill_fp8_entry);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(attention_decode_bf16,
                               hpc::attention::attention_decode_bf16_entry);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(attention_decode_fp8,
                               hpc::attention::attention_decode_fp8_entry);
