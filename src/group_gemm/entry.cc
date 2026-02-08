// Copyright (C) 2026 Tencent.

#include <cuda_runtime_api.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include "tvm_ffi_utils.h"

#include "src/group_gemm/group_gemm.h"

namespace hpc {
namespace group_gemm {

tvm::ffi::Tensor group_gemm_pertensor_fp8_entry(
    const tvm::ffi::TensorView &x, const tvm::ffi::TensorView &weight,
    const tvm::ffi::TensorView &seqlens, const tvm::ffi::TensorView &cu_seqlens,
    const tvm::ffi::TensorView &y_scale, int64_t num_seq_per_group_avg,
    tvm::ffi::Optional<tvm::ffi::Tensor> output,
    tvm::ffi::Optional<tvm::ffi::Tensor> tma_desc) {
  auto stream = TVM_FFI_GET_CUDA_STREAM(x);
  TVM_FFI_CHECK_CUDA(x);
  TVM_FFI_CHECK_CUDA(weight);
  TVM_FFI_CHECK_CUDA(seqlens);
  TVM_FFI_CHECK_CUDA(cu_seqlens);
  TVM_FFI_CHECK_CONTIGUOUS(x);
  TVM_FFI_CHECK_CONTIGUOUS(weight);
  TVM_FFI_ICHECK(seqlens.shape().at(0) == weight.shape().at(0))
      << "seqlens and weight must share the same num_group";
  TVM_FFI_ICHECK(x.shape().at(1) == weight.shape().at(2))
      << "x and weight must share the same k";

  int m = x.shape().at(0);
  int k = x.shape().at(1);
  int n = weight.shape().at(1);
  int num_group = seqlens.shape().at(0);

  auto device = x.device();

  tvm::ffi::Tensor y;
  if (output.has_value()) {
    y = tvm::ffi::Tensor(output.value());
  } else {
    y = tvm_ffi_empty({m, n}, dl_bfloat16, device);
  }

  tvm::ffi::Tensor tmas;
  bool update_tma = true;
  if (tma_desc.has_value()) {
    tmas = tvm::ffi::Tensor(tma_desc.value());
    update_tma = false;
  } else {
    tmas = tvm_ffi_empty({num_group * 2, 128}, x.dtype(), device);
  }

  tvm::ffi::Tensor tiles = tvm_ffi_empty({num_group}, dl_int32, device);
  tvm::ffi::Tensor cu_tiles = tvm_ffi_empty({num_group + 1}, dl_int32, device);

  const auto *x_ptr = x.data_ptr();
  const auto *weight_ptr = weight.data_ptr();
  const auto *seqlens_ptr = seqlens.data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.data_ptr();
  const auto *yscale_ptr = y_scale.data_ptr();
  auto *tmas_ptr = tmas.data_ptr();
  auto *y_ptr = y.data_ptr();

  auto *tiles_ptr = tiles.data_ptr();
  auto *cu_tiles_ptr = cu_tiles.data_ptr();

  group_gemm_pertensor_fp8_async(y_ptr, x_ptr, weight_ptr, seqlens_ptr, cu_seqlens_ptr, yscale_ptr,
                                 tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k,
                                 num_seq_per_group_avg, update_tma, stream);

  return y;
}

tvm::ffi::Tensor group_gemm_blockwise_fp8_entry(
    const tvm::ffi::TensorView &x, const tvm::ffi::TensorView &weight,
    const tvm::ffi::TensorView &seqlens, const tvm::ffi::TensorView &cu_seqlens,
    const tvm::ffi::TensorView &x_scale, const tvm::ffi::TensorView &w_scale,
    int64_t num_seq_per_group_avg, tvm::ffi::Optional<tvm::ffi::Tensor> output,
    tvm::ffi::Optional<tvm::ffi::Tensor> tma_desc) {
  auto stream = TVM_FFI_GET_CUDA_STREAM(x);
  TVM_FFI_CHECK_CUDA(x);
  TVM_FFI_CHECK_CUDA(weight);
  TVM_FFI_CHECK_CUDA(seqlens);
  TVM_FFI_CHECK_CUDA(cu_seqlens);
  TVM_FFI_CHECK_CONTIGUOUS(x);
  TVM_FFI_CHECK_CONTIGUOUS(weight);
  TVM_FFI_ICHECK(seqlens.shape().at(0) == weight.shape().at(0))
      << "seqlens and weight must share the same num_group";
  TVM_FFI_ICHECK(x.shape().at(1) == weight.shape().at(2))
      << "x and weight must share the same k";
  TVM_FFI_ICHECK(w_scale.shape().at(2) % 4 == 0) << "w_scale must be multiple of 4";

  int m = x.shape().at(0);
  int k = x.shape().at(1);
  int n = weight.shape().at(1);
  int m_pad = x_scale.shape().at(1);
  int num_block_k_pad4 = w_scale.shape().at(2);
  int num_group = seqlens.shape().at(0);

  auto device = x.device();

  tvm::ffi::Tensor y;
  if (output.has_value()) {
    y = tvm::ffi::Tensor(output.value());
  } else {
    y = tvm_ffi_empty({m, n}, dl_bfloat16, device);
  }

  tvm::ffi::Tensor tmas;
  bool update_tma = true;
  if (tma_desc.has_value()) {
    tmas = tvm::ffi::Tensor(tma_desc.value());
    update_tma = false;
  } else {
    tmas = tvm_ffi_empty({num_group * 2, 128}, x.dtype(), device);
  }

  tvm::ffi::Tensor tiles = tvm_ffi_empty({num_group}, dl_int32, device);
  tvm::ffi::Tensor cu_tiles = tvm_ffi_empty({num_group + 1}, dl_int32, device);

  const auto *x_ptr = x.data_ptr();
  const auto *weight_ptr = weight.data_ptr();
  const auto *seqlens_ptr = seqlens.data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.data_ptr();
  const auto *xscale_ptr = x_scale.data_ptr();
  const auto *wscale_ptr = w_scale.data_ptr();
  auto *tmas_ptr = tmas.data_ptr();
  auto *y_ptr = y.data_ptr();

  auto *tiles_ptr = tiles.data_ptr();
  auto *cu_tiles_ptr = cu_tiles.data_ptr();

  group_gemm_blockwise_fp8_async(y_ptr, x_ptr, weight_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr,
                                 wscale_ptr, tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k,
                                 m_pad, num_block_k_pad4, num_seq_per_group_avg, update_tma,
                                 stream);

  return y;
}

tvm::ffi::Tensor reformat_x_scale_entry(const tvm::ffi::TensorView &x_scale,
                                         const tvm::ffi::TensorView &seqlens,
                                         const tvm::ffi::TensorView &cu_seqlens,
                                         tvm::ffi::Optional<tvm::ffi::Tensor> out_x_scale,
                                         int64_t num_seq_per_group_avg) {
  auto stream = TVM_FFI_GET_CUDA_STREAM(x_scale);
  TVM_FFI_CHECK_CUDA(x_scale);
  TVM_FFI_CHECK_CUDA(seqlens);
  TVM_FFI_CHECK_CUDA(cu_seqlens);
  TVM_FFI_CHECK_CONTIGUOUS(x_scale);
  TVM_FFI_CHECK_CONTIGUOUS(seqlens);
  TVM_FFI_CHECK_CONTIGUOUS(cu_seqlens);

  int m = x_scale.shape().at(0);
  int n = x_scale.shape().at(1);
  TVM_FFI_ICHECK(n == 16 || n == 32)
      << "n must be 16 or 32(for dsv4 group gemm k=2048 or 4096)";

  int num_group = seqlens.shape().at(0);
  int tilem = 0;
  // careful!!! here logit must be corresponds with group_gemm_blockwise_fp8_async
  if (num_seq_per_group_avg <= 16) {
    tilem = 16;
  } else if (num_seq_per_group_avg <= 32) {
    tilem = 32;
  } else {
    tilem = 64;
  }
  int num_seq_pad_per_group = m / num_group;
  TVM_FFI_ICHECK(num_seq_pad_per_group % tilem == 0)
      << "The sparse pad length of x_scale for each group must be aligned to multiple of "
         "16/32/64 according to num_seq_per_group_avg";

  auto device = x_scale.device();

  tvm::ffi::Tensor output;
  if (out_x_scale.has_value()) {
    output = tvm::ffi::Tensor(out_x_scale.value());
  } else {
    output = tvm_ffi_empty({n, m}, dl_float32, device);
  }

  const auto *xscale_ptr = x_scale.data_ptr();
  const auto *seqlens_ptr = seqlens.data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.data_ptr();
  auto *output_ptr = output.data_ptr();

  reformat_x_scale_async(output_ptr, xscale_ptr, seqlens_ptr, cu_seqlens_ptr, num_group, m, n,
                         tilem, stream);

  return output;
}

}  // namespace group_gemm
}  // namespace hpc

TVM_FFI_DLL_EXPORT_TYPED_FUNC(group_gemm_pertensor_fp8,
                               hpc::group_gemm::group_gemm_pertensor_fp8_entry);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(group_gemm_blockwise_fp8,
                               hpc::group_gemm::group_gemm_blockwise_fp8_entry);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(reformat_x_scale,
                               hpc::group_gemm::reformat_x_scale_entry);
