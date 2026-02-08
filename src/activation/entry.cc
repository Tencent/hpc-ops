// Copyright (C) 2026 Tencent.

#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <tvm/ffi/container/tuple.h>
#include <tuple>
#include <vector>

#include "tvm_ffi_utils.h"

#include "src/activation/activation.h"

namespace hpc {
namespace activation {

tvm::ffi::Tensor act_mul_and_quant_entry(const tvm::ffi::TensorView &input,
                                          const tvm::ffi::TensorView &scale, bool use_bf16_mul,
                                          tvm::ffi::Optional<tvm::ffi::Tensor> output) {
  auto stream = TVM_FFI_GET_CUDA_STREAM(input);
  auto device = input.device();

  int ndim = input.ndim();
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; ++i) {
    output_shape.push_back(input.shape().at(i));
  }
  output_shape[ndim - 1] /= 2;

  tvm::ffi::Tensor output_tensor;
  if (output.has_value()) {
    output_tensor = tvm::ffi::Tensor(output.value());
  } else {
    output_tensor = tvm_ffi_empty(output_shape, dl_float8_e4m3, device);
  }

  using Tin = __nv_bfloat16;
  using Tout = __nv_fp8_e4m3;

  const auto *input_ptr = reinterpret_cast<const Tin *>(input.data_ptr());
  auto *output_ptr = reinterpret_cast<Tout *>(output_tensor.data_ptr());
  const float *scale_ptr = reinterpret_cast<const float *>(scale.data_ptr());

  int num_col = input.shape().at(ndim - 1);
  int num_row = 1;
  for (int i = 0; i < ndim - 1; ++i) {
    num_row *= input.shape().at(i);
  }

  act_mul_and_quant_async(output_ptr, input_ptr, scale_ptr, num_row, num_col, use_bf16_mul, stream);

  return output_tensor;
}

tvm::ffi::Tensor masked_act_mul_and_quant_entry(const tvm::ffi::TensorView &input,
                                                 const tvm::ffi::TensorView &scale,
                                                 const tvm::ffi::TensorView &num_per_expert,
                                                 tvm::ffi::Optional<tvm::ffi::Tensor> output) {
  auto stream = TVM_FFI_GET_CUDA_STREAM(input);
  auto device = input.device();

  TVM_FFI_CHECK_CONTIGUOUS(input);
  TVM_FFI_CHECK_CONTIGUOUS(scale);
  TVM_FFI_CHECK_CONTIGUOUS(num_per_expert);

  TVM_FFI_CHECK_CUDA(input);
  TVM_FFI_CHECK_CUDA(scale);
  TVM_FFI_CHECK_CUDA(num_per_expert);

  TVM_FFI_ICHECK(input.shape().at(input.ndim() - 1) / 2 % 8 == 0)
      << "hidden dim must be divided by 8";
  TVM_FFI_ICHECK(num_per_expert.numel() > 0) << "num_per_expert must not be empty";

  int ndim = input.ndim();
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; ++i) {
    output_shape.push_back(input.shape().at(i));
  }
  output_shape[ndim - 1] /= 2;

  tvm::ffi::Tensor output_tensor;
  if (output.has_value()) {
    output_tensor = tvm::ffi::Tensor(output.value());
  } else {
    output_tensor = tvm_ffi_empty(output_shape, dl_float8_e4m3, device);
  }

  using Tin = __nv_bfloat16;
  using Tout = __nv_fp8_e4m3;

  const auto *input_ptr = reinterpret_cast<const Tin *>(input.data_ptr());
  const auto *scale_ptr = reinterpret_cast<const float *>(scale.data_ptr());
  auto *output_ptr = reinterpret_cast<Tout *>(output_tensor.data_ptr());

  const auto *num_per_expert_ptr = reinterpret_cast<const int *>(num_per_expert.data_ptr());

  int num_experts = num_per_expert.shape().at(0);
  int num_total_tokens = input.shape().at(0);
  int num_tokens_per_expert = num_total_tokens / num_experts;

  int num_intermediate_size = input.shape().at(1) / 2;

  masked_act_mul_and_quant_async(output_ptr, input_ptr, scale_ptr, num_per_expert_ptr,
                                 num_total_tokens, num_intermediate_size, num_tokens_per_expert,
                                 stream);

  return output_tensor;
}

tvm::ffi::Tuple<tvm::ffi::Tensor, tvm::ffi::Tensor> masked_act_mul_and_blockwise_quant_entry(
    const tvm::ffi::TensorView &input, const tvm::ffi::TensorView &num_per_expert,
    tvm::ffi::Optional<tvm::ffi::Tensor> output,
    tvm::ffi::Optional<tvm::ffi::Tensor> output_scale) {
  auto stream = TVM_FFI_GET_CUDA_STREAM(input);
  auto device = input.device();

  TVM_FFI_CHECK_CONTIGUOUS(input);
  TVM_FFI_CHECK_CONTIGUOUS(num_per_expert);

  TVM_FFI_CHECK_CUDA(input);
  TVM_FFI_CHECK_CUDA(num_per_expert);

  TVM_FFI_ICHECK(input.shape().at(input.ndim() - 1) / 2 % 128 == 0)
      << "hidden dim must be divided by 128";

  int ndim = input.ndim();
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; ++i) {
    output_shape.push_back(input.shape().at(i));
  }
  output_shape[ndim - 1] /= 2;

  auto output_scale_shape = output_shape;
  output_scale_shape[output_scale_shape.size() - 1] /= 128;

  tvm::ffi::Tensor output_tensor;
  if (output.has_value()) {
    output_tensor = tvm::ffi::Tensor(output.value());
  } else {
    output_tensor = tvm_ffi_empty(output_shape, dl_float8_e4m3, device);
  }
  tvm::ffi::Tensor output_scale_tensor;
  if (output_scale.has_value()) {
    output_scale_tensor = tvm::ffi::Tensor(output_scale.value());
  } else {
    output_scale_tensor = tvm_ffi_empty(output_scale_shape, dl_float32, device);
  }

  using Tin = __nv_bfloat16;
  using Tout = __nv_fp8_e4m3;

  const auto *input_ptr = reinterpret_cast<const Tin *>(input.data_ptr());
  auto *output_ptr = reinterpret_cast<Tout *>(output_tensor.data_ptr());
  auto *output_scale_ptr = reinterpret_cast<float *>(output_scale_tensor.data_ptr());

  const auto *num_per_expert_ptr = reinterpret_cast<const int *>(num_per_expert.data_ptr());

  int num_experts = num_per_expert.shape().at(0);
  int num_total_tokens = input.shape().at(0);
  int num_tokens_per_expert = num_total_tokens / num_experts;

  int num_intermediate_size = input.shape().at(1) / 2;

  masked_act_mul_and_blockwise_quant_async(output_ptr, output_scale_ptr, input_ptr,
                                           num_per_expert_ptr, num_total_tokens,
                                           num_intermediate_size, num_tokens_per_expert, stream);

  return tvm::ffi::Tuple<tvm::ffi::Tensor, tvm::ffi::Tensor>(output_tensor, output_scale_tensor);
}

}  // namespace activation
}  // namespace hpc

TVM_FFI_DLL_EXPORT_TYPED_FUNC(act_mul_and_quant,
                               hpc::activation::act_mul_and_quant_entry);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(masked_act_mul_and_quant,
                               hpc::activation::masked_act_mul_and_quant_entry);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(masked_act_mul_and_blockwise_quant,
                               hpc::activation::masked_act_mul_and_blockwise_quant_entry);
