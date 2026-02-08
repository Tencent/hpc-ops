// Copyright (C) 2026 Tencent.

#pragma once

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <dlpack/dlpack.h>

inline constexpr int64_t encode_dlpack_dtype(DLDataType dtype) {
  return (dtype.code << 16) | (dtype.bits << 8) | dtype.lanes;
}

constexpr DLDataType dl_float16 = DLDataType{kDLFloat, 16, 1};
constexpr DLDataType dl_float32 = DLDataType{kDLFloat, 32, 1};
constexpr DLDataType dl_float64 = DLDataType{kDLFloat, 64, 1};
constexpr DLDataType dl_bfloat16 = DLDataType{kDLBfloat, 16, 1};
constexpr DLDataType dl_int32 = DLDataType{kDLInt, 32, 1};
constexpr DLDataType dl_int64 = DLDataType{kDLInt, 64, 1};
constexpr DLDataType dl_uint8 = DLDataType{kDLUInt, 8, 1};
constexpr DLDataType dl_int8 = DLDataType{kDLInt, 8, 1};
constexpr DLDataType dl_float8_e4m3 = DLDataType{6 /*kDLFloat8_e4m3fn*/, 8, 1};

constexpr int64_t float16_code = encode_dlpack_dtype(dl_float16);
constexpr int64_t float32_code = encode_dlpack_dtype(dl_float32);
constexpr int64_t float64_code = encode_dlpack_dtype(dl_float64);
constexpr int64_t bfloat16_code = encode_dlpack_dtype(dl_bfloat16);
constexpr int64_t int32_code = encode_dlpack_dtype(dl_int32);

#define TVM_FFI_GET_CUDA_STREAM(data) \
  static_cast<cudaStream_t>(TVMFFIEnvGetStream(data.device().device_type, data.device().device_id))

#define CHECK_CUDA_SUCCESS(err)                                                            \
  do {                                                                                     \
    TVM_FFI_ICHECK(err == cudaSuccess) << "CUDA Failure: " << cudaGetErrorString(err);     \
  } while (0)

#define TVM_FFI_CHECK_CUDA(input)                                                          \
  TVM_FFI_ICHECK(input.device().device_type == kDLCUDA) << #input " tensor must be cuda"

#define TVM_FFI_CHECK_CONTIGUOUS(input)                                                    \
  TVM_FFI_ICHECK(input.IsContiguous()) << #input " tensor must be contiguous"

inline tvm::ffi::Tensor tvm_ffi_empty(std::vector<int64_t> shape, DLDataType dtype,
                                       DLDevice device) {
  tvm::ffi::ShapeView sv(shape.data(), shape.size());
  return tvm::ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, sv, dtype, device);
}

inline tvm::ffi::Tensor tvm_ffi_zeros(std::vector<int64_t> shape, DLDataType dtype,
                                       DLDevice device) {
  tvm::ffi::ShapeView sv(shape.data(), shape.size());
  auto t = tvm::ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, sv, dtype, device);
  cudaMemset(t.data_ptr(), 0, t.numel() * (dtype.bits / 8));
  return t;
}
