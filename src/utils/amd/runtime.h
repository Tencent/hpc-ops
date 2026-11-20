// Copyright (C) 2026 Tencent.
#ifndef SRC_UTILS_AMD_RUNTIME_H_
#define SRC_UTILS_AMD_RUNTIME_H_

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace hpc::amd {

inline void check_tensor(const at::Tensor& t, const at::Tensor& reference, at::ScalarType dtype,
                         const char* name) {
  TORCH_CHECK(t.is_cuda() && t.device() == reference.device(), name,
              " must be on the same GPU as the input");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has an unsupported dtype");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

inline hip_bfloat16* bf16_data(const at::Tensor& t) {
  return reinterpret_cast<hip_bfloat16*>(t.data_ptr<at::BFloat16>());
}

inline hipStream_t current_stream(const at::Tensor& t) {
  return c10::hip::getCurrentHIPStream(t.get_device()).stream();
}

}  // namespace hpc::amd
#endif  // SRC_UTILS_AMD_RUNTIME_H_
