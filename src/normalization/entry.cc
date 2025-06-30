#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/normalization/fused_rms_norm_with_scale/fused_rms_norm_with_scale.h"

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def("fused_rms_norm_with_scale",
        &hpc::normalization::fused_rms_norm_with_scale::entry);
}
