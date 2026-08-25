// Copyright (C) 2026 Tencent.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>

namespace hpc {
namespace fuse_moe {

namespace {

template <typename IdType>
__global__ void prepare_indexed_input_kernel(
    const __nv_bfloat16 *__restrict__ source, const IdType *__restrict__ source_ids,
    const float *__restrict__ source_weights, __nv_fp8_e4m3 *__restrict__ activation,
    float *__restrict__ activation_scale, int32_t *__restrict__ output_ids,
    float *__restrict__ output_weights, int num_tokens, int capacity, int hidden_size,
    int num_groups, int top_k) {
  constexpr int kGroupSize = 128;
  __shared__ float reductions[kGroupSize];

  int program = blockIdx.x;
  int quant_programs = num_tokens * num_groups;
  if (program < quant_programs) {
    int token = program / num_groups;
    int group = program - token * num_groups;
    int column = group * kGroupSize + threadIdx.x;
    float value = __bfloat162float(source[static_cast<int64_t>(token) * hidden_size + column]);
    reductions[threadIdx.x] = fabsf(value);
    __syncthreads();
    for (int stride = kGroupSize / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + stride]);
      }
      __syncthreads();
    }
    float scale = fmaxf(reductions[0], 1.0e-10f) / 448.0f;
    if (threadIdx.x == 0) {
      activation_scale[static_cast<int64_t>(token) * num_groups + group] = scale;
    }
    float quantized = fminf(fmaxf(value / scale, -448.0f), 448.0f);
    activation[static_cast<int64_t>(token) * hidden_size + column] = __nv_fp8_e4m3(quantized);
  }

  if (program < capacity && threadIdx.x < top_k) {
    int64_t output_index = static_cast<int64_t>(program) * top_k + threadIdx.x;
    if (program < num_tokens) {
      int64_t source_index = static_cast<int64_t>(program) * top_k + threadIdx.x;
      output_ids[output_index] = static_cast<int32_t>(source_ids[source_index]);
      output_weights[output_index] = source_weights[source_index];
    } else {
      output_ids[output_index] = -1;
      output_weights[output_index] = 0.0f;
    }
  }
}

void check_cuda_contiguous(const at::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void prepare_indexed_input(const at::Tensor &source, const at::Tensor &topk_ids,
                           const at::Tensor &topk_weights, const at::Tensor &activation,
                           const at::Tensor &activation_scale, const at::Tensor &output_ids,
                           const at::Tensor &output_weights) {
  check_cuda_contiguous(source, "indexed input source");
  TORCH_CHECK(source.dim() == 2, "indexed input source must be two-dimensional");
  TORCH_CHECK(source.scalar_type() == at::kBFloat16, "indexed input source must be bfloat16");

  check_cuda_contiguous(topk_ids, "indexed input topk_ids");
  TORCH_CHECK(topk_ids.dim() == 2, "indexed input topk_ids must be two-dimensional");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt || topk_ids.scalar_type() == at::kLong,
              "indexed input topk_ids must be int32 or int64");
  check_cuda_contiguous(topk_weights, "indexed input topk_weights");
  TORCH_CHECK(topk_weights.dim() == 2, "indexed input topk_weights must be two-dimensional");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat,
              "indexed input topk_weights must be float32");
  TORCH_CHECK(topk_ids.sizes() == topk_weights.sizes(),
              "indexed input route ids and weights must have the same shape");
  TORCH_CHECK(topk_ids.size(0) == source.size(0),
              "indexed input routes and source must have the same row count");

  check_cuda_contiguous(activation, "indexed input activation output");
  TORCH_CHECK(activation.dim() == 2, "indexed input activation output must be two-dimensional");
  TORCH_CHECK(activation.scalar_type() == at::kFloat8_e4m3fn,
              "indexed input activation output must be float8_e4m3fn");
  TORCH_CHECK(activation.size(1) == source.size(1),
              "indexed input activation hidden size must match source hidden size");
  TORCH_CHECK(source.size(1) > 0 && source.size(1) % 128 == 0,
              "indexed input hidden size must be a positive multiple of 128");
  TORCH_CHECK(source.size(0) <= activation.size(0),
              "indexed input source rows must not exceed buffer capacity");

  check_cuda_contiguous(activation_scale, "indexed input activation scale output");
  TORCH_CHECK(activation_scale.dim() == 2,
              "indexed input activation scale output must be two-dimensional");
  TORCH_CHECK(activation_scale.scalar_type() == at::kFloat,
              "indexed input activation scale output must be float32");
  TORCH_CHECK(activation_scale.size(0) == activation.size(0) &&
                  activation_scale.size(1) == source.size(1) / 128,
              "indexed input activation scale shape is invalid");

  check_cuda_contiguous(output_ids, "indexed input route id output");
  check_cuda_contiguous(output_weights, "indexed input route weight output");
  TORCH_CHECK(output_ids.scalar_type() == at::kInt, "indexed input route id output must be int32");
  TORCH_CHECK(output_weights.scalar_type() == at::kFloat,
              "indexed input route weight output must be float32");
  TORCH_CHECK(output_ids.dim() == 2 && output_weights.dim() == 2,
              "indexed input route outputs must be two-dimensional");
  TORCH_CHECK(output_ids.sizes() == output_weights.sizes(),
              "indexed input route output shapes must match");
  TORCH_CHECK(output_ids.size(0) == activation.size(0) && output_ids.size(1) == topk_ids.size(1),
              "indexed input route output shape is invalid");
  TORCH_CHECK(topk_ids.size(1) > 0 && topk_ids.size(1) <= 128,
              "indexed input top_k must be in [1, 128]");

  int device = source.get_device();
  for (const at::Tensor *tensor :
       {&topk_ids, &topk_weights, &activation, &activation_scale, &output_ids, &output_weights}) {
    TORCH_CHECK(tensor->get_device() == device,
                "indexed input tensors must use the same CUDA device");
  }

  int num_tokens = source.size(0);
  int capacity = activation.size(0);
  int hidden_size = source.size(1);
  int num_groups = hidden_size / 128;
  int programs = std::max(capacity, num_tokens * num_groups);
  if (programs == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream(device);
  if (topk_ids.scalar_type() == at::kInt) {
    prepare_indexed_input_kernel<<<programs, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(source.const_data_ptr()),
        topk_ids.const_data_ptr<int32_t>(), topk_weights.const_data_ptr<float>(),
        reinterpret_cast<__nv_fp8_e4m3 *>(activation.data_ptr()),
        activation_scale.data_ptr<float>(), output_ids.data_ptr<int32_t>(),
        output_weights.data_ptr<float>(), num_tokens, capacity, hidden_size, num_groups,
        topk_ids.size(1));
  } else {
    prepare_indexed_input_kernel<<<programs, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(source.const_data_ptr()),
        topk_ids.const_data_ptr<int64_t>(), topk_weights.const_data_ptr<float>(),
        reinterpret_cast<__nv_fp8_e4m3 *>(activation.data_ptr()),
        activation_scale.data_ptr<float>(), output_ids.data_ptr<int32_t>(),
        output_weights.data_ptr<float>(), num_tokens, capacity, hidden_size, num_groups,
        topk_ids.size(1));
  }
  auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "prepare indexed input launch failed: ", cudaGetErrorString(error));
}

}  // namespace fuse_moe
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "prepare_indexed_input(Tensor source, Tensor topk_ids, Tensor topk_weights, "
      "Tensor(a!) activation, Tensor(b!) activation_scale, Tensor(c!) output_ids, "
      "Tensor(d!) output_weights) -> ()");
  m.impl("prepare_indexed_input", c10::DispatchKey::CUDA, &hpc::fuse_moe::prepare_indexed_input);
}
