// Copyright 2025 hpc-ops authors

#ifndef SRC_ROUTING_METHOD_ROUTING_METHOD_H_
#define SRC_ROUTING_METHOD_ROUTING_METHOD_H_

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace hpc {
namespace routing_method {

void deepseekv4_routing_method_async(float* out_weights_ptr, int* out_indices_ptr,
                                     const __nv_bfloat16* score_ptr, const float* bias_ptr,
                                     const int32_t* input_ids_ptr, const int32_t* tid2eid_ptr,
                                     int batch_size, int num_expert, int topk, float route_scale,
                                     bool is_hash, cudaStream_t stream);

}  // namespace routing_method
}  // namespace hpc

#endif  // SRC_ROUTING_METHOD_ROUTING_METHOD_H_
