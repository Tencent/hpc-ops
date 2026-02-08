// Copyright (C) 2026 Tencent.

#include <cuda_runtime_api.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <tvm/ffi/container/tuple.h>
#include <tuple>

#include "tvm_ffi_utils.h"

#include "src/fuse_moe/fuse_moe.h"

namespace hpc {
namespace fuse_moe {

tvm::ffi::Tuple<tvm::ffi::Tensor, tvm::ffi::Tensor, tvm::ffi::Tensor, tvm::ffi::Tensor,
           tvm::ffi::Tensor, tvm::ffi::Tensor, tvm::ffi::Tensor, tvm::ffi::Tensor,
           tvm::ffi::Tensor>
count_and_gather_entry(const tvm::ffi::TensorView &x, const tvm::ffi::TensorView &topk_ids,
                       int64_t num_expert, int64_t rank_ep, int64_t intermediate_size,
                       int64_t num_seq_per_group_avg) {
  auto stream = TVM_FFI_GET_CUDA_STREAM(x);
  TVM_FFI_CHECK_CUDA(x);
  TVM_FFI_CHECK_CUDA(topk_ids);
  TVM_FFI_CHECK_CONTIGUOUS(x);
  TVM_FFI_CHECK_CONTIGUOUS(topk_ids);
  TVM_FFI_ICHECK(x.shape().at(0) == topk_ids.shape().at(0))
      << "x and topk_ids must share the same k";

  int num_seq = x.shape().at(0);
  int hidden_size = x.shape().at(1);
  int num_topk = topk_ids.shape().at(1);

  auto device = x.device();
  tvm::ffi::Tensor gate_up_input =
      tvm_ffi_empty({num_seq * num_topk, hidden_size}, x.dtype(), device);
  tvm::ffi::Tensor gate_up_output =
      tvm_ffi_empty({num_seq * num_topk, intermediate_size}, dl_bfloat16, device);
  tvm::ffi::Tensor down_input =
      tvm_ffi_empty({num_seq * num_topk, intermediate_size / 2}, x.dtype(), device);
  tvm::ffi::Tensor down_output =
      tvm_ffi_empty({num_seq * num_topk, hidden_size}, dl_bfloat16, device);

  tvm::ffi::Tensor topk_pos = tvm_ffi_empty({num_seq, num_topk}, dl_int32, device);
  tvm::ffi::Tensor seqlens = tvm_ffi_zeros({num_expert}, dl_int32, device);
  tvm::ffi::Tensor cu_seqlens = tvm_ffi_empty({num_expert + 1}, dl_int32, device);
  tvm::ffi::Tensor tiles = tvm_ffi_empty({num_expert}, dl_int32, device);
  tvm::ffi::Tensor cu_tiles = tvm_ffi_empty({num_expert + 1}, dl_int32, device);
  tvm::ffi::Tensor gate_up_tmas = tvm_ffi_empty({num_expert * 2, 128}, dl_int8, device);
  tvm::ffi::Tensor dowm_tmas = tvm_ffi_empty({num_expert * 2, 128}, dl_int8, device);

  const auto *x_ptr = x.data_ptr();
  const auto *topk_ids_ptr = topk_ids.data_ptr();

  auto *gate_up_input_ptr = gate_up_input.data_ptr();
  auto *gate_up_output_ptr = gate_up_output.data_ptr();
  auto *down_input_ptr = down_input.data_ptr();
  auto *down_output_ptr = down_output.data_ptr();
  auto *topk_pos_ptr = topk_pos.data_ptr();
  auto *seqlens_ptr = seqlens.data_ptr();
  auto *cu_seqlens_ptr = cu_seqlens.data_ptr();
  auto *tiles_ptr = tiles.data_ptr();
  auto *cu_tiles_ptr = cu_tiles.data_ptr();
  auto *gate_up_tmas_ptr = gate_up_tmas.data_ptr();
  auto *dowm_tmas_ptr = dowm_tmas.data_ptr();

  count_and_gather_async(gate_up_input_ptr, gate_up_output_ptr, down_input_ptr, down_output_ptr,
                         x_ptr, topk_ids_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr,
                         gate_up_tmas_ptr, dowm_tmas_ptr, tiles_ptr, cu_tiles_ptr, num_seq,
                         hidden_size, intermediate_size, num_topk, num_expert, rank_ep,
                         num_seq_per_group_avg, stream);

  return tvm::ffi::Tuple<tvm::ffi::Tensor, tvm::ffi::Tensor, tvm::ffi::Tensor, tvm::ffi::Tensor,
                         tvm::ffi::Tensor, tvm::ffi::Tensor, tvm::ffi::Tensor, tvm::ffi::Tensor,
                         tvm::ffi::Tensor>(gate_up_input, gate_up_output, topk_pos, seqlens,
                                           cu_seqlens, tiles, cu_tiles, gate_up_tmas, dowm_tmas);
}

tvm::ffi::Tensor reduce_entry(const tvm::ffi::TensorView &x, const tvm::ffi::TensorView &topk_pos,
                               const tvm::ffi::TensorView &topk_scale,
                               tvm::ffi::Optional<tvm::ffi::Tensor> shared_output) {
  auto stream = TVM_FFI_GET_CUDA_STREAM(x);
  TVM_FFI_CHECK_CUDA(x);
  TVM_FFI_CHECK_CUDA(topk_pos);
  TVM_FFI_CHECK_CUDA(topk_scale);
  TVM_FFI_CHECK_CONTIGUOUS(x);
  TVM_FFI_CHECK_CONTIGUOUS(topk_pos);
  TVM_FFI_CHECK_CONTIGUOUS(topk_scale);
  TVM_FFI_ICHECK(topk_pos.shape().at(0) == topk_scale.shape().at(0))
      << "topk_pos and topk_scale must share the same num_seq";
  TVM_FFI_ICHECK(topk_pos.shape().at(1) == topk_scale.shape().at(1))
      << "topk_pos and topk_scale must share the same num_topk";

  const void *shared_output_ptr = nullptr;
  if (shared_output.has_value()) {
    const auto &shared_output_tensor = shared_output.value();
    TVM_FFI_CHECK_CUDA(shared_output_tensor);
    TVM_FFI_CHECK_CONTIGUOUS(shared_output_tensor);
    shared_output_ptr = shared_output_tensor.data_ptr();
  }

  int total_num_seq = x.shape().at(0);
  int hidden_size = x.shape().at(1);
  int num_seq = topk_pos.shape().at(0);
  int num_topk = topk_pos.shape().at(1);
  TVM_FFI_ICHECK(num_topk <= 128) << "num_topk must less than or equal to 128";

  auto device = x.device();
  tvm::ffi::Tensor y = tvm_ffi_empty({num_seq, hidden_size}, dl_bfloat16, device);

  const auto *x_ptr = x.data_ptr();
  const auto *topk_pos_ptr = topk_pos.data_ptr();
  const auto *topk_scale_ptr = topk_scale.data_ptr();

  auto *y_ptr = y.data_ptr();

  reduce_async(y_ptr, x_ptr, topk_pos_ptr, topk_scale_ptr, shared_output_ptr, total_num_seq,
               num_seq, hidden_size, num_topk, stream);

  return y;
}

tvm::ffi::Tensor fuse_moe_pertensor_fp8_entry(
    const tvm::ffi::TensorView &x, const tvm::ffi::TensorView &gate_up_weight,
    const tvm::ffi::TensorView &down_weight, const tvm::ffi::TensorView &gate_up_scale,
    const tvm::ffi::TensorView &down_scale, const tvm::ffi::TensorView &act_and_mul_scale,
    const tvm::ffi::TensorView &topk_ids, const tvm::ffi::TensorView &topk_scale,
    tvm::ffi::Optional<tvm::ffi::Tensor> shared_output, int64_t rank_ep,
    int64_t num_expert_total, bool use_bf16_mul) {
  auto stream = TVM_FFI_GET_CUDA_STREAM(x);

  TVM_FFI_CHECK_CUDA(x);
  TVM_FFI_CHECK_CUDA(gate_up_weight);
  TVM_FFI_CHECK_CUDA(gate_up_scale);
  TVM_FFI_CHECK_CUDA(down_scale);
  TVM_FFI_CHECK_CUDA(act_and_mul_scale);
  TVM_FFI_CHECK_CUDA(topk_ids);
  TVM_FFI_CHECK_CUDA(topk_scale);

  TVM_FFI_CHECK_CONTIGUOUS(x);
  TVM_FFI_CHECK_CONTIGUOUS(gate_up_weight);
  TVM_FFI_CHECK_CONTIGUOUS(gate_up_scale);
  TVM_FFI_CHECK_CONTIGUOUS(down_weight);
  TVM_FFI_CHECK_CONTIGUOUS(down_scale);
  TVM_FFI_CHECK_CONTIGUOUS(topk_ids);
  TVM_FFI_CHECK_CONTIGUOUS(topk_scale);

  TVM_FFI_ICHECK(x.shape().at(0) == topk_ids.shape().at(0))
      << "x and topk_ids must share the same num_seq";
  TVM_FFI_ICHECK(topk_ids.shape().at(0) == topk_scale.shape().at(0))
      << "topk_ids and topk_scale must share the same num_seq";
  TVM_FFI_ICHECK(topk_ids.shape().at(1) == topk_scale.shape().at(1))
      << "topk_ids and topk_scale must share the same num_topk";
  TVM_FFI_ICHECK(x.shape().at(1) == gate_up_weight.shape().at(2))
      << "x and weight must share the same k";
  TVM_FFI_ICHECK(gate_up_weight.shape().at(0) == down_weight.shape().at(0))
      << "gate_up_weight and down_weight must share the same num_expert";

  const void *shared_output_ptr = nullptr;
  if (shared_output.has_value()) {
    const auto &shared_output_tensor = shared_output.value();
    TVM_FFI_CHECK_CUDA(shared_output_tensor);
    TVM_FFI_CHECK_CONTIGUOUS(shared_output_tensor);
    shared_output_ptr = shared_output_tensor.data_ptr();
  }

  int num_seq = x.shape().at(0);
  int hidden_size = x.shape().at(1);
  int num_expert = gate_up_weight.shape().at(0);
  int intermediate_size = gate_up_weight.shape().at(1);
  int num_topk = topk_ids.shape().at(1);
  TVM_FFI_ICHECK(num_topk <= 128) << "num_topk must less than or equal to 128";

  auto device = x.device();
  tvm::ffi::Tensor y = tvm_ffi_empty({num_seq, hidden_size}, dl_bfloat16, device);

  tvm::ffi::Tensor gate_up_input =
      tvm_ffi_empty({num_seq * num_topk, hidden_size}, x.dtype(), device);
  tvm::ffi::Tensor gate_up_output =
      tvm_ffi_empty({num_seq * num_topk, intermediate_size}, dl_bfloat16, device);
  tvm::ffi::Tensor gate_up_tmas = tvm_ffi_empty({num_expert * 2, 128}, dl_int8, device);
  tvm::ffi::Tensor down_input =
      tvm_ffi_empty({num_seq * num_topk, intermediate_size / 2}, x.dtype(), device);
  tvm::ffi::Tensor down_output =
      tvm_ffi_empty({num_seq * num_topk, hidden_size}, dl_bfloat16, device);
  tvm::ffi::Tensor down_tmas = tvm_ffi_empty({num_expert * 2, 128}, dl_int8, device);

  tvm::ffi::Tensor topk_pos_tensor = tvm_ffi_empty({num_seq, num_topk}, dl_int32, device);
  tvm::ffi::Tensor seqlens = tvm_ffi_zeros({num_expert}, dl_int32, device);
  tvm::ffi::Tensor cu_seqlens = tvm_ffi_empty({num_expert + 1}, dl_int32, device);
  tvm::ffi::Tensor tiles = tvm_ffi_empty({num_expert}, dl_int32, device);
  tvm::ffi::Tensor cu_tiles = tvm_ffi_empty({num_expert + 1}, dl_int32, device);

  const auto *x_ptr = x.data_ptr();
  const auto *topk_ids_ptr = topk_ids.data_ptr();
  const auto *topk_scale_ptr = topk_scale.data_ptr();
  const auto *gate_up_weight_ptr = gate_up_weight.data_ptr();
  const auto *gate_up_scale_ptr = gate_up_scale.data_ptr();
  const auto *act_and_mul_scale_ptr = act_and_mul_scale.data_ptr();
  const auto *down_weight_ptr = down_weight.data_ptr();
  const auto *down_scale_ptr = down_scale.data_ptr();

  auto *y_ptr = y.data_ptr();
  auto *topk_pos_ptr = topk_pos_tensor.data_ptr();
  auto *seqlens_ptr = seqlens.data_ptr();
  auto *cu_seqlens_ptr = cu_seqlens.data_ptr();
  auto *tiles_ptr = tiles.data_ptr();
  auto *cu_tiles_ptr = cu_tiles.data_ptr();
  auto *gate_up_input_ptr = gate_up_input.data_ptr();
  auto *gate_up_output_ptr = gate_up_output.data_ptr();
  auto *gate_up_tmas_ptr = gate_up_tmas.data_ptr();
  auto *down_input_ptr = down_input.data_ptr();
  auto *down_output_ptr = down_output.data_ptr();
  auto *down_tmas_ptr = down_tmas.data_ptr();

  fuse_moe_pertensor_fp8_async(
      y_ptr, x_ptr, gate_up_input_ptr, gate_up_output_ptr, gate_up_weight_ptr, gate_up_scale_ptr,
      gate_up_tmas_ptr, act_and_mul_scale_ptr, down_input_ptr, down_output_ptr, down_weight_ptr,
      down_scale_ptr, down_tmas_ptr, topk_ids_ptr, topk_scale_ptr, topk_pos_ptr, seqlens_ptr,
      cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr, shared_output_ptr, num_seq, hidden_size,
      intermediate_size, num_topk, num_expert_total, num_expert, rank_ep, use_bf16_mul, stream);

  return y;
}

tvm::ffi::Tensor fuse_moe_blockwise_fp8_entry(
    const tvm::ffi::TensorView &x, const tvm::ffi::TensorView &x_scale,
    const tvm::ffi::TensorView &gate_up_weight,
    const tvm::ffi::TensorView &gate_up_weight_scale,
    const tvm::ffi::TensorView &down_weight, const tvm::ffi::TensorView &down_weight_scale,
    const tvm::ffi::TensorView &topk_ids, const tvm::ffi::TensorView &topk_scale,
    tvm::ffi::Optional<tvm::ffi::Tensor> shared_output, int64_t rank_ep,
    int64_t num_expert_total) {
  auto stream = TVM_FFI_GET_CUDA_STREAM(x);

  TVM_FFI_CHECK_CUDA(x);
  TVM_FFI_CHECK_CUDA(x_scale);
  TVM_FFI_CHECK_CUDA(gate_up_weight);
  TVM_FFI_CHECK_CUDA(gate_up_weight_scale);
  TVM_FFI_CHECK_CUDA(down_weight);
  TVM_FFI_CHECK_CUDA(down_weight_scale);
  TVM_FFI_CHECK_CUDA(topk_ids);
  TVM_FFI_CHECK_CUDA(topk_scale);

  TVM_FFI_CHECK_CONTIGUOUS(x);
  TVM_FFI_CHECK_CONTIGUOUS(x_scale);
  TVM_FFI_CHECK_CONTIGUOUS(gate_up_weight);
  TVM_FFI_CHECK_CONTIGUOUS(gate_up_weight_scale);
  TVM_FFI_CHECK_CONTIGUOUS(down_weight);
  TVM_FFI_CHECK_CONTIGUOUS(down_weight_scale);
  TVM_FFI_CHECK_CONTIGUOUS(topk_ids);
  TVM_FFI_CHECK_CONTIGUOUS(topk_scale);

  TVM_FFI_ICHECK(x.shape().at(0) == topk_ids.shape().at(0))
      << "x and topk_ids must share the same num_tokens";
  TVM_FFI_ICHECK(topk_ids.shape().at(0) == topk_scale.shape().at(0))
      << "topk_ids and topk_scale must share the same num_tokens";
  TVM_FFI_ICHECK(topk_ids.shape().at(1) == topk_scale.shape().at(1))
      << "topk_ids and topk_scale must share the same num_topk";
  TVM_FFI_ICHECK(x.shape().at(1) == gate_up_weight.shape().at(2))
      << "x and weight must share the same k";
  TVM_FFI_ICHECK(gate_up_weight.shape().at(0) == down_weight.shape().at(0))
      << "gate_up_weight and down_weight must share the same num_expert";

  const void *shared_output_ptr = nullptr;
  if (shared_output.has_value()) {
    const auto &shared_output_tensor = shared_output.value();
    TVM_FFI_CHECK_CUDA(shared_output_tensor);
    TVM_FFI_CHECK_CONTIGUOUS(shared_output_tensor);
    shared_output_ptr = shared_output_tensor.data_ptr();
  }

  int num_tokens = x.shape().at(0);
  int hidden_size = x.shape().at(1);
  int num_experts = gate_up_weight.shape().at(0);
  int intermediate_size = gate_up_weight.shape().at(1);
  int num_topk = topk_ids.shape().at(1);
  int num_tokens_per_group_avg = num_tokens * num_topk / num_expert_total;
  int aligned_size = 0;
  int gate_up_weight_scale_lastdim_pad4 = gate_up_weight_scale.shape().at(
      gate_up_weight_scale.ndim() - 1);
  int down_weight_scale_lastdim_pad4 = down_weight_scale.shape().at(
      down_weight_scale.ndim() - 1);

  TVM_FFI_ICHECK(num_topk <= 128) << "num_topk must less than or equal to 128";

  if (num_tokens_per_group_avg <= 16) {
    aligned_size = 16;
  } else if (num_tokens_per_group_avg <= 32) {
    aligned_size = 32;
  } else if (num_tokens_per_group_avg <= 48) {
    aligned_size = 48;
  } else {
    aligned_size = 64;
  }
  int num_padded_tokens =
      (num_tokens * num_topk + num_expert_total * aligned_size + aligned_size - 1) / aligned_size *
      aligned_size;

  auto device = x.device();
  tvm::ffi::Tensor y = tvm_ffi_empty({num_tokens, hidden_size}, dl_bfloat16, device);
  tvm::ffi::Tensor gate_up_input =
      tvm_ffi_empty({num_tokens * num_topk, hidden_size}, dl_float8_e4m3, device);
  tvm::ffi::Tensor gate_up_input_scale =
      tvm_ffi_empty({x_scale.shape().at(1), num_padded_tokens}, dl_float32, device);
  tvm::ffi::Tensor gate_up_output =
      tvm_ffi_empty({num_tokens * num_topk, intermediate_size}, dl_bfloat16, device);
  tvm::ffi::Tensor gate_up_tmas = tvm_ffi_empty({num_experts * 2, 128}, dl_int8, device);
  tvm::ffi::Tensor down_input =
      tvm_ffi_empty({num_tokens * num_topk, intermediate_size / 2}, dl_float8_e4m3, device);
  tvm::ffi::Tensor down_input_scale =
      tvm_ffi_empty({intermediate_size / 2 / 128, num_padded_tokens}, dl_float32, device);
  tvm::ffi::Tensor down_output =
      tvm_ffi_empty({num_tokens * num_topk, hidden_size}, dl_bfloat16, device);
  tvm::ffi::Tensor down_tmas = tvm_ffi_empty({num_experts * 2, 128}, dl_int8, device);
  tvm::ffi::Tensor topk_pos_tensor =
      tvm_ffi_empty({num_tokens, num_topk}, dl_int32, device);
  tvm::ffi::Tensor num_tokens_per_group = tvm_ffi_zeros({num_experts}, dl_int32, device);
  tvm::ffi::Tensor cu_num_tokens_per_group =
      tvm_ffi_empty({num_experts + 1}, dl_int32, device);
  tvm::ffi::Tensor tiles = tvm_ffi_empty({num_experts}, dl_int32, device);
  tvm::ffi::Tensor cu_tiles = tvm_ffi_empty({num_experts + 1}, dl_int32, device);

  const auto *x_ptr = x.data_ptr();
  const auto *x_scale_ptr = x_scale.data_ptr();
  const auto *topk_ids_ptr = topk_ids.data_ptr();
  const auto *topk_scale_ptr = topk_scale.data_ptr();
  const auto *gate_up_weight_ptr = gate_up_weight.data_ptr();
  const auto *gate_up_weight_scale_ptr = gate_up_weight_scale.data_ptr();
  const auto *down_weight_ptr = down_weight.data_ptr();
  const auto *down_weight_scale_ptr = down_weight_scale.data_ptr();

  auto *y_ptr = y.data_ptr();
  auto *topk_pos_ptr = topk_pos_tensor.data_ptr();
  auto *num_tokens_per_group_ptr = num_tokens_per_group.data_ptr();
  auto *cu_num_tokens_per_group_ptr = cu_num_tokens_per_group.data_ptr();
  auto *tiles_ptr = tiles.data_ptr();
  auto *cu_tiles_ptr = cu_tiles.data_ptr();
  auto *gate_up_input_ptr = gate_up_input.data_ptr();
  auto *gate_up_input_scale_ptr = gate_up_input_scale.data_ptr();
  auto *gate_up_output_ptr = gate_up_output.data_ptr();
  auto *gate_up_tmas_ptr = gate_up_tmas.data_ptr();
  auto *down_input_ptr = down_input.data_ptr();
  auto *down_input_scale_ptr = down_input_scale.data_ptr();
  auto *down_output_ptr = down_output.data_ptr();
  auto *down_tmas_ptr = down_tmas.data_ptr();

  fuse_moe_blockwise_fp8_async(
      y_ptr, x_ptr, x_scale_ptr, gate_up_input_ptr, gate_up_input_scale_ptr, gate_up_output_ptr,
      gate_up_weight_ptr, gate_up_weight_scale_ptr, gate_up_tmas_ptr, down_input_ptr,
      down_input_scale_ptr, down_output_ptr, down_weight_ptr, down_weight_scale_ptr, down_tmas_ptr,
      topk_ids_ptr, topk_scale_ptr, topk_pos_ptr, num_tokens_per_group_ptr,
      cu_num_tokens_per_group_ptr, tiles_ptr, cu_tiles_ptr, shared_output_ptr, num_tokens,
      num_padded_tokens, hidden_size, intermediate_size, num_topk, num_expert_total, num_experts,
      gate_up_weight_scale_lastdim_pad4, down_weight_scale_lastdim_pad4, rank_ep, stream);
  return y;
}

}  // namespace fuse_moe
}  // namespace hpc

TVM_FFI_DLL_EXPORT_TYPED_FUNC(count_and_gather, hpc::fuse_moe::count_and_gather_entry);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(reduce, hpc::fuse_moe::reduce_entry);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fuse_moe_pertensor_fp8,
                               hpc::fuse_moe::fuse_moe_pertensor_fp8_entry);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fuse_moe_blockwise_fp8,
                               hpc::fuse_moe::fuse_moe_blockwise_fp8_entry);
