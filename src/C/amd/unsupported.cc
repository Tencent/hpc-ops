// Copyright (C) 2026 Tencent.
// Schemas mirror the CUDA entries so public wrappers/fake registration remain importable.
// No unsupported entry silently falls back to CPU or a different numerical operation.
#include <torch/library.h>

namespace hpc::amd {

void unsupported(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, op.schema().name(),
                              ": TODO(amd): kernel not yet ported to HIP");
}
void unsupported_fp8(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, op.schema().name(),
                              ": TODO(amd): FP8 kernels are excluded from this batch");
}
void unsupported_matrix(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, op.schema().name(),
                              ": TODO(amd): matrix-unit kernel; WMMA/MFMA port is deferred");
}
void unsupported_communication(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, op.schema().name(),
                              ": TODO(amd): replace NVIDIA NVLink/multicast communication");
}

}  // namespace hpc::amd

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  // TODO(amd): act_mul_and_quant: FP8 data path. Source: src/activation/entry.cc.
  m.def(
      "act_mul_and_quant(Tensor input, Tensor scale, bool use_bf16_mul, Tensor? output) -> "
      "(Tensor)");
  m.impl("act_mul_and_quant", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): masked_act_mul_and_quant: FP8 data path. Source: src/activation/entry.cc.
  m.def(
      "masked_act_mul_and_quant(Tensor input, Tensor scale, Tensor num_per_expert, Tensor? "
      "output) -> (Tensor)");
  m.impl("masked_act_mul_and_quant", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): masked_act_mul_and_blockwise_quant: FP8 data path. Source: src/activation/entry.cc.
  m.def(
      "masked_act_mul_and_blockwise_quant(Tensor input, Tensor num_per_expert, Tensor? output, "
      "Tensor? output_scale) -> (Tensor output, Tensor output_scale)");
  m.impl("masked_act_mul_and_blockwise_quant", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): scaled_fp8_quant: FP8 data path. Source: src/activation/entry.cc.
  m.def("scaled_fp8_quant(Tensor input, Tensor? scale, Tensor? output) -> (Tensor, Tensor)");
  m.impl("scaled_fp8_quant", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): fuse_allreduce_rmsnorm_high_throughput: NVIDIA communication primitives. Source:
  // src/allreduce/entry.cc.
  m.def(
      "fuse_allreduce_rmsnorm_high_throughput(Tensor input, Tensor mc_input, Tensor "
      "in_residual, Tensor weight, Tensor signal, int rank, int world_size, int num_max_blocks, "
      "float rms_norm_eps, Tensor output, Tensor mc_output, Tensor out_residual) -> ()");
  m.impl("fuse_allreduce_rmsnorm_high_throughput", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_communication>());

  // TODO(amd): fuse_allreduce_rmsnorm_low_latency: NVIDIA communication primitives. Source:
  // src/allreduce/entry.cc.
  m.def(
      "fuse_allreduce_rmsnorm_low_latency(Tensor input_x, Tensor multicast_x, Tensor "
      "data_buffer_ptrs, Tensor! multinode_x, Tensor buffer_flags, int world_size, int rank, "
      "bool rmsnorm_fusion, bool launch_with_pdl, bool use_two_shot, Tensor! output_x, Tensor! "
      "residual_out, Tensor residual_in, Tensor weight_gamma, float rms_norm_eps) -> ()");
  m.impl("fuse_allreduce_rmsnorm_low_latency", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_communication>());

  // TODO(amd): attention_prefill_bf16: matrix-unit implementation. Source: src/attention/entry.cc.
  m.def(
      "attention_prefill_bf16(Tensor q, Tensor k, Tensor v, Tensor seqlens_q, Tensor "
      "cu_seqlens_q, int max_seqlens_q, Tensor? output) -> (Tensor)");
  m.impl("attention_prefill_bf16", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_matrix>());

  // TODO(amd): attention_with_kvcache_prefill_bf16: matrix-unit implementation. Source:
  // src/attention/entry.cc.
  m.def(
      "attention_with_kvcache_prefill_bf16(Tensor q, Tensor kcache, Tensor vcache,Tensor "
      "cu_seqlens_q, Tensor block_ids, Tensor num_seq_kvcache, int max_seqlens_q, Tensor? "
      "output) -> (Tensor)");
  m.impl("attention_with_kvcache_prefill_bf16", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_matrix>());

  // TODO(amd): attention_with_kvcache_prefill_fp8: FP8 data path. Source: src/attention/entry.cc.
  m.def(
      "attention_with_kvcache_prefill_fp8(Tensor q, Tensor kcache, Tensor vcache,Tensor qscale, "
      "Tensor kscale, Tensor vscale, Tensor cu_seqlens_q,Tensor block_ids, Tensor "
      "num_seq_kvcache, int max_seqlens_q, int quant_type,Tensor? output) -> (Tensor)");
  m.impl("attention_with_kvcache_prefill_fp8", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): attention_with_kvcache_blocksparse_prefill_fp8: FP8 data path. Source:
  // src/attention/entry.cc.
  m.def(
      "attention_with_kvcache_blocksparse_prefill_fp8(Tensor q, Tensor kcache, Tensor "
      "vcache,Tensor qscale, Tensor kscale, Tensor vscale, Tensor cu_seqlens_q,Tensor "
      "block_ids, Tensor num_seq_kvcache, int max_seqlens_q, int quant_type,Tensor? block_mask, "
      "Tensor? output) -> (Tensor)");
  m.impl("attention_with_kvcache_blocksparse_prefill_fp8", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): attention_decode_bf16: matrix-unit implementation. Source: src/attention/entry.cc.
  m.def(
      "attention_decode_bf16(Tensor q, Tensor! kcache, Tensor! vcache, Tensor block_ids, Tensor "
      "num_seq_kvcache, int mtp, bool new_kv_included, bool use_splitk, Tensor? task_map, "
      "Tensor? split_flag, Tensor? output) -> (Tensor)");
  m.impl("attention_decode_bf16", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_matrix>());

  // TODO(amd): attention_decode_fp8: FP8 data path. Source: src/attention/entry.cc.
  m.def(
      "attention_decode_fp8(Tensor q, Tensor! kcache, Tensor! vcache, Tensor block_ids, Tensor "
      "num_seq_kvcache, Tensor qscale, Tensor kscale, Tensor vscale, int mtp, bool "
      "new_kv_included, int quant_type, bool use_splitk, Tensor? task_map, Tensor? split_flag, "
      "Tensor? output) -> (Tensor)");
  m.impl("attention_decode_fp8", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): assign_attention_decode_task: non-FP8 kernel or helper not yet ported. Source:
  // src/attention/entry.cc.
  m.def(
      "assign_attention_decode_task(Tensor num_seq_kvcache, int num_head_kv, int num_seq_q, "
      "bool new_kv_included, int min_process_len, Tensor? task_map) -> (Tensor)");
  m.impl("assign_attention_decode_task", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported>());

  // TODO(amd): count_and_gather: FP8 data path. Source: src/fuse_moe/entry.cc.
  m.def(
      "count_and_gather(Tensor x, Tensor topk_ids, int num_expert, int rank_ep, int "
      "intermediate_size, int num_seq_per_group_avg) -> (Tensor, Tensor, Tensor, Tensor, "
      "Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.impl("count_and_gather", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): fuse_moe: FP8 data path. Source: src/fuse_moe/entry.cc.
  m.def(
      "fuse_moe(Tensor x, Tensor gate_up_weight, Tensor down_weight, Tensor gate_up_scale, "
      "Tensor down_scale, Tensor act_and_mul_scale, Tensor topk_ids, Tensor topk_scale, Tensor "
      "? shared_output, int rank_ep, int num_expert_total, bool use_bf16_mul, Tensor ? output) "
      "-> (Tensor)");
  m.impl("fuse_moe", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): fuse_moe_pertensor_fp8: FP8 data path. Source: src/fuse_moe/entry.cc.
  m.def(
      "fuse_moe_pertensor_fp8(Tensor x, Tensor gate_up_weight, Tensor down_weight, Tensor "
      "gate_up_scale, Tensor down_scale, Tensor act_and_mul_scale, Tensor topk_ids, Tensor "
      "topk_scale, Tensor ? shared_output, int rank_ep, int num_expert_total, bool "
      "use_bf16_mul, Tensor ? output) -> (Tensor)");
  m.impl("fuse_moe_pertensor_fp8", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): fuse_moe_blockwise: FP8 data path. Source: src/fuse_moe/entry.cc.
  m.def(
      "fuse_moe_blockwise(Tensor x, Tensor x_scale, Tensor gate_up_weight, Tensor "
      "gate_up_weight_scale, Tensor down_weight, Tensor down_weight_scale, Tensor topk_ids, "
      "Tensor topk_scale, Tensor ? shared_output, int rank_ep, int num_expert_total, Tensor ? "
      "output) -> (Tensor)");
  m.impl("fuse_moe_blockwise", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): fuse_moe_blockwise_fp8: FP8 data path. Source: src/fuse_moe/entry.cc.
  m.def(
      "fuse_moe_blockwise_fp8(Tensor x, Tensor x_scale, Tensor gate_up_weight, Tensor "
      "gate_up_weight_scale, Tensor down_weight, Tensor down_weight_scale, Tensor topk_ids, "
      "Tensor topk_scale, Tensor ? shared_output, int rank_ep, int num_expert_total, Tensor ? "
      "output) -> (Tensor)");
  m.impl("fuse_moe_blockwise_fp8", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): gemm_bf16xfp32: matrix-unit implementation. Source: src/gemm/entry.cc.
  m.def(
      "gemm_bf16xfp32(Tensor x, Tensor w_high, Tensor w_low, float scale, bool use_fp32_output, "
      "bool use_splitk, Tensor? split_flag) -> (Tensor)");
  m.impl("gemm_bf16xfp32", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_matrix>());

  // TODO(amd): gated_mla_gemm: matrix-unit implementation. Source: src/gemm/entry.cc.
  m.def("gated_mla_gemm(Tensor x, Tensor w, Tensor x2) -> Tensor");
  m.impl("gated_mla_gemm", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_matrix>());

  // TODO(amd): group_gemm_fp8: FP8 data path. Source: src/group_gemm/entry.cc.
  m.def(
      "group_gemm_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, Tensor "
      "y_scale, int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc, Tensor? "
      "task_map_workspace) -> (Tensor)");
  m.impl("group_gemm_fp8", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): group_gemm_pertensor_fp8: FP8 data path. Source: src/group_gemm/entry.cc.
  m.def(
      "group_gemm_pertensor_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, "
      "Tensor y_scale, int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc, Tensor? "
      "task_map_workspace) -> (Tensor)");
  m.impl("group_gemm_pertensor_fp8", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): group_gemm_blockwise_fp8: FP8 data path. Source: src/group_gemm/entry.cc.
  m.def(
      "group_gemm_blockwise_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, "
      "Tensor xscale, Tensor wscale,int num_seq_per_group_avg, Tensor? output, Tensor? "
      "tma_desc, Tensor? task_map_workspace) -> (Tensor)");
  m.impl("group_gemm_blockwise_fp8", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): reformat_x_scale: non-FP8 kernel or helper not yet ported. Source:
  // src/group_gemm/entry.cc.
  m.def(
      "reformat_x_scale(Tensor x_scale, Tensor seqlens, Tensor cu_seqlens, Tensor? out_x_scale, "
      "int num_seq_per_group_avg) -> (Tensor)");
  m.impl("reformat_x_scale", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported>());

  // TODO(amd): group_gemm_fp8_cp_async: FP8 data path. Source: src/group_gemm/entry.cc.
  m.def(
      "group_gemm_fp8_cp_async(Tensor x, Tensor weight, Tensor y_scale, Tensor seqlens, Tensor "
      "cu_seqlens, Tensor tiles, Tensor cu_tiles, bool use_task_map=False) -> (Tensor)");
  m.impl("group_gemm_fp8_cp_async", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): group_gemm_fp8_scatter_cp_async: FP8 data path. Source: src/group_gemm/entry.cc.
  m.def(
      "group_gemm_fp8_scatter_cp_async(Tensor x, Tensor weight, Tensor y_scale, Tensor "
      "row_indices, Tensor seqlens, Tensor cu_seqlens, Tensor tiles, Tensor cu_tiles, bool "
      "use_task_map=False) -> (Tensor)");
  m.impl("group_gemm_fp8_scatter_cp_async", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): fuse_ihc_post_pre: non-FP8 fused post/pre and stage1/stage2 kernels, deferred.
  // Source: src/iHC/entry.cc.
  m.def(
      "fuse_ihc_post_pre(Tensor xa, Tensor residual, Tensor H_post_in, Tensor w, Tensor "
      "hc_scale, Tensor hc_base, float norm_eps, float hc_eps, float magnitude, Tensor? "
      "rms_weight=None, float rms_eps=0.0, bool cast_bfloat_for_norm=False) -> (Tensor y, "
      "Tensor z, Tensor H_post_out)");
  m.impl("fuse_ihc_post_pre", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported>());

  // TODO(amd): fused_rmsnorm_with_scale: FP8 data path. Source: src/normalization/entry.cc.
  m.def(
      "fused_rmsnorm_with_scale(Tensor input, Tensor weight, Tensor scale, float eps, bool "
      "is_moe) -> (Tensor, Tensor, Tensor)");
  m.impl("fused_rmsnorm_with_scale", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): rope_norm_store_kv_fp8: FP8 data path. Source: src/rope/entry.cc.
  m.def(
      "rope_norm_store_kv_fp8(Tensor! kcache, Tensor! vcache, Tensor qkv, Tensor cos_sin, "
      "Tensor num_seqlen_per_req, Tensor q_index, Tensor kvcache_indices, bool is_prefill, "
      "Tensor k_scale, Tensor v_scale, int quant_policy, int max_seqlens, float? upper_max, "
      "Tensor? q_scale_inv, Tensor? q_norm_weight, Tensor? k_norm_weight, Tensor? out_q=None, "
      "Tensor? out_k=None, Tensor? out_v=None, int qk_norm_policy=0) -> (Tensor, Tensor, "
      "Tensor)");
  m.impl("rope_norm_store_kv_fp8", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): fused_sampler: non-FP8 kernel or helper not yet ported. Source:
  // src/sampler/entry.cc.
  m.def(
      "fused_sampler(Tensor logits, Tensor? penalty_mask, Tensor? slot_id, Tensor? "
      "repetition_penalty, float repetition_penalty_val, Tensor? temperature, float "
      "temperature_val, int softmax_policy, Tensor? topk, int topk_val, Tensor? topp, float "
      "topp_val, int max_topk, Tensor? gumbel_noise=None, int seed=0) -> Tensor");
  m.impl("fused_sampler", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported>());

  // TODO(amd): fused_sampler_temperature_sample: non-FP8 kernel or helper not yet ported. Source:
  // src/sampler/entry.cc.
  m.def(
      "fused_sampler_temperature_sample(Tensor logits, Tensor? temperature, float "
      "temperature_val, Tensor? gumbel_noise=None, Tensor? draft_token_ids=None, int seed=0) -> "
      "Tensor");
  m.impl("fused_sampler_temperature_sample", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported>());

  // TODO(amd): stem_oam_prep_paged_kv: FP8 data path. Source: src/stem/entry.cc.
  m.def(
      "stem_oam_prep_paged_kv(Tensor kcache, Tensor vcache, Tensor kscale, Tensor vscale, "
      "Tensor kv_indices, Tensor kv_seq_lens, float lambda_mag, int stem_block_size, int "
      "stem_stride, int quant_type) -> (Tensor, Tensor)");
  m.impl("stem_oam_prep_paged_kv", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): stem_oam_prep_varlen_q: FP8 data path. Source: src/stem/entry.cc.
  m.def(
      "stem_oam_prep_varlen_q(Tensor q_fp8, Tensor qscale, Tensor q_seq_lens, Tensor "
      "cu_seqlens_q, int stem_block_size, int stem_stride) -> Tensor");
  m.impl("stem_oam_prep_varlen_q", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_fp8>());

  // TODO(amd): stem_oam_gemm: matrix-unit implementation. Source: src/stem/entry.cc.
  m.def(
      "stem_oam_gemm(Tensor qflat, Tensor kflat, Tensor vbias, Tensor q_seq_lens, Tensor "
      "kv_seq_lens, int stem_block_size, int stem_stride, bool causal) -> Tensor");
  m.impl("stem_oam_gemm", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported_matrix>());

  // TODO(amd): stem_tpd: non-FP8 kernel or helper not yet ported. Source: src/stem/entry.cc.
  m.def(
      "stem_tpd(Tensor block_logits, Tensor q_seq_lens, Tensor kv_seq_lens, Tensor "
      "num_prompt_tokens, int block_size, float alpha, int initial_blocks, int window_size, "
      "float k_block_num_rate_medium, int k_block_num_bias_medium, float "
      "k_block_num_rate_large, int k_block_num_bias_large) -> Tensor");
  m.impl("stem_tpd", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported>());

  // TODO(amd): topk_filtered: non-FP8 kernel or helper not yet ported. Source: src/topk/entry.cc.
  m.def(
      "topk_filtered(Tensor logits, Tensor ke, Tensor! output, int topk, Tensor num_valid_rows, "
      "Tensor? counters, Tensor? workspace) -> (Tensor)");
  m.impl("topk_filtered", c10::DispatchKey::CUDA,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported>());

  // TODO(amd): topk_filtered_workspace_size: non-FP8 kernel or helper not yet ported. Source:
  // src/topk/entry.cc.
  m.def("topk_filtered_workspace_size(int num_rows, int max_kv_len) -> (int, int)");
  m.impl("topk_filtered_workspace_size", c10::DispatchKey::CompositeExplicitAutograd,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported>());

  // TODO(amd): topk_filtered_min_workspace_size: non-FP8 kernel or helper not yet ported. Source:
  // src/topk/entry.cc.
  m.def("topk_filtered_min_workspace_size(int max_kv_len) -> int");
  m.impl("topk_filtered_min_workspace_size", c10::DispatchKey::CompositeExplicitAutograd,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported>());

  // TODO(amd): topk_filtered_peak_workspace_size: non-FP8 kernel or helper not yet ported. Source:
  // src/topk/entry.cc.
  m.def("topk_filtered_peak_workspace_size(int max_kv_len) -> int");
  m.impl("topk_filtered_peak_workspace_size", c10::DispatchKey::CompositeExplicitAutograd,
         torch::CppFunction::makeFromBoxedFunction<&hpc::amd::unsupported>());
}
