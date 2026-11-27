# Copyright (C) 2026 Tencent.
# Copyright 2026, The FlagOS Contributors.

SCHEMAS = (
    "act_mul_and_quant(Tensor input, Tensor scale, bool use_bf16_mul, Tensor? output) -> (Tensor)",
    "masked_act_mul_and_quant(Tensor input, Tensor scale, Tensor num_per_expert, Tensor? output) -> (Tensor)",
    "masked_act_mul_and_blockwise_quant(Tensor input, Tensor num_per_expert, Tensor? output, Tensor? output_scale) -> (Tensor output, Tensor output_scale)",
    "scaled_fp8_quant(Tensor input, Tensor? scale, Tensor? output) -> (Tensor, Tensor)",
    "fuse_allreduce_rmsnorm_high_throughput(Tensor input, Tensor mc_input, Tensor in_residual, Tensor weight, Tensor signal, int rank, int world_size, int num_max_blocks, float rms_norm_eps, Tensor output, Tensor mc_output, Tensor out_residual) -> ()",
    "fuse_allreduce_rmsnorm_low_latency(Tensor input_x, Tensor multicast_x, Tensor data_buffer_ptrs, Tensor! multinode_x, Tensor buffer_flags, int world_size, int rank, bool rmsnorm_fusion, bool launch_with_pdl, bool use_two_shot, Tensor! output_x, Tensor! residual_out, Tensor residual_in, Tensor weight_gamma, float rms_norm_eps) -> ()",
    "attention_prefill_bf16(Tensor q, Tensor k, Tensor v, Tensor seqlens_q, Tensor cu_seqlens_q, int max_seqlens_q, Tensor? output) -> (Tensor)",
    "attention_with_kvcache_prefill_bf16(Tensor q, Tensor kcache, Tensor vcache,Tensor cu_seqlens_q, Tensor block_ids, Tensor num_seq_kvcache, int max_seqlens_q, Tensor? output) -> (Tensor)",
    "attention_with_kvcache_prefill_fp8(Tensor q, Tensor kcache, Tensor vcache,Tensor qscale, Tensor kscale, Tensor vscale, Tensor cu_seqlens_q,Tensor block_ids, Tensor num_seq_kvcache, int max_seqlens_q, int quant_type,Tensor? output) -> (Tensor)",
    "attention_with_kvcache_blocksparse_prefill_fp8(Tensor q, Tensor kcache, Tensor vcache,Tensor qscale, Tensor kscale, Tensor vscale, Tensor cu_seqlens_q,Tensor block_ids, Tensor num_seq_kvcache, int max_seqlens_q, int quant_type,Tensor? block_mask, Tensor? output) -> (Tensor)",
    "attention_decode_bf16(Tensor q, Tensor! kcache, Tensor! vcache, Tensor block_ids, Tensor num_seq_kvcache, int mtp, bool new_kv_included, bool use_splitk, Tensor? task_map, Tensor? split_flag, Tensor? output) -> (Tensor)",
    "attention_decode_fp8(Tensor q, Tensor! kcache, Tensor! vcache, Tensor block_ids, Tensor num_seq_kvcache, Tensor qscale, Tensor kscale, Tensor vscale, int mtp, bool new_kv_included, int quant_type, bool use_splitk, Tensor? task_map, Tensor? split_flag, Tensor? output) -> (Tensor)",
    "assign_attention_decode_task(Tensor num_seq_kvcache, int num_head_kv, int num_seq_q, bool new_kv_included, int min_process_len, Tensor? task_map) -> (Tensor)",
    "count_and_gather(Tensor x, Tensor topk_ids, int num_expert, int rank_ep, int intermediate_size, int num_seq_per_group_avg) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
    "fuse_moe(Tensor x, Tensor gate_up_weight, Tensor down_weight, Tensor gate_up_scale, Tensor down_scale, Tensor act_and_mul_scale, Tensor topk_ids, Tensor topk_scale, Tensor ? shared_output, int rank_ep, int num_expert_total, bool use_bf16_mul, Tensor ? output) -> (Tensor)",
    "fuse_moe_pertensor_fp8(Tensor x, Tensor gate_up_weight, Tensor down_weight, Tensor gate_up_scale, Tensor down_scale, Tensor act_and_mul_scale, Tensor topk_ids, Tensor topk_scale, Tensor ? shared_output, int rank_ep, int num_expert_total, bool use_bf16_mul, Tensor ? output) -> (Tensor)",
    "fuse_moe_blockwise(Tensor x, Tensor x_scale, Tensor gate_up_weight, Tensor gate_up_weight_scale, Tensor down_weight, Tensor down_weight_scale, Tensor topk_ids, Tensor topk_scale, Tensor ? shared_output, int rank_ep, int num_expert_total, Tensor ? output) -> (Tensor)",
    "fuse_moe_blockwise_fp8(Tensor x, Tensor x_scale, Tensor gate_up_weight, Tensor gate_up_weight_scale, Tensor down_weight, Tensor down_weight_scale, Tensor topk_ids, Tensor topk_scale, Tensor ? shared_output, int rank_ep, int num_expert_total, Tensor ? output) -> (Tensor)",
    "gemm_bf16xfp32(Tensor x, Tensor w_high, Tensor w_low, float scale, bool use_fp32_output, bool use_splitk, Tensor? split_flag) -> (Tensor)",
    "gated_mla_gemm(Tensor x, Tensor w, Tensor x2) -> Tensor",
    "group_gemm_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, Tensor y_scale, int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc, Tensor? task_map_workspace) -> (Tensor)",
    "group_gemm_pertensor_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, Tensor y_scale, int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc, Tensor? task_map_workspace) -> (Tensor)",
    "group_gemm_blockwise_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, Tensor xscale, Tensor wscale,int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc, Tensor? task_map_workspace) -> (Tensor)",
    "group_gemm_fp8_cp_async(Tensor x, Tensor weight, Tensor y_scale, Tensor seqlens, Tensor cu_seqlens, Tensor tiles, Tensor cu_tiles, bool use_task_map=False) -> (Tensor)",
    "group_gemm_fp8_scatter_cp_async(Tensor x, Tensor weight, Tensor y_scale, Tensor row_indices, Tensor seqlens, Tensor cu_seqlens, Tensor tiles, Tensor cu_tiles, bool use_task_map=False) -> (Tensor)",
    "fused_rmsnorm_with_scale(Tensor input, Tensor weight, Tensor scale, float eps, bool is_moe) -> (Tensor, Tensor, Tensor)",
    "rope_norm_store_kv_fp8(Tensor! kcache, Tensor! vcache, Tensor qkv, Tensor cos_sin, Tensor num_seqlen_per_req, Tensor q_index, Tensor kvcache_indices, bool is_prefill, Tensor k_scale, Tensor v_scale, int quant_policy, int max_seqlens, float? upper_max, Tensor? q_scale_inv, Tensor? q_norm_weight, Tensor? k_norm_weight, Tensor? out_q=None, Tensor? out_k=None, Tensor? out_v=None, int qk_norm_policy=0) -> (Tensor, Tensor, Tensor)",
    "stem_oam_prep_paged_kv(Tensor kcache, Tensor vcache, Tensor kscale, Tensor vscale, Tensor kv_indices, Tensor kv_seq_lens, float lambda_mag, int stem_block_size, int stem_stride, int quant_type) -> (Tensor, Tensor)",
    "stem_oam_prep_varlen_q(Tensor q_fp8, Tensor qscale, Tensor q_seq_lens, Tensor cu_seqlens_q, int stem_block_size, int stem_stride) -> Tensor",
    "stem_oam_gemm(Tensor qflat, Tensor kflat, Tensor vbias, Tensor q_seq_lens, Tensor kv_seq_lens, int stem_block_size, int stem_stride, bool causal) -> Tensor",
)

DEFERRED = {
    # TODO(amd): act_mul_and_quant: FP8 data path.
    "act_mul_and_quant": "FP8 data path",
    # TODO(amd): masked_act_mul_and_quant: FP8 data path.
    "masked_act_mul_and_quant": "FP8 data path",
    # TODO(amd): masked_act_mul_and_blockwise_quant: FP8 data path.
    "masked_act_mul_and_blockwise_quant": "FP8 data path",
    # TODO(amd): scaled_fp8_quant: FP8 data path.
    "scaled_fp8_quant": "FP8 data path",
    # TODO(amd): fuse_allreduce_rmsnorm_high_throughput: communication kernel not yet implemented.
    "fuse_allreduce_rmsnorm_high_throughput": "communication kernel not yet implemented",
    # TODO(amd): fuse_allreduce_rmsnorm_low_latency: communication kernel not yet implemented.
    "fuse_allreduce_rmsnorm_low_latency": "communication kernel not yet implemented",
    # TODO(amd): attention_prefill_bf16: matrix-unit kernel.
    "attention_prefill_bf16": "matrix-unit kernel",
    # TODO(amd): attention_with_kvcache_prefill_bf16: matrix-unit kernel.
    "attention_with_kvcache_prefill_bf16": "matrix-unit kernel",
    # TODO(amd): attention_with_kvcache_prefill_fp8: FP8 data path.
    "attention_with_kvcache_prefill_fp8": "FP8 data path",
    # TODO(amd): attention_with_kvcache_blocksparse_prefill_fp8: FP8 data path.
    "attention_with_kvcache_blocksparse_prefill_fp8": "FP8 data path",
    # TODO(amd): attention_decode_bf16: matrix-unit kernel.
    "attention_decode_bf16": "matrix-unit kernel",
    # TODO(amd): attention_decode_fp8: FP8 data path.
    "attention_decode_fp8": "FP8 data path",
    # TODO(amd): assign_attention_decode_task: kernel not yet ported to Triton.
    "assign_attention_decode_task": "kernel not yet ported to Triton",
    # TODO(amd): count_and_gather: FP8 data path.
    "count_and_gather": "FP8 data path",
    # TODO(amd): fuse_moe: FP8 data path.
    "fuse_moe": "FP8 data path",
    # TODO(amd): fuse_moe_pertensor_fp8: FP8 data path.
    "fuse_moe_pertensor_fp8": "FP8 data path",
    # TODO(amd): fuse_moe_blockwise: FP8 data path.
    "fuse_moe_blockwise": "FP8 data path",
    # TODO(amd): fuse_moe_blockwise_fp8: FP8 data path.
    "fuse_moe_blockwise_fp8": "FP8 data path",
    # TODO(amd): gemm_bf16xfp32: matrix-unit kernel.
    "gemm_bf16xfp32": "matrix-unit kernel",
    # TODO(amd): gated_mla_gemm: matrix-unit kernel.
    "gated_mla_gemm": "matrix-unit kernel",
    # TODO(amd): group_gemm_fp8: FP8 data path.
    "group_gemm_fp8": "FP8 data path",
    # TODO(amd): group_gemm_pertensor_fp8: FP8 data path.
    "group_gemm_pertensor_fp8": "FP8 data path",
    # TODO(amd): group_gemm_blockwise_fp8: FP8 data path.
    "group_gemm_blockwise_fp8": "FP8 data path",
    # TODO(amd): group_gemm_fp8_cp_async: FP8 data path.
    "group_gemm_fp8_cp_async": "FP8 data path",
    # TODO(amd): group_gemm_fp8_scatter_cp_async: FP8 data path.
    "group_gemm_fp8_scatter_cp_async": "FP8 data path",
    # TODO(amd): fused_rmsnorm_with_scale: FP8 data path.
    "fused_rmsnorm_with_scale": "FP8 data path",
    # TODO(amd): rope_norm_store_kv_fp8: FP8 data path.
    "rope_norm_store_kv_fp8": "FP8 data path",
    # TODO(amd): stem_oam_prep_paged_kv: FP8 data path.
    "stem_oam_prep_paged_kv": "FP8 data path",
    # TODO(amd): stem_oam_prep_varlen_q: FP8 data path.
    "stem_oam_prep_varlen_q": "FP8 data path",
    # TODO(amd): stem_oam_gemm: matrix-unit kernel.
    "stem_oam_gemm": "matrix-unit kernel",
}


def _unimplemented(name, reason):
    def entry(*args, **kwargs):
        raise NotImplementedError(f"hpc::{name}: TODO(amd): {reason}")

    return entry


def register(library):
    for schema in SCHEMAS:
        library.define(schema)
        name = schema.split("(", 1)[0]
        entry = _unimplemented(name, DEFERRED[name])
        library.impl(name, entry, "CUDA")
        if name == "assign_attention_decode_task":
            library.impl(name, entry, "CPU")
