# Copyright (C) 2026 Tencent.

import torch

from hpc._amd.utils.validation import check_bf16, check_tensor, require

from .stem_tpd import stem_tpd_async


SCHEMAS = (
    "stem_tpd(Tensor block_logits, Tensor q_seq_lens, Tensor kv_seq_lens, Tensor num_prompt_tokens, int block_size, float alpha, int initial_blocks, int window_size, float k_block_num_rate_medium, int k_block_num_bias_medium, float k_block_num_rate_large, int k_block_num_bias_large) -> Tensor",
)


def stem_tpd(
    block_logits,
    q_seq_lens,
    kv_seq_lens,
    num_prompt_tokens,
    block_size,
    alpha,
    initial_blocks,
    window_size,
    k_block_num_rate_medium,
    k_block_num_bias_medium,
    k_block_num_rate_large,
    k_block_num_bias_large,
):
    check_bf16(block_logits, block_logits, "block_logits")
    require(block_logits.ndim == 4, "block_logits must have shape [B,H,Q,K]")
    batch, heads, qrows, kcols = block_logits.shape
    require(kcols <= 32768 and block_size > 0, "K must be <= 32768 and block_size positive")
    for tensor, name in (
        (q_seq_lens, "q_seq_lens"),
        (kv_seq_lens, "kv_seq_lens"),
        (num_prompt_tokens, "num_prompt_tokens"),
    ):
        check_tensor(tensor, block_logits, torch.int32, name)
        require(tensor.shape == (batch,), f"{name} must have shape [B]")
    output = torch.empty_like(block_logits, dtype=torch.uint8)
    stem_tpd_async(
        block_logits,
        output,
        q_seq_lens,
        kv_seq_lens,
        num_prompt_tokens,
        block_size,
        alpha,
        initial_blocks,
        window_size,
        k_block_num_rate_medium,
        k_block_num_bias_medium,
        k_block_num_rate_large,
        k_block_num_bias_large,
    )
    return output


# TODO(amd): STEM preprocessing consumes FP8; OAM GEMM needs matrix-unit kernels.


def register(library):
    for schema in SCHEMAS:
        library.define(schema)
    library.impl("stem_tpd", stem_tpd, "CUDA")
