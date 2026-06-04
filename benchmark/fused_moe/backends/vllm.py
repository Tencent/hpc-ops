# Copyright (C) 2026 Tencent.

"""vLLM Triton FusedMoE backend."""
from __future__ import annotations

from typing import Callable

import torch

from . import register
from .base import (
    Backend, BenchSpec, build_activation, build_a_scale,
    build_fp8_weights,
)


def _import_vllm_apis():
    """Resolve vLLM symbols used by this backend."""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.config import (
        fp8_w8a8_moe_quant_config,
    )
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
    except ImportError:
        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (  # noqa: E501
            fused_topk,
        )
    return fused_experts, fused_topk, fp8_w8a8_moe_quant_config


class TritonBackend(Backend):
    name = "vllm"

    def setup(self, spec: BenchSpec) -> Callable[[], None]:
        fused_experts, fused_topk, fp8_w8a8_moe_quant_config = (
            _import_vllm_apis()
        )

        E = spec.num_expert_local
        N = spec.intermediate_per_rank
        K = spec.hidden

        w1_fp8, w2_fp8, w1_scale, w2_scale = build_fp8_weights(E, N, K)
        a_half = build_activation(spec.num_seq, K)
        a_scale = build_a_scale()
        # vllm's per-tensor a-scale tensor uses the same shape convention.
        a2_scale = build_a_scale()

        score = torch.randn(
            (spec.num_seq, spec.num_expert_total), dtype=torch.half,
            device="cuda",
        )
        topk_w, topk_ids, _ = fused_topk(
            a_half, score, spec.num_topk, renormalize=False,
        )

        qc = fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale, w2_scale=w2_scale,
            a1_scale=a_scale, a2_scale=a2_scale,
            per_act_token_quant=False, per_out_ch_quant=False,
        )

        def call_fn():
            fused_experts(
                a_half, w1_fp8, w2_fp8, topk_w, topk_ids,
                quant_config=qc,
            )

        self._tensors = (
            a_half, a_scale, a2_scale, w1_fp8, w2_fp8, w1_scale, w2_scale,
            topk_ids, topk_w, score, qc,
        )
        return call_fn


register("vllm", TritonBackend)
