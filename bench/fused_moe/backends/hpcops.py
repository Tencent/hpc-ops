# Copyright (C) 2026 Tencent.

"""hpc-ops FusedMoE backend."""
from __future__ import annotations

from typing import Callable

import torch

from . import register
from .base import (
    A_SCALE_VALUE, Backend, BenchSpec, build_a_scale, build_activation,
    build_fp8_weights, build_routing,
)


class HpcBackend(Backend):
    def __init__(self, label: str):
        assert label == "hpcops", label
        self.label = label
        self.name = label

    def setup(self, spec: BenchSpec) -> Callable[[], None]:
        import hpc
        from sglang.srt.layers.quantization.fp8_kernel import (
            scaled_fp8_quant,
        )

        E = spec.num_expert_local
        N = spec.intermediate_per_rank
        K = spec.hidden

        a_half = build_activation(spec.num_seq, K)
        a_scale = build_a_scale()

        w1_fp8, w2_fp8, w1_scale, w2_scale = build_fp8_weights(E, N, K)

        gate_up_scale = (w1_scale.flatten() * A_SCALE_VALUE).contiguous()
        down_scale = (w2_scale.flatten() * A_SCALE_VALUE).contiguous()
        act_and_mul_scale = torch.full(
            (1,), A_SCALE_VALUE, device="cuda", dtype=torch.float32,
        )

        topk_ids, topk_w = build_routing(
            spec.num_seq, spec.num_expert_total, spec.num_topk,
        )

        rank_ep = 0
        num_expert_total = spec.num_expert_total

        def call_fn():
            x_fp8, _ = scaled_fp8_quant(a_half, a_scale)
            hpc.fuse_moe(
                x_fp8, w1_fp8, w2_fp8,
                gate_up_scale, down_scale, act_and_mul_scale,
                topk_ids, topk_w,
                rank_ep, num_expert_total,
                use_bf16_mul=True,
            )

        self._tensors = (
            a_half, a_scale, w1_fp8, w2_fp8, w1_scale, w2_scale,
            gate_up_scale, down_scale, act_and_mul_scale, topk_ids, topk_w,
        )
        return call_fn

    def extra_metadata(self) -> dict:
        try:
            import hpc
            return {"hpc_module_path": getattr(hpc, "__file__", "?")}
        except Exception as e:
            return {"hpc_import_error": str(e)}


register("hpcops", lambda: HpcBackend("hpcops"))
