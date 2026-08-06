# Copyright (C) 2026 Tencent.

"""hpc-ops FusedMoE backend."""
from __future__ import annotations

from typing import Callable

import torch

from . import register
from .base import (
    A_SCALE_VALUE, Backend, BenchSpec, DTYPE_BF16, build_a_scale, build_activation,
    build_bf16_weights, build_fp8_weights, build_routing,
)


class HpcBackend(Backend):
    def __init__(self, label: str):
        assert label == "hpcops", label
        self.label = label
        self.name = label

    def setup(self, spec: BenchSpec) -> Callable[[], None]:
        import hpc

        if spec.dtype == "bf16":
            return self._setup_bf16(spec, hpc)
        return self._setup_fp8(spec, hpc)

    def _setup_bf16(self, spec: BenchSpec, hpc) -> Callable[[], None]:
        E = spec.num_expert_local
        N = spec.intermediate_per_rank
        K = spec.hidden

        x_bf16 = build_activation(spec.num_seq, K, seed=spec.seed, dtype=DTYPE_BF16)
        w1_bf16, w2_bf16 = build_bf16_weights(E, N, K, seed=spec.seed + 1)
        topk_ids, topk_w = build_routing(
            spec.num_seq, spec.num_expert_total, spec.num_topk, seed=spec.seed + 2,
        )

        rank_ep = 0
        num_expert_total = spec.num_expert_total

        def call_fn():
            hpc.fuse_moe_bf16(
                x_bf16, w1_bf16, w2_bf16,
                topk_ids, topk_w,
                rank_ep, num_expert_total,
            )

        self._tensors = (x_bf16, w1_bf16, w2_bf16, topk_ids, topk_w)
        return call_fn

    def _setup_fp8(self, spec: BenchSpec, hpc) -> Callable[[], None]:
        E = spec.num_expert_local
        N = spec.intermediate_per_rank
        K = spec.hidden
        scaled_fp8_quant = lambda x, scale=None: torch.ops.hpc.scaled_fp8_quant(x, scale, None)

        a_half = build_activation(spec.num_seq, K, seed=spec.seed)
        a_scale = build_a_scale()

        w1_fp8, w2_fp8, w1_scale, w2_scale = build_fp8_weights(E, N, K, seed=spec.seed + 1)

        gate_up_scale = (w1_scale.flatten() * A_SCALE_VALUE).contiguous()
        down_scale = (w2_scale.flatten() * A_SCALE_VALUE).contiguous()
        act_and_mul_scale = torch.full(
            (1,), A_SCALE_VALUE, device="cuda", dtype=torch.float32,
        )

        topk_ids, topk_w = build_routing(
            spec.num_seq, spec.num_expert_total, spec.num_topk, seed=spec.seed + 2,
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
