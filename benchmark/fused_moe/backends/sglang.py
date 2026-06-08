# Copyright (C) 2025 Tencent.

"""SGLang Triton FusedMoE backend."""
from __future__ import annotations

import os
import sys
import types
from typing import Callable

import torch

from . import register
from .base import (
    Backend, BenchSpec, build_a_scale, build_activation,
    build_fp8_weights,
)


class SglangBackend(Backend):
    name = "sglang"

    def __init__(self):
        self._cfg_source: str | None = None

    def setup(self, spec: BenchSpec) -> Callable[[], None]:
        import triton.language as tl

        sgl_root = os.environ.get("SGLANG_ROOT")
        if sgl_root:
            sgl_python = os.path.join(sgl_root, "python")
            if not os.path.isdir(os.path.join(sgl_python, "sglang")):
                raise RuntimeError(
                    "SGLANG_ROOT must point to a local sglang checkout containing "
                    "python/sglang"
                )
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
            if sgl_python not in sys.path:
                sys.path.insert(0, sgl_python)

        from sglang.srt.server_args import set_global_server_args_for_scheduler
        set_global_server_args_for_scheduler(types.SimpleNamespace(
            enable_deterministic_inference=False,
            enable_fused_moe_sum_all_reduce=False,
        ))

        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (  # noqa: E501
            try_get_optimal_moe_config,
        )
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (  # noqa: E501
            act_and_mul_triton,
            invoke_fused_moe_kernel,
            moe_sum_reduce_triton,
        )
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (  # noqa: E501
            moe_align_block_size,
        )

        try:
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                fused_topk,
            )
        except ImportError:
            from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (  # noqa: E501
                fused_topk,
            )

        E = spec.num_expert_local
        N = spec.intermediate_per_rank
        K = spec.hidden
        M = spec.num_seq

        w1_fp8, w2_fp8, w1_scale, w2_scale = build_fp8_weights(E, N, K, seed=spec.seed + 1)
        a_half = build_activation(M, K, seed=spec.seed)
        a1s = build_a_scale()
        a2s = build_a_scale()

        g = torch.Generator(device="cuda").manual_seed(spec.seed + 2)
        score = torch.randn((M, spec.num_expert_total), dtype=torch.half,
                            device="cuda", generator=g)
        topk_w, topk_ids, _ = fused_topk(
            a_half, score, spec.num_topk, renormalize=False,
        )

        config = try_get_optimal_moe_config(
            w1_shape=tuple(w1_fp8.shape),
            w2_shape=tuple(w2_fp8.shape),
            top_k=spec.num_topk,
            dtype="fp8_w8a8",
            M=M,
            is_marlin=False,
            block_shape=None,
        )
        self._cfg_source = "sglang.try_get_optimal_moe_config"

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], E,
        )

        N2 = N * 2
        inter1 = torch.empty((M * spec.num_topk, N2),
                             dtype=torch.half, device="cuda")
        inter2 = torch.empty((M * spec.num_topk, N),
                             dtype=torch.half, device="cuda")
        inter3 = torch.empty((M, spec.num_topk, K),
                             dtype=torch.half, device="cuda")
        out = torch.empty((M, K), dtype=torch.half, device="cuda")

        def call_fn():
            invoke_fused_moe_kernel(
                A=a_half, B=w1_fp8, bias=None, C=inter1,
                A_scale=a1s, B_scale=w1_scale, B_zp=None,
                topk_weights=topk_w, topk_ids=topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False, top_k=spec.num_topk, config=config,
                compute_type=tl.float16,
                use_fp8_w8a8=True, use_int8_w8a8=False,
                use_int8_w8a16=False, use_int4_w4a16=False,
                per_channel_quant=False, block_shape=None,
                filter_expert=False,
            )
            act_and_mul_triton(
                inter1, inter2, config, topk_ids, expert_ids, False, "silu")
            invoke_fused_moe_kernel(
                A=inter2, B=w2_fp8, bias=None, C=inter3,
                A_scale=a2s, B_scale=w2_scale, B_zp=None,
                topk_weights=topk_w, topk_ids=topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=True, top_k=1, config=config,
                compute_type=tl.float16,
                use_fp8_w8a8=True, use_int8_w8a8=False,
                use_int8_w8a16=False, use_int4_w4a16=False,
                per_channel_quant=False, block_shape=None,
                filter_expert=False, router_topk=spec.num_topk,
            )
            moe_sum_reduce_triton(
                inter3, out, routed_scaling_factor=1.0)

        self._tensors = (
            a_half, a1s, a2s, w1_fp8, w2_fp8, w1_scale, w2_scale,
            topk_ids, topk_w, score,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            inter1, inter2, inter3, out,
        )
        return call_fn

    def extra_metadata(self) -> dict:
        return {"sgl_cfg_source": self._cfg_source}


register("sglang", SglangBackend)
