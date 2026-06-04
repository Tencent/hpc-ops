# Copyright (C) 2026 Tencent.

"""vLLM CUTLASS FusedMoE backend."""
from __future__ import annotations

from typing import Callable

import torch

from . import register
from .base import (
    Backend, BenchSpec, build_activation, build_a_scale,
    build_fp8_weights,
)


def _import_vllm_cutlass_apis():
    """Resolve the CUTLASS MoE API exposed by the selected vLLM checkout."""
    from vllm.model_executor.layers.fused_moe.config import (
        fp8_w8a8_moe_quant_config,
    )
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
    except ImportError:
        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (  # noqa: E501
            fused_topk,
        )

    try:
        from vllm.model_executor.layers.fused_moe import cutlass_moe as _cm_mod
        cutlass_moe_fp8 = _cm_mod.cutlass_moe_fp8
        return (
            fused_topk, fp8_w8a8_moe_quant_config, "legacy",
            _patch_legacy_asserts(_cm_mod, cutlass_moe_fp8),
        )
    except (ImportError, AttributeError):
        pass

    try:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEConfig,
            FusedMoEParallelConfig,
            RoutingMethodType,
        )
        try:
            from vllm.model_executor.layers.fused_moe.cutlass_moe import (
                CutlassExpertsFp8,
            )
        except ImportError:
            from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
                CutlassExpertsFp8,
            )
        from vllm.model_executor.layers.fused_moe.prepare_finalize import (
            MoEPrepareAndFinalizeNoDPEPModular,
        )
        import vllm.model_executor.layers.fused_moe.modular_kernel as mk
        from vllm.v1.worker.workspace import init_workspace_manager
        return (
            fused_topk, fp8_w8a8_moe_quant_config, "modular",
            (
                MoEActivation, FusedMoEConfig, FusedMoEParallelConfig,
                RoutingMethodType, CutlassExpertsFp8,
                MoEPrepareAndFinalizeNoDPEPModular, mk, init_workspace_manager,
            ),
        )
    except (ImportError, AttributeError) as e:
        raise ImportError(
            "Unable to resolve either legacy cutlass_moe_fp8 or latest "
            f"modular CUTLASS MoE APIs from this vLLM checkout: {e}"
        ) from e


def _patch_legacy_asserts(cm_mod, original_fn):
    """Wrap a legacy helper so scalar activation scales are accepted."""
    import torch
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.cutlass_moe import (
        CutlassExpertsFp8,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNoEP,
    )

    def patched_cutlass_moe_fp8(
        a, w1_q, w2_q, topk_weights, topk_ids,
        ab_strides1, ab_strides2, c_strides1, c_strides2,
        quant_config, activation="silu", global_num_experts=-1,
        expert_map=None, apply_router_weight_on_input=False,
    ):
        assert quant_config is not None
        assert (quant_config.w1_scale is None or (
            quant_config.per_out_ch_quant == (
                quant_config.w1_scale.size(1) == w1_q.size(1))))

        num_experts = (
            global_num_experts if global_num_experts != -1 else w1_q.size(0))

        fn = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            CutlassExpertsFp8(
                out_dtype=a.dtype,
                ab_strides1=ab_strides1,
                ab_strides2=ab_strides2,
                c_strides1=c_strides1,
                c_strides2=c_strides2,
                quant_config=quant_config,
            ),
        )
        return fn(
            a, w1_q, w2_q, topk_weights, topk_ids,
            activation=activation,
            global_num_experts=num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

    return patched_cutlass_moe_fp8


class CutlassBackend(Backend):
    name = "vllm_cutlass"

    def setup(self, spec: BenchSpec) -> Callable[[], None]:
        fused_topk, fp8_w8a8_moe_quant_config, cutlass_mode, cutlass_api = (
            _import_vllm_cutlass_apis()
        )

        E = spec.num_expert_local
        N = spec.intermediate_per_rank
        K = spec.hidden

        w1_fp8, w2_fp8, w1_scale, w2_scale = build_fp8_weights(E, N, K)
        a_half = build_activation(spec.num_seq, K)
        a_scale = build_a_scale()
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

        ab_strides1 = torch.full((E,), K, dtype=torch.int64, device="cuda")
        ab_strides2 = torch.full((E,), N, dtype=torch.int64, device="cuda")
        c_strides1 = torch.full((E,), 2 * N, dtype=torch.int64, device="cuda")
        c_strides2 = torch.full((E,), K, dtype=torch.int64, device="cuda")

        if cutlass_mode == "legacy":
            cutlass_moe_fp8 = cutlass_api

            def call_fn():
                cutlass_moe_fp8(
                    a=a_half, w1_q=w1_fp8, w2_q=w2_fp8,
                    topk_weights=topk_w, topk_ids=topk_ids,
                    ab_strides1=ab_strides1, ab_strides2=ab_strides2,
                    c_strides1=c_strides1, c_strides2=c_strides2,
                    quant_config=qc, activation="silu",
                    global_num_experts=E,
                )
            extra_keepalive = (cutlass_moe_fp8,)
        else:
            (
                MoEActivation, FusedMoEConfig, FusedMoEParallelConfig,
                RoutingMethodType, CutlassExpertsFp8,
                MoEPrepareAndFinalizeNoDPEPModular, mk, init_workspace_manager,
            ) = cutlass_api
            init_workspace_manager(torch.device("cuda"))
            parallel_config = FusedMoEParallelConfig.make_no_parallel()
            moe_config = FusedMoEConfig(
                num_experts=E,
                experts_per_token=spec.num_topk,
                hidden_dim=K,
                intermediate_size_per_partition=N,
                num_local_experts=E,
                num_logical_experts=E,
                activation=MoEActivation.SILU,
                device=a_half.device,
                routing_method=RoutingMethodType.Default,
                moe_parallel_config=parallel_config,
                in_dtype=a_half.dtype,
                max_num_tokens=spec.num_seq,
            )
            experts = CutlassExpertsFp8(moe_config, qc)
            prepare_finalize = MoEPrepareAndFinalizeNoDPEPModular()
            try:
                cutlass_kernel = mk.FusedMoEKernel(
                    prepare_finalize, experts,
                    moe_parallel_config=parallel_config,
                    inplace=False,
                )
            except TypeError:
                cutlass_kernel = mk.FusedMoEKernel(prepare_finalize, experts)

            def call_fn():
                cutlass_kernel.apply(
                    a_half, w1_fp8, w2_fp8, topk_w, topk_ids,
                    MoEActivation.SILU, E, None, False,
                )
            extra_keepalive = (
                parallel_config, moe_config, experts,
                prepare_finalize, cutlass_kernel,
            )

        self._tensors = (
            a_half, a_scale, a2_scale, w1_fp8, w2_fp8, w1_scale, w2_scale,
            topk_ids, topk_w, score, qc,
            ab_strides1, ab_strides2, c_strides1, c_strides2,
            extra_keepalive,
        )
        return call_fn


register("vllm_cutlass", CutlassBackend)
