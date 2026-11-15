# Copyright (C) 2026 Tencent.

"""``torch.library`` registration for the CuTe DSL fused MoE.

The DSL path registers its own operator names (``..._dsl``) so they stay
distinct from the SM90 C++ operator names in the ``hpc`` namespace. Registration
is deferred to first use because it pulls in ``cutlass``.

The operator writes into a caller-provided ``output`` and returns nothing: a
custom op may not return a tensor that aliases one of its inputs, so the
allocate-or-reuse choice lives in the Python wrapper below.
"""

from typing import Optional

import torch
from torch import Tensor

_registered = False


def register() -> None:
    """Register ``hpc::fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl``. Idempotent."""
    global _registered
    if _registered:
        return
    _registered = True

    from ._pipeline import (
        fuse_moe_blockwise_fp8,
        fuse_moe_blockwise_w4a8_mxfp4_mxfp8,
    )

    @torch.library.custom_op(
        "hpc::fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl",
        mutates_args={"output"},
        device_types="cuda",
    )
    def _blockwise_dsl(
        x: Tensor,
        x_scale: Optional[Tensor],
        gate_up_weight: Tensor,
        gate_up_weight_scale: Tensor,
        down_weight: Tensor,
        down_weight_scale: Tensor,
        topk_ids: Tensor,
        topk_scale: Tensor,
        shared_output: Optional[Tensor],
        rank_ep: int,
        num_expert_total: int,
        output: Tensor,
        swiglu_limit: float,
    ) -> None:
        fuse_moe_blockwise_w4a8_mxfp4_mxfp8(
            x,
            x_scale,
            gate_up_weight,
            gate_up_weight_scale,
            down_weight,
            down_weight_scale,
            topk_ids,
            topk_scale,
            shared_output=shared_output,
            rank_ep=rank_ep,
            num_expert_total=num_expert_total,
            output=output,
            swiglu_limit=swiglu_limit,
        )

    @_blockwise_dsl.register_fake
    def _(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        shared_output,
        rank_ep,
        num_expert_total,
        output,
        swiglu_limit,
    ):
        return None

    @torch.library.custom_op(
        "hpc::fuse_moe_blockwise_fp8_dsl",
        mutates_args={"output"},
        device_types="cuda",
    )
    def _blockwise_fp8_dsl(
        x: Tensor,
        x_scale: Tensor,
        gate_up_weight: Tensor,
        gate_up_weight_scale: Tensor,
        down_weight: Tensor,
        down_weight_scale: Tensor,
        topk_ids: Tensor,
        topk_scale: Tensor,
        shared_output: Optional[Tensor],
        rank_ep: int,
        num_expert_total: int,
        output: Tensor,
        swiglu_limit: float,
        mx_mma: bool,
    ) -> None:
        fuse_moe_blockwise_fp8(
            x,
            x_scale,
            gate_up_weight,
            gate_up_weight_scale,
            down_weight,
            down_weight_scale,
            topk_ids,
            topk_scale,
            shared_output=shared_output,
            rank_ep=rank_ep,
            num_expert_total=num_expert_total,
            output=output,
            swiglu_limit=swiglu_limit,
            mx_mma=mx_mma,
        )

    @_blockwise_fp8_dsl.register_fake
    def _(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        shared_output,
        rank_ep,
        num_expert_total,
        output,
        swiglu_limit,
        mx_mma,
    ):
        return None


def fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl(
    x: Tensor,
    x_scale: Optional[Tensor],
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int = 0,
    num_expert_total: int = 0,
    shared_output: Optional[Tensor] = None,
    output: Optional[Tensor] = None,
    swiglu_limit: Optional[float] = None,
) -> Tensor:
    """CuTe DSL counterpart of ``hpc.fuse_moe.fuse_moe_blockwise_w4a8_mxfp4_mxfp8``."""
    if output is None:
        output = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    register()
    experts = num_expert_total if num_expert_total else gate_up_weight.shape[0]
    limit = -1.0 if swiglu_limit is None else swiglu_limit
    if torch.compiler.is_compiling():
        torch.ops.hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl(
            x,
            x_scale,
            gate_up_weight,
            gate_up_weight_scale,
            down_weight,
            down_weight_scale,
            topk_ids,
            topk_scale,
            shared_output,
            rank_ep,
            experts,
            output,
            limit,
        )
        return output
    _eager_impl("fuse_moe_blockwise_w4a8_mxfp4_mxfp8")(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        shared_output=shared_output,
        rank_ep=rank_ep,
        num_expert_total=experts,
        output=output,
        swiglu_limit=limit,
    )
    return output


_pipeline_impls: dict = {}


def _eager_impl(name: str):
    """The pipeline function behind an operator, for calls no tracer is watching.

    Going through ``torch.library.custom_op`` costs ~88us per call on top of
    the pipeline itself -- schema validation and the mutation bookkeeping --
    which at 48 MoE layers is milliseconds of a decode step spent in the
    dispatcher. The boundary only has to exist for a tracer: it is what keeps
    the kernels opaque to Dynamo and tells functionalization what is mutated.
    In eager the two are the same call, so take the short way.
    """
    impl = _pipeline_impls.get(name)
    if impl is None:
        from . import _pipeline

        impl = _pipeline_impls[name] = getattr(_pipeline, name)
    return impl


def _prepared_fp8_scales(
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
):
    """Pack both weight scales for the MX MMA, requantizing the weights if needed.

    Prepared scales take the early return, which is the only path a server ever
    takes and so the only one that has to be cheap.
    """
    if (
        gate_up_weight_scale.dtype == torch.uint8
        and gate_up_weight_scale.dim() == 2
        and down_weight_scale.dtype == torch.uint8
        and down_weight_scale.dim() == 2
    ):
        return gate_up_weight_scale, down_weight_scale

    from ._pipeline import prepare_fp8_mx_mma_weights

    # Before the capture check, which Dynamo cannot trace: reaching it under a
    # tracer would surface as a graph break about a non-Tensor return rather
    # than the thing the caller needs to hear. Requantizing in place is not
    # something a traced graph can carry anyway.
    if torch.compiler.is_compiling():
        raise RuntimeError(
            "FP8 MoE weights are not prepared for the mx_mma path and the call is "
            "being traced; call prepare_fp8_mx_mma_weights at load time and pass "
            "the packed scales"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FP8 MoE weights are not prepared for the mx_mma path and a CUDA graph "
            "capture is in progress; call prepare_fp8_mx_mma_weights at load time"
        )
    _, gate_up_packed = prepare_fp8_mx_mma_weights(gate_up_weight, gate_up_weight_scale)
    _, down_packed = prepare_fp8_mx_mma_weights(down_weight, down_weight_scale)
    return gate_up_packed, down_packed


def fuse_moe_blockwise_fp8_dsl(
    x: Tensor,
    x_scale: Tensor,
    gate_up_weight: Tensor,
    gate_up_weight_scale: Tensor,
    down_weight: Tensor,
    down_weight_scale: Tensor,
    topk_ids: Tensor,
    topk_scale: Tensor,
    rank_ep: int = 0,
    num_expert_total: int = 0,
    shared_output: Optional[Tensor] = None,
    output: Optional[Tensor] = None,
    swiglu_limit: Optional[float] = None,
    mx_mma: bool = True,
) -> Tensor:
    """CuTe DSL SM120 counterpart of ``hpc.fuse_moe.fuse_moe_blockwise_fp8``.

    ``mx_mma=True`` (default) packs 128-block float32 scales as UE8M0 and runs
    the V4 fused GEMM. Pass ``False`` for the exact float32 128-block path.

    Float32 weight scales are prepared here rather than inside the operator,
    because preparing them requantizes the weights in place -- a mutation the
    operator would have to declare, and one that a graph capture would record.
    Servers should call :func:`prepare_fp8_mx_mma_weights` at load time and
    hand the packed scales straight in; this is the convenience path for eager
    callers.
    """
    if mx_mma:
        gate_up_weight_scale, down_weight_scale = _prepared_fp8_scales(
            gate_up_weight, gate_up_weight_scale, down_weight, down_weight_scale
        )
    if output is None:
        output = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    # Ahead of the branch below: registering an operator is itself not
    # traceable, so a tracer must never be the first caller to reach it.
    register()
    experts = num_expert_total if num_expert_total else gate_up_weight.shape[0]
    limit = -1.0 if swiglu_limit is None else swiglu_limit
    if torch.compiler.is_compiling():
        torch.ops.hpc.fuse_moe_blockwise_fp8_dsl(
            x,
            x_scale,
            gate_up_weight,
            gate_up_weight_scale,
            down_weight,
            down_weight_scale,
            topk_ids,
            topk_scale,
            shared_output,
            rank_ep,
            experts,
            output,
            limit,
            mx_mma,
        )
        return output
    _eager_impl("fuse_moe_blockwise_fp8")(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        shared_output=shared_output,
        rank_ep=rank_ep,
        num_expert_total=experts,
        output=output,
        swiglu_limit=limit,
        mx_mma=mx_mma,
    )
    return output
