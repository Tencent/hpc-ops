# Copyright (C) 2026 Tencent.

"""CuTe DSL implementation of the SM120 fused MoE.

CuTe DSL kernels for the full SM120 MoE pipeline: routing, gather,
block-scaled grouped GEMM, SwiGLU requantization, and top-k reduction. Two
weight formats share it: W4A8 MXFP4 x MXFP8, with weights in the interleaved
UE8M0 scale layout from :func:`pack_mxfp_weight_scale_sm120`, and 128-block
FP8, requantized against UE8M0 scales by :func:`prepare_fp8_mx_mma_weights`.

Not auto-imported by ``hpc/__init__.py``, which only scans top-level modules.
What callers use is re-exported from ``hpc`` on first access (``_CUTE_EXPORTS``);
the entry points are also wrapped in ``hpc/fuse_moe.py``.
"""

from .. import dsl_available, pack_mxfp_weight_scale_sm120
from ._ops import (
    fuse_moe_blockwise_fp8_dsl,
    fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl,
    register,
)
from ._pipeline import (
    fuse_moe_blockwise_fp8,
    fuse_moe_blockwise_w4a8_mxfp4_mxfp8,
    prepare_fp8_mx_mma_weights,
)
from ._warmup import BlockwiseShape, WarmupUnit, blockwise_warmup_units, fp8_blockwise_warmup_units

__all__ = [
    "BlockwiseShape",
    "WarmupUnit",
    "blockwise_warmup_units",
    "dsl_available",
    "fp8_blockwise_warmup_units",
    "fuse_moe_blockwise_fp8",
    "fuse_moe_blockwise_fp8_dsl",
    "prepare_fp8_mx_mma_weights",
    "fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl",
    "pack_mxfp_weight_scale_sm120",
    "register",
    "fuse_moe_blockwise_w4a8_mxfp4_mxfp8",
]
