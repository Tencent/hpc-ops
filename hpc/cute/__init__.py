# Copyright (C) 2026 Tencent.

"""Operators written in the CuTe DSL, JIT-compiled on first use.

Shared pieces live here: the ``cutlass`` import and architecture gate
(``_compile``), UE8M0 / MX quantization helpers (``_quant``), and the SM120
block-scaled scale-factor layout (``_layouts``, ``_pack_scale``). Each operator
is a subpackage, e.g. :mod:`hpc.cute.fuse_moe`.

Nothing here imports ``cutlass`` at module import time; it takes several
seconds, so it is deferred to the first compile.
"""

from ._compile import dsl_available, is_sm120, require_sm120
from ._layouts import scale_offset, scale_storage_size
from ._pack_scale import pack_mxfp_weight_scale_sm120

__all__ = [
    "dsl_available",
    "is_sm120",
    "pack_mxfp_weight_scale_sm120",
    "require_sm120",
    "scale_offset",
    "scale_storage_size",
]
