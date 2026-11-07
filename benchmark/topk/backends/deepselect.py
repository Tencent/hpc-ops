# Copyright (C) 2026 Tencent.

"""DeepSelect SM90 provider (LightSeek tokenspeed PR 1737 rendition, MIT).

Thin adapter over the vendored ``deep_select.py`` CuTe DSL kernel. Requires
the optional ``nvidia-cutlass-dsl`` and ``apache-tvm-ffi`` packages; when
they are missing the backend reports itself unavailable and the benchmark
skips it. The kernel's per-row offsets are 32-bit element indices, so
launches with ``rows * width >= 2**31`` are rejected by ``supports``.
"""

import os
import sys

import torch

from .base import TopKBackend


_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

try:
    from deep_select import capacity_for, choose_cluster_size, deepselect_topk
    from deep_select import warmup as _compile_selectors

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on optional packages
    IMPORT_ERROR = exc


class DeepSelectBackend(TopKBackend):
    name = "deepselect"
    available = IMPORT_ERROR is None

    def __init__(self):
        super().__init__()
        self._compiled_ks = None

    def warmup(self, device, ks):
        """Compile one selector per (capacity, cluster size) ahead of timing."""
        if not self.available or self._compiled_ks:
            return
        capacities = sorted({capacity_for(k) for k in ks})
        _compile_selectors(tuple(capacities), device)
        self._compiled_ks = capacities

    def supports(self, m, n):
        return m * n < 2**31

    def prepare(self, logits, ke, num_valid, top_k, sm_count):
        m, n = logits.shape
        capacity = capacity_for(top_k)
        cluster_size = choose_cluster_size(m, n, capacity, sm_count)
        indices = torch.empty(m, top_k, dtype=torch.int32, device="cuda")
        values = torch.empty(m, top_k, dtype=torch.float32, device="cuda")

        def run():
            deepselect_topk(
                logits,
                ke,
                top_k,
                capacity=capacity,
                cluster_size=cluster_size,
                indices=indices,
                values=values,
            )

        return run, indices
