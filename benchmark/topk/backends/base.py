# Copyright (C) 2026 Tencent.

"""Provider interface shared by every Top-K benchmark backend."""

import torch


class TopKBackend:
    """One selector call per (logits, ends, top_k) triple.

    ``warmup`` runs once before any timing (JIT compilation, plan caches);
    ``prepare`` allocates the per-shape scratch and returns the callable to
    time together with its int32 ``[m, top_k]`` index output.
    """

    name = ""

    def warmup(self, device, ks):
        """Compile whatever the backend caches per (capacity, cluster) shape."""

    def supports(self, m, n):
        """Whether this backend can address a launch of ``m`` rows of ``n``."""
        return True

    def prepare(self, logits, ke, num_valid, top_k, sm_count):
        raise NotImplementedError
