# Copyright (C) 2026 Tencent.

"""HPC-Ops exact FP32 Top-K provider."""

import torch

import hpc

from .base import TopKBackend


class HpcOpsBackend(TopKBackend):
    name = "hpc-ops"

    def prepare(self, logits, ke, num_valid, top_k, sm_count):
        m, n = logits.shape
        cnt_bytes, ws_bytes = hpc.topk_workspace_size(m, n)
        counters = torch.zeros(cnt_bytes, dtype=torch.uint8, device="cuda")
        workspace = torch.empty(ws_bytes, dtype=torch.uint8, device="cuda")
        out = torch.empty(m, top_k, dtype=torch.int32, device="cuda")

        def run():
            hpc.topk(logits, ke, out, num_valid, top_k, counters, workspace)

        return run, out
