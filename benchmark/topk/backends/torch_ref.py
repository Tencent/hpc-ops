# Copyright (C) 2026 Tencent.

"""torch.topk reference provider."""

import torch

from .base import TopKBackend


class TorchRefBackend(TopKBackend):
    name = "torch"

    def prepare(self, logits, ke, num_valid, top_k, sm_count):
        out = torch.empty(logits.shape[0], top_k, dtype=torch.int32, device="cuda")

        def run():
            out.copy_(torch.topk(logits, top_k, dim=-1).indices.to(torch.int32))

        return run, out
