import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import torch.nn.functional as F
import pytest
from utils import allclose


def naive_deepseekv4_routing_method(scores, bias, tid2eid, input_ids, topk, route_scale, is_hash):
    scores = F.softplus(scores.float()).sqrt()
    original_scores = scores
    if bias is not None:
        scores = scores + bias
    if is_hash:
        indices = tid2eid[input_ids]
    else:
        indices = scores.topk(topk, dim=-1)[1]
    weights = original_scores.gather(1, indices.to(dtype=torch.int64))
    weights /= weights.sum(dim=-1, keepdim=True)
    weights *= route_scale
    return weights, indices.int()


@pytest.mark.parametrize("batch_size", [128, 16384])
@pytest.mark.parametrize("vocab_size", [129280])
@pytest.mark.parametrize("dim", [4096])
@pytest.mark.parametrize("n_routed_experts", [256])
@pytest.mark.parametrize("n_activated_experts", [6])
@pytest.mark.parametrize("route_scale", [1.5])
@pytest.mark.parametrize("is_hash", [True, False])
def test_deepseekv4_routing_method(
    batch_size, vocab_size, dim, n_routed_experts, n_activated_experts, route_scale, is_hash
):
    torch.cuda.manual_seed(13)

    scores = torch.randn((batch_size, n_routed_experts), device="cuda").to(dtype=torch.bfloat16)
    output_weights = torch.empty(
        (batch_size, n_activated_experts), dtype=torch.float32, device="cuda"
    )
    output_indices = torch.empty(
        (batch_size, n_activated_experts), dtype=torch.int32, device="cuda"
    )
    if is_hash:
        bias = None
        tid2eid = (
            torch.randint(0, n_routed_experts, (vocab_size, n_activated_experts), device="cuda")
            .int()
            .contiguous()
        )
        input_ids = torch.randint(0, vocab_size, (batch_size,), device="cuda").int().contiguous()
    else:
        bias = torch.randn((n_routed_experts,), dtype=torch.float32, device="cuda")
        tid2eid = None
        input_ids = None

    ref_weights, ref_indices = naive_deepseekv4_routing_method(
        scores, bias, tid2eid, input_ids, n_activated_experts, route_scale, is_hash
    )

    real_weights, real_indices = hpc.deepseekv4_routing_method(
        scores,
        bias,
        input_ids,
        tid2eid,
        n_activated_experts,
        route_scale,
        is_hash,
        output_weights,
        output_indices,
    )

    assert allclose(ref_weights, real_weights)
    assert allclose(ref_indices, real_indices)
