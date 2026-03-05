import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
import pytest
from pathlib import Path
from utils import allclose
from typing import Optional


def grouped_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    else:
        group_scores = (
            scores.view(num_token, num_expert_group, -1).max(dim=-1).values
        )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

    print(topk_weights)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    print(topk_weights)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


@pytest.mark.parametrize("num_tokens", [16])
@pytest.mark.parametrize("num_experts", [192])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("renormalize", [True])
@pytest.mark.parametrize("num_expert_group", [1])
@pytest.mark.parametrize("topk_group", [1])
@pytest.mark.parametrize("scoring_func", ["sigmoid"])
@pytest.mark.parametrize("routed_scaling_factor", [2.826])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_grouped_topk(
    num_tokens: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
    routed_scaling_factor: float,
    dtype: torch.dtype,
):
    torch.cuda.manual_seed(10086)
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")

    e_score_correction_bias = torch.randn((num_experts,), dtype=torch.float32, device="cuda")

    gt_topk_weights, gt_topk_ids = grouped_topk(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
    )

    my_topk_weights, my_topk_ids = hpc.grouped_topk(
        gating_output,
        topk,
        num_expert_group,
        topk_group,
        routed_scaling_factor,
        renormalize,
        e_score_correction_bias,
    )

    assert allclose(gt_topk_weights.sort()[0], my_topk_weights.sort()[0])
    assert allclose(gt_topk_ids.sort()[0], my_topk_ids.sort()[0])
