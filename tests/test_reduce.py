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


def naive_reduce(x_bf16, topk_pos, topk_scale, shared_output=None):
    num_tokens, num_topk = topk_pos.shape
    total_num_tokens, hidden_size = x_bf16.shape

    y_bf16 = torch.zeros((num_tokens, hidden_size), dtype=torch.bfloat16, device=x_bf16.device)
    for i in range(num_tokens):
        acc = torch.zeros((1, hidden_size), dtype=torch.float, device="cuda")
        cur_topk_pos = topk_pos[i]
        cur_topk_scale = topk_scale[i]
        for j, pos in enumerate(cur_topk_pos):
            if pos >= 0:
                acc += x_bf16[pos].float() * cur_topk_scale[j].float()
        if shared_output is not None:
            acc += shared_output[i].float()
        y_bf16[i] = acc.to(torch.bfloat16)
    return y_bf16


@pytest.mark.parametrize("num_tokens", [128])
@pytest.mark.parametrize("num_topk", [8])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("num_expert", [128])
def test_reduce(num_tokens, num_topk, hidden_size, num_expert):
    torch.cuda.manual_seed(0)

    total_tokens = num_tokens * num_topk
    topk_pos = torch.randint(
        low=-1, high=total_tokens, size=(num_tokens, num_topk), dtype=torch.int32, device="cuda"
    )
    topk_scales = torch.rand(size=(num_tokens, num_topk), dtype=torch.float32, device="cuda")
    x = torch.randn((total_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")

    print(x)
    print(topk_pos)
    print(topk_scales)
    gt = naive_reduce(x.clone(), topk_pos.clone(), topk_scales.clone())
    my = hpc.reduce(x.clone(), topk_pos.clone(), topk_scales.clone())
    print(my)
    print(gt)
    assert allclose(gt.to(torch.float32), my.to(torch.float32), rtol=0.08, atol=0.01)
