import os
import sys
import math
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import hpc
from utils import allclose


def precompute_freqs_cis(
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def rope_interleave_ref(input, freq_cis, cu_seqlenq, num_seq_kv, output):

    num_batch = num_seq_kv.shape[0]
    seqlenq = cu_seqlenq[1:] - cu_seqlenq[:-1]

    y = torch.empty_like(input)

    for bi in range(num_batch):
        x = input[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlenq[bi]]
        pos = torch.arange(seqlenq[bi], device="cuda") + num_seq_kv[bi] - seqlenq[bi]
        freq = freq_cis[pos]
        x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
        # import pdb;pdb.set_trace()
        x = torch.view_as_real(x * freq).flatten(-2)
        y[cu_seqlenq[bi] : cu_seqlenq[bi] + seqlenq[bi]] = x

    output.copy_(y)

    return output


@pytest.mark.parametrize("num_batch", [1, 10, 20, 40, 60, 80])
@pytest.mark.parametrize("max_seq_q", [1, 2, 2 * 1024])
@pytest.mark.parametrize("max_seq_kv", [2 * 1024])
@pytest.mark.parametrize("dim", [64])
@pytest.mark.parametrize("max_seqlen", [8 * 1024])
def test_rope_interleave(
    num_batch,
    max_seq_q,
    max_seq_kv,
    dim,
    max_seqlen,
):
    # torch.manual_seed(41)
    # torch.cuda.manual_seed(41)

    freq_cis = precompute_freqs_cis(dim, max_seqlen, max_seqlen, 40000, 40, 32, 1).cuda()

    num_seq_q = torch.randint(1, max_seq_q + 1, (num_batch,), dtype=torch.int32, device="cuda")
    num_seq_kvcache = torch.randint(
        1, max_seq_kv + 1, (num_batch,), dtype=torch.int32, device="cuda"
    )
    num_seq_kv = num_seq_q + num_seq_kvcache

    total_seq_q = sum(num_seq_q)

    cu_seqlenq = torch.cumsum(num_seq_q, dtype=torch.int32, dim=0)
    cu_seqlenq = torch.concat(
        [torch.tensor([0], dtype=torch.int32, device="cuda"), cu_seqlenq], dim=0
    )

    input = torch.randn(total_seq_q, 512, device="cuda", dtype=torch.bfloat16)
    input_clone = input.clone()
    input = input[:, -dim:]
    input_clone = input_clone[:, -dim:]

    # import pdb;pdb.set_trace()
    my = hpc.rope_interleave(
        input,
        torch.view_as_real(freq_cis),
        cu_seqlenq,
        num_seq_kv,
        output=input,
    )

    gt = rope_interleave_ref(
        input_clone,
        freq_cis,
        cu_seqlenq,
        num_seq_kv,
        output=input_clone,
    )

    # import pdb;pdb.set_trace()

    assert allclose(gt, my, atol=1e-3, rtol=1e-2)
