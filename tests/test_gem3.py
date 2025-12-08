import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
from utils import allclose


def test_gem3():

    # torch.cuda.manual_seed(10086)

    num_batch = 70 * 8
    num_batch = 37 * 8  # (9288 + 255) / 256  * 8

    num_seq = 256
    num_qk_dim = 128
    num_v_dim = 80

    Q = torch.randn(
        (num_batch, num_seq, num_qk_dim), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(num_qk_dim)
    K = torch.randn(
        (num_batch, num_seq, num_qk_dim), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(num_qk_dim)
    V = torch.randn((num_batch, num_seq, num_v_dim), dtype=torch.bfloat16, device="cuda")

    qscale = (
        torch.rand((num_batch, num_seq, 1), dtype=torch.float32, device="cuda")
        .to(torch.bfloat16)
        .to(torch.float32)
    )

    kscale = (
        torch.rand((num_batch, 1, num_seq), dtype=torch.float32, device="cuda")
        .to(torch.bfloat16)
        .to(torch.float32)
    )

    P = torch.tril(
        (Q.to(torch.float32) @ K.to(torch.float32).permute(0, 2, 1)) * qscale * kscale
    ).to(torch.bfloat16)
    gt = (P.to(torch.float32) @ V.to(torch.float32)).to(torch.bfloat16)
    my = hpc.gem3(Q, K, V, qscale, kscale)

    print("\ngt\n")
    print(gt[0, :, :])
    print("\nmy\n")
    print(my[0, :, :])

    abs_diff = torch.abs(gt - my)
    vals, idxs = torch.topk(abs_diff.view(-1), 10)
    idxs = [torch.unravel_index(idx, gt.shape) for idx in idxs]

    for i, idx in enumerate(idxs):
        cpu_idx = tuple(tensor.cpu().item() for tensor in idx)
        print(
            "{:+.4f} vs {:+.4f} with diff = {:.4f}, @ {}".format(gt[idx], my[idx], vals[i], cpu_idx)
        )

    assert allclose(gt, my, atol=0.0156)
