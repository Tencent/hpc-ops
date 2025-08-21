import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch


def test_gem3():
    num_batch = 70 * 8
    num_batch = 37 * 8  # (9288 + 255) / 256  * 8
    num_batch = 1
    num_batch = 560

    num_seq = 256
    num_qk_dim = 128
    num_v_dim = 80

    Q = torch.randn(
        (num_batch, num_seq, num_qk_dim), dtype=torch.bfloat16, device="cuda"
    )  # * 0 + 1.
    K = torch.randn(
        (num_batch, num_seq, num_qk_dim), dtype=torch.bfloat16, device="cuda"
    )  # * 0 + 1.
    V = torch.randn(
        (num_batch, num_seq, num_v_dim), dtype=torch.bfloat16, device="cuda"
    )  # * 0 + 0.01

    gt = Q @ K.permute(0, 2, 1) @ V
    my = hpc.gem3(Q, K, V)

    print("\ngt\n")
    print(gt)
    print("\nmy\n")
    print(my)

    """
    my = my.flatten()
    gt = gt.flatten()
    """

    idx = torch.nonzero(torch.abs(gt - my) > 1.0)
    print(idx)

    """
    for i in idx:
        print(i)
    """

    for i in idx[:5]:
        t = tuple(i.tolist())
        print("{} {} vs {}".format(t, gt[t].item(), my[t].item()))
    """
    """

    assert torch.allclose(my, gt)
    assert gt.device == my.device
    assert gt.dtype == my.dtype
    assert gt.shape == my.shape
