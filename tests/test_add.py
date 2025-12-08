import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
from utils import allclose


def test_add():
    a = torch.randn(3, 5, device="cuda")
    b = torch.randn(3, 5, device="cuda")

    gt = a + b
    my = hpc.add(a, b)

    assert allclose(gt, my)
