import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
import pytest
import torch.nn.functional as F
from utils import allclose


def test_transpose():
    m, n = 9614, 108
    x = torch.randn((m, n), dtype=torch.float, device="cuda")
    print(x)
    print("\n")

    gt = x.clone().transpose(0, 1)
    print(gt)
    print("\n")

    my = hpc.pad_and_transpose(x)
    print(my)

    assert allclose(gt, my[:, 0:m])
