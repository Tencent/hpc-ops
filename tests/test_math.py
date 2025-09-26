import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import random
import string
import pytest


def generate_random_string(n):
    characters = string.digits + string.ascii_uppercase
    return "".join(random.choices(characters, k=n))


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float,
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
        torch.float8_e8m0fnu,
    ],
)
@pytest.mark.parametrize("tag_length", [0, 20])
@pytest.mark.parametrize("tensor_size", [1, 2 * 1024 * 1024])
def test_has_nan(capfd, dtype, tag_length, tensor_size):
    a = torch.tensor([float("nan")] * tensor_size, dtype=dtype, device="cuda")
    tag = generate_random_string(tag_length)
    hpc.has_nan(a, tag)
    torch.cuda.synchronize()
    out, _ = capfd.readouterr()
    assert len(out) > 0
    print(out)
