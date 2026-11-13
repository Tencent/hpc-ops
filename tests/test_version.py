import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc

import json
import pytest
import torch


def test_version_and_built():
    print(hpc.__version__)
    print(hpc.__built_json__)

    j = json.loads(hpc.__built_json__)
    print(j)
    assert j["sm-arch"] == f"sm{hpc.__loaded_arch__}"
    assert j["version"] == hpc.__version__


@pytest.mark.skipif(hpc.__loaded_arch__ != 120, reason="checks SM120 unsupported dispatch")
def test_sm120_unsupported_gemm_dispatch():
    x = torch.zeros((1, 64), dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros((256, 64), dtype=torch.bfloat16, device="cuda")
    output = torch.zeros((1, 256), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(
        RuntimeError,
        match="gemm_bf16xfp32 does not run on sm120: implemented for 90, loaded module built for sm120",
    ):
        hpc.gemm_bf16xfp32(x, weight, weight, 1.0 / 256)
    with pytest.raises(
        RuntimeError,
        match="gated_mla_gemm does not run on sm120: implemented for 100, 103, loaded module built for sm120",
    ):
        hpc.gated_mla_gemm(x, weight, output)
