# Copyright (C) 2026 Tencent.

import os
import subprocess
import sys

# A fresh interpreter: this one has already loaded a C++ module, and which one
# is decided once, at import.
_CHECK = """
import torch
import hpc
import hpc.cute.fuse_moe as dsl

assert hpc.__loaded_arch__ == 120, hpc.__loaded_arch__
assert not hasattr(torch.ops.hpc, "version"), "a C++ module was loaded"
for name in hpc._CUTE_EXPORTS:
    assert getattr(hpc, name) is getattr(dsl, name), name
from hpc import fuse_moe_blockwise_fp8_dsl
try:
    hpc.fuse_moe_blockwise_fp8
except AttributeError as e:
    assert "sm120" in str(e), e
else:
    raise AssertionError("a C++ operator wrapper was exported")
"""


def test_sm120_imports_without_a_cpp_module():
    """SM120 has only the CuTe DSL fused MoE; any wheel has to import there."""
    env = dict(os.environ, HPC_SM_ARCH="120")
    result = subprocess.run([sys.executable, "-c", _CHECK], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
