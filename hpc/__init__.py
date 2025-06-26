import torch
from pathlib import Path

so_files = list(Path(__file__).parent.glob("_C.*.so"))
assert (len(so_files) == 1), f"Expected one _C*.so file, found {len(so_files)}"
torch.ops.load_library(so_files[0])

from . import ops

__doc__ = '''
High Performance Computing Operators Library

This library provides optimized CUDA kernels for tensor operations.
'''
