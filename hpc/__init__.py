import os
import torch


def _load():
  lib_path = os.path.join(
      os.path.dirname(__file__), "_C.cpython-311-x86_64-linux-gnu.so")
  if os.path.exists(lib_path):
    torch.ops.load_library(lib_path)
  else:
    raise ImportError(f"Cannot find library at {lib_path}")


_load()
__all__ = []
