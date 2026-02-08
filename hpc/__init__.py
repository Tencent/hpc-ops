from functools import lru_cache
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict

import torch
import tvm_ffi


_pkg_dir = Path(__file__).parent

LIB_ROOT = _pkg_dir.parent / "build"
if not LIB_ROOT.exists():
    LIB_ROOT = _pkg_dir / "ops"

# Define the torch library for op registration (needed for torch.compile tracing)
_torch_lib = torch.library.Library("hpc", "DEF")


@lru_cache(maxsize=None)
def load_ffi_lib(name: str):
    """
    Libraries would be in `<repo>/build` or `<site-packages>/hpc/ops`.
    """
    p = Path(name)
    return tvm_ffi.load_module(LIB_ROOT / p.name)


def _discover_modules() -> Dict[str, ModuleType]:
    modules = {}

    for file in _pkg_dir.iterdir():
        if file.suffix != ".py" or file.name.startswith("_") or file.name == __file__:
            continue

        module_name = file.stem

        try:
            module = importlib.import_module(f".{module_name}", package=__package__)
            modules[module_name] = module
        except ImportError as e:
            print(f"WARNING: Failed to import {module_name}: {str(e)}", file=sys.stderr)

    return modules


def _export_functions(modules: Dict[str, ModuleType]):
    for module_name, module in modules.items():
        funcs = {
            name: obj
            for name, obj in vars(module).items()
            if callable(obj) and not name.startswith("_")
        }

        globals().update(funcs)

        __all__.extend(funcs.keys())


__all__ = []

_export_functions(_discover_modules())

_lib = load_ffi_lib("_C.so")
__version__ = _lib.version()
__built_json__ = _lib.built_json()

__doc__ = """
High Performance Computing Operators Library

This library provides optimized CUDA kernels for tensor operations.
"""
