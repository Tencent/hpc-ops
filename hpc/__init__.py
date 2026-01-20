import torch
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict

_pkg_dir = Path(__file__).parent


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


so_files = list(Path(__file__).parent.glob("_C.*.so"))
assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"
torch.ops.load_library(so_files[0])

__all__ = []

_export_functions(_discover_modules())

__version__ = torch.ops.hpc.version()
__built_json__ = torch.ops.hpc.built_json()

__doc__ = """
High Performance Computing Operators Library

This library provides optimized CUDA kernels for tensor operations.
"""
