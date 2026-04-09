import torch
import ctypes
import importlib
import os
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Dict

_pkg_dir = Path(__file__).resolve().parent

_NVSHMEM_REL = Path("3rd", "ucl", "nvshmem", "lib", "libnvshmem_host.so")
_PROJECT_ROOT_MARKERS = ("CMakeLists.txt", "setup.py")


def _find_project_root() -> "Path | None":
    """Walk up from the package directory to locate the project root."""
    d = _pkg_dir.parent
    for _ in range(5):
        if any((d / m).is_file() for m in _PROJECT_ROOT_MARKERS):
            return d
        d = d.parent
    return None


def _load_nvshmem_library():
    """Load ``libnvshmem_host.so`` at import time via ``ctypes.CDLL``.

    Search order:
      1. ``HPC_OPS_ROOT`` environment variable (explicit override).
      2. Auto-detected project root (editable install / local cmake build).
    """
    search_roots: list[Path] = []

    env_root = os.environ.get("HPC_OPS_ROOT")
    if env_root:
        search_roots.append(Path(env_root).resolve())

    project_root = _find_project_root()
    if project_root and project_root not in search_roots:
        search_roots.append(project_root)

    errors: list[tuple[Path, Exception]] = []
    for root in search_roots:
        so_path = root / _NVSHMEM_REL
        if not so_path.exists():
            continue
        try:
            ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
            return
        except OSError as e:
            errors.append((so_path, e))

    msg = "hpc-ops: failed to load libnvshmem_host.so.\n  Searched:\n"
    for root in search_roots:
        msg += f"    {root / _NVSHMEM_REL}\n"
    for so_path, e in errors:
        msg += f"  Load error for {so_path}: {e}\n"
    if not search_roots:
        msg += "    (no search roots — set HPC_OPS_ROOT or use editable install)\n"
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def _discover_modules() -> Dict[str, ModuleType]:
    modules = {}
    for file in _pkg_dir.iterdir():
        if file.suffix != ".py" or file.name.startswith("_"):
            continue
        module_name = file.stem
        try:
            module = importlib.import_module(f".{module_name}", package=__package__)
            modules[module_name] = module
        except (ImportError, RuntimeError) as e:
            print(f"WARNING: Failed to import {module_name}: {e}", file=sys.stderr)
    return modules


def _export_functions(modules: Dict[str, ModuleType]):
    for _module_name, module in modules.items():
        funcs = {
            name: obj
            for name, obj in vars(module).items()
            if callable(obj) and not name.startswith("_")
        }
        globals().update(funcs)
        __all__.extend(funcs.keys())


# Bootstrap

_load_nvshmem_library()

so_files = list(_pkg_dir.glob("_C.*.so"))
if len(so_files) != 1:
    raise ImportError(
        f"Expected exactly one _C.*.so in {_pkg_dir}, " f"found {len(so_files)}: {so_files}"
    )
torch.ops.load_library(so_files[0])

__all__: list[str] = []

_export_functions(_discover_modules())

__version__ = torch.ops.hpc.version()
__built_json__ = torch.ops.hpc.built_json()

__doc__ = """
High Performance Computing Operators Library

This library provides optimized CUDA kernels for tensor operations.
"""
