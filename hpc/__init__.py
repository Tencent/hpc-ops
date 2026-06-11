import torch
import ctypes
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict

_pkg_dir = Path(__file__).resolve().parent

_NVSHMEM_NAMES = ("libnvshmem_host.so.3", "libnvshmem_host.so")
_NVSHMEM_SYSTEM_PREFIX = "/usr/local/nvshmem"


def _detect_sm_arch() -> "str | None":
    """Detect SM architecture of the current GPU, e.g. '90', '100'. Returns None if no GPU."""
    try:
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return str(major * 10 + minor)
    except Exception:
        return None


def _ld_library_path_has_system_nvshmem() -> bool:
    """Return True if any entry of ``LD_LIBRARY_PATH`` is under ``/usr/local/nvshmem``."""
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if not cur:
        return False
    prefix = _NVSHMEM_SYSTEM_PREFIX.rstrip("/")
    for entry in cur.split(":"):
        if not entry:
            continue
        e = entry.rstrip("/")
        if e == prefix or e.startswith(prefix + "/"):
            return True
    return False


def _load_nvshmem_library():
    """Ensure ``libnvshmem_host.so[.3]`` can be resolved when loading the
    ``_C_sm*.abi3.so`` extension (which has ``NEEDED libnvshmem_host.so.3``).
    """
    # -- Mode 1: trust the user-provided LD_LIBRARY_PATH --
    if _ld_library_path_has_system_nvshmem():
        return

    # -- Mode 2: manual preload from per-arch cmake build output --
    arch = _detect_sm_arch()
    if arch is None:
        return

    lib_dir = _pkg_dir.parent.parent / f"sm{arch}" / "hpc" / "nvshmem-install" / "lib"
    tried: list[str] = []
    if lib_dir.is_dir():
        for name in _NVSHMEM_NAMES:
            so_path = lib_dir / name
            tried.append(str(so_path))
            if not so_path.exists():
                continue
            try:
                ctypes.CDLL(str(so_path), mode=ctypes.RTLD_LOCAL)
            except OSError:
                continue
            return

    print("WARNING: libnvshmem_host.so[.3] not found")


def _load_extension_library():
    """Load the _C extension library.

    Loading priority:
      1. Auto-detect current GPU architecture and load matching _C_sm*.abi3.so
      2. Fallback to legacy _C.*.so glob pattern (backward compatible with old single-arch packages)
    """
    # Scan for new-format _C_sm*.abi3.so files: {arch: path}
    arch_sos: Dict[str, Path] = {
        f.name[len("_C_sm") : -len(".abi3.so")]: f for f in _pkg_dir.glob("_C_sm*.abi3.so")
    }

    if arch_sos:
        # -- New format: auto-detect GPU architecture --
        detected_arch = _detect_sm_arch()
        if detected_arch is None:
            raise ImportError(
                f"hpc-ops: no GPU detected, cannot auto-select architecture.\n"
                f"  Available architectures: {sorted(arch_sos)}\n"
                f"  Search directory: {_pkg_dir}"
            )
        so_path = arch_sos.get(detected_arch)
        if so_path is None:
            raise ImportError(
                f"hpc-ops: current GPU architecture sm{detected_arch} is not supported.\n"
                f"  _C_sm{detected_arch}.abi3.so not found\n"
                f"  Available architectures: {sorted(arch_sos)}\n"
                f"  Search directory: {_pkg_dir}"
            )
        torch.ops.load_library(so_path)
        return

    # -- Fallback: legacy _C.*.so format (backward compatible with old single-arch packages) --
    legacy_files = list(_pkg_dir.glob("_C.*.so"))
    if len(legacy_files) == 1:
        torch.ops.load_library(legacy_files[0])
        return

    raise ImportError(
        f"hpc-ops: no extension library (.so file) found in {_pkg_dir}.\n"
        f"  Please make sure hpc-ops is properly installed."
    )


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
_load_extension_library()

from .attention import QuantType

torch.serialization.add_safe_globals([QuantType])

__all__: list[str] = []

_export_functions(_discover_modules())

__version__ = torch.ops.hpc.version()
__built_json__ = torch.ops.hpc.built_json()

__doc__ = """
High Performance Computing Operators Library

This library provides optimized CUDA kernels for tensor operations.
"""
