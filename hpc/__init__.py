import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict

import torch

_pkg_dir = Path(__file__).parent


def _nvml_arch():
    """SM architecture of the first device torch would see, or None if NVML cannot say.

    Detection runs at import, so it must not initialize CUDA: both
    torch.cuda.is_available() and torch.cuda.get_device_capability() reach
    cuInit, and a process that has initialized CUDA cannot use CUDA in a
    fork()ed child. That would break fork-based workers in anything that merely
    imports hpc. NVML answers the same question without a context.
    """
    try:
        import pynvml
    except ImportError:
        return None

    # The visible-device resolution torch.cuda.device_count() uses, so an NVML
    # index or UUID here means what an ordinal means to torch.
    try:
        visible = torch.cuda._parse_visible_devices()
    except Exception:
        visible = None

    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        if visible is None:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        elif not visible:
            return None
        elif isinstance(visible[0], str):
            handle = pynvml.nvmlDeviceGetHandleByUUID(visible[0].encode())
        else:
            handle = pynvml.nvmlDeviceGetHandleByIndex(visible[0])
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return major * 10 + minor


def _load_module() -> int:
    """Loads the extension module for the local device; returns its architecture.

    A package carries one `_C_sm<arch>.abi3.so` per architecture it was built
    for. The match is exact: architecture-specific cubins are not forward
    compatible, so another module cannot stand in. HPC_SM_ARCH overrides the
    detection, for a build machine with no matching GPU.
    """
    override = os.getenv("HPC_SM_ARCH", "").strip()
    if override:
        if not override.isdigit():
            raise ImportError(f"HPC_SM_ARCH={override!r} is not an architecture number such as 90.")
        arch = int(override)
    else:
        arch = _nvml_arch()
    if arch is None:
        # No NVML answer. This falls back to torch, which initializes CUDA.
        if not torch.cuda.is_available():
            raise ImportError(
                "hpc found no CUDA device to pick an operator module for. Set HPC_SM_ARCH=<arch> "
                "to load one anyway, e.g. HPC_SM_ARCH=90 for an H800."
            )
        major, minor = torch.cuda.get_device_capability()
        arch = major * 10 + minor

    modules = _pkg_dir.glob("_C_sm*.abi3.so")
    built = sorted(int(p.name.removeprefix("_C_sm").removesuffix(".abi3.so")) for p in modules)
    if arch not in built:
        raise ImportError(
            f"hpc has no operator module for this sm{arch} device; this install covers "
            f"{built or 'no architecture'}. Rebuild including it, e.g. 'SM_ARCH={arch} make'."
        )

    torch.ops.load_library(str(_pkg_dir / f"_C_sm{arch}.abi3.so"))
    return arch


__loaded_arch__ = _load_module()


def _discover_modules() -> Dict[str, ModuleType]:
    modules = {}

    for file in _pkg_dir.iterdir():
        if file.suffix != ".py" or file.name.startswith("_") or file.name == __file__:
            continue

        module_name = file.stem

        selected = os.getenv("HPC_OPS_IMPORT_MODULES")
        if selected:
            selected_modules = {x.strip() for x in selected.split(",") if x.strip()}
            if module_name not in selected_modules:
                continue

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


def _call_optional_hpc_op(name: str, default: str) -> str:
    try:
        return getattr(torch.ops.hpc, name)()
    except AttributeError:
        if os.getenv("HPC_OPS_IMPORT_MODULES"):
            return default
        raise


__version__ = _call_optional_hpc_op("version", "unknown")
__built_json__ = _call_optional_hpc_op("built_json", "{}")

__doc__ = """
High Performance Computing Operators Library

This library provides optimized CUDA kernels for tensor operations.
"""
