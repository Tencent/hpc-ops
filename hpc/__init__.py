import torch
import ctypes
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict

_pkg_dir = Path(__file__).resolve().parent

# Filenames to try, in order. Versioned SONAME first because that is what the
# extension .so actually links against (NEEDED libnvshmem_host.so.3).
_NVSHMEM_NAMES = ("libnvshmem_host.so.3", "libnvshmem_host.so")
# Legacy in-source layout (master-branch style, kept for backward compatibility).
_NVSHMEM_LEGACY_REL = Path("3rd", "ucl", "nvshmem", "lib")
_PROJECT_ROOT_MARKERS = ("CMakeLists.txt", "setup.py")


def _find_project_root() -> "Path | None":
    """Walk up from the package directory to locate the project root."""
    d = _pkg_dir.parent
    for _ in range(5):
        if any((d / m).is_file() for m in _PROJECT_ROOT_MARKERS):
            return d
        d = d.parent
    return None


def _detect_sm_arch() -> "str | None":
    """Detect SM architecture of the current GPU, e.g. '90', '100'. Returns None if no GPU."""
    try:
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return str(major * 10 + minor)
    except Exception:
        return None


def _nvshmem_search_dirs(arch: "str | None") -> "list[Path]":
    """Build the ordered list of directories to search for libnvshmem_host.

    NVSHMEM's ``libnvshmem_host.so`` statically embeds a per-arch device
    fatbin (see comment in setup.py). Therefore the search order MUST
    prefer per-arch subdirectories matching the current GPU architecture
    over generic flat directories. Otherwise CUDA Runtime fails to
    register the embedded module on a GPU of a different arch and the
    test reports ``Unable to access device state. 500``.

    The lookup priority is:
      1. Wheel install per-arch layout: ``<pkg>/nvshmem/sm{ARCH}/``
      2. Wheel install flat layout (legacy / single-arch wheels): ``<pkg>/``
      3. Per-arch CMake build outputs: ``<root>/build/sm{ARCH}/hpc/nvshmem-install/lib``
      4. Generic per-arch CMake build outputs (any arch).
      5. Legacy in-source layout: ``<root>/3rd/ucl/nvshmem/lib``
    """
    dirs: list[Path] = []

    # 1. Package per-arch subdirectory (current MULTI_ARCH wheel layout).
    if arch is not None:
        per_arch = _pkg_dir / "nvshmem" / f"sm{arch}"
        if per_arch not in dirs:
            dirs.append(per_arch)

    # 2. Package flat directory (legacy single-arch wheel layout).
    dirs.append(_pkg_dir)

    # 3. Explicit override via env var.
    env_root = os.environ.get("HPC_OPS_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        if arch is not None:
            arch_build = root / "build" / f"sm{arch}" / "hpc" / "nvshmem-install" / "lib"
            if arch_build not in dirs:
                dirs.append(arch_build)
        # Generic per-arch cmake layout (build/sm*/hpc/nvshmem-install/lib),
        # used as a last-resort fallback when arch detection fails.
        for d in (root / "build").glob("sm*/hpc/nvshmem-install/lib"):
            if d not in dirs:
                dirs.append(d)
        legacy = root / _NVSHMEM_LEGACY_REL
        if legacy not in dirs:
            dirs.append(legacy)

    # 4. Auto-detected project root (editable / local cmake build).
    project_root = _find_project_root()
    if project_root:
        if arch is not None:
            arch_build = project_root / "build" / f"sm{arch}" / "hpc" / "nvshmem-install" / "lib"
            if arch_build not in dirs:
                dirs.append(arch_build)
        for d in (project_root / "build").glob("sm*/hpc/nvshmem-install/lib"):
            if d not in dirs:
                dirs.append(d)
        legacy = project_root / _NVSHMEM_LEGACY_REL
        if legacy not in dirs:
            dirs.append(legacy)

    return dirs


def _setup_nvshmem_dlopen_path(lib_dir: Path) -> None:
    """Make sure NVSHMEM's runtime ``dlopen("nvshmem_bootstrap_*.so.3")``
    calls (issued from libnvshmem_host) can locate their plugins.

    NVSHMEM loads bootstrap plugins by *bare filename* (no ``/``), so
    ``ctypes.CDLL`` preload alone is not enough on every glibc/ld.so combo:
    we also need to inject ``lib_dir`` into ``LD_LIBRARY_PATH`` so the
    subsequent ``dlopen`` can resolve the plugin via the standard search
    path. Doing both is intentional belt-and-braces, since glibc's runtime
    ``dlopen`` re-reads ``LD_LIBRARY_PATH`` at call time.
    """
    # 1. Preload every plugin we shipped, so the dynamic linker has the
    #    SONAME already in its loaded-list cache.
    for so in sorted(lib_dir.glob("nvshmem_bootstrap_*.so*")):
        if not so.is_file():
            continue
        try:
            ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            # Non-fatal: LD_LIBRARY_PATH below is the real safety net.
            print(f"WARNING: failed to preload {so}: {e}", file=sys.stderr)

    # 2. Prepend lib_dir to LD_LIBRARY_PATH so plain-name dlopen() resolves.
    lib_dir_s = str(lib_dir)
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = cur.split(":") if cur else []
    if lib_dir_s not in parts:
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir_s}:{cur}" if cur else lib_dir_s


def _load_nvshmem_library():
    """Preload ``libnvshmem_host.so[.3]`` via ``ctypes.CDLL`` so that the
    subsequent ``torch.ops.load_library`` call on ``_C_sm*.abi3.so`` can
    resolve its ``NEEDED libnvshmem_host.so.3`` dependency.

    The host lib is per-arch (it statically links libnvshmem_device.a whose
    fatbin is per-arch), so we MUST pick the copy that matches the current
    GPU's compute capability. ``_nvshmem_search_dirs`` enforces this
    ordering by placing ``<pkg>/nvshmem/sm{ARCH}/`` first.

    On success, also configure ``LD_LIBRARY_PATH`` and preload bootstrap
    plugins so that NVSHMEM's runtime ``dlopen`` calls succeed.

    Failures are reported on stderr (instead of being silently swallowed) to
    make CI diagnostics tractable.
    """
    arch = _detect_sm_arch()
    tried: list[str] = []
    for d in _nvshmem_search_dirs(arch):
        if not d.is_dir():
            continue
        for name in _NVSHMEM_NAMES:
            so_path = d / name
            tried.append(str(so_path))
            if not so_path.exists():
                continue
            try:
                ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                print(
                    f"WARNING: failed to preload {so_path}: {e}",
                    file=sys.stderr,
                )
                continue
            # Host lib loaded: also set up bootstrap plugin discovery.
            _setup_nvshmem_dlopen_path(d)
            return

    print(
        "WARNING: libnvshmem_host.so[.3] not found; "
        "_C extension load may fail with 'libnvshmem_host.so.3: cannot open "
        "shared object file'. Tried:\n  " + "\n  ".join(tried),
        file=sys.stderr,
    )


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
