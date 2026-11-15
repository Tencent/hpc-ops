# Copyright (C) 2026 Tencent.

"""Compilation helpers shared by the CuTe DSL SM120 fused MoE kernels.

The DSL toolchain is an optional dependency, so ``cutlass`` is imported lazily.
Importing it eagerly would add several seconds to ``import hpc`` and would make
the package unusable on machines that only need the C++ operators.
"""

import functools
import os
import threading
from typing import Any, Callable, Dict, Optional, Tuple

import torch

_SM120_CAPABILITY = (12, 0)

_import_lock = threading.Lock()
_cutlass_modules: Optional[Tuple[Any, Any]] = None


def dsl_available() -> Tuple[bool, str]:
    """Report whether the CuTe DSL toolchain can be imported.

    Returns:
        A ``(available, reason)`` pair. ``reason`` is empty when available.
    """
    try:
        import cutlass  # noqa: F401
    except ImportError as error:
        return False, f"nvidia-cutlass-dsl is not installed: {error}"
    return True, ""


def device_capability(device: Optional[torch.device] = None) -> Tuple[int, int]:
    index = None if device is None else torch.device(device).index
    return torch.cuda.get_device_capability(index)


def is_sm120(device: Optional[torch.device] = None) -> bool:
    return device_capability(device) == _SM120_CAPABILITY


def require_sm120(device: Optional[torch.device] = None) -> None:
    capability = device_capability(device)
    if capability != _SM120_CAPABILITY:
        raise RuntimeError(
            "the CuTe DSL W4A8 MoE kernels require an SM120 device, but the current "
            f"device reports compute capability {capability[0]}.{capability[1]}"
        )


def _arch_string(device: Optional[torch.device] = None) -> str:
    major, minor = device_capability(device)
    return f"sm_{major}{minor}a"


def import_cutlass(device: Optional[torch.device] = None) -> Tuple[Any, Any]:
    """Import ``cutlass`` with ``CUTE_DSL_ARCH`` pinned to the current device.

    The DSL infers the target architecture once, at import time. Auto-detection
    already yields ``sm_120a`` on this hardware, but the block-scaled MMA ops
    reject anything other than ``sm_120a``/``sm_121a``, so the value is pinned
    explicitly to avoid a confusing arch mismatch when detection is wrong (for
    example when the visible devices are heterogeneous).
    """
    global _cutlass_modules
    if _cutlass_modules is not None:
        return _cutlass_modules

    with _import_lock:
        if _cutlass_modules is not None:
            return _cutlass_modules

        os.environ.setdefault("CUTE_DSL_ARCH", _arch_string(device))

        available, reason = dsl_available()
        if not available:
            raise ImportError(
                f"{reason}. Install it with 'pip install nvidia-cutlass-dsl' to use the "
                "CuTe DSL fused MoE path."
            )

        import cutlass
        import cutlass.cute as cute

        _cutlass_modules = (cutlass, cute)
        return _cutlass_modules


def dsl_arch() -> str:
    """Return the architecture string the DSL is currently targeting."""
    import_cutlass()
    return os.environ["CUTE_DSL_ARCH"]


_stream_cache: Dict[int, Any] = {}
_raw_stream = torch._C._cuda_getCurrentRawStream
_current_device = torch.cuda.current_device


def current_stream() -> Any:
    """Return the current torch CUDA stream as a CUstream for DSL launches.

    Both halves of this are on the critical path of every launch: the handle is
    read through the raw C entry point (``torch.cuda.current_stream()`` builds a
    Python ``Stream`` object and costs ~3.5us), and the ``CUstream`` wrapper is
    cached on that handle because constructing it is a few microseconds more.
    """
    handle = _raw_stream(_current_device())
    stream = _stream_cache.get(handle)
    if stream is None:
        import cuda.bindings.driver as cuda

        stream = cuda.CUstream(handle)
        _stream_cache[handle] = stream
    return stream


def dynamic_tensor(tensor: torch.Tensor, *, assumed_align: Optional[int] = None) -> Any:
    """Wrap a torch tensor for the DSL with a runtime (non-baked) layout.

    Marking the layout dynamic is what lets one compiled kernel serve every
    token count, so the compilation cache stays keyed on configuration rather
    than on shape.
    """
    from cutlass.cute.runtime import from_dlpack

    wrapped = from_dlpack(tensor, assumed_align=assumed_align)
    return wrapped.mark_layout_dynamic()


_ptr_factory: Optional[Callable[..., Any]] = None
_gmem_space: Any = None

# A pointer object is a description of an address -- dtype, value, address
# space, alignment -- that the launch path reads and does not write, so one
# built for an address is good for every later use of that address. Building
# one costs ~1.1us, and a fused MoE call makes 46 of them, which is a quarter
# of the host time of a decode-sized call; a lookup is ~0.05us. Addresses
# repeat because the buffers do: the caching allocator hands back the same
# blocks, and the weights never move.
_ptr_cache: Dict[Tuple[Any, int, int], Any] = {}
# Bounds the cache against a caller that keeps allocating fresh buffers. The
# entries are tiny and the working set is a few dozen, so this is far above
# any steady state and just stops unbounded growth.
_PTR_CACHE_LIMIT = 8192


def make_ptr(dtype: Any, tensor: torch.Tensor, *, assumed_align: int = 16) -> Any:
    """Build a DSL pointer of ``dtype`` aimed at ``tensor``'s storage.

    Every kernel in this package passes its buffers this way. Pointers are both
    cheaper to marshal than ``from_dlpack`` tensors (~1us versus ~6us, which
    matters when a fused MoE call issues half a dozen launches) and the only way
    to describe the interleaved scale-factor layouts, which no compact torch
    tensor can express.

    Memoized on the address, which is what makes the launch path cheap enough
    for 48 layers of decode. Reusing one is safe even after the tensor that
    produced it is freed and the address handed to something else: the object
    carries no more than the address itself.
    """
    global _ptr_factory, _gmem_space
    if _ptr_factory is None:
        _, cute = import_cutlass()
        _ptr_factory = cute.runtime.make_ptr
        _gmem_space = cute.AddressSpace.gmem

    address = tensor.data_ptr()
    key = (dtype, address, assumed_align)
    pointer = _ptr_cache.get(key)
    if pointer is None:
        if len(_ptr_cache) >= _PTR_CACHE_LIMIT:
            _ptr_cache.clear()
        pointer = _ptr_cache[key] = _ptr_factory(
            dtype, address, _gmem_space, assumed_align=assumed_align
        )
    return pointer


# Same trade for the extents and the handful of float constants that ride
# along with them: ~0.96us to box an Int32, ~26 of them per fused MoE call.
# Only for host-side argument lists -- inside a trace these constructors build
# graph values, and a cache keyed on the Python value would be wrong.
_scalar_cache: Dict[Tuple[Any, Any], Any] = {}
_SCALAR_CACHE_LIMIT = 8192


def scalar(dtype: Any, value: Any) -> Any:
    """Box ``value`` as a DSL ``dtype``, memoized. Host-side argument lists only."""
    key = (dtype, value)
    boxed = _scalar_cache.get(key)
    if boxed is None:
        if len(_scalar_cache) >= _SCALAR_CACHE_LIMIT:
            _scalar_cache.clear()
        boxed = _scalar_cache[key] = dtype(value)
    return boxed


# Marshaling a launch's arguments into the C argument array is ~12us of the
# compiled function's call, and four launches a fused MoE call make it a fifth
# of the host time -- the part a prefill that outruns its CUDA graphs waits on.
# The arguments are the memoized pointers and scalars above, so a repeated
# launch passes the very same objects; keyed on their identities, the array
# built the first time is the array every later call would build. An entry
# holds its arguments, which keeps the identities from being reused.
_LAUNCH_CACHE_LIMIT = 4096


class _Launcher:
    """A compiled DSL function whose marshaled arguments are reused across calls."""

    __slots__ = ("_fn", "_entries")

    def __init__(self, fn: Any) -> None:
        self._fn = fn
        self._entries: Dict[Tuple[int, ...], Tuple[Any, Any, Tuple[Any, ...]]] = {}

    def __call__(self, *args: Any) -> Any:
        key = tuple(map(id, args))
        entry = self._entries.get(key)
        if entry is None:
            if len(self._entries) >= _LAUNCH_CACHE_LIMIT:
                self._entries.clear()
            exe_args, adapted = self._fn.execution_args.generate_execution_args(args, {})
            entry = self._entries[key] = (exe_args, adapted, args)
        executor = self._fn._default_executor
        if executor is None:
            return self._fn.run_compiled_program(entry[0])
        return executor.run_compiled_program(entry[0])


def launcher(fn: Any) -> Any:
    """Wrap a ``cute.compile`` result for host-side argument lists built from
    :func:`make_ptr`, :func:`scalar` and :func:`current_stream`.

    Falls back to ``fn`` itself if the DSL's executor does not expose the
    split marshal/run calls this relies on.
    """
    if not (
        hasattr(fn, "execution_args")
        and hasattr(fn, "_default_executor")
        and hasattr(fn, "run_compiled_program")
    ):
        return fn
    return _Launcher(fn)


def compiled(builder: Callable[..., Any]) -> Callable[..., Any]:
    """Memoize a kernel factory on its (hashable) configuration arguments.

    ``cute.compile`` is expensive enough that every kernel in this package is
    built once per configuration and reused for the lifetime of the process.
    """

    cache: dict = {}
    lock = threading.Lock()

    @functools.wraps(builder)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        try:
            return cache[key]
        except KeyError:
            pass
        with lock:
            if key not in cache:
                cache[key] = builder(*args, **kwargs)
            return cache[key]

    wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
    return wrapper
