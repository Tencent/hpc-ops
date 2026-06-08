# Copyright (C) 2026 Tencent.

"""Backend registry."""
from __future__ import annotations

from typing import Callable, Dict

from .base import Backend, BenchSpec  # re-export for convenience

_REGISTRY: Dict[str, Callable[[], Backend]] = {}


def register(name: str, factory: Callable[[], Backend]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"backend already registered: {name}")
    _REGISTRY[name] = factory


def make(name: str) -> Backend:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown backend: {name!r} (known: {sorted(_REGISTRY.keys())})")
    return _REGISTRY[name]()


def known() -> list[str]:
    return sorted(_REGISTRY.keys())


def _import_all():
    """Trigger registration side effects for all known modules."""
    from . import hpcops        # noqa: F401
    from . import vllm          # noqa: F401
    from . import vllm_cutlass  # noqa: F401
    from . import sglang        # noqa: F401


_import_all()


__all__ = ["Backend", "BenchSpec", "register", "make", "known"]
