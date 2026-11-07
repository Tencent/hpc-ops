# Copyright (C) 2026 Tencent.

"""Top-K benchmark provider registry."""

from .base import TopKBackend
from .hpcops import HpcOpsBackend
from .torch_ref import TorchRefBackend

BACKENDS = {
    HpcOpsBackend.name: HpcOpsBackend,
    TorchRefBackend.name: TorchRefBackend,
}

try:
    from .deepselect import DeepSelectBackend

    if DeepSelectBackend.available:
        BACKENDS[DeepSelectBackend.name] = DeepSelectBackend
except Exception:  # optional provider: registry works without it
    pass


def known():
    return list(BACKENDS)


def create(name):
    return BACKENDS[name]()
