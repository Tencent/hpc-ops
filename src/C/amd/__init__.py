# Copyright (C) 2026 Tencent.

import importlib.metadata
import json

import torch

try:
    FLAGTREE_VERSION = importlib.metadata.version("flagtree")
    import triton
    from triton.experimental import tle
except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
    raise ImportError("The AMD backend requires FlagTree >= 0.7 with AMD TLE support") from exc

from . import unsupported
from .fuse_moe import entry as moe
from .group_gemm import entry as group_gemm
from .iHC import entry as ihc
from .rope import entry as rope
from .sampler import entry as sampler
from .stem import entry as stem
from .topk import entry as topk

try:
    from ..version import git_hash, version
except ImportError:
    git_hash, version = "unknown", "unknown"

_library = torch.library.Library("hpc", "FRAGMENT")


def register():
    for entry in (moe, group_gemm, ihc, rope, sampler, stem, topk, unsupported):
        entry.register(_library)
    _library.define("version() -> str")
    _library.impl("version", lambda: version, "CompositeExplicitAutograd")
    _library.define("built_json() -> str")
    metadata = json.dumps(
        {
            "version": version,
            "git-hash": git_hash,
            "backend": "triton",
            "compiler": "flagtree",
            "flagtree": FLAGTREE_VERSION,
            "triton": triton.__version__,
        }
    )
    _library.impl("built_json", lambda: metadata, "CompositeExplicitAutograd")
