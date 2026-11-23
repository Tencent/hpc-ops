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

from . import ihc, moe, rope
from ._schemas import DEFERRED, SCHEMAS

try:
    from ..version import git_hash, version
except ImportError:
    git_hash, version = "unknown", "unknown"

_library = torch.library.Library("hpc", "FRAGMENT")


def _unimplemented(name, reason):
    def entry(*args, **kwargs):
        raise NotImplementedError(f"hpc::{name}: TODO(amd): {reason}")

    return entry


def register():
    implementations = {
        "fuse_ihc_pre": ihc.pre,
        "fuse_ihc_head": ihc.head,
        "fuse_ihc_post": ihc.post,
        "reduce": moe.reduce,
        "rope_norm_store_kv": rope.rope_norm_store_kv,
    }
    for schema in SCHEMAS:
        _library.define(schema)
        name = schema.split("(", 1)[0]
        if name in implementations:
            _library.impl(name, implementations[name], "CUDA")
        else:
            entry = _unimplemented(name, DEFERRED[name])
            dispatch = "CompositeExplicitAutograd" if "workspace_size" in name else "CUDA"
            _library.impl(name, entry, dispatch)
            if name == "assign_attention_decode_task":
                _library.impl(name, entry, "CPU")
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
