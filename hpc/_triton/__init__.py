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

from hpc._ops.iHC.amd import iHC as ihc
from hpc._ops.fuse_moe.amd import reduce as moe
from hpc._ops.rope.amd import rope
from hpc._ops.group_gemm.amd import reformat
from hpc._ops.stem.amd import stem_tpd as stem
from hpc._ops.topk.amd import topk
from hpc._ops.sampler.amd import sampler
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
        "fuse_ihc_post_pre": ihc.post_pre,
        "reduce": moe.reduce,
        "rope_norm_store_kv": rope.rope_norm_store_kv,
        "reformat_x_scale": reformat.reformat_x_scale,
        "stem_tpd": stem.stem_tpd,
        "topk_filtered": topk.topk_filtered,
        "topk_filtered_workspace_size": topk.workspace_size,
        "topk_filtered_min_workspace_size": topk.min_workspace_size,
        "topk_filtered_peak_workspace_size": topk.peak_workspace_size,
        "fused_sampler": sampler.fused_sampler,
        "fused_sampler_temperature_sample": sampler.temperature_sample,
    }
    for schema in SCHEMAS:
        _library.define(schema)
        name = schema.split("(", 1)[0]
        if name in implementations:
            dispatch = "CompositeExplicitAutograd" if "workspace_size" in name else "CUDA"
            _library.impl(name, implementations[name], dispatch)
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
