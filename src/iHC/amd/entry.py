# Copyright (C) 2026 Tencent.

import os

import torch
import triton
from triton.experimental.tle import require_tle
from triton.experimental.tle import language as tle

from hpc._amd.utils.validation import check_bf16, check_tensor, require

from .iHC import _post_kernel, _pre_head_kernel

USE_TLE = os.getenv("HPC_TRITON_USE_TLE", "0") == "1"


SCHEMAS = (
    "fuse_ihc_pre(Tensor x, Tensor w, Tensor hc_scale, Tensor hc_base, float norm_eps, float hc_eps, float magnitude, Tensor? rms_weight=None, float rms_eps=0.0, bool cast_bfloat_for_norm=False) -> (Tensor output_y, Tensor output_H_post)",
    "fuse_ihc_post_pre(Tensor xa, Tensor residual, Tensor H_post_in, Tensor w, Tensor hc_scale, Tensor hc_base, float norm_eps, float hc_eps, float magnitude, Tensor? rms_weight=None, float rms_eps=0.0, bool cast_bfloat_for_norm=False) -> (Tensor y, Tensor z, Tensor H_post_out)",
    "fuse_ihc_post(Tensor x, Tensor residual, Tensor H_post) -> Tensor output",
    "fuse_ihc_head(Tensor x, Tensor w, Tensor hc_scale, Tensor hc_base, float norm_eps, float hc_eps, Tensor? rms_weight=None, float rms_eps=0.0, bool cast_bfloat_for_norm=False) -> Tensor output",
)


def _pre_head(
    x,
    w,
    hc_scale,
    hc_base,
    norm_eps,
    hc_eps,
    magnitude,
    rms_weight,
    rms_eps,
    cast_bfloat_for_norm,
    head,
    post_inputs=None,
):
    if USE_TLE:
        require_tle("amd", tle.extract_tile)
    check_bf16(x, x, "x")
    check_tensor(w, x, torch.float32, "w")
    check_tensor(hc_scale, x, torch.float32, "hc_scale")
    check_tensor(hc_base, x, torch.float32, "hc_base")
    require(x.ndim == 3 and x.shape[1] == 4, "x must have shape [N,4,D]")
    batch, _, hidden = x.shape
    require(hidden in (4096, 6144), "hidden dim must be 4096 or 6144")
    projections = 4 if head else 8
    require(w.shape == (projections, 4 * hidden), "w has an invalid projection shape")
    require(hc_scale.shape == (1 if head else 2,), "hc_scale has an invalid shape")
    require(hc_base.shape == (projections,), "hc_base has an invalid shape")
    if rms_weight is not None:
        check_bf16(rms_weight, x, "rms_weight")
        require(rms_weight.shape == (hidden,), "rms_weight must have shape [D]")
    y = torch.empty((batch, hidden), device=x.device, dtype=x.dtype)
    post = None if head else torch.empty((batch, 4), device=x.device, dtype=torch.float32)
    xa, post_in, residual_out = post_inputs if post_inputs is not None else (None, None, None)
    if batch:
        with torch.cuda.device(x.device):
            _pre_head_kernel[(batch,)](
                x,
                w,
                hc_scale,
                hc_base,
                y,
                post,
                rms_weight,
                xa,
                post_in,
                residual_out,
                hidden,
                triton.next_power_of_2(hidden),
                head,
                rms_weight is not None,
                cast_bfloat_for_norm,
                USE_TLE,
                post_inputs is not None,
                norm_eps,
                hc_eps,
                magnitude,
                rms_eps,
                num_warps=16 if hidden == 6144 and (USE_TLE or post_inputs is not None) else 8,
                enable_fp_fusion=False,
            )
    return y, post


def pre(
    x,
    w,
    hc_scale,
    hc_base,
    norm_eps,
    hc_eps,
    magnitude,
    rms_weight=None,
    rms_eps=0.0,
    cast_bfloat_for_norm=False,
):
    return _pre_head(
        x,
        w,
        hc_scale,
        hc_base,
        norm_eps,
        hc_eps,
        magnitude,
        rms_weight,
        rms_eps,
        cast_bfloat_for_norm,
        False,
    )


def head(
    x,
    w,
    hc_scale,
    hc_base,
    norm_eps,
    hc_eps,
    rms_weight=None,
    rms_eps=0.0,
    cast_bfloat_for_norm=False,
):
    return _pre_head(
        x,
        w,
        hc_scale,
        hc_base,
        norm_eps,
        hc_eps,
        0.0,
        rms_weight,
        rms_eps,
        cast_bfloat_for_norm,
        True,
    )[0]


def post(x, residual, H_post):
    check_bf16(x, x, "x")
    check_bf16(residual, x, "residual")
    check_tensor(H_post, x, torch.float32, "H_post")
    require(x.ndim == 2, "x must have shape [N,D]")
    batch, hidden = x.shape
    require(hidden in (4096, 6144), "hidden dim must be 4096 or 6144")
    require(residual.shape == (batch, 4, hidden), "residual must have shape [N,4,D]")
    require(H_post.shape == (batch, 4), "H_post must have shape [N,4]")
    y = torch.empty_like(residual)
    if batch:
        if batch <= 16:
            block, waves = 512, 8
        elif batch < 512:
            block, waves = 256, 1
        else:
            block, waves = 1024, 2
        with torch.cuda.device(x.device):
            _post_kernel[(batch, triton.cdiv(hidden, block))](
                x,
                residual,
                H_post,
                y,
                hidden,
                block,
                batch > 16,
                num_warps=waves,
                enable_fp_fusion=False,
            )
    return y


def post_pre(
    xa,
    residual,
    H_post_in,
    w,
    hc_scale,
    hc_base,
    norm_eps,
    hc_eps,
    magnitude,
    rms_weight=None,
    rms_eps=0.0,
    cast_bfloat_for_norm=False,
):
    check_bf16(xa, residual, "xa")
    check_tensor(H_post_in, residual, torch.float32, "H_post_in")
    require(residual.ndim == 3 and residual.shape[1] == 4, "residual must have shape [N,4,D]")
    require(xa.shape == (residual.shape[0], residual.shape[2]), "xa must have shape [N,D]")
    require(H_post_in.shape == residual.shape[:2], "H_post_in must have shape [N,4]")
    y = torch.empty_like(residual)
    z, gates = _pre_head(
        residual,
        w,
        hc_scale,
        hc_base,
        norm_eps,
        hc_eps,
        magnitude,
        rms_weight,
        rms_eps,
        cast_bfloat_for_norm,
        False,
        (xa, H_post_in, y),
    )
    return y, z, gates


def register(library):
    implementations = {
        "fuse_ihc_pre": pre,
        "fuse_ihc_post_pre": post_pre,
        "fuse_ihc_post": post,
        "fuse_ihc_head": head,
    }
    for schema in SCHEMAS:
        library.define(schema)
        name = schema.split("(", 1)[0]
        library.impl(name, implementations[name], "CUDA")
