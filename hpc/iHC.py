import os

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


def fuse_ihc_pre(
    x: Tensor,
    w: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    norm_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    magnitude: float = 2.0,
    rms_weight: Tensor = None,
    rms_eps: float = 0.0,
    cast_bfloat_for_norm: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Fully fused iHC (independent Hyper-Connections) pre block.
    Specifically:
    1. x_flat = x.flatten(1).float()
    2. r = torch.rsqrt(x_flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    3. mixes = F.linear(x_flat, w) * r
    4. H_pre = sigmoid(hc_scale[0] * mixes[:, :hc_mult] + hc_base[:hc_mult]) + hc_eps
    5. H_post = magnitude * sigmoid(hc_scale[1] * mixes[:, hc_mult:] + hc_base[hc_mult:]) + hc_eps
    6. y = torch.sum(H_pre.unsqueeze(-1) * x_flat.view_as(x), dim=1).to(torch.bfloat16)

    Args:
      x: input tensor.
          Shape: [N, hc_mult, d] (N = batch size, hc_mult = HC expand ratio, d = hidden dim)
          Dtype: bfloat16
      w: hc linear proj weight. The checkpoint keeps hc_fn out of fp8 quantization
          (quantization_config.modules_to_not_convert), so this stays float32.
          Shape: [2 * hc_mult, hc_mult * d]
          Dtype: float32
      hc_scale: hc scale tensor, composed of scale_pre, scale_post in last dim.
          Shape: [2, ]
          Dtype: float32
      hc_base: hc base tensor, composed of base_pre, base_post in last dim.
          Shape: [2 * hc_mult, ]
          Dtype: float32
      norm_eps: rms norm eps, float, default value is 1e-5
      hc_eps: eps added to both gates, float, default value is 1e-6
      magnitude: H_post multiplier, float, default value is 2.0
      rms_weight: optional RMSNorm weight. When given, step 6 continues with
          y = y * rsqrt(y.square().mean(-1) + rms_eps) * rms_weight, i.e. the
          caller's input_layernorm is folded into this kernel. y then never
          reaches HBM unnormalized, saving a [N, d] round trip and one launch.
          Measured cost of folding it in: +1.7% on this kernel.
          Shape: [d, ]
          Dtype: bfloat16
      rms_eps: eps inside the folded RMSNorm. Only read when rms_weight is given.
      cast_bfloat_for_norm: cast output to bfloat16 for layernorm.
    Returns:
      y: pre-gated reduction of x over the hc dim, RMS-normalized when rms_weight
          was supplied.
          Shape: [N, d]
          Dtype: bfloat16
      H_post: H_post mapping weight tensor, consumed by fuse_ihc_post.
          Shape: [N, hc_mult]
          Dtype: float32
    """

    y, H_post = torch.ops.hpc.fuse_ihc_pre(
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
    )
    return y, H_post


def fuse_ihc_post(x: Tensor, residual: Tensor, H_post: Tensor) -> Tensor:
    """Apply H_post mapping to x and add the multi-channel residual.

    Args:
      x: input tensor, the attention / MLP output.
          Shape: [N, d] (N = batch size, d = hidden dim)
          Dtype: bfloat16
      residual: multi-channel residual tensor.
          Shape: [N, hc_mult, d] (hc_mult = HC expand ratio)
          Dtype: bfloat16
      H_post: H_post mapping weight tensor.
          Shape: [N, hc_mult]
          Dtype: float32

    Returns:
      y: output tensor.
          Shape: [N, hc_mult, d]
          Dtype: bfloat16
    """

    y = torch.ops.hpc.fuse_ihc_post(x, residual, H_post)
    return y


def fuse_ihc_head(
    x: Tensor,
    w: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    norm_eps: float = 1e-5,
    hc_eps: float = 1e-6,
) -> Tensor:
    """Fused iHC head block, merges the hc channels back into a single hidden state.

    Args:
      x: input tensor.
          Shape: [N, hc_mult, d] (N = batch size, hc_mult = HC expand ratio, d = hidden dim)
          Dtype: bfloat16
      w: hc head linear proj weight. Kept out of fp8 quantization like hc_fn.
          Shape: [hc_mult, hc_mult * d]
          Dtype: float32
      hc_scale: hc head scale tensor.
          Shape: [1, ]
          Dtype: float32
      hc_base: hc head base tensor.
          Shape: [hc_mult, ]
          Dtype: float32
      norm_eps: rms norm eps, float, default value is 1e-5
      hc_eps: eps added to the gate, float, default value is 1e-6

    Returns:
      y: output tensor.
          Shape: [N, d]
          Dtype: bfloat16
    """

    y = torch.ops.hpc.fuse_ihc_head(x, w, hc_scale, hc_base, norm_eps, hc_eps)
    return y


def fuse_ihc_post_pre(
    xa: Tensor,
    residual: Tensor,
    H_post_in: Tensor,
    w: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    norm_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    magnitude: float = 2.0,
    rms_weight: Tensor = None,
    rms_eps: float = 0.0,
    cast_bfloat_for_norm: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """One segment's iHC post block plus the next segment's pre block, in one kernel.

    Args:
      xa: the first segment's attention/MLP output.
          Shape: [N, d]
          Dtype: bfloat16
      residual: the first segment's multi-channel residual.
          Shape: [N, hc_mult, d]
          Dtype: bfloat16
      H_post_in: the first segment's H_post gates (from its own pre block).
          Shape: [N, hc_mult]
          Dtype: float32
      w: the SECOND segment's hc linear proj weight. Stays float32, see fuse_ihc_pre.
          Shape: [2 * hc_mult, hc_mult * d]
          Dtype: float32
      hc_scale: the second segment's hc scale (scale_pre, scale_post).
          Shape: [2, ]
          Dtype: float32
      hc_base: the second segment's hc base (base_pre, base_post).
          Shape: [2 * hc_mult, ]
          Dtype: float32
      norm_eps: rms norm eps inside the pre block, float, default 1e-5
      hc_eps: eps added to both gates, float, default 1e-6
      magnitude: H_post multiplier, float, default 2.0
      rms_weight: optional RMSNorm weight folded onto z, same contract as in
          fuse_ihc_pre. Lets one kernel cover post + pre + the following norm.
          Shape: [d, ]
          Dtype: bfloat16
      rms_eps: eps inside the folded RMSNorm. Only read when rms_weight is given.
      cast_bfloat_for_norm: cast output to bfloat16 for layernorm.
    Returns:
      y: the post block's output. Still materialized because the next segment takes
          it as its residual.
          Shape: [N, hc_mult, d]
          Dtype: bfloat16
      z: the pre block's reduction of y over the hc dim, RMS-normalized when
          rms_weight was supplied.
          Shape: [N, d]
          Dtype: bfloat16
      H_post_out: the second segment's H_post gates, consumed by its fuse_ihc_post.
          Shape: [N, hc_mult]
          Dtype: float32
    """
    y, z, H_post_out = torch.ops.hpc.fuse_ihc_post_pre(
        xa,
        residual,
        H_post_in,
        w,
        hc_scale,
        hc_base,
        norm_eps,
        hc_eps,
        magnitude,
        rms_weight,
        rms_eps,
        cast_bfloat_for_norm,
    )
    return y, z, H_post_out
