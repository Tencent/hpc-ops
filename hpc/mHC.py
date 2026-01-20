import torch
from torch import Tensor
from typing import Tuple


def fuse_cal_mixes_hat_hat_H_and_r(
    x: Tensor, w_a: Tensor, w_b: Tensor, norm_eps: float = 1e-6
) -> Tensor:
    """Applies inverse root mean square of x, and linear porj for x and w.
    Specifically:
    1. r = torch.rsqrt(x.square().mean(dim=-1) + norm_eps)
    2. x = x / r
    3. mixes_hat_hat_H = torch.mm(x, w_a.T, out_dtype=torch.float32) + (2**-8) * torch.mm(x, w_b.T, out_dtype=torch.float32)

    Executes via a custom high-performance GPU kernel.

    Args:
      x: input tensor.
          Shape: [N, hc_dim] (N = batch size, hc_dim = hc_mult * d, hc_mult = HC expand ratio, d = hidden dim)
          Dtype: bfloat16

      w_a: high bit of hc linear proj weight.
          Shape: [mix_hc, hc_dim] (mix_hc = (2 + hc_mult) * hc_mult, hc_mult = HC expand ratio)
          Dtype: bfloat16

      w_b: low bit of hc linear proj weight.
          Shape: [mix_hc, hc_dim] (mix_hc = (2 + hc_mult) * hc_mult, hc_mult = HC expand ratio)
          Dtype: bfloat16

      norm_eps: norm eps, float, default value is 1e-6

    Returns:
      mixes_hat_hat_H: output tensor, composed of hat_hat_H_pre, hat_hat_H_post and hat_hat_H_res in last dim.
          Shape: [N, mix_hc]
          Dtype: float32
    """

    norm_x = torch.ops.hpc.fuse_cal_mixes_hat_hat_H_and_r(x, w_a, w_b, norm_eps)
    # TODO(lando): remove torch impl
    mixes_hat_hat_H = torch.mm(norm_x, w_a.T, out_dtype=torch.float32) + (1 / 256) * torch.mm(
        norm_x, w_b.T, out_dtype=torch.float32
    )
    return mixes_hat_hat_H


def fuse_cal_three_H(
    mixes_hat_hat_H: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int = 4,
    hc_sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Calculate H_pre, H_post and H_res
    Specifically:
    1. hat_hat_H_pre, hat_hat_H_post, hat_hat_H_res = mixes_hat_hat_H.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    2. scale_pre, scale_post, scale_res = hc_scale.split([1, 1, 1], dim=-1)
    3. base_pre, base_post, base_res = hc_scale.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    4. hat_H_pre = scale_pre * hat_hat_H_pre + base_pre
    5. hat_H_post = scale_post * hat_hat_H_post + base_post
    6. hat_H_res = scale_res * hat_hat_H_res + base_res
    7. H_pre = sigmoid(hat_H_pre) + hc_eps
    8. H_post = 2 * sigmoid(hat_H_post)
    9. H_res = Sinkhorn-Knopp(har_H_res, hc_sinkhorn_iters, hc_eps)

    Executes via a custom high-performance GPU kernel.

    Args:
      mixes_hat_hat_H: input tensor, composed of hat_hat_H_pre, hat_hat_H_post and hat_hat_H_res in last dim.
          Shape: [N, mix_hc] (N = batch size, mix_hc = (2 + hc_mult) * hc_mult, hc_mult = HC expand ratio)
          Dtype: float32
      hc_scale: hc scale tensor, composed of scale_pre, scale_post, scale_res in last dim.
          Shape: [3, ]
          Dtype: float32
      hc_base: hc base tensor, composed of base_pre, base_post, base_res in last dim.
          Shape: [mix_hc, ]
          Dtype: float32
      hc_mult: HC expand ratio, int, default value is 4
      hc_sinkhorn_iters: Iteration number of Sinkhorn-Knopp algorithm, int, default value is 20
      hc_eps: eps of norm in Sinkhorn-Knopp algorithm, float, default value is 1e-6

    Returns:
      H_pre: H_pre mapping weight tensor.
          Shape: [N, hc_mult]
          Dtype: float32
      H_post: H_post mapping weight tensor.
          Shape: [N, hc_mult]
          Dtype: float32
      H_res: H_res mapping weight tensor.
          Shape: [N, hc_mult, hc_mult]
          Dtype: float32
    """
    H_pre, H_post, H_res = torch.ops.hpc.fuse_cal_three_H(
        mixes_hat_hat_H,
        hc_scale,
        hc_base,
        hc_mult,
        hc_sinkhorn_iters,
        hc_eps,
    )
    return H_pre, H_post, H_res


def fuse_hc_pre_mapping(x: Tensor, H_pre: Tensor) -> Tensor:
    """Apply H_pre mapping to x, then cast result from float32 to bfloat16
    Specifically:
    1. x = x.float()
    2. y = torch.sum(H_pre.unsqueeze(-1) * x, dim=1)
    3. y = y.to(dtype=torch.bfloat16)

    Executes via a custom high-performance GPU kernel.

    Args:
      x: input tensor.
          Shape: [N, hc_mult, d] (N = batch size, hc_mult = HC expand ratio, d = hidden dim)
          Dtype: bfloat16
      H_pre: H_pre mapping weight tensor.
          Shape: [N, hc_mult]
          Dtype: float32

    Returns:
      y: output tensor.
          Shape: [N, d] (N = batch size, d = hidden dim)
          Dtype: bfloat16

    """

    y = torch.ops.hpc.fuse_hc_pre_mapping(x, H_pre)
    return y


def fuse_H_post_mapping_H_res_mapping_and_residual_add(
    x: Tensor, residual: Tensor, H_post: Tensor, H_res: Tensor
) -> Tensor:
    """Apply H_post mapping to x, H_res mapping to residual, and residual add
    Specifically:
    1. x = x.float()
    2. y1 = post.unsqueeze(-1) * x.unsqueeze(-2)
    3. residual = residual.float()
    4. y2 = torch.sum(res.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    5. y = (y1 + y2).to(dtype=torch.bfloat16)

    Executes via a custom high-performance GPU kernel.

    Args:
      x: input tensor.
          Shape: [N, d] (N = batch size, hc_mult = HC expand ratio, d = hidden dim)
          Dtype: bfloat16
      residual: input tensor.
          Shape: [N, hc_mult, d] (N = batch size, hc_mult = HC expand ratio, d = hidden dim)
          Dtype: bfloat16
      H_post: H_post mapping weight tensor.
          Shape: [N, hc_mult]
          Dtype: float32
      H_res: H_res mapping weight tensor.
          Shape: [N, hc_mult, hc_mult]
          Dtype: float32

    Returns:
      y: output tensor.
          Shape: [N, hc_mult, d] (N = batch size, hc_mult = HC expand ratio, d = hidden dim)
          Dtype: bfloat16
    """

    y = torch.ops.hpc.fuse_H_post_mapping_H_res_mapping_and_residual_add(x, residual, H_post, H_res)
    return y
