import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import torch.nn.functional as F
import pytest
from utils import allclose
from packaging.version import Version


def naive_fuse_cal_mixes_hat_hat_H_and_r(x, w, norm_eps):
    x = x.float()
    r = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + norm_eps)
    mixes_hat_hat_H = F.linear(x, w) * r
    return mixes_hat_hat_H


def is_torch_version_less_than(target_version: str) -> bool:
    current_version = torch.__version__.split("+")[0]  # remove cuda version like "+cu118"
    current_ver = Version(current_version)
    target_ver = Version(target_version)
    return current_ver < target_ver


@pytest.mark.skipif(
    is_torch_version_less_than("2.8.0"), reason="torch.mm has out_dtype arg since torch 2.8.0"
)
@pytest.mark.parametrize("num_batch", [128, 16384])
@pytest.mark.parametrize("hc_dim", [4 * 4096])
@pytest.mark.parametrize("mix_hc", [(2 + 4) * 4])
@pytest.mark.parametrize("norm_eps", [1e-6])
def test_fuse_cal_mixes_hat_hat_H_and_r(num_batch, hc_dim, mix_hc, norm_eps):
    torch.cuda.manual_seed(13)

    x = torch.rand([num_batch, hc_dim], dtype=torch.float32, device="cuda").to(dtype=torch.bfloat16)
    w = torch.rand([mix_hc, hc_dim], dtype=torch.float32, device="cuda")

    ref_mixes = naive_fuse_cal_mixes_hat_hat_H_and_r(x, w, norm_eps)

    w_a = w.to(dtype=torch.bfloat16)
    w_b = ((w - w_a.float()) * (256)).to(dtype=torch.bfloat16)
    new_w = torch.cat([w_a, w_b], dim=0)
    real_mixes = hpc.fuse_cal_mixes_hat_hat_H_and_r(x, new_w, norm_eps)

    assert allclose(ref_mixes, real_mixes, atol=0.3, rtol=4e-4)


def naive_fuse_cal_three_H(mixes_hat_hat_H, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps):
    hat_hat_H_pre, hat_hat_H_post, hat_hat_H_res = mixes_hat_hat_H.split(
        [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    hc_pre_base, hc_post_base, hc_res_base = hc_base.split(
        [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    H_pre = torch.sigmoid(hc_scale[0] * hat_hat_H_pre + hc_pre_base) + hc_eps
    H_post = 2.0 * torch.sigmoid(hc_scale[1] * hat_hat_H_post + hc_post_base)
    H_res = hc_scale[2] * hat_hat_H_res + hc_res_base

    H_res = H_res.reshape([-1, hc_mult, hc_mult])
    H_res = H_res.softmax(dim=-1) + hc_eps
    H_res = H_res / (H_res.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(hc_sinkhorn_iters - 1):
        H_res = H_res / (H_res.sum(dim=-1, keepdim=True) + hc_eps)
        H_res = H_res / (H_res.sum(dim=-2, keepdim=True) + hc_eps)

    return H_pre, H_post, H_res


@pytest.mark.parametrize("num_batch", [128, 16384])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("hc_sinkhorn_iters", [20])
@pytest.mark.parametrize("hc_eps", [1e-6])
def test_fuse_cal_three_H(num_batch, hc_mult, hc_sinkhorn_iters, hc_eps):
    torch.cuda.manual_seed(13)

    hc_dim = 2 * hc_mult + (hc_mult * hc_mult)
    mixes_hat_hat_H = torch.rand((num_batch, hc_dim), dtype=torch.float32, device="cuda")
    hc_scale = torch.rand((3,), dtype=torch.float32, device="cuda")
    hc_base = torch.rand((hc_dim,), dtype=torch.float32, device="cuda")

    ref_H_pre, ref_H_post, ref_H_res = naive_fuse_cal_three_H(
        mixes_hat_hat_H, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps
    )

    real_H_pre, real_H_post, real_H_res = hpc.fuse_cal_three_H(
        mixes_hat_hat_H, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps
    )

    assert allclose(ref_H_pre, real_H_pre)
    assert allclose(ref_H_post, real_H_post)
    assert allclose(ref_H_res, real_H_res, atol=2e-5, rtol=1e-4)


def naive_fuse_hc_pre_mapping(x, H_pre):
    x = x.float()
    y = torch.sum(H_pre.unsqueeze(-1) * x, dim=1)
    y = y.to(dtype=torch.bfloat16)
    return y


@pytest.mark.parametrize("num_batch", [128, 16384])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("hidden_dim", [4096])
def test_fuse_hc_pre_mapping(num_batch, hc_mult, hidden_dim):
    torch.cuda.manual_seed(13)

    x = torch.rand((num_batch, hc_mult, hidden_dim), dtype=torch.float32, device="cuda").to(
        dtype=torch.bfloat16
    )
    H_pre = torch.rand((num_batch, hc_mult), dtype=torch.float32, device="cuda")

    ref_y = naive_fuse_hc_pre_mapping(x, H_pre)
    real_y = hpc.fuse_hc_pre_mapping(x, H_pre)

    assert allclose(ref_y, real_y, atol=2e-2, rtol=1e-2)


def naive_fuse_H_post_mapping_H_res_mapping_and_residual_add(x, residual, H_post, H_res):
    x = x.float()
    y1 = H_post.unsqueeze(-1) * x.unsqueeze(-2)
    residual = residual.float()
    y2 = torch.sum(H_res.unsqueeze(-1) * residual.unsqueeze(-2), dim=1)
    y = (y1 + y2).to(dtype=torch.bfloat16)
    return y


@pytest.mark.parametrize("num_batch", [128, 16384])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("hidden_dim", [4096])
def test_fuse_H_post_mapping_H_res_mapping_and_residual_add(num_batch, hc_mult, hidden_dim):
    torch.cuda.manual_seed(13)

    x = torch.rand((num_batch, hidden_dim), dtype=torch.float32, device="cuda").to(
        dtype=torch.bfloat16
    )
    residual = torch.rand((num_batch, hc_mult, hidden_dim), dtype=torch.float32, device="cuda").to(
        dtype=torch.bfloat16
    )
    H_post = torch.rand((num_batch, hc_mult), dtype=torch.float32, device="cuda")
    H_res = torch.rand((num_batch, hc_mult, hc_mult), dtype=torch.float32, device="cuda")

    ref_y = naive_fuse_H_post_mapping_H_res_mapping_and_residual_add(x, residual, H_post, H_res)

    real_y = hpc.fuse_H_post_mapping_H_res_mapping_and_residual_add(x, residual, H_post, H_res)

    assert allclose(ref_y, real_y, atol=2e-2, rtol=1e-2)
