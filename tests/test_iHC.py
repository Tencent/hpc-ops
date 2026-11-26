import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import torch.nn.functional as F
import pytest
from utils import allclose

torch.backends.cuda.matmul.allow_tf32 = True


def naive_fuse_ihc_pre(x, w, hc_scale, hc_base, hc_mult, norm_eps, hc_eps, magnitude):
    x_flat = x.flatten(1).float()
    r = torch.rsqrt(x_flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, w) * r

    H_pre = torch.sigmoid(hc_scale[0] * mixes[:, :hc_mult] + hc_base[:hc_mult]) + hc_eps
    H_post = (
        magnitude * torch.sigmoid(hc_scale[1] * mixes[:, hc_mult:] + hc_base[hc_mult:]) + hc_eps
    )

    y = torch.sum(H_pre.unsqueeze(-1) * x.float(), dim=1).to(dtype=torch.bfloat16)
    return y, H_post


@pytest.mark.parametrize("num_batch", [1, 7, 128, 16384])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
@pytest.mark.parametrize("norm_eps", [1e-5])
@pytest.mark.parametrize("hc_eps", [1e-6])
@pytest.mark.parametrize("magnitude", [2.0])
def test_fuse_ihc_pre(num_batch, hc_mult, hidden_dim, norm_eps, hc_eps, magnitude):
    torch.cuda.manual_seed(13)

    x = torch.rand((num_batch, hc_mult, hidden_dim), dtype=torch.float32, device="cuda").to(
        dtype=torch.bfloat16
    )
    w = torch.rand((2 * hc_mult, hc_mult * hidden_dim), dtype=torch.float32, device="cuda") * 6e-3
    hc_scale = torch.rand((2,), dtype=torch.float32, device="cuda")
    hc_base = torch.rand((2 * hc_mult,), dtype=torch.float32, device="cuda")

    ref_y, ref_H_post = naive_fuse_ihc_pre(
        x, w, hc_scale, hc_base, hc_mult, norm_eps, hc_eps, magnitude
    )
    real_y, real_H_post = hpc.fuse_ihc_pre(x, w, hc_scale, hc_base, norm_eps, hc_eps, magnitude)

    assert allclose(ref_y, real_y, atol=2e-2, rtol=1e-2)
    assert allclose(ref_H_post, real_H_post, atol=2e-5, rtol=1e-4)


def naive_fuse_ihc_post(x, residual, H_post):
    y = H_post.unsqueeze(-1) * x.float().unsqueeze(-2) + residual.float()
    return y.to(dtype=torch.bfloat16)


@pytest.mark.parametrize("num_batch", [1, 7, 128, 16384])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
def test_fuse_ihc_post(num_batch, hc_mult, hidden_dim):
    torch.cuda.manual_seed(13)

    x = torch.rand((num_batch, hidden_dim), dtype=torch.float32, device="cuda").to(
        dtype=torch.bfloat16
    )
    residual = torch.rand((num_batch, hc_mult, hidden_dim), dtype=torch.float32, device="cuda").to(
        dtype=torch.bfloat16
    )
    H_post = torch.rand((num_batch, hc_mult), dtype=torch.float32, device="cuda")

    ref_y = naive_fuse_ihc_post(x, residual, H_post)
    real_y = hpc.fuse_ihc_post(x, residual, H_post)

    assert allclose(ref_y, real_y, atol=2e-2, rtol=1e-2)


def naive_fuse_ihc_head(x, w, hc_scale, hc_base, norm_eps, hc_eps, rms_weight=None, rms_eps=0.0):
    x_flat = x.flatten(1).float()
    r = torch.rsqrt(x_flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, w) * r
    H_pre = torch.sigmoid(hc_scale * mixes + hc_base) + hc_eps
    y = torch.sum(H_pre.unsqueeze(-1) * x.float(), dim=1)
    if rms_weight is not None:
        y = naive_rms_norm(y, rms_weight, rms_eps)
    return y.to(dtype=torch.bfloat16)


@pytest.mark.parametrize("num_batch", [1, 7, 128, 16384])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
@pytest.mark.parametrize("norm_eps", [1e-5])
@pytest.mark.parametrize("hc_eps", [1e-6])
def test_fuse_ihc_head(num_batch, hc_mult, hidden_dim, norm_eps, hc_eps):
    torch.cuda.manual_seed(13)

    x = torch.rand((num_batch, hc_mult, hidden_dim), dtype=torch.float32, device="cuda").to(
        dtype=torch.bfloat16
    )
    w = torch.rand((hc_mult, hc_mult * hidden_dim), dtype=torch.float32, device="cuda") * 6e-3
    hc_scale = torch.rand((1,), dtype=torch.float32, device="cuda")
    hc_base = torch.rand((hc_mult,), dtype=torch.float32, device="cuda")

    ref_y = naive_fuse_ihc_head(x, w, hc_scale, hc_base, norm_eps, hc_eps)
    real_y = hpc.fuse_ihc_head(x, w, hc_scale, hc_base, norm_eps, hc_eps)

    assert allclose(ref_y, real_y, atol=2e-2, rtol=1e-2)


def naive_fuse_ihc_post_pre(
    xa,
    residual,
    H_post_in,
    w,
    hc_scale,
    hc_base,
    hc_mult,
    norm_eps,
    hc_eps,
    magnitude,
    rms_weight=None,
    rms_eps=0.0,
):
    y = naive_fuse_ihc_post(xa, residual, H_post_in)
    z, H_post_out = naive_fuse_ihc_pre(
        y, w, hc_scale, hc_base, hc_mult, norm_eps, hc_eps, magnitude
    )
    if rms_weight is not None:
        zf = z.float()
        z = (
            zf * torch.rsqrt(zf.square().mean(dim=-1, keepdim=True) + rms_eps) * rms_weight.float()
        ).to(dtype=torch.bfloat16)
    return y, z, H_post_out


@pytest.mark.parametrize(
    ("num_batch", "hidden_dim", "weight_offset"),
    [
        (num_batch, hidden_dim, 0)
        for hidden_dim in (6144, 4096, 3072)
        for num_batch in (1, 7, 128, 512, 513, 545, 16384)
    ]
    + [
        (148, 4096, 0),
        (149, 4096, 0),
        (296, 4096, 0),
        (297, 4096, 0),
        (79, 6144, 0),
        (80, 6144, 0),
        (296, 6144, 0),
        (297, 6144, 0),
        (80, 6144, 4),
        (296, 6144, 4),
        (297, 6144, 4),
    ],
)
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("fuse_norm", [False, True])
def test_fuse_ihc_post_pre(num_batch, hc_mult, hidden_dim, fuse_norm, weight_offset):
    torch.cuda.manual_seed(13)
    norm_eps, hc_eps, magnitude, rms_eps = 1e-5, 1e-6, 2.0, 1e-5

    xa = torch.rand((num_batch, hidden_dim), dtype=torch.float32, device="cuda").to(
        dtype=torch.bfloat16
    )
    residual = torch.rand((num_batch, hc_mult, hidden_dim), dtype=torch.float32, device="cuda").to(
        dtype=torch.bfloat16
    )
    H_post_in = torch.rand((num_batch, hc_mult), dtype=torch.float32, device="cuda")
    weight_rows, weight_cols = 2 * hc_mult, hc_mult * hidden_dim
    w = torch.rand(weight_rows * weight_cols + weight_offset, dtype=torch.float32, device="cuda")
    w = w[weight_offset:].view(weight_rows, weight_cols).mul_(6e-3)
    assert w.data_ptr() % 32 == weight_offset * w.element_size()
    hc_scale = torch.rand((2,), dtype=torch.float32, device="cuda")
    hc_base = torch.rand((2 * hc_mult,), dtype=torch.float32, device="cuda")
    rms_weight = (
        torch.rand((hidden_dim,), dtype=torch.float32, device="cuda").to(dtype=torch.bfloat16)
        if fuse_norm
        else None
    )

    ref_y, ref_z, ref_H_post = naive_fuse_ihc_post_pre(
        xa,
        residual,
        H_post_in,
        w,
        hc_scale,
        hc_base,
        hc_mult,
        norm_eps,
        hc_eps,
        magnitude,
        rms_weight,
        rms_eps,
    )
    if fuse_norm:
        real_y, real_z, real_H_post = hpc.fuse_ihc_post_pre(
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
        )
    else:
        real_y, real_z, real_H_post = hpc.fuse_ihc_post_pre(
            xa, residual, H_post_in, w, hc_scale, hc_base, norm_eps, hc_eps, magnitude
        )

    assert allclose(ref_y, real_y, atol=2e-2, rtol=1e-2)
    assert allclose(ref_z, real_z, atol=2e-2, rtol=1e-2)
    assert allclose(ref_H_post, real_H_post, atol=2e-5, rtol=1e-4)


HC_MULT = 4
MAGNITUDE = 2.0
HC_EPS = 1e-6
RMS_EPS = 1e-5


def bit_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    """Number of elements differing in their raw bit pattern."""
    assert a.dtype == b.dtype and a.shape == b.shape
    view = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    return int((a.view(view) != b.view(view)).sum().item())


def max_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    """Max ULP distance. Only meaningful for same-sign, non-NaN values."""
    view = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    ia = a.view(view).to(torch.int64)
    ib = b.view(view).to(torch.int64)
    return int((ia - ib).abs().max().item())


def torch_ihc_pre(x, w, hc_scale, hc_base):
    hc = HC_MULT
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + RMS_EPS)
    mixes = F.linear(x_flat, w) * rsqrt
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + HC_EPS
    post = (
        MAGNITUDE * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
        + HC_EPS
    )
    y = torch.sum(pre.unsqueeze(-1) * x.float(), dim=1)
    return y.to(x.dtype), post


def torch_ihc_post(x, residual, H_post):
    return (H_post.float().unsqueeze(-1) * x.float().unsqueeze(-2) + residual.float()).to(x.dtype)


def torch_ihc_head(x, w, hc_scale, hc_base):
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + RMS_EPS)
    mixes = F.linear(x_flat, w) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + HC_EPS
    return torch.sum(pre.unsqueeze(-1) * x.float(), dim=1).to(x.dtype)


def make_pre_inputs(num_batch, hidden_dim, seed=13):
    torch.manual_seed(seed)
    dev = "cuda"
    x = torch.rand((num_batch, HC_MULT, hidden_dim), device=dev).to(torch.bfloat16)
    w = torch.rand((2 * HC_MULT, HC_MULT * hidden_dim), dtype=torch.float32, device=dev) * 6e-3
    hc_scale = torch.rand((2,), dtype=torch.float32, device=dev)
    hc_base = torch.rand((2 * HC_MULT,), dtype=torch.float32, device=dev)
    return x, w, hc_scale, hc_base


def make_head_inputs(num_batch, hidden_dim, seed=13):
    torch.manual_seed(seed)
    dev = "cuda"
    x = torch.rand((num_batch, HC_MULT, hidden_dim), device=dev).to(torch.bfloat16)
    w = torch.rand((HC_MULT, HC_MULT * hidden_dim), dtype=torch.float32, device=dev) * 6e-3
    hc_scale = torch.rand((1,), dtype=torch.float32, device=dev)
    hc_base = torch.rand((HC_MULT,), dtype=torch.float32, device=dev)
    return x, w, hc_scale, hc_base


@pytest.mark.parametrize("num_batch", [1, 7, 16, 128, 512, 4096])
@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
def test_fuse_ihc_post_bitwise(num_batch, hidden_dim):
    torch.manual_seed(13)
    dev = "cuda"
    x = torch.rand((num_batch, hidden_dim), device=dev).to(torch.bfloat16)
    residual = torch.rand((num_batch, HC_MULT, hidden_dim), device=dev).to(torch.bfloat16)
    H_post = torch.rand((num_batch, HC_MULT), dtype=torch.float32, device=dev) + 0.5

    ref = torch_ihc_post(x, residual, H_post)
    got = hpc.fuse_ihc_post(x, residual, H_post)

    ndiff = bit_diff(ref, got)
    total = ref.numel()
    assert (
        max_ulp(ref, got) <= 1
    ), f"post exceeded 1 ULP: {ndiff}/{total} elements differ, max_ulp={max_ulp(ref, got)}"
    assert ndiff <= total * 1e-4, f"too many 1-ULP diffs: {ndiff}/{total}"


@pytest.mark.parametrize("num_batch", [1, 7, 16, 128, 512, 4096])
@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
def test_fuse_ihc_pre_matches_torch(num_batch, hidden_dim):
    x, w, hc_scale, hc_base = make_pre_inputs(num_batch, hidden_dim)

    ref_y, ref_post = torch_ihc_pre(x, w, hc_scale, hc_base)
    got_y, got_post = hpc.fuse_ihc_pre(x, w, hc_scale, hc_base, RMS_EPS, HC_EPS, MAGNITUDE)

    assert got_y.dtype == torch.bfloat16
    assert got_post.dtype == torch.float32

    torch.testing.assert_close(got_y.float(), ref_y.float(), atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(got_post, ref_post, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("num_batch", [1, 7, 16, 128, 512, 4096])
@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
def test_fuse_ihc_head_matches_torch(num_batch, hidden_dim):
    x, w, hc_scale, hc_base = make_head_inputs(num_batch, hidden_dim)

    ref = torch_ihc_head(x, w, hc_scale, hc_base)
    got = hpc.fuse_ihc_head(x, w, hc_scale, hc_base, RMS_EPS, HC_EPS)

    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), ref.float(), atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
def test_fuse_ihc_pre_gate_structure(hidden_dim):
    """x=1, w=0, scale=0 isolates the constant terms exactly."""
    dev = "cuda"
    num_batch = 8
    x = torch.ones((num_batch, HC_MULT, hidden_dim), dtype=torch.bfloat16, device=dev)
    w = torch.zeros((2 * HC_MULT, HC_MULT * hidden_dim), dtype=torch.float32, device=dev)
    hc_scale = torch.zeros((2,), dtype=torch.float32, device=dev)
    hc_base = torch.zeros((2 * HC_MULT,), dtype=torch.float32, device=dev)

    y, H_post = hpc.fuse_ihc_pre(x, w, hc_scale, hc_base, RMS_EPS, HC_EPS, MAGNITUDE)

    assert torch.allclose(
        H_post, torch.full_like(H_post, MAGNITUDE * 0.5 + HC_EPS), atol=1e-6
    ), f"H_post structure wrong: got {H_post[0, 0].item()}"
    expected_y = HC_MULT * (0.5 + HC_EPS)
    assert (
        abs(y[0, 0].float().item() - expected_y) < 2e-2
    ), f"H_pre/eps structure wrong: y={y[0, 0].float().item()} expected {expected_y}"


@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
def test_fuse_ihc_pre_rsqrt_divisor(hidden_dim):
    """The rsqrt divides by hc_mult*hidden_dim, not hidden_dim (a 2x error otherwise)."""
    x, w, hc_scale, hc_base = make_pre_inputs(64, hidden_dim)
    ref_y, ref_post = torch_ihc_pre(x, w, hc_scale, hc_base)
    got_y, got_post = hpc.fuse_ihc_pre(x, w, hc_scale, hc_base, RMS_EPS, HC_EPS, MAGNITUDE)
    torch.testing.assert_close(got_post, ref_post, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(got_y.float(), ref_y.float(), atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
def test_fuse_ihc_post_identity(hidden_dim):
    """H_post=0 must leave the residual bit-for-bit untouched."""
    dev = "cuda"
    num_batch = 32
    torch.manual_seed(13)
    x = torch.rand((num_batch, hidden_dim), device=dev).to(torch.bfloat16)
    residual = torch.rand((num_batch, HC_MULT, hidden_dim), device=dev).to(torch.bfloat16)
    H_post = torch.zeros((num_batch, HC_MULT), dtype=torch.float32, device=dev)

    got = hpc.fuse_ihc_post(x, residual, H_post)
    assert torch.equal(got, residual), "H_post=0 must pass the residual through unchanged"


@pytest.mark.parametrize("num_batch", [7, 512])
@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
def test_fuse_ihc_pre_deterministic(num_batch, hidden_dim):
    x, w, hc_scale, hc_base = make_pre_inputs(num_batch, hidden_dim)
    first_y, first_post = hpc.fuse_ihc_pre(x, w, hc_scale, hc_base, RMS_EPS, HC_EPS, MAGNITUDE)
    first_y, first_post = first_y.clone(), first_post.clone()
    for _ in range(5):
        y, post = hpc.fuse_ihc_pre(x, w, hc_scale, hc_base, RMS_EPS, HC_EPS, MAGNITUDE)
        assert torch.equal(y, first_y), "pre y is not deterministic"
        assert torch.equal(post, first_post), "pre H_post is not deterministic"


@pytest.mark.parametrize("num_batch", [7, 512])
@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
def test_fuse_ihc_post_deterministic(num_batch, hidden_dim):
    dev = "cuda"
    torch.manual_seed(13)
    x = torch.rand((num_batch, hidden_dim), device=dev).to(torch.bfloat16)
    residual = torch.rand((num_batch, HC_MULT, hidden_dim), device=dev).to(torch.bfloat16)
    H_post = torch.rand((num_batch, HC_MULT), dtype=torch.float32, device=dev) + 0.5
    first = hpc.fuse_ihc_post(x, residual, H_post).clone()
    for _ in range(5):
        assert torch.equal(
            hpc.fuse_ihc_post(x, residual, H_post), first
        ), "post is not deterministic"


def naive_rms_norm(y, rms_weight, rms_eps):
    yf = y.float()
    scale = torch.rsqrt(yf.square().mean(dim=-1, keepdim=True) + rms_eps)
    return (yf * scale * rms_weight.float()).to(dtype=torch.bfloat16)


@pytest.mark.parametrize("num_batch", [1, 7, 16, 128, 512, 4096])
@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
@pytest.mark.parametrize("norm_eps", [1e-5])
@pytest.mark.parametrize("hc_eps", [1e-6])
@pytest.mark.parametrize("magnitude", [2.0])
def test_fuse_ihc_pre_with_rms_weight(num_batch, hidden_dim, norm_eps, hc_eps, magnitude):
    torch.cuda.manual_seed(13)
    hc_mult = HC_MULT
    dev = "cuda"

    x = torch.rand((num_batch, hc_mult, hidden_dim), dtype=torch.float32, device=dev).to(
        dtype=torch.bfloat16
    )
    w = torch.rand((2 * hc_mult, hc_mult * hidden_dim), dtype=torch.float32, device=dev) * 6e-3
    hc_scale = torch.rand((2,), dtype=torch.float32, device=dev)
    hc_base = torch.rand((2 * hc_mult,), dtype=torch.float32, device=dev)
    rms_weight = torch.rand((hidden_dim,), dtype=torch.float32, device=dev).to(dtype=torch.bfloat16)

    ref_y, ref_H_post = naive_fuse_ihc_pre(
        x, w, hc_scale, hc_base, hc_mult, norm_eps, hc_eps, magnitude
    )
    ref_y = naive_rms_norm(ref_y, rms_weight, RMS_EPS)

    real_y, real_H_post = hpc.fuse_ihc_pre(
        x, w, hc_scale, hc_base, norm_eps, hc_eps, magnitude, rms_weight, RMS_EPS
    )

    assert allclose(ref_y, real_y, atol=2e-2, rtol=1e-2)
    assert allclose(ref_H_post, real_H_post, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("num_batch", [1, 7, 16, 128, 512, 4096])
@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
@pytest.mark.parametrize("norm_eps", [1e-5])
@pytest.mark.parametrize("hc_eps", [1e-6])
def test_fuse_ihc_head_with_rms_weight(num_batch, hidden_dim, norm_eps, hc_eps):
    torch.cuda.manual_seed(13)
    hc_mult = HC_MULT
    dev = "cuda"

    x = torch.rand((num_batch, hc_mult, hidden_dim), dtype=torch.float32, device=dev).to(
        dtype=torch.bfloat16
    )
    w = torch.rand((hc_mult, hc_mult * hidden_dim), dtype=torch.float32, device=dev) * 6e-3
    hc_scale = torch.rand((1,), dtype=torch.float32, device=dev)
    hc_base = torch.rand((hc_mult,), dtype=torch.float32, device=dev)
    rms_weight = torch.rand((hidden_dim,), dtype=torch.float32, device=dev).to(dtype=torch.bfloat16)

    ref_y = naive_fuse_ihc_head(x, w, hc_scale, hc_base, norm_eps, hc_eps, rms_weight, RMS_EPS)

    real_y = torch.ops.hpc.fuse_ihc_head(
        x, w, hc_scale, hc_base, norm_eps, hc_eps, rms_weight, RMS_EPS
    )

    assert allclose(ref_y, real_y, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("op", ["pre", "post_pre", "head"])
@pytest.mark.parametrize(
    ("num_batch", "hidden_dim"),
    [(1, 6144), (80, 6144), (297, 6144), (7, 4096), (512, 4096), (1, 3072), (128, 3072)],
)
def test_fuse_ihc_cast_bfloat_for_norm(op, num_batch, hidden_dim):
    """Use exact half gates and BF16 midpoints to distinguish the two RMSNorm modes."""
    channels = torch.zeros((num_batch, HC_MULT, hidden_dim), device="cuda", dtype=torch.bfloat16)
    channels[:, 0, :] = 1.0
    channels[:, 1, ::2] = 1.0 + 2**-7
    channels[:, 1, 1::2] = 1.0 - 2**-8

    num_scales = 1 if op == "head" else 2
    num_gates = num_scales * HC_MULT
    weight = torch.zeros((num_gates, HC_MULT * hidden_dim), device="cuda", dtype=torch.float32)
    hc_scale = torch.ones(num_scales, device="cuda", dtype=torch.float32)
    hc_base = torch.zeros(num_gates, device="cuda", dtype=torch.float32)
    rms_weight = torch.ones(hidden_dim, device="cuda", dtype=torch.bfloat16)
    hc_args = (weight, hc_scale, hc_base, RMS_EPS, 0.0)

    mixed = channels.float().sum(dim=1) * 0.5
    ref_fp32 = naive_rms_norm(mixed, rms_weight, RMS_EPS)
    ref_bfloat = naive_rms_norm(mixed.bfloat16(), rms_weight, RMS_EPS)
    assert not torch.equal(ref_fp32, ref_bfloat)

    if op == "post_pre":
        post_input = torch.zeros_like(mixed, dtype=torch.bfloat16)
        post_gates = torch.ones((num_batch, HC_MULT), device="cuda", dtype=torch.float32)

    for cast_bfloat_for_norm, ref_y in [(False, ref_fp32), (True, ref_bfloat)]:
        norm_kwargs = dict(
            rms_weight=rms_weight, rms_eps=RMS_EPS, cast_bfloat_for_norm=cast_bfloat_for_norm
        )
        if op == "pre":
            output, _ = hpc.fuse_ihc_pre(channels, *hc_args, MAGNITUDE, **norm_kwargs)
        elif op == "post_pre":
            _, output, _ = hpc.fuse_ihc_post_pre(
                post_input, channels, post_gates, *hc_args, MAGNITUDE, **norm_kwargs
            )
        else:
            output = torch.ops.hpc.fuse_ihc_head(channels, *hc_args, **norm_kwargs)

        torch.testing.assert_close(output, ref_y, atol=0, rtol=0)


@pytest.mark.parametrize("num_batch", [1, 7, 64, 512])
@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
def test_fuse_ihc_pre_splitk_force_matches_row(num_batch, hidden_dim):
    x, w, hc_scale, hc_base = make_pre_inputs(num_batch, hidden_dim)

    args = (x, w, hc_scale, hc_base, RMS_EPS, HC_EPS, MAGNITUDE)
    row_y, row_post = torch.ops.hpc.fuse_ihc_pre(*args, splitk_mode=2)
    sk_y, sk_post = torch.ops.hpc.fuse_ihc_pre(*args, splitk_mode=1)

    torch.testing.assert_close(sk_y.float(), row_y.float(), atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(sk_post, row_post, atol=2e-5, rtol=1e-4)

    if num_batch in (1, 512):
        expected_y, expected_post = (sk_y, sk_post) if num_batch == 1 else (row_y, row_post)
        auto_y, auto_post = hpc.fuse_ihc_pre(*args)
        torch.testing.assert_close(auto_y, expected_y, atol=0, rtol=0)
        torch.testing.assert_close(auto_post, expected_post, atol=0, rtol=0)


@pytest.mark.parametrize("num_batch", [1, 7, 64, 512])
@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
def test_fuse_ihc_pre_splitk_force_with_rms_weight(num_batch, hidden_dim):
    """The folded-RMSNorm split-K variant must match the row kernel too."""
    x, w, hc_scale, hc_base = make_pre_inputs(num_batch, hidden_dim)
    rms_weight = torch.rand((hidden_dim,), dtype=torch.float32, device="cuda").to(
        dtype=torch.bfloat16
    )

    args = (x, w, hc_scale, hc_base, RMS_EPS, HC_EPS, MAGNITUDE, rms_weight, RMS_EPS, True)
    row_y, row_post = torch.ops.hpc.fuse_ihc_pre(*args, splitk_mode=2)
    sk_y, sk_post = torch.ops.hpc.fuse_ihc_pre(*args, splitk_mode=1)

    torch.testing.assert_close(sk_y.float(), row_y.float(), atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(sk_post, row_post, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("num_batch", [1, 7, 64, 512])
@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
def test_fuse_ihc_head_splitk_force_matches_row(num_batch, hidden_dim):
    x, w, hc_scale, hc_base = make_head_inputs(num_batch, hidden_dim)

    args = (x, w, hc_scale, hc_base, RMS_EPS, HC_EPS)
    row = torch.ops.hpc.fuse_ihc_head(*args, splitk_mode=2)
    sk = torch.ops.hpc.fuse_ihc_head(*args, splitk_mode=1)

    torch.testing.assert_close(sk.float(), row.float(), atol=2e-2, rtol=1e-2)

    if num_batch in (1, 512):
        torch.testing.assert_close(
            hpc.fuse_ihc_head(*args), sk if num_batch == 1 else row, atol=0, rtol=0
        )


@pytest.mark.parametrize("num_batch", [1, 7, 64])
@pytest.mark.parametrize("hidden_dim", [6144, 4096, 3072])
def test_fuse_ihc_head_splitk_force_with_rms_weight(num_batch, hidden_dim):
    x, w, hc_scale, hc_base = make_head_inputs(num_batch, hidden_dim)
    rms_weight = torch.rand((hidden_dim,), dtype=torch.float32, device="cuda").to(
        dtype=torch.bfloat16
    )

    args = (x, w, hc_scale, hc_base, RMS_EPS, HC_EPS, rms_weight, RMS_EPS)
    row = torch.ops.hpc.fuse_ihc_head(*args, splitk_mode=2)
    sk = torch.ops.hpc.fuse_ihc_head(*args, splitk_mode=1)

    torch.testing.assert_close(sk.float(), row.float(), atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("op", ["pre", "head"])
def test_fuse_ihc_splitk_invalid_mode(op):
    if op == "pre":
        fn = torch.ops.hpc.fuse_ihc_pre
        args = (*make_pre_inputs(1, 4096), RMS_EPS, HC_EPS, MAGNITUDE)
    else:
        fn = torch.ops.hpc.fuse_ihc_head
        args = (*make_head_inputs(1, 4096), RMS_EPS, HC_EPS)
    with pytest.raises(RuntimeError, match="splitk_mode must be"):
        fn(*args, splitk_mode=3)
