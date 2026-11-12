import sys
import os
from pathlib import Path

try:
    sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))
except IndexError:
    pass

import hpc
import torch

import pytest

pytestmark = [
    pytest.mark.skipif(torch.cuda.get_device_capability() != (12, 0), reason="skip on non sm120"),
    pytest.mark.skipif(
        bool(os.getenv("SANITIZER_CHECK")),
        reason="skip sanitizer: racecheck cannot model cp.async.mbarrier.arrive",
    ),
]


FP8_MAX = 448
FP4_MAX = 6.0
FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _ue8m0_scale(amax, value_max):
    """Per-block UE8M0 (power-of-two) dequant scale, stored as float8_e8m0fnu."""
    scale = (amax / value_max).clamp_min(1e-30).log2().ceil().exp2()
    return scale.to(torch.float8_e8m0fnu)


def quantize_mxfp8(w, block=32):
    """MXFP8 quantization along the last dim: fp8-e4m3 elements + UE8M0 scale."""
    shape = w.shape
    blocks = w.reshape(*shape[:-1], shape[-1] // block, block)
    scale = _ue8m0_scale(blocks.abs().amax(dim=-1, keepdim=True), FP8_MAX)
    w_fp8 = (blocks / scale.to(torch.float32)).reshape(shape).to(torch.float8_e4m3fn)
    return w_fp8, scale.squeeze(-1)


def dequantize_mxfp8(w_fp8, scale, block=32):
    """Inverse of quantize_mxfp8 -> fp32 tensor of w_fp8's shape."""
    shape = w_fp8.shape
    blocks = w_fp8.to(torch.float32).reshape(*shape[:-1], shape[-1] // block, block)
    return (blocks * scale.to(torch.float32).unsqueeze(-1)).reshape(shape)


def _fp4_encode(x):
    levels = FP4_LEVELS.to(x.device)
    bounds = (levels[1:] + levels[:-1]) / 2
    mag_idx = torch.bucketize(x.abs(), bounds).to(torch.uint8)
    sign_bit = (x < 0).to(torch.uint8)
    return (sign_bit << 3) | mag_idx


def _fp4_decode(code):
    levels = FP4_LEVELS.to(code.device)
    mag = levels[(code & 0x7).long()]
    sign = 1.0 - 2.0 * ((code >> 3) & 0x1).to(torch.float32)
    return sign * mag


def quantize_mxfp4(w, block=32):
    """MXFP4 quantization along the last dim, packed two values per uint8."""
    shape = w.shape
    blocks = w.reshape(*shape[:-1], shape[-1] // block, block)
    scale = _ue8m0_scale(blocks.abs().amax(dim=-1, keepdim=True), FP4_MAX)
    codes = _fp4_encode((blocks / scale.to(torch.float32)).reshape(shape))
    lo, hi = codes[..., 0::2], codes[..., 1::2]
    packed = (hi << 4) | lo
    return packed, scale.squeeze(-1)


def dequantize_mxfp4(packed, scale, block=32):
    """Inverse of quantize_mxfp4 -> fp32 tensor with last dim K = 2 * packed_K."""
    lo = packed & 0x0F
    hi = packed >> 4
    codes = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], -1)
    values = _fp4_decode(codes)
    shape = values.shape
    blocks = values.reshape(*shape[:-1], shape[-1] // block, block)
    return (blocks * scale.to(torch.float32).unsqueeze(-1)).reshape(shape)


def _quantize(w, fp4=False):
    """quantize_mxfp8 / quantize_mxfp4 with the scale viewed as the uint8 UE8M0 bytes the op
    takes. mxfp4 comes back packed two per uint8, the even index in the low 4 bits."""
    q, sf = (quantize_mxfp4 if fp4 else quantize_mxfp8)(w.float())
    return q, sf.view(torch.uint8)


def _dequantize_w(q, sf):
    """Weights are packed mxfp4 when uint8, mxfp8 otherwise."""
    dequantize = dequantize_mxfp4 if q.dtype == torch.uint8 else dequantize_mxfp8
    return dequantize(q, sf.view(torch.float8_e8m0fnu))


def to_layout_128x4(sf):
    """(E, rows, k/32) row-major -> flashinfer layout_128x4, laid out per expert.

    Every 128 rows x 4 scales form one 512B block, blocks run (rows/128, k/128) with the row block
    outermost, and within a block the offset is (r % 32) * 16 + (r % 128 / 32) * 4 + c % 4. The
    shape is unchanged, only the contents move.
    """
    e, r, c = sf.shape
    return (
        sf.reshape(e, r // 128, 4, 32, c // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(e, r, c)
    )


def interleave_gate_up(w):
    """(E, 2*inter, K) [gate; up] -> 16-row interleaved [g16, u16, g16, u16, ...]."""
    e, n, k = w.shape
    gate, up = w[:, : n // 2].reshape(e, -1, 16, k), w[:, n // 2 :].reshape(e, -1, 16, k)
    return torch.stack([gate, up], dim=2).reshape(e, n, k).contiguous()


def make_inputs(
    num_seq, hidden, inter, num_expert_local, num_expert_total, seed=0, num_topk=8, w_fp4=False
):
    torch.manual_seed(seed)
    x = _quantize(torch.randn((num_seq, hidden), device="cuda"))
    w_gu = _quantize(
        torch.randn((num_expert_local, 2 * inter, hidden), device="cuda") * 0.05, fp4=w_fp4
    )
    w_dn = _quantize(
        torch.randn((num_expert_local, hidden, inter), device="cuda") * 0.05, fp4=w_fp4
    )
    logits = torch.rand((num_seq, num_expert_total), device="cuda")
    topk_ids = logits.argsort(dim=-1, descending=True)[:, :num_topk].to(torch.int32).contiguous()
    topk_scale = torch.softmax(torch.rand((num_seq, num_topk), device="cuda"), dim=-1)
    return x, w_gu, w_dn, topk_ids, topk_scale


def torch_ref(x, w_gu, w_dn, topk_ids, topk_scale, shared_output, rank_ep, limit):
    """fp32 reference: gate/up GEMM -> clamp -> silu(gate) * up -> mxfp8 requant -> down GEMM ->
    accumulate by topk_scale."""
    num_expert_local, inter = w_gu[0].shape[0], w_gu[0].shape[1] // 2
    x_f = dequantize_mxfp8(x[0], x[1].view(torch.float8_e8m0fnu))
    y = torch.zeros((x_f.shape[0], w_dn[0].shape[1]), device="cuda")
    for le in range(num_expert_local):
        tok, slot = torch.nonzero(topk_ids == rank_ep * num_expert_local + le, as_tuple=True)
        if tok.numel() == 0:
            continue
        w = _dequantize_w(w_gu[0][le], w_gu[1][le])
        h = x_f[tok] @ w.t()
        # Matches the kernel: the gate is clamped from above only, the up projection both sides.
        gate = h[:, :inter].clamp(max=limit)
        up = h[:, inter:].clamp(min=-limit, max=limit)
        act = dequantize_mxfp8(*quantize_mxfp8(torch.nn.functional.silu(gate) * up))
        w_d = _dequantize_w(w_dn[0][le], w_dn[1][le])
        y.index_add_(0, tok, act @ w_d.t() * topk_scale[tok, slot, None])
    return y if shared_output is None else y + shared_output.float()


def run_op(
    x,
    w_gu,
    w_dn,
    topk_ids,
    topk_scale,
    shared_output,
    rank_ep,
    num_expert_total,
    limit,
):
    shared = None if shared_output is None else shared_output.clone()
    # The weight scales are interleaved with the weights first, then moved into layout_128x4.
    y = hpc.fuse_moe_mxfp8(
        x[0],
        x[1],
        interleave_gate_up(w_gu[0]),
        to_layout_128x4(interleave_gate_up(w_gu[1])),
        w_dn[0],
        to_layout_128x4(w_dn[1]),
        topk_ids,
        topk_scale,
        rank_ep,
        num_expert_total,
        limit=limit,
        shared_output=shared,
        output=shared,
    )
    if shared is not None:
        assert y.data_ptr() == shared.data_ptr(), "the epilogue must accumulate into shared_output"
    return y


def _op_weights(w_gu, w_dn):
    """The weight scales are interleaved with the weights first, then moved into layout_128x4."""
    return (
        interleave_gate_up(w_gu[0]),
        to_layout_128x4(interleave_gate_up(w_gu[1])),
        w_dn[0],
        to_layout_128x4(w_dn[1]),
    )


def _shifted(t, offset_bytes):
    """A contiguous copy of t whose storage starts offset_bytes past an aligned address."""
    pad = offset_bytes // t.element_size()
    buf = torch.empty(t.numel() + pad, dtype=t.dtype, device=t.device)
    view = buf[pad:].view(t.shape)
    view.copy_(t)
    return view


@pytest.mark.parametrize("w_fp4", [False, True])
@pytest.mark.parametrize(
    "num_seq,hidden,inter,num_expert_local,num_expert_total,rank_ep,with_shared,num_topk",
    [
        (17, 1024, 384, 16, 16, 0, False, 8),  # tile_m 32, m tile tail
        (90, 1024, 384, 16, 16, 0, False, 8),  # tile_m 64
        (256, 1024, 384, 16, 16, 0, True, 8),  # tile_m 64, with shared_output
        (256, 1024, 384, 8, 16, 1, False, 8),  # rank_ep 1: half the routes are filtered out
        (64, 7168, 2048, 32, 256, 0, False, 8),  # tile_m 32, target shape
        (48, 1024, 384, 32, 64, 0, False, 16),  # tile_m 32
        (300, 1024, 384, 16, 16, 0, False, 6),  # tile_m 128
        (200, 2560, 640, 32, 32, 0, True, 10),
        (33, 1024, 384, 8, 8, 0, False, 1),
        (200, 512, 256, 320, 640, 1, False, 8),  # more local experts than routing threads
        (100, 512, 256, 512, 512, 0, False, 8),
    ],
)
def test_fuse_moe_mxfp8(
    num_seq,
    hidden,
    inter,
    num_expert_local,
    num_expert_total,
    rank_ep,
    with_shared,
    num_topk,
    w_fp4,
):
    _check(
        num_seq,
        hidden,
        inter,
        num_expert_local,
        num_expert_total,
        rank_ep,
        with_shared,
        num_topk,
        float("inf"),
        w_fp4,
    )


# A finite limit clamps gate/up before the SwiGLU; 0.5 is small enough that most rows hit it.
@pytest.mark.parametrize("w_fp4", [False, True])
@pytest.mark.parametrize("limit", [0.5, 2.0])
def test_fuse_moe_mxfp8_swiglu_limit(limit, w_fp4):
    _check(256, 1024, 384, 16, 16, 0, False, 8, limit, w_fp4)


def _make_shared(kind, num_seq, hidden):
    if kind is None:
        return None
    values = torch.randn((num_seq, hidden), device="cuda").to(torch.bfloat16)
    if kind == "misaligned":
        return _shifted(values, 2)
    if kind == "strided":
        return values.t().contiguous().t()
    return values


@pytest.mark.parametrize("num_seq", [37, 4096])
@pytest.mark.parametrize("shared_kind", [None, "aligned", "misaligned", "strided"])
@pytest.mark.parametrize("with_output", [True, False])
def test_fuse_moe_mxfp8_output(num_seq, shared_kind, with_output):
    x, w_gu, w_dn, topk_ids, topk_scale = make_inputs(num_seq, 1024, 384, 16, 16)
    shared = _make_shared(shared_kind, num_seq, 1024)
    args = (x[0], x[1], *_op_weights(w_gu, w_dn), topk_ids, topk_scale, 0, 16)
    expected = torch_ref(x, w_gu, w_dn, topk_ids, topk_scale, shared, 0, float("inf"))
    # A stale output catches a missing init.
    output = (
        torch.full((num_seq, 1024), 777.0, dtype=torch.bfloat16, device="cuda")
        if with_output
        else None
    )
    shared_before = None if shared is None else shared.clone()
    y = hpc.fuse_moe_mxfp8(*args, shared_output=shared, output=output)
    if output is not None:
        assert y.data_ptr() == output.data_ptr()
    if shared is not None:
        assert torch.equal(shared, shared_before), "shared_output must only be read"
    torch.testing.assert_close(
        y.float(), expected, rtol=0.02, atol=0.02 * expected.abs().max().item()
    )


# A TP shard padded with zero weights gives all-zero quantization groups, which must not be NaN.
@pytest.mark.parametrize("w_fp4", [False, True])
@pytest.mark.parametrize("real", [128, 0])
def test_fuse_moe_mxfp8_zero_padded_intermediate(real, w_fp4):
    num_seq, hidden, inter, num_expert = 64, 1024, 256, 16
    x, _, _, topk_ids, topk_scale = make_inputs(num_seq, hidden, inter, num_expert, num_expert)
    gu = torch.randn((num_expert, 2 * inter, hidden), device="cuda") * 0.05
    gu[:, real:inter] = 0
    gu[:, inter + real :] = 0
    dn = torch.randn((num_expert, hidden, inter), device="cuda") * 0.05
    dn[:, :, real:] = 0
    w_gu, w_dn = _quantize(gu, fp4=w_fp4), _quantize(dn, fp4=w_fp4)
    y = hpc.fuse_moe_mxfp8(
        x[0], x[1], *_op_weights(w_gu, w_dn), topk_ids, topk_scale, 0, num_expert
    )
    expected = torch_ref(x, w_gu, w_dn, topk_ids, topk_scale, None, 0, float("inf"))
    assert torch.isfinite(y).all()
    torch.testing.assert_close(
        y.float(), expected, rtol=0.02, atol=0.02 * max(expected.abs().max().item(), 1e-6)
    )


@pytest.mark.parametrize(
    "name,offset", [("x", 1), ("x_scale", 1), ("w_gu", 1), ("w_down", 1), ("output", 2)]
)
def test_fuse_moe_mxfp8_misaligned_rejected(name, offset):
    x, w_gu, w_dn, topk_ids, topk_scale = make_inputs(37, 1024, 384, 16, 16)
    tensors = dict(zip(("x", "x_scale"), x))
    tensors["w_gu"], gu_scale, tensors["w_down"], down_scale = _op_weights(w_gu, w_dn)
    tensors["output"] = torch.empty((37, 1024), dtype=torch.bfloat16, device="cuda")
    tensors[name] = _shifted(tensors[name], offset)
    with pytest.raises(RuntimeError, match="aligned"):
        hpc.fuse_moe_mxfp8(
            tensors["x"],
            tensors["x_scale"],
            tensors["w_gu"],
            gu_scale,
            tensors["w_down"],
            down_scale,
            topk_ids,
            topk_scale,
            0,
            16,
            output=tensors["output"],
        )


def test_fuse_moe_mxfp8_opcheck():
    x, w_gu, w_dn, topk_ids, topk_scale = make_inputs(37, 1024, 384, 16, 16)
    shared = torch.randn((37, 1024), device="cuda").to(torch.bfloat16)
    args = (x[0], x[1], *_op_weights(w_gu, w_dn), topk_ids, topk_scale, shared, 0, 16, 2.0, None)
    # test_schema is skipped: the bf16 atomic sum differs between runs in its last bits.
    torch.library.opcheck(
        torch.ops.hpc.fuse_moe_mxfp8.default,
        args,
        test_utils=("test_autograd_registration", "test_faketensor"),
    )


def _check(
    num_seq,
    hidden,
    inter,
    num_expert_local,
    num_expert_total,
    rank_ep,
    with_shared,
    num_topk,
    limit,
    w_fp4=False,
):
    inputs = make_inputs(
        num_seq, hidden, inter, num_expert_local, num_expert_total, num_topk=num_topk, w_fp4=w_fp4
    )
    shared = (
        torch.randn((num_seq, hidden), device="cuda").to(torch.bfloat16) if with_shared else None
    )
    ref = torch_ref(*inputs, shared, rank_ep, limit)
    ours = run_op(*inputs, shared, rank_ep, num_expert_total, limit).float()
    rel = ((ours - ref).abs().sum() / ref.abs().sum()).item()
    print(
        f"\n[seq={num_seq} h={hidden} i={inter} e={num_expert_local}/{num_expert_total} "
        f"topk={num_topk} limit={limit} w_fp4={w_fp4}] mean_rel_err={rel:.3e}"
    )
    torch.testing.assert_close(ours, ref, rtol=0.02, atol=0.02 * ref.abs().max().item())
    assert rel < 1e-2
