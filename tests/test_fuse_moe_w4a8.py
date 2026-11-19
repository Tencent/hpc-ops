# Copyright (C) 2026 Tencent.

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

from hpc.w4a8 import prepare_w4a8_weights

requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="The native W4A8 MoE is implemented only for SM90",
)


@pytest.fixture(scope="module")
def moe_case():
    """Keep two independent raw MXFP4 sets, without a Humming dependency."""
    generator = torch.Generator(device="cuda").manual_seed(1701)
    raw, prepared = {}, []
    for name, n, k in (("w13", 1024, 2048), ("w2", 2048, 512)):
        weight = torch.randint(
            0,
            256,
            (256, n, k // 2),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )
        scale = torch.randint(
            120,
            123,
            (256, n, k // 32),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )
        raw[name] = weight, scale
        prepared.extend(prepare_w4a8_weights(weight, scale))
    source = torch.randn(2048, 2048, device="cuda", generator=generator).bfloat16()
    experts = torch.tensor([0, 1, 17, 31, 63, 127, 192, 255], device="cuda", dtype=torch.int32)
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False

    def get(m=4, top_k=8, zero=False):
        x = source[:m].clone()
        if zero:
            x.zero_()
        positions = torch.arange(m, device="cuda")[:, None] + torch.arange(top_k, device="cuda")
        ids = experts[positions % 8].contiguous()
        weights = (positions.float() % 13 * 0.1).softmax(-1).contiguous()
        return (x, *prepared, ids, weights), raw

    yield get
    torch.backends.cuda.matmul.allow_tf32 = original_tf32


def _quant_dequant(values):
    groups = values.float().reshape(values.shape[0], -1, 128)
    scale = (groups.abs().amax(-1, keepdim=True) / 448).clamp_min(1e-30)
    return ((groups / scale).to(torch.float8_e4m3fn).float() * scale).reshape(values.shape)


def _decode_expert(raw, expert):
    packed, scales = raw
    data = packed[expert].int()
    codes = torch.stack((data & 15, data >> 4), dim=-1).reshape(data.shape[0], -1)
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
    )
    return lut[codes] * torch.exp2(scales[expert].float() - 127).repeat_interleave(32, -1)


def _reference(arguments, raw):
    x, *_, ids, weights = arguments
    m, top_k = ids.shape
    flat_ids = ids.flatten()
    active = flat_ids.unique().cpu().tolist()
    qx = _quant_dequant(x)
    gate_up = torch.empty(m * top_k, 1024, device=x.device, dtype=torch.bfloat16)
    for expert in active:
        slots = torch.where(flat_ids == expert)[0]
        gate_up[slots] = (qx[slots // top_k] @ _decode_expert(raw["w13"], expert).T).bfloat16()
    middle = (F.silu(gate_up[:, :512].float()) * gate_up[:, 512:].float()).bfloat16()
    qm = _quant_dequant(middle)
    down = torch.empty(m * top_k, 2048, device=x.device, dtype=torch.bfloat16)
    for expert in active:
        slots = torch.where(flat_ids == expert)[0]
        down[slots] = (qm[slots] @ _decode_expert(raw["w2"], expert).T).bfloat16()
    return (down.reshape(m, top_k, 2048).float() * weights[..., None]).sum(1).bfloat16()


def _assert_reference(actual, reference):
    assert actual.shape == reference.shape and actual.dtype == torch.bfloat16
    assert bool(torch.isfinite(actual).all())
    error = (actual.float() - reference.float()).norm()
    assert float(error / reference.float().norm().clamp_min(1e-30)) < 0.035


@requires_sm90
@pytest.mark.parametrize(
    "m,top_k", [(1, 1), (1, 8), (4, 8), (16, 8), (17, 2), (129, 8), (1024, 8), (2048, 8)]
)
def test_w4a8_moe_original_mxfp4_reference(moe_case, m, top_k):
    arguments, raw = moe_case(m, top_k)
    reference = _reference(arguments, raw)
    output = torch.full_like(reference, float("nan"))
    actual = hpc.fuse_moe_w4a8(*arguments, output=output)
    assert actual.data_ptr() == output.data_ptr()
    _assert_reference(actual, reference)


@requires_sm90
def test_w4a8_moe_default_allocation_and_zero_input(moe_case):
    arguments, _ = moe_case(4, 8, zero=True)
    actual = hpc.fuse_moe_w4a8(*arguments)
    assert actual.shape == (4, 2048) and actual.dtype == torch.bfloat16
    assert torch.equal(actual, torch.zeros_like(actual))


@requires_sm90
def test_w4a8_moe_full_relative_exponent_range(moe_case):
    arguments, raw = moe_case(1, 1)
    extreme, prepared = {}, []
    for name in ("w13", "w2"):
        weight, scales = raw[name]
        scales = scales.clone()
        scales[0].copy_(
            (torch.arange(scales[0].numel(), device=scales.device) % 12 + 120).reshape_as(scales[0])
        )
        extreme[name] = (weight, scales)
        prepared.extend(prepare_w4a8_weights(weight, scales))
    inputs = (arguments[0], *prepared, torch.zeros_like(arguments[-2]), arguments[-1])
    _assert_reference(hpc.fuse_moe_w4a8(*inputs), _reference(inputs, extreme))


@requires_sm90
@pytest.mark.parametrize("m", [1, 16, 64, 256, 1024])
def test_w4a8_moe_graph_updates_routes_and_reuses_output(moe_case, m):
    arguments, raw = moe_case(m, 8)
    x, *_, ids, weights = arguments
    output = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            hpc.fuse_moe_w4a8(*arguments, output=output)
    torch.cuda.current_stream().wait_stream(stream)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = hpc.fuse_moe_w4a8(*arguments, output=output)
    assert actual.data_ptr() == output.data_ptr()
    distributed = (torch.arange(ids.numel(), device=ids.device).reshape_as(ids) % 256).int()
    concentrated = (
        torch.tensor([255, 255, 17, 0, 17, 1, 1, 255], device=ids.device, dtype=torch.int32)
        if m == 1
        else torch.arange(8, device=ids.device, dtype=torch.int32)
    ).expand_as(ids)
    routes = [distributed, concentrated] * 2
    if m == 1024:
        skewed = distributed.clone()
        donors = (skewed.flatten() >= 128).nonzero().flatten()[:162]
        skewed.flatten()[donors] = torch.arange(18, device=ids.device, dtype=ids.dtype).repeat(9)
        # Uneven expert populations exercise partial M tiles and task-map changes.
        routes = [distributed, concentrated, skewed, concentrated, skewed]
    for route in routes:
        # Change expert histograms and source values inside the captured shapes.
        # Cover repeated slots for M=1 and stale counters/workspaces for large M.
        ids.copy_(route)
        weights.copy_(weights.roll(1, -1))
        x.copy_(x.roll(1, -1))
        reference = _reference(arguments, raw)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _assert_reference(actual, reference)


@requires_sm90
def test_w4a8_moe_empty_and_fake_outputs(moe_case):
    arguments, _ = moe_case(0, 8)
    actual = hpc.fuse_moe_w4a8(*arguments)
    assert actual.shape == (0, 2048) and actual.device == arguments[0].device
    arguments, _ = moe_case(1, 8)
    with FakeTensorMode() as mode:
        fake = tuple(mode.from_tensor(tensor) for tensor in arguments)
        actual = hpc.fuse_moe_w4a8(*fake)
        assert actual.shape == (1, 2048) and actual.dtype == torch.bfloat16
        assert actual.device == fake[0].device
        output = torch.empty_like(actual)
        assert hpc.fuse_moe_w4a8(*fake, output=output) is output


@requires_sm90
@pytest.mark.parametrize(
    "invalid", ["input_dtype", "weight_dtype", "scale_shape", "ids", "weights"]
)
def test_w4a8_moe_rejects_invalid_metadata(moe_case, invalid):
    arguments, _ = moe_case(4, 8)
    arguments = list(arguments)
    if invalid == "input_dtype":
        arguments[0] = arguments[0].float()
    elif invalid == "weight_dtype":
        arguments[1] = arguments[1].view(torch.uint8)
    elif invalid == "scale_shape":
        arguments[3] = arguments[3][:-1]
    elif invalid == "ids":
        arguments[7] = arguments[7].long()
    else:
        arguments[8] = arguments[8].double()
    with pytest.raises(RuntimeError):
        hpc.fuse_moe_w4a8(*arguments)


@requires_sm90
@pytest.mark.parametrize("top_k", [0, 9])
def test_w4a8_moe_rejects_topk_shape(moe_case, top_k):
    arguments, _ = moe_case(4, 8)
    arguments = list(arguments)
    arguments[7] = torch.zeros(4, top_k, dtype=torch.int32, device="cuda")
    arguments[8] = torch.ones(4, top_k, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        hpc.fuse_moe_w4a8(*arguments)


@requires_sm90
def test_w4a8_moe_rejects_alignment_overlap_and_output(moe_case):
    arguments, _ = moe_case(4, 8)
    x = arguments[0]
    storage = torch.empty(x.numel() + 1, dtype=x.dtype, device=x.device)
    unaligned = storage[1:].reshape_as(x)
    with pytest.raises(RuntimeError):
        hpc.fuse_moe_w4a8(unaligned, *arguments[1:])
    for output in (x, unaligned, torch.empty_like(x, dtype=torch.float32)):
        with pytest.raises(RuntimeError):
            hpc.fuse_moe_w4a8(*arguments, output=output)
    cpu_ids = (*arguments[:7], arguments[7].cpu(), arguments[8])
    with pytest.raises(RuntimeError):
        hpc.fuse_moe_w4a8(*cpu_ids)
