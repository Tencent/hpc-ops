"""Hunyuan-v3 TP8 per-tensor FP8 decode tests with an independent reference."""

import inspect
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

_builds = sorted((Path(__file__).resolve().parents[1] / "build").glob("lib.*"))
if _builds:
    sys.path.insert(0, str(_builds[0]))

import hpc

HIDDEN, INTER, EXPERTS, TOPK = 4096, 192, 192, 8
FP8 = torch.float8_e4m3fn
pytestmark = pytest.mark.skipif(hpc.__loaded_arch__ != 90, reason="SM90 decode backend")


@pytest.fixture(scope="module", autouse=True)
def fp32_reference_math():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def _weights(seed, local_experts=EXPERTS):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return dict(
        seed=seed,
        gate_up_weight=torch.randn(
            local_experts, 2 * INTER, HIDDEN, device="cuda", generator=generator
        ).to(FP8),
        down_weight=torch.randn(
            local_experts, HIDDEN, INTER, device="cuda", generator=generator
        ).to(FP8),
        gate_up_scale=torch.rand(local_experts, device="cuda", generator=generator) + 0.5,
        down_scale=torch.rand(local_experts, device="cuda", generator=generator) * 0.01 + 0.01,
        act_and_mul_scale=torch.tensor([16.0], device="cuda"),
    )


@pytest.fixture(scope="module", params=[20261028, 20261029], ids=lambda seed: f"seed{seed}")
def weights(request):
    return _weights(request.param)


@pytest.fixture(scope="module")
def small_weights():
    return _weights(20261028)


def _inputs(weights, batch, use_bf16_mul=True, rank_ep=0):
    generator = torch.Generator(device="cuda").manual_seed(weights["seed"] + 100 + batch)
    ids = torch.rand(batch, EXPERTS, device="cuda", generator=generator)
    return dict(
        x=(torch.randn(batch, HIDDEN, device="cuda", generator=generator) * 0.025).to(FP8),
        gate_up_weight=weights["gate_up_weight"],
        down_weight=weights["down_weight"],
        gate_up_scale=weights["gate_up_scale"].clone(),
        down_scale=weights["down_scale"].clone(),
        act_and_mul_scale=weights["act_and_mul_scale"].clone(),
        topk_ids=ids.topk(TOPK, dim=-1).indices.int().contiguous(),
        topk_scale=torch.rand(batch, TOPK, device="cuda", generator=generator).softmax(-1),
        rank_ep=rank_ep,
        num_expert_total=EXPERTS,
        use_bf16_mul=use_bf16_mul,
    )


def _run(args, **updates):
    return hpc.fuse_moe(**(args | updates))


def _profile(args):
    _run(args)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        actual = _run(args)
        torch.cuda.synchronize()
    kernels = [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert kernels, "CUDA profiler did not record any kernels"
    return actual, kernels


def _bits(tensor):
    return tensor.contiguous().view(torch.uint8)


def _reference(args):
    """Reproduce CUDA rounding boundaries, independently of the decode code."""
    x = args["x"].float()
    ids, route = args["topk_ids"], args["topk_scale"]
    gate_up, down = args["gate_up_weight"], args["down_weight"]
    local_experts = gate_up.shape[0]
    local_ids = ids.long() - args["rank_ep"] * local_experts
    result = torch.zeros_like(x)
    for slot in range(TOPK):
        tokens = (
            ((local_ids[:, slot] >= 0) & (local_ids[:, slot] < local_experts)).nonzero().flatten()
        )
        if not tokens.numel():
            continue
        selected = local_ids[tokens, slot]
        projection = torch.bmm(gate_up[selected].float(), x[tokens, :, None]).squeeze(-1)
        projection = (projection * args["gate_up_scale"][selected, None]).bfloat16().float()
        gate, up = projection.chunk(2, dim=-1)
        silu = F.silu(gate)
        if args["use_bf16_mul"]:
            activation = (silu.bfloat16() * up.bfloat16()).float()
        else:
            activation = silu * up
        # CUDA FP8 conversion uses finite saturation, including negative scales.
        activation = (activation * args["act_and_mul_scale"].reshape(())).clamp(-448, 448).to(FP8)
        contribution = torch.bmm(down[selected].float(), activation.float()[..., None]).squeeze(-1)
        contribution = (contribution * args["down_scale"][selected, None]).bfloat16().float()
        result[tokens] += contribution * route[tokens, slot, None]
    if args.get("shared_output") is not None:
        result += args["shared_output"].float()
    # Keep FP32 here so the error also covers the final BF16 output conversion.
    return result


def _check(actual, reference):
    assert actual.dtype == torch.bfloat16
    assert actual.shape == reference.shape and actual.device == reference.device
    assert torch.isfinite(actual).all()
    if not actual.numel():
        return dict(nrmse=0.0, max_abs=0.0, reference_rms=0.0)
    error = actual.float() - reference.float()
    rms = reference.float().square().mean().sqrt().item()
    nrmse = error.square().mean().sqrt().item() / max(rms, 1e-20)
    maximum = error.abs().max().item()
    assert nrmse < 0.006, (nrmse, maximum, rms)
    assert maximum < max(0.15 * rms, 1e-6), (nrmse, maximum, rms)
    return dict(nrmse=nrmse, max_abs=maximum, reference_rms=rms)


def _graph_check(args):
    fn = lambda: _run(args)
    expected = fn().clone()
    for _ in range(10):
        assert torch.equal(_bits(fn()), _bits(expected))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = fn()
    torch.cuda.synchronize()
    errors = []
    for _ in range(3):
        args["x"].copy_((args["x"].float() * -1.03125).clamp(-448, 448).to(FP8))
        args["gate_up_scale"].mul_(1.03125)
        args["down_scale"].mul_(0.96875)
        args["act_and_mul_scale"].mul_(1.0625)
        args["topk_ids"].copy_(args["topk_ids"].roll(1, -1))
        args["topk_scale"].copy_(args["topk_scale"].roll(1, -1) * 0.9375)
        expected = fn()
        graph.replay()
        assert torch.equal(_bits(captured), _bits(expected))
        errors.append(_check(captured, _reference(args)))
    return errors


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("use_bf16_mul", [False, True])
def test_default_selects_decode(small_weights, batch, use_bf16_mul):
    actual, kernels = _profile(_inputs(small_weights, batch, use_bf16_mul))
    assert actual.shape == (batch, HIDDEN)
    assert any(
        "warp_decode" in name and "gate_up_pertensor_kernel" in name for name in kernels
    ), kernels
    assert any(
        "warp_decode" in name and "down_pertensor_kernel" in name for name in kernels
    ), kernels


@pytest.mark.parametrize("batch", range(1, 9))
@pytest.mark.parametrize("use_bf16_mul", [False, True])
def test_hunyuan_pertensor_shapes(weights, batch, use_bf16_mul, record_property):
    args = _inputs(weights, batch, use_bf16_mul)
    errors = [_check(_run(args), _reference(args))]
    errors += _graph_check(args)
    record_property("precision_metrics", json.dumps(errors))
    record_property("max_nrmse", max(error["nrmse"] for error in errors))
    record_property("max_abs", max(error["max_abs"] for error in errors))


@pytest.mark.parametrize("batch", [2, 7, 8])
@pytest.mark.parametrize("use_bf16_mul", [False, True])
def test_output_and_shared_output(small_weights, batch, use_bf16_mul):
    args = _inputs(small_weights, batch, use_bf16_mul)
    shared = torch.randn(batch, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.01
    output = torch.empty_like(shared)
    args.update(shared_output=shared, output=output)
    reference = _reference(args)
    actual = _run(args)
    assert actual.data_ptr() == output.data_ptr()
    _check(actual, reference)
    # An exact shared/output alias is safe; compute the reference before mutation.
    args["output"] = shared
    reference = _reference(args)
    actual = _run(args)
    assert actual.data_ptr() == shared.data_ptr()
    _check(actual, reference)


@pytest.mark.parametrize("batch", [3, 7, 8])
@pytest.mark.parametrize("rank_ep,local_experts", [(0, 192), (1, 48), (7, 24)])
@pytest.mark.parametrize("all_nonlocal", [False, True])
def test_expert_parallel_and_invalid_routes(batch, rank_ep, local_experts, all_nonlocal):
    weights = _weights(20261028, local_experts)
    args = _inputs(weights, batch, rank_ep=rank_ep)
    start = rank_ep * local_experts
    args["topk_ids"][:] = torch.tensor(
        [start, start + local_experts - 1, 0, 191, -1, -(2**31), 2**31 - 1, start],
        dtype=torch.int32,
        device="cuda",
    )
    if all_nonlocal:
        args["topk_ids"].fill_(-1)
    invalid = (args["topk_ids"] < start) | (args["topk_ids"] >= start + local_experts)
    # Nonlocal slots must be skipped rather than evaluating 0 * NaN.
    args["topk_scale"][invalid] = float("nan")
    shared = torch.randn(batch, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.01
    args["shared_output"] = shared
    actual = _run(args)
    _check(actual, _reference(args))
    if all_nonlocal:
        assert torch.equal(actual, shared)


@pytest.mark.parametrize("use_bf16_mul", [False, True])
def test_padded_scale_tails_ignored(small_weights, use_bf16_mul):
    args = _inputs(small_weights, 2, use_bf16_mul)
    reference = _reference(args)
    expected = _run(args)
    _check(expected, reference)
    for name in ("gate_up_scale", "down_scale", "act_and_mul_scale"):
        args[name] = torch.cat((args[name], args[name].new_full((3,), float("nan"))))
    actual = _run(args)
    assert torch.equal(_bits(actual), _bits(expected))
    _check(actual, reference)


@pytest.mark.parametrize("scale_name", ["gate_up_scale", "down_scale", "act_and_mul_scale"])
@pytest.mark.parametrize("scale_value", [0.0, -1.0])
@pytest.mark.parametrize("use_bf16_mul", [False, True])
def test_zero_and_negative_scales(small_weights, scale_name, scale_value, use_bf16_mul):
    args = _inputs(small_weights, 2, use_bf16_mul)
    args[scale_name].mul_(scale_value)
    actual = _run(args)
    _check(actual, _reference(args))
    if scale_value == 0:
        assert torch.count_nonzero(actual) == 0


@pytest.mark.parametrize("use_bf16_mul", [False, True])
def test_fp8_saturation(small_weights, use_bf16_mul):
    args = _inputs(small_weights, 1, use_bf16_mul)
    args["act_and_mul_scale"].fill_(65536.0)
    _check(_run(args), _reference(args))


@pytest.mark.parametrize("batch", [2, 7, 8])
@pytest.mark.parametrize("use_bf16_mul", [False, True])
def test_large_fp8_values(small_weights, batch, use_bf16_mul):
    args = _inputs(small_weights, batch, use_bf16_mul)
    raw_multiplier = 64.0
    # New tensors preserve the module fixture while covering large FP8 values
    # together with the smaller scales used to dequantize real model weights.
    for name in ("x", "gate_up_weight", "down_weight"):
        args[name] = (args[name].float() * raw_multiplier).clamp(-448, 448).to(FP8)
    args["gate_up_scale"].div_(raw_multiplier * raw_multiplier)
    args["down_scale"].div_(raw_multiplier)
    assert args["gate_up_weight"].data_ptr() != small_weights["gate_up_weight"].data_ptr()
    assert args["down_weight"].data_ptr() != small_weights["down_weight"].data_ptr()
    _check(_run(args), _reference(args))


def test_empty_and_zero_input(small_weights):
    args = _inputs(small_weights, 0)
    output = torch.empty(0, HIDDEN, device="cuda", dtype=torch.bfloat16)
    actual = _run(args, output=output)
    assert actual.shape == output.shape and actual.dtype == output.dtype
    assert actual.data_ptr() == output.data_ptr()
    args = _inputs(small_weights, 2)
    args["x"].view(torch.uint8).zero_()
    assert torch.count_nonzero(_run(args)) == 0
    args = _inputs(small_weights, 2)
    args["topk_scale"].zero_()
    assert torch.count_nonzero(_run(args)) == 0


def test_legacy_alias_and_api_contract(small_weights):
    args = _inputs(small_weights, 1)
    assert torch.equal(_bits(_run(args)), _bits(hpc.fuse_moe_pertensor_fp8(**args)))
    for name in ("fuse_moe", "fuse_moe_pertensor_fp8"):
        assert "backend" not in inspect.signature(getattr(hpc, name)).parameters
        arguments = {
            argument.name for argument in getattr(torch.ops.hpc, name).default._schema.arguments
        }
        assert "backend" not in arguments
        assert "output" in arguments


@pytest.mark.parametrize("fallback", ["batch", "shape", "topk", "scale_layout"])
def test_automatic_fallback(small_weights, fallback):
    args = _inputs(small_weights, 9 if fallback == "batch" else 2)
    if fallback == "shape":
        args["x"] = args["x"][:, :512].contiguous()
        args["gate_up_weight"] = args["gate_up_weight"][:, :, :512].contiguous()
        args["down_weight"] = args["down_weight"][:, :512].contiguous()
    elif fallback == "topk":
        args["topk_ids"] = args["topk_ids"][:, :7].contiguous()
        args["topk_scale"] = args["topk_scale"][:, :7].contiguous()
    elif fallback == "scale_layout":
        args["gate_up_scale"] = args["gate_up_scale"].view(EXPERTS, 1)
        args["down_scale"] = args["down_scale"].view(EXPERTS, 1)
    # Keep fallback assertions independent of the general path's rounding.
    args["x"].view(torch.uint8).zero_()
    shared = torch.randn_like(args["x"], dtype=torch.bfloat16) * 0.01
    output = torch.empty_like(shared)
    args.update(shared_output=shared, output=output)
    actual, kernels = _profile(args)
    assert not any("warp_decode" in name for name in kernels), kernels
    assert any("group_gemm_cp_async" in name for name in kernels), kernels
    assert actual.data_ptr() == output.data_ptr()
    assert torch.equal(actual, shared)


@pytest.mark.parametrize(
    "invalid",
    [
        "input_dtype",
        "scale_dtype",
        "gu_scale_shape",
        "down_scale_shape",
        "act_scale_shape",
        "ids_dtype",
        "route_dtype",
        "route_shape",
        "rank",
        "total_experts",
        "input_layout",
        "input_alignment",
        "output_dtype",
        "output_layout",
        "weight_shape",
    ],
)
def test_rejected_inputs(small_weights, invalid):
    args = _inputs(small_weights, 2)
    if invalid == "input_dtype":
        args["x"] = args["x"].bfloat16()
    elif invalid == "scale_dtype":
        args["gate_up_scale"] = args["gate_up_scale"].bfloat16()
    elif invalid == "gu_scale_shape":
        args["gate_up_scale"] = args["gate_up_scale"][:-1]
    elif invalid == "down_scale_shape":
        args["down_scale"] = args["down_scale"][:-1]
    elif invalid == "act_scale_shape":
        args["act_and_mul_scale"] = args["act_and_mul_scale"][:0]
    elif invalid == "ids_dtype":
        args["topk_ids"] = args["topk_ids"].long()
    elif invalid == "route_dtype":
        args["topk_scale"] = args["topk_scale"].bfloat16()
    elif invalid == "route_shape":
        args["topk_scale"] = args["topk_scale"][:, :-1].contiguous()
    elif invalid == "rank":
        args["rank_ep"] = -1
    elif invalid == "total_experts":
        args["num_expert_total"] = 0
    elif invalid == "input_layout":
        args["x"] = args["x"].t().contiguous().t()
    elif invalid == "input_alignment":
        args["x"] = torch.empty(2 * HIDDEN + 1, device="cuda", dtype=FP8)[1:].view(2, HIDDEN)
    elif invalid == "output_dtype":
        args["output"] = torch.empty(2, HIDDEN, device="cuda")
    elif invalid == "output_layout":
        args["output"] = torch.empty(HIDDEN, 2, device="cuda", dtype=torch.bfloat16).t()
    elif invalid == "weight_shape":
        args["down_weight"] = args["down_weight"][:, :, :-16].contiguous()
    with pytest.raises(RuntimeError):
        _run(args)


def test_partial_output_overlap_rejected(small_weights):
    args = _inputs(small_weights, 2)
    storage = torch.empty(2 * HIDDEN + 1, device="cuda", dtype=torch.bfloat16)
    args["shared_output"] = storage[:-1].view(2, HIDDEN)
    args["output"] = storage[1:].view(2, HIDDEN)
    with pytest.raises(RuntimeError):
        _run(args)


def test_fake_tensor_and_compile(small_weights):
    from torch._subclasses.fake_tensor import FakeTensorMode

    args = _inputs(small_weights, 2)
    with FakeTensorMode() as mode:
        fake_args = {
            name: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
            for name, value in args.items()
        }
        actual = _run(fake_args)
        assert actual.shape == (2, HIDDEN) and actual.dtype == torch.bfloat16
        assert actual.device == fake_args["x"].device
        output = torch.empty(2, HIDDEN, dtype=torch.bfloat16, device=fake_args["x"].device)
        assert _run(fake_args, output=output) is output

    compiled = torch.compile(lambda **kwargs: _run(kwargs), backend="eager", fullgraph=True)
    assert torch.equal(_bits(compiled(**args)), _bits(_run(args)))
