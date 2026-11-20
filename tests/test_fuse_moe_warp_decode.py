"""Independent validation of automatic blockwise FP8 decode dispatch."""

import inspect
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

_builds = sorted((Path(__file__).resolve().parents[1] / "build").glob("lib.*"))
if _builds:
    sys.path.insert(0, str(_builds[0]))

import hpc


GROUP = 128
FP8 = torch.float8_e4m3fn
CASES = [
    ("qwen35_36_35b_tp1", 2048, 512, 256, 8),
    ("qwen35_36_35b_tp2", 2048, 256, 256, 8),
    ("qwen35_122b_tp2", 3072, 512, 256, 8),
    ("qwen35_122b_tp4", 3072, 256, 256, 8),
    ("qwen35_397b_tp4", 4096, 256, 512, 10),
    ("qwen35_397b_tp8", 4096, 128, 512, 10),
    ("qwen3_next_80b_tp1", 2048, 512, 512, 10),
    ("qwen3_next_80b_tp2", 2048, 256, 512, 10),
]
pytestmark = pytest.mark.skipif(hpc.__loaded_arch__ != 90, reason="SM90 decode backend")


@pytest.fixture(scope="module", autouse=True)
def fp32_reference_math():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def _quantize(x):
    blocks = x.float().reshape(x.shape[0], x.shape[1] // GROUP, GROUP)
    # FP64 division independently reproduces rounded FP32 scale/reciprocal.
    scale = (blocks.abs().amax(-1).clamp_min(1e-10).double() / 448).float()
    reciprocal = scale.double().reciprocal().float()
    quantized = (blocks * reciprocal[..., None]).clamp(-448, 448).to(FP8)
    return quantized.reshape(x.shape), scale


def _make_weights(case, local_experts=None):
    _, hidden, inter, total_experts, topk = case
    experts = local_experts or total_experts
    generator = torch.Generator(device="cuda").manual_seed(20261027)
    gate_up = (
        (torch.randn(experts, 2 * inter, hidden, device="cuda", generator=generator) * 64)
        .clamp(-448, 448)
        .to(FP8)
    )
    down = (
        (torch.randn(experts, hidden, inter, device="cuda", generator=generator) * 64)
        .clamp(-448, 448)
        .to(FP8)
    )

    def scales(rows, cols):
        padded_cols = (cols + 3) // 4 * 4
        result = torch.full((experts, rows, padded_cols), float("nan"), device="cuda")
        result[..., :cols] = (
            torch.rand(experts, rows, cols, device="cuda", generator=generator) * 0.0005 + 0.0005
        )
        return result

    return dict(
        hidden=hidden,
        inter=inter,
        total_experts=total_experts,
        topk=topk,
        gate_up_weight=gate_up,
        down_weight=down,
        gate_up_weight_scale=scales(2 * inter // GROUP, hidden // GROUP),
        down_weight_scale=scales(hidden // GROUP, inter // GROUP),
    )


@pytest.fixture(scope="module", params=CASES, ids=[case[0] for case in CASES])
def weights(request):
    return _make_weights(request.param)


@pytest.fixture(scope="module")
def small_weights():
    return _make_weights(CASES[1])


def _inputs(weights, batch, rank_ep=0):
    generator = torch.Generator(device="cuda").manual_seed(41 + batch)
    x = (
        torch.randn(
            batch,
            weights["hidden"],
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    )
    x, x_scale = _quantize(x)
    ids = torch.rand(batch, weights["total_experts"], device="cuda", generator=generator)
    ids = ids.topk(weights["topk"], dim=-1).indices.int().contiguous()
    route = torch.rand(batch, weights["topk"], device="cuda", generator=generator).softmax(-1)
    return dict(
        x=x,
        x_scale=x_scale,
        gate_up_weight=weights["gate_up_weight"],
        gate_up_weight_scale=weights["gate_up_weight_scale"],
        down_weight=weights["down_weight"],
        down_weight_scale=weights["down_weight_scale"],
        topk_ids=ids,
        topk_scale=route,
        rank_ep=rank_ep,
        num_expert_total=weights["total_experts"],
    )


def _run(args, **updates):
    return hpc.fuse_moe_blockwise(**(args | updates))


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


def _dequant_weight(weight, scales):
    cols = weight.shape[-1] // GROUP
    scales = scales[..., :cols]  # The padded tail deliberately contains NaNs.
    return weight.float() * scales.repeat_interleave(GROUP, 1).repeat_interleave(GROUP, 2)


def _reference(args):
    x = args["x"].float() * args["x_scale"].repeat_interleave(GROUP, -1)
    ids, route = args["topk_ids"], args["topk_scale"]
    gate_up, down = args["gate_up_weight"], args["down_weight"]
    experts = gate_up.shape[0]
    local_ids = ids.long() - args["rank_ep"] * experts
    result = torch.zeros_like(x)
    for slot in range(ids.shape[1]):
        tokens = ((local_ids[:, slot] >= 0) & (local_ids[:, slot] < experts)).nonzero().flatten()
        if not tokens.numel():
            continue
        selected = local_ids[tokens, slot]
        weight = _dequant_weight(gate_up[selected], args["gate_up_weight_scale"][selected])
        gate, up = torch.bmm(weight, x[tokens, :, None]).squeeze(-1).bfloat16().float().chunk(2, -1)
        activation = (F.silu(gate) * up).bfloat16()
        quantized, scale = _quantize(activation)
        activation = quantized.float() * scale.repeat_interleave(GROUP, -1)
        weight = _dequant_weight(down[selected], args["down_weight_scale"][selected])
        contribution = torch.bmm(weight, activation[..., None]).squeeze(-1)
        result[tokens] += contribution * route[tokens, slot, None]
    if args.get("shared_output") is not None:
        result += args["shared_output"].float()
    return result


def _check(actual, reference):
    assert actual.dtype == torch.bfloat16
    assert actual.shape == reference.shape and actual.device == reference.device
    assert torch.isfinite(actual).all()
    if not actual.numel():
        return
    error = actual.float() - reference.float()
    rms = reference.float().square().mean().sqrt().item()
    nrmse = error.square().mean().sqrt().item() / max(rms, 1e-20)
    maximum = error.abs().max().item()
    assert nrmse < 0.006, (nrmse, maximum, rms)
    assert maximum < max(0.15 * rms, 1e-6), (maximum, rms)


@pytest.mark.parametrize("batch", [1, 8])
def test_default_selects_decode(weights, batch):
    actual, kernels = _profile(_inputs(weights, batch))
    assert actual.shape == (batch, weights["hidden"])
    assert any("warp_decode" in name and "moe_gate_up" in name for name in kernels), kernels
    assert any("warp_decode" in name and "moe_down" in name for name in kernels), kernels


@pytest.mark.parametrize("batch", range(1, 9))
def test_decode_shapes(weights, batch):
    args = _inputs(weights, batch)
    eager = _run(args)
    _check(eager, _reference(args))
    for _ in range(2):
        assert torch.equal(eager, _run(args))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _run(args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = _run(args)
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
        assert torch.equal(captured, eager)


def test_weight_scale_expert_padding_ignored(small_weights):
    args = _inputs(small_weights, 2)
    reference = _reference(args)
    expected = _run(args)
    _check(expected, reference)
    for name in ("gate_up_weight_scale", "down_weight_scale"):
        scale = args[name]
        padding = scale.new_full((1, *scale.shape[1:]), float("nan"))
        args[name] = torch.cat((scale, padding), dim=0)
    actual, kernels = _profile(args)
    assert any("warp_decode" in name and "moe_gate_up" in name for name in kernels), kernels
    assert any("warp_decode" in name and "moe_down" in name for name in kernels), kernels
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    _check(actual, reference)


def test_output_and_shared_output(small_weights):
    args = _inputs(small_weights, 2)
    shared = torch.randn(2, small_weights["hidden"], dtype=torch.bfloat16, device="cuda") * 0.01
    output = torch.empty_like(shared)
    args.update(shared_output=shared, output=output)
    reference = _reference(args)
    actual = _run(args)
    assert actual.data_ptr() == output.data_ptr()
    _check(actual, reference)

    # Reading a shared expert result and writing the same storage is supported.
    args["output"] = shared
    reference = _reference(args)
    actual = _run(args)
    assert actual.data_ptr() == shared.data_ptr()
    _check(actual, reference)


@pytest.mark.parametrize("overlap", ["shared_output", "x_scale"])
def test_reject_output_overlap(small_weights, overlap):
    args = _inputs(small_weights, 2)
    hidden = small_weights["hidden"]
    if overlap == "shared_output":
        storage = torch.empty(2 * hidden + 1, device="cuda", dtype=torch.bfloat16)
        args["shared_output"] = storage[:-1].view(2, hidden)
        args["output"] = storage[1:].view(2, hidden)
    else:
        output = torch.empty(2, hidden, device="cuda", dtype=torch.bfloat16)
        args["x_scale"] = (
            output.view(torch.float32).flatten()[: 2 * hidden // GROUP].view(2, hidden // GROUP)
        )
        args["x_scale"].fill_(1)
        args["output"] = output
    with pytest.raises(RuntimeError, match="overlap|memory location|some elements"):
        _run(args)


@pytest.mark.parametrize("batch", [1, 2, 3])
@pytest.mark.parametrize("all_nonlocal", [False, True])
def test_expert_parallel_routing(all_nonlocal, batch):
    weights = _make_weights(CASES[1], local_experts=128)
    args = _inputs(weights, batch, rank_ep=1)
    args["topk_ids"][:] = torch.tensor(
        [128, 255, 0, 127, 256, -1, 129, 128], device="cuda", dtype=torch.int32
    )
    if all_nonlocal:
        args["topk_ids"].fill_(-1)
    shared = torch.randn(batch, weights["hidden"], device="cuda", dtype=torch.bfloat16) * 0.01
    args["shared_output"] = shared
    actual = _run(args)
    _check(actual, _reference(args))
    if all_nonlocal:
        assert torch.equal(actual, shared)


def test_fused_pair_ep_shared_graph():
    weights = _make_weights(CASES[5], local_experts=256)
    args = _inputs(weights, 2, rank_ep=1)
    args["topk_ids"][:] = torch.tensor(
        [256, 511, 256, 0, 255, -1, 512, -(1 << 31), (1 << 31) - 1, 300],
        device="cuda",
        dtype=torch.int32,
    )
    shared = torch.randn(2, weights["hidden"], device="cuda", dtype=torch.bfloat16) * 0.01
    output = torch.empty_like(shared)
    args.update(shared_output=shared, output=output)
    reference = _reference(args)
    actual = _run(args)
    assert actual.data_ptr() == output.data_ptr()
    _check(actual, reference)

    alias_shared = shared.clone()
    alias_args = args | {"shared_output": alias_shared, "output": alias_shared}
    reference = _reference(alias_args | {"shared_output": alias_shared.clone()})
    actual = _run(alias_args)
    assert actual.data_ptr() == alias_shared.data_ptr()
    _check(actual, reference)

    # Capture with a separate output to avoid repeatedly adding the shared value.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _run(args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = _run(args)
    torch.cuda.synchronize()
    assert captured.data_ptr() == output.data_ptr()
    for _ in range(3):
        args["x_scale"].mul_(0.97)
        args["topk_ids"].copy_(args["topk_ids"].roll(1, -1))
        args["topk_scale"].copy_(args["topk_scale"].roll(2, -1))
        args["gate_up_weight_scale"].mul_(1.003)
        args["down_weight_scale"].mul_(0.997)
        # Eager and Graph share preallocated output, so preserve the eager bytes.
        eager = _run(args).clone()
        reference = _reference(args)
        graph.replay()
        assert torch.equal(captured, eager)
        _check(captured, reference)


def test_empty_batch(small_weights):
    args = _inputs(small_weights, 0)
    output = torch.empty(0, small_weights["hidden"], device="cuda", dtype=torch.bfloat16)
    actual = _run(args, output=output)
    assert actual.shape == output.shape and actual.dtype == output.dtype
    assert actual.data_ptr() == output.data_ptr()


def test_zero_input_and_routes(small_weights):
    args = _inputs(small_weights, 2)
    args["topk_scale"].zero_()
    assert torch.count_nonzero(_run(args)) == 0
    args["topk_scale"].fill_(1 / small_weights["topk"])
    args["x"].view(torch.uint8).zero_()
    assert torch.count_nonzero(_run(args)) == 0


def test_legacy_alias_and_api_contract(small_weights):
    args = _inputs(small_weights, 1)
    assert torch.equal(hpc.fuse_moe_blockwise_fp8(**args), _run(args))
    for name in ("fuse_moe_blockwise", "fuse_moe_blockwise_fp8"):
        assert "backend" not in inspect.signature(getattr(hpc, name)).parameters
        arguments = {
            argument.name for argument in getattr(torch.ops.hpc, name).default._schema.arguments
        }
        assert "backend" not in arguments
        assert "output" in arguments


@pytest.mark.parametrize("fallback", ["batch", "shape"])
def test_automatic_fallback(small_weights, fallback):
    weights = small_weights
    if fallback == "shape":
        weights = _make_weights(("general", 512, 256, 32, 8))
    args = _inputs(weights, 9 if fallback == "batch" else 2)
    # Zero projections make shared addition exact for either rounding policy.
    # General-path random-GEMM accuracy remains covered by its own test suite.
    args["x"].view(torch.uint8).zero_()
    shared = torch.randn_like(args["x"], dtype=torch.bfloat16) * 0.01
    output = torch.empty_like(shared)
    args.update(shared_output=shared, output=output)
    actual, kernels = _profile(args)
    assert not any("warp_decode" in name for name in kernels), kernels
    assert any("blockwise_count" in name for name in kernels), kernels
    assert any("group_gemm" in name for name in kernels), kernels
    assert actual.data_ptr() == output.data_ptr()
    assert torch.equal(actual, shared)


def test_reject_invalid_topk(small_weights):
    args = _inputs(small_weights, 1)
    args["topk_ids"] = torch.zeros(1, 129, device="cuda", dtype=torch.int32)
    args["topk_scale"] = torch.ones(1, 129, device="cuda")
    with pytest.raises(RuntimeError, match="topk|top-k"):
        _run(args)


def test_invalid_expert_partition():
    weights = _make_weights(CASES[1], local_experts=127)
    with pytest.raises(RuntimeError, match="partition"):
        _run(_inputs(weights, 1))


@pytest.mark.parametrize("rank_ep", [-1, 1, 1 << 31])
def test_invalid_expert_rank(small_weights, rank_ep):
    with pytest.raises(RuntimeError, match="partition|rank_ep"):
        _run(_inputs(small_weights, 1, rank_ep=rank_ep))


@pytest.mark.parametrize(
    "invalid",
    [
        "x_dtype",
        "x_scale_dtype",
        "weight_scale_dtype",
        "scale_shape",
        "noncontiguous_x",
        "noncontiguous_scale",
        "ids_dtype",
        "route_dtype",
        "output_dtype",
        "noncontiguous_output",
    ],
)
def test_reject_invalid_layouts(small_weights, invalid):
    args = _inputs(small_weights, 2)
    if invalid == "x_dtype":
        args["x"] = args["x"].bfloat16()
    elif invalid == "x_scale_dtype":
        args["x_scale"] = args["x_scale"].half()
    elif invalid == "weight_scale_dtype":
        args["down_weight_scale"] = args["down_weight_scale"].half()
    elif invalid == "scale_shape":
        args["down_weight_scale"] = args["down_weight_scale"][..., :-1].contiguous()
    elif invalid == "noncontiguous_x":
        args["x"] = torch.empty(2, 2 * small_weights["hidden"], device="cuda", dtype=FP8)[:, ::2]
    elif invalid == "noncontiguous_scale":
        args["x_scale"] = torch.empty(2, 2 * args["x_scale"].shape[1], device="cuda")[:, ::2]
    elif invalid == "ids_dtype":
        args["topk_ids"] = args["topk_ids"].long()
    elif invalid == "route_dtype":
        args["topk_scale"] = args["topk_scale"].half()
    elif invalid == "output_dtype":
        args["output"] = torch.empty(2, small_weights["hidden"], device="cuda", dtype=torch.float16)
    elif invalid == "noncontiguous_output":
        args["output"] = torch.empty(
            small_weights["hidden"], 2, device="cuda", dtype=torch.bfloat16
        ).t()
    with pytest.raises(RuntimeError):
        _run(args)


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_graph_updates(small_weights, batch):
    args = _inputs(small_weights, batch)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _run(args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = _run(args)
    torch.cuda.synchronize()
    for _ in range(3):
        args["x"].view(torch.uint8).bitwise_xor_(128)
        args["x_scale"].mul_(0.97)
        args["topk_ids"].add_(17).remainder_(small_weights["total_experts"])
        args["topk_scale"].copy_(args["topk_scale"].roll(1, -1))
        args["gate_up_weight_scale"].mul_(1.003)
        args["down_weight_scale"].mul_(0.997)
        eager = _run(args)
        graph.replay()
        assert torch.equal(captured, eager)
        _check(captured, _reference(args))


def test_fake_tensor(small_weights):
    from torch._subclasses.fake_tensor import FakeTensorMode

    args = _inputs(small_weights, 2)
    with FakeTensorMode() as mode:
        fake_args = {
            name: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
            for name, value in args.items()
        }
        actual = _run(fake_args)
        assert actual.shape == args["x"].shape
        assert actual.dtype == torch.bfloat16 and actual.device == args["x"].device
        output = torch.empty_like(actual)
        assert _run(fake_args, output=output) is output


def test_compile_fullgraph(small_weights):
    args = _inputs(small_weights, 2)

    def forward(x, scale, ids, route):
        return hpc.fuse_moe_blockwise(
            x,
            scale,
            args["gate_up_weight"],
            args["gate_up_weight_scale"],
            args["down_weight"],
            args["down_weight_scale"],
            ids,
            route,
            args["rank_ep"],
            args["num_expert_total"],
        )

    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    actual = compiled(args["x"], args["x_scale"], args["topk_ids"], args["topk_scale"])
    assert torch.equal(actual, _run(args))
