"""Benchmark automatically selected Hunyuan-v3 TP8 per-tensor FP8 warp_decode.

The timed function calls hpc.fuse_moe with already-quantized inputs.
Weights and routing remain fixed during warm CUDA Graph timing. Initial input
quantization, router scoring, weight setup, shared experts and TP communication
are excluded. Run the focused independent-reference tests before benchmarking.

Outputs must pass the FP32-reference NRMSE < 0.006 and max-error checks, the
BF16-reference allclose contract (rtol=0.08, atol=0.1), finite output,
determinism, and eager/Graph byte agreement.
"""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
HIDDEN, INTER, EXPERTS, TOPK = 4096, 192, 192, 8
BACKENDS = ("warp_decode",)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--seeds", default="20261028,20261029")
    parser.add_argument("--use-bf16-mul", choices=("both", "true", "false"), default="both")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iters", type=int, default=100, help="Graph replays per timing sample")
    parser.add_argument("--calls-per-graph", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("warp_decode_hunyuan_pertensor.json"))
    args = parser.parse_args()
    for field in ("batches", "seeds"):
        try:
            values = [int(value) for value in getattr(args, field).split(",")]
        except ValueError:
            parser.error(field + " must be comma-separated integers")
        if not values or len(values) != len(set(values)):
            parser.error(field + " must be nonempty and must not contain duplicates")
        setattr(args, field, values)
    if any(not 1 <= batch <= 8 for batch in args.batches):
        parser.error("batches must be in 1..8")
    if min(args.rounds, args.iters, args.calls_per_graph) < 1:
        parser.error("rounds, iters and calls-per-graph must be positive")
    if args.output.suffix != ".json":
        parser.error("output must use .json; a sibling CSV is also written")
    args.mul_modes = [False, True] if args.use_bf16_mul == "both" else [args.use_bf16_mul == "true"]
    return args


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


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


def _check(actual, reference, backend):
    assert backend in BACKENDS, backend
    assert actual.dtype == torch.bfloat16, (backend, str(actual.dtype))
    assert actual.shape == reference.shape and actual.device == reference.device, backend
    assert torch.isfinite(actual).all(), (backend, "nonfinite output")
    if actual.numel():
        error = actual.float() - reference.float()
        rms = reference.float().square().mean().sqrt().item()
        nrmse = error.square().mean().sqrt().item() / max(rms, 1e-20)
        maximum = error.abs().max().item()
    else:
        rms = nrmse = maximum = 0.0
    max_abs_limit = max(0.15 * rms, 1e-6)
    strict_passed = nrmse < 0.006 and maximum < max_abs_limit
    legacy_passed = torch.allclose(
        actual.float(), reference.bfloat16().float(), rtol=0.08, atol=0.1
    )
    metrics = dict(
        nrmse=nrmse,
        max_abs=maximum,
        reference_rms=rms,
        strict_reference_passed=strict_passed,
        legacy_allclose_passed=legacy_passed,
        criteria=dict(
            strict_reference=dict(
                nrmse_less_than=0.006,
                max_abs_less_than=max_abs_limit,
                reference="FP32 before final BF16 conversion",
            ),
            legacy_allclose=dict(
                rtol=0.08, atol=0.1, reference="independent reference converted to BF16"
            ),
            precision_policy="required",
            required="strict_reference and legacy_allclose",
        ),
    )
    assert legacy_passed, (backend, metrics)
    assert strict_passed, (backend, metrics)
    return metrics


def capture(fn, calls):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            output = fn()
    torch.cuda.synchronize()
    return graph, output


def measure(graph, calls, replays):
    for _ in range(3):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (calls * replays)


def point(args, weights, seed, batch, use_bf16_mul):
    inputs = _inputs(weights, batch, use_bf16_mul)
    tensor_names = [
        "x",
        "gate_up_scale",
        "down_scale",
        "act_and_mul_scale",
        "topk_ids",
        "topk_scale",
    ]
    hashes = {name: tensor_sha256(inputs[name]) for name in tensor_names}
    reference = _reference(inputs)
    functions = {"warp_decode": lambda: hpc.fuse_moe(**inputs)}
    expected, graphs, captured, errors = {}, {}, {}, {}
    for backend, fn in functions.items():
        expected[backend] = fn().clone()
        errors[backend] = _check(expected[backend], reference, backend)
        for _ in range(10):
            assert torch.equal(_bits(expected[backend]), _bits(fn())), (
                backend + " is not deterministic"
            )
        graphs[backend], captured[backend] = capture(fn, args.calls_per_graph)
        for _ in range(3):
            graphs[backend].replay()
            torch.cuda.synchronize()
            assert torch.equal(_bits(captured[backend]), _bits(expected[backend])), (
                backend + " Graph mismatch"
            )

    samples = {backend: [] for backend in BACKENDS}
    for _ in range(args.rounds):
        for backend in BACKENDS:
            samples[backend].append(measure(graphs[backend], args.calls_per_graph, args.iters))

    timed_errors = {}
    for backend in BACKENDS:
        assert torch.equal(_bits(captured[backend]), _bits(expected[backend])), (
            backend + " changed after timing"
        )
        timed_errors[backend] = _check(captured[backend], reference, backend)
    assert hashes == {
        name: tensor_sha256(inputs[name]) for name in tensor_names
    }, "input fixture mutated"
    medians = {backend: statistics.median(values) for backend, values in samples.items()}
    return dict(
        seed=seed,
        batch=batch,
        use_bf16_mul=use_bf16_mul,
        shape=dict(hidden=HIDDEN, inter=INTER, experts=EXPERTS, topk=TOPK, tp=8, ep=1),
        selection="automatic warp_decode",
        fixture=dict(
            weight_seed=seed,
            input_seed=seed + 100 + batch,
            routing_ids=inputs["topk_ids"].cpu().tolist(),
            tensor_sha256=hashes,
        ),
        validation=dict(
            errors=errors,
            timed_errors=timed_errors,
            eager_repeats_bitwise=True,
            eager_graph_bitwise=True,
            input_fixture_unchanged=True,
        ),
        samples_us=samples,
        median_us=medians,
    )


def metadata(args):
    sources = [ROOT / "CMakeLists.txt", ROOT / "hpc/fuse_moe.py", Path(__file__).resolve()]
    for directory in ("src/fuse_moe", "src/activation", "src/group_gemm/sm90/cp_async"):
        sources += [
            path
            for path in (ROOT / directory).rglob("*")
            if path.is_file() and path.suffix in (".cc", ".cu", ".h", ".cuh")
        ]
    package = Path(hpc.__file__).resolve().parent
    binary = package / f"_C_sm{hpc.__loaded_arch__}.abi3.so"
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    return dict(
        software=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            torch_cuda=torch.version.cuda,
            hpc=hpc.__version__,
            hpc_build=hpc.__built_json__,
        ),
        gpu=dict(
            name=device.name,
            capability=list(torch.cuda.get_device_capability()),
            total_memory_bytes=device.total_memory,
        ),
        git=dict(
            head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            dirty=bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()
            ),
        ),
        source_sha256={str(path.relative_to(ROOT)): sha_file(path) for path in sources},
        loaded_binary=dict(path=str(binary), sha256=sha_file(binary)),
        loaded_wrapper=dict(
            path=str(package / "fuse_moe.py"), sha256=sha_file(package / "fuse_moe.py")
        ),
        protocol=dict(
            seeds=args.seeds,
            batches=args.batches,
            use_bf16_mul=args.mul_modes,
            rounds=args.rounds,
            graph_replays=args.iters,
            calls_per_graph=args.calls_per_graph,
            backends=list(BACKENDS),
            selection="automatic warp_decode",
            statistic="median of raw round samples",
            weights="fixed addresses; no cache flush or weight rotation",
            includes="gate/up, fixed-scale activation quantization, down, expert sum",
            excludes="initial input quantization, router, shared expert, TP communication, weight setup",
            input="prequantized E4M3; FP32 per-expert gate_up/down scales and one activation multiplier",
            rounding="BF16 gate/up projection; optional BF16 SiLU and multiply; SATFINITE E4M3; BF16 per-expert down; FP32 weighted sum then BF16",
            accuracy=dict(
                common_reference="independent FP32 reference before final BF16 conversion",
                warp_decode="requires NRMSE < 0.006, max_abs < max(0.15 * reference_rms, 1e-6), and BF16-reference allclose",
                legacy_allclose="actual.float() versus reference.bfloat16().float(), rtol=0.08, atol=0.1",
                recording="all metrics and strict_reference_passed/legacy_allclose_passed flags are retained before and after timing",
            ),
            validation="reference precision checks before and after timing; finite output, 10 eager repeats, eager/Graph byte equality and preserved fixture",
            complete_definition="all requested points timed and required checks passed",
        ),
    )


def write_results(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with path.with_suffix(".csv").open("w", newline="") as file:
        fields = [
            "seed",
            "batch",
            "use_bf16_mul",
            "warp_decode_us",
        ]
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in result["results"]:
            writer.writerow(
                dict(
                    seed=row["seed"],
                    batch=row["batch"],
                    use_bf16_mul=row["use_bf16_mul"],
                    warp_decode_us=row["median_us"]["warp_decode"],
                )
            )


def main():
    args = parse_args()
    global torch, F, FP8, hpc
    import torch
    import torch.nn.functional as F

    FP8 = torch.float8_e4m3fn
    sys.path.insert(0, str(ROOT))
    builds = sorted((ROOT / "build").glob("lib.*"))
    if builds:
        sys.path.insert(0, str(builds[0]))
    import hpc

    if hpc.__loaded_arch__ != 90:
        raise RuntimeError("warp_decode benchmark requires the SM90 hpc-ops module")
    torch.backends.cuda.matmul.allow_tf32 = False
    result = dict(metadata=metadata(args), results=[], errors=[], complete=False)
    write_results(args.output, result)
    for seed in args.seeds:
        weights = _weights(seed)
        for use_bf16_mul in args.mul_modes:
            for batch in args.batches:
                try:
                    row = point(args, weights, seed, batch, use_bf16_mul)
                    result["results"].append(row)
                    print(
                        f"seed={seed} B={batch} bf16_mul={use_bf16_mul}: "
                        f"warp_decode={row['median_us']['warp_decode']:.3f} us",
                        flush=True,
                    )
                except Exception as error:
                    failure = dict(
                        seed=seed,
                        batch=batch,
                        use_bf16_mul=use_bf16_mul,
                        error=traceback.format_exc(),
                    )
                    evidence = (
                        error.args[0] if isinstance(error, AssertionError) and error.args else None
                    )
                    if (
                        isinstance(evidence, tuple)
                        and len(evidence) == 2
                        and evidence[0] in BACKENDS
                    ):
                        failure["backend"] = evidence[0]
                        if isinstance(evidence[1], dict):
                            failure["metrics"] = evidence[1]
                    result["errors"].append(failure)
                    print(failure["error"], file=sys.stderr, flush=True)
                write_results(args.output, result)
        del weights
        gc.collect()
        torch.cuda.empty_cache()
    result["complete"] = not result["errors"] and len(result["results"]) == (
        len(args.seeds) * len(args.mul_modes) * len(args.batches)
    )
    write_results(args.output, result)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
