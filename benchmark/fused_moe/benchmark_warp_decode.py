"""Benchmark automatically selected blockwise FP8 warp_decode on Qwen decode shapes.

Inputs are already FP8. Timings include gate/up, SiLU, intermediate activation
quantization, down and expert combination. Router scoring, shared experts,
initial activation quantization, weight setup and TP communication are excluded.
Fixed weights and routes are reused by warm CUDA Graph replay.
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
BACKENDS = ("warp_decode",)
CASES = {
    "qwen35_36_35b_tp1": dict(hidden=2048, inter=512, experts=256, topk=8, tp=1),
    "qwen35_36_35b_tp2": dict(hidden=2048, inter=256, experts=256, topk=8, tp=2),
    "qwen35_122b_tp2": dict(hidden=3072, inter=512, experts=256, topk=8, tp=2),
    "qwen35_122b_tp4": dict(hidden=3072, inter=256, experts=256, topk=8, tp=4),
    "qwen35_397b_tp4": dict(hidden=4096, inter=256, experts=512, topk=10, tp=4),
    "qwen35_397b_tp8": dict(hidden=4096, inter=128, experts=512, topk=10, tp=8),
    "qwen3_next_80b_tp1": dict(hidden=2048, inter=512, experts=512, topk=10, tp=1),
    "qwen3_next_80b_tp2": dict(hidden=2048, inter=256, experts=512, topk=10, tp=2),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="all", help="all, or comma-separated case names")
    parser.add_argument("--list-cases", action="store_true", help="list cases without CUDA imports")
    parser.add_argument("--batches", default="1,2,4,8", help="comma-separated batches in 1..8")
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100, help="graph replays per timing sample")
    parser.add_argument("--calls-per-graph", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("warp_decode.json"))
    args = parser.parse_args()
    if args.list_cases:
        print(json.dumps(CASES, indent=2))
        parser.exit()
    args.cases = list(CASES) if args.cases == "all" else args.cases.split(",")
    if not args.cases or any(name not in CASES for name in args.cases):
        parser.error("unknown case; use --list-cases for supported names")
    if len(set(args.cases)) != len(args.cases):
        parser.error("cases must not contain duplicates")
    try:
        args.batches = [int(value) for value in args.batches.split(",")]
    except ValueError:
        parser.error("batches must be comma-separated integers")
    if not args.batches or any(not 1 <= batch <= 8 for batch in args.batches):
        parser.error("batches must be in 1..8")
    if len(set(args.batches)) != len(args.batches):
        parser.error("batches must not contain duplicates")
    if min(args.rounds, args.iters, args.calls_per_graph) < 1:
        parser.error("rounds, iters and calls-per-graph must be positive")
    if args.output.suffix != ".json":
        parser.error("output must use the .json suffix; a sibling .csv is also written")
    return args


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixture_seed(seed, case, part):
    return int(hashlib.sha256(f"{seed}:{case}:{part}".encode()).hexdigest()[:16], 16)


def tensor_sha256(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().flatten().tolist()
    return hashlib.sha256(bytes(raw)).hexdigest()


def quantize_input(x):
    blocks = x.float().reshape(x.shape[0], x.shape[1] // 128, 128)
    scales = (blocks.abs().amax(-1).clamp_min(1e-10).double() / 448).float()
    reciprocal = scales.double().reciprocal().float()
    values = (blocks * reciprocal[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return values.reshape(x.shape), scales


def make_weights(shape, seed):
    hidden, inter, experts = (shape[key] for key in ("hidden", "inter", "experts"))
    generator = torch.Generator(device="cuda").manual_seed(seed)
    gate_up = (
        (torch.randn(experts, 2 * inter, hidden, device="cuda", generator=generator) * 64)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    down = (
        (torch.randn(experts, hidden, inter, device="cuda", generator=generator) * 64)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )

    def weight_scales(rows, cols):
        result = torch.zeros((experts, rows, (cols + 3) // 4 * 4), device="cuda")
        result[..., :cols] = (
            torch.rand(experts, rows, cols, device="cuda", generator=generator) * 0.0005 + 0.0005
        )
        return result

    return dict(
        gate_up_weight=gate_up,
        gate_up_weight_scale=weight_scales(2 * inter // 128, hidden // 128),
        down_weight=down,
        down_weight_scale=weight_scales(hidden // 128, inter // 128),
    )


def make_inputs(shape, batch, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = (
        torch.randn(
            batch, shape["hidden"], device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.1
    )
    x, scales = quantize_input(x)
    scores = torch.rand(batch, shape["experts"], device="cuda", generator=generator)
    ids = scores.topk(shape["topk"], dim=-1).indices.int().contiguous()
    route = torch.rand(batch, shape["topk"], device="cuda", generator=generator).softmax(-1)
    return dict(
        x=x,
        x_scale=scales,
        topk_ids=ids,
        topk_scale=route,
        rank_ep=0,
        num_expert_total=shape["experts"],
    )


def same_bits(left, right):
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


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
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (calls * replays)


def benchmark_case(args, case, shape, batch, weights):
    seed = fixture_seed(args.seed, case, f"batch={batch}")
    inputs = make_inputs(shape, batch, seed)
    kwargs = weights | inputs
    fixture = dict(
        seed=seed,
        routing_ids=inputs["topk_ids"].cpu().tolist(),
        tensor_sha256={
            key: tensor_sha256(inputs[key]) for key in ("x", "x_scale", "topk_ids", "topk_scale")
        },
    )
    functions = {"warp_decode": lambda: hpc.fuse_moe_blockwise(**kwargs)}
    expected, graphs, captured, validation = {}, {}, {}, {}
    for backend, fn in functions.items():
        expected[backend] = fn().clone()
        assert torch.isfinite(expected[backend]).all(), backend + " eager output is not finite"
        for _ in range(10):
            assert same_bits(expected[backend], fn()), (
                backend + " eager output is not deterministic"
            )
        graphs[backend], captured[backend] = capture(fn, args.calls_per_graph)
        for _ in range(3):
            graphs[backend].replay()
            torch.cuda.synchronize()
            assert same_bits(expected[backend], captured[backend]), (
                backend + " graph output differs"
            )
        validation[backend] = dict(
            finite=True, eager_repeats_bitwise=True, eager_graph_bitwise=True
        )

    samples = {backend: [] for backend in BACKENDS}
    for _ in range(args.rounds):
        for backend in BACKENDS:
            samples[backend].append(measure(graphs[backend], args.calls_per_graph, args.iters))
    # Timing must not change the recorded fixture or captured outputs.
    for backend in BACKENDS:
        assert same_bits(expected[backend], captured[backend]), backend + " changed after timing"
    assert fixture["tensor_sha256"] == {
        key: tensor_sha256(inputs[key]) for key in fixture["tensor_sha256"]
    }, "benchmark mutated its input fixture"
    medians = {backend: statistics.median(values) for backend, values in samples.items()}
    return dict(
        case=case,
        shape=shape,
        batch=batch,
        tokens=batch * shape["topk"],
        fixture=fixture,
        validation=validation,
        backend_parameters=dict(
            rank_ep=0, num_expert_total=shape["experts"], shared_output=None, output_provided=False
        ),
        samples_us=samples,
        median_us=medians,
    )


def metadata(args):
    sources = [
        "CMakeLists.txt",
        "hpc/fuse_moe.py",
        "src/fuse_moe/entry.cc",
        "src/fuse_moe/fuse_moe_warp_decode.h",
        "src/fuse_moe/sm90/warp_decode/gate_up.cu",
        "src/fuse_moe/sm90/warp_decode/down.cu",
        str(Path(__file__).resolve().relative_to(ROOT)),
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
        source_sha256={name: sha_file(ROOT / name) for name in sources},
        loaded_binary=dict(path=str(binary), sha256=sha_file(binary)),
        loaded_wrapper=dict(
            path=str(package / "fuse_moe.py"), sha256=sha_file(package / "fuse_moe.py")
        ),
        protocol=dict(
            seed=args.seed,
            cases=args.cases,
            batches=args.batches,
            rounds=args.rounds,
            graph_replays=args.iters,
            calls_per_graph=args.calls_per_graph,
            selection="automatic warp_decode",
            statistic="median of raw round samples",
            fixture_seeds="SHA256(base seed, case name, weights or batch number)",
            weights="fixed addresses; no cache flush or weight rotation",
            input="prequantized E4M3 plus compact row/group-128 scales",
            weight_scales="FP32 per 128x128 block, last dimension padded to four",
            includes="gate/up, SiLU, intermediate quantization, down, expert sum",
            excludes="initial activation quantization, router, shared expert, TP communication, weight setup",
            validation="finite and deterministic eager output; eager/Graph byte equality; fixture preserved",
            accuracy="Run tests/test_fuse_moe_warp_decode.py separately before timing",
        ),
    )


def write_results(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with path.with_suffix(".csv").open("w", newline="") as file:
        fields = [
            "case",
            "batch",
            "tokens",
            "hidden",
            "inter",
            "experts",
            "topk",
            "tp",
            "warp_decode_us",
        ]
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in result["results"]:
            writer.writerow(
                dict(
                    case=row["case"],
                    batch=row["batch"],
                    tokens=row["tokens"],
                    **row["shape"],
                    warp_decode_us=row["median_us"]["warp_decode"],
                )
            )


def main():
    args = parse_args()
    global torch, hpc
    import torch

    sys.path.insert(0, str(ROOT))
    builds = sorted((ROOT / "build").glob("lib.*"))
    if builds:
        sys.path.insert(0, str(builds[0]))
    import hpc

    if hpc.__loaded_arch__ != 90:
        raise RuntimeError("warp_decode benchmark requires the SM90 hpc-ops module")
    torch.backends.cuda.matmul.allow_tf32 = False
    result = dict(metadata=metadata(args), results=[], errors=[])
    write_results(args.output, result)
    for case in args.cases:
        shape = CASES[case]
        seed = fixture_seed(args.seed, case, "weights")
        weights = make_weights(shape, seed)
        for batch in args.batches:
            try:
                row = benchmark_case(args, case, shape, batch, weights)
                row["weight_seed"] = seed
                result["results"].append(row)
                print(
                    f"{case} B={batch}: warp_decode={row['median_us']['warp_decode']:.3f} us",
                    flush=True,
                )
            except Exception:
                failure = dict(case=case, batch=batch, error=traceback.format_exc())
                result["errors"].append(failure)
                print(failure["error"], file=sys.stderr, flush=True)
            write_results(args.output, result)
        del weights
        gc.collect()
        torch.cuda.empty_cache()
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
