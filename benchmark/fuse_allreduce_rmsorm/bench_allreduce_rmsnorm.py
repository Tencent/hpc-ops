"""Benchmark fused AllReduce + Residual + RMSNorm on a single 8-GPU node.

Compares four implementations of ``RMSNorm(AllReduce(x) + residual, weight)``:

  - hpc_ops_ht : hpc.fuse_allreduce_rmsnorm_high_throughput
  - hpc_ops_ll : hpc.fuse_allreduce_rmsnorm_low_latency    
  - nccl       : torch.distributed all_reduce + flashinfer fused_add_rmsnorm
  - flashinfer : flashinfer fused allreduce (``--fi-backend mnnvl`` or ``trtllm``)

world_size is fixed to 8; token counts are configurable. The script is
self-contained: it spawns its own processes and bootstraps both the hpc
multicast communicator and a NCCL process group, so it runs like a test (no
torchrun needed). Timing uses CUDA-graph replay by default, which measures pure
device time; pass ``--no-graph`` for eager timing (includes host dispatch).

Example:

  cd hpc-ops
  python3 benchmark/fuse_allreduce_rmsorm/bench_allreduce_rmsnorm.py --hidden 7168 \
      --tokens 8 32 128 512 4096 8192 16384 32768
"""

import argparse
import csv
import json
import math
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../../build/lib.*/"))[0]))

import hpc
import torch
import torch.distributed as dist

try:
    import flashinfer.comm as fi_comm
    import flashinfer.norm as fi_norm

    HAS_FLASHINFER = True
except Exception as e:  # pragma: no cover
    HAS_FLASHINFER = False
    _FI_ERR = repr(e)

NUM_LAMPORT_BUFFERS = 3
SEED = 10001


def rmsnorm(x, w, eps):
    mean_square = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(mean_square + eps)).to(torch.bfloat16) * w.reshape(1, -1)


def ref_allreduce_rmsnorm(input_list, residual, weight, eps):
    input_sum = torch.zeros_like(input_list[0])
    for x in input_list:
        input_sum += x
    out_residual = input_sum + residual
    return out_residual, rmsnorm(out_residual, weight, eps)


def _reduce_max_us(local_us):
    t = torch.tensor([local_us], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def _reduce_max_us_list(local_us):
    if not local_us:
        return []
    t = torch.tensor(local_us, device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.cpu().tolist()


def bench_us(step_fn, warmup, iters, nvtx_label=None):
    """Per-call latency (us), reduced across ranks via MAX.

    Each call is isolated by a sync + barrier so all ranks launch the kernel in
    lockstep (the lamport collectives spin-wait on peer data and deadlock if
    ranks drift). NOTE: start.record() sits *before* the host-side dispatch, so
    the measured time includes python launch overhead (heavy for the flashinfer
    wrapper, light for our direct torch.ops call) -- it is wall, not pure device
    time. This is the ``--no-graph`` path; graph replay measures device time.
    """
    for _ in range(warmup):
        step_fn()
        torch.cuda.synchronize()
        dist.barrier()

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for _ in range(iters):
        torch.cuda.synchronize()
        dist.barrier()
        start, end = events[_]
        if nvtx_label is not None:
            torch.cuda.nvtx.range_push(f"step:{nvtx_label}")
        start.record()
        step_fn()
        end.record()
        if nvtx_label is not None:
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    local_us = [start.elapsed_time(end) * 1000.0 for start, end in events]
    return median(_reduce_max_us_list(local_us))


def bench_us_graph(step_fn, warmup, iters, nvtx_label=None):
    """Pure device latency (us) via CUDA graph capture+replay.

    A fixed-shape lamport collective replayed back-to-back is exactly its
    steady state: buffer_flags rotate through NUM_LAMPORT_BUFFERS in device
    memory across replays, and each kernel clears the buffer it dirtied 3 iters
    ago. Graph capture only *records* launches (no execution, so no spin-wait
    deadlock at capture); replay re-runs only the kernels, dropping all python
    dispatch overhead so the number reflects real GPU time. Ranks stay in
    lockstep because every rank replays the same count and the lamport spin-wait
    self-synchronizes.
    """
    # Eager lockstep warmup so the lamport flag state is in steady rotation
    # before we freeze the launch sequence into a graph.
    for _ in range(warmup):
        step_fn()
        torch.cuda.synchronize()
        dist.barrier()

    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    dist.barrier()
    with torch.cuda.graph(g):
        step_fn()
    torch.cuda.synchronize()
    dist.barrier()

    # A couple of lockstep replays to settle flags after capture.
    for _ in range(2):
        g.replay()
        torch.cuda.synchronize()
        dist.barrier()

    torch.cuda.synchronize()
    dist.barrier()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for _ in range(iters):
        start, end = events[_]
        if nvtx_label is not None:
            torch.cuda.nvtx.range_push(f"step:{nvtx_label}")
        start.record()
        g.replay()
        end.record()
        if nvtx_label is not None:
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    local_us = [start.elapsed_time(end) * 1000.0 for start, end in events]
    return median(_reduce_max_us_list(local_us))


def check(name, got, ref, rank, atol=1e-1, rtol=1e-1):
    ok = torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol)
    flag = torch.tensor([1 if ok else 0], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"    [check] {name:<12}: {'PASS' if flag.item() else 'FAIL'}")


# ---------------------------------------------------------------------------
# hpc_ops_ht: two-shot fuse_allreduce_rmsnorm_high_throughput over NVLink multicast
# ---------------------------------------------------------------------------
class HpcOpsHighThroughput:
    def __init__(self, comm, ws, rank, N, H, num_max_blocks, eps, device):
        self.ws, self.rank, self.num_max_blocks, self.eps = ws, rank, num_max_blocks, eps
        N_pad = (N + ws - 1) // ws * ws
        self.N, self.N_pad = N, N_pad
        self.in_x, self.in_hdl = hpc.empty_multimem(comm, [N_pad, H], dtype=torch.bfloat16, device=device)
        self.out_x, self.out_hdl = hpc.empty_multimem(comm, [N_pad, H], dtype=torch.bfloat16, device=device)
        self.residual = torch.empty((N_pad, H), dtype=torch.bfloat16, device=device)
        self.weight = torch.empty((H,), dtype=torch.bfloat16, device=device)
        self.out_residual = torch.empty((N_pad, H), dtype=torch.bfloat16, device=device)
        self.start = N_pad // ws * rank
        self.end = N_pad // ws * (rank + 1)
        self.offset = self.start * H * self.in_x.element_size()

    def load(self, input_rank, residual, weight):
        self.in_x.zero_()
        self.in_x[: self.N, :] = input_rank[: self.N, :]
        self.residual.copy_(residual)
        self.weight.copy_(weight)

    def step(self):
        s, e, off = self.start, self.end, self.offset
        hpc.fuse_allreduce_rmsnorm_high_throughput(
            self.in_x[s:e, :],
            self.in_hdl.get_multimem_buff(self.in_x[s:e, :].shape, dtype=self.in_x.dtype, storage_offset=off),
            self.residual[s:e, :],
            self.weight,
            self.eps,
            self.in_hdl.signal_buffer_ptrs_dev,
            self.rank,
            self.ws,
            self.num_max_blocks,
            self.out_x[s:e, :],
            self.out_hdl.get_multimem_buff(self.out_x[s:e, :].shape, dtype=self.out_x.dtype, storage_offset=off),
            self.out_residual[s:e, :],
        )

    def free(self):
        del self.in_x, self.in_hdl, self.out_x, self.out_hdl
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# hpc_ops_ll: two-shot fuse_allreduce_rmsnorm_low_latency over NVLink multicast
# ---------------------------------------------------------------------------
class HpcOpsLowLatency:
    def __init__(self, comm, ws, rank, N, H, num_max_blocks, eps, device):
        self.ws, self.rank, self.num_max_blocks, self.eps = ws, rank, num_max_blocks, eps
        N_pad = (N + ws - 1) // ws * ws
        self.N, self.N_pad = N, N_pad
        M_pad = 2 * math.ceil(N / ws) * ws * NUM_LAMPORT_BUFFERS
        self.multinode_x, self.hdl = hpc.empty_multimem(comm, [M_pad, H], dtype=torch.bfloat16, device=device)
        self.multinode_x.view(torch.uint32).fill_(0x80000000)
        self.multicast_x = self.hdl.get_multimem_buff([M_pad, H], dtype=torch.bfloat16)
        self.data_buffer_ptrs = self.hdl.data_buffer_ptrs_dev

        ws_bytes = M_pad * H * 2
        lamport_bytes = math.floor(ws_bytes / NUM_LAMPORT_BUFFERS) // 16 * 16
        self.buffer_flags = torch.tensor(
            [0, 2, lamport_bytes, 0, 0, 0, 0, 0, 0], dtype=torch.uint32, device=device
        )
        self.input = torch.empty((N_pad, H), dtype=torch.bfloat16, device=device)
        self.residual = torch.empty((N_pad, H), dtype=torch.bfloat16, device=device)
        self.weight = torch.empty((H,), dtype=torch.bfloat16, device=device)
        self.output = torch.empty_like(self.input)
        self.out_residual = torch.empty_like(self.residual)

    def load(self, input_rank, residual, weight):
        self.input.copy_(input_rank)
        self.residual.copy_(residual)
        self.weight.copy_(weight)

    def step(self):
        hpc.fuse_allreduce_rmsnorm_low_latency(
            self.input,
            self.multicast_x,
            self.data_buffer_ptrs,
            self.multinode_x,
            self.buffer_flags,
            self.ws,
            self.rank,
            self.residual,
            self.weight,
            self.eps,
            self.num_max_blocks,
            self.output,
            self.out_residual,
            True,
        )

    def free(self):
        del self.multinode_x, self.multicast_x, self.hdl
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# nccl baseline: NCCL all_reduce + flashinfer fused_add_rmsnorm (separate)
# ---------------------------------------------------------------------------
class NcclRunner:
    def __init__(self, ws, rank, N, H, num_max_blocks, eps, device):
        self.eps = eps
        N_pad = (N + ws - 1) // ws * ws
        self.N = N
        self.buf = torch.empty((N_pad, H), dtype=torch.bfloat16, device=device)
        self.residual = torch.empty((N_pad, H), dtype=torch.bfloat16, device=device)
        self.weight = torch.empty((H,), dtype=torch.bfloat16, device=device)

    def load(self, input_rank, residual, weight):
        self.buf.copy_(input_rank)
        self.residual.copy_(residual)
        self.weight.copy_(weight)

    def step(self):
        dist.all_reduce(self.buf, op=dist.ReduceOp.SUM)
        fi_norm.fused_add_rmsnorm(self.buf, self.residual, self.weight, self.eps)

    def free(self):
        pass


# ---------------------------------------------------------------------------
# flashinfer baseline: fused allreduce + residual + rmsnorm
#   backend="trtllm": single fused kernel
#   backend="mnnvl" : two kernels (allreduce + rmsnorm) with PDL overlap
# ---------------------------------------------------------------------------
class FlashInferRunner:
    def __init__(self, ws, rank, N, H, num_max_blocks, eps, device, backend="trtllm"):
        self.eps = eps
        self.backend = backend
        N_pad = (N + ws - 1) // ws * ws
        self.N = N
        # Fresh workspace per shape. The lamport buffer is initialized to its
        # sentinel only at workspace creation; reusing one workspace while sweeping
        # many token counts leaves stale lamport flags, so the 2nd shape already
        # reads garbage (wrong result) and larger shapes deadlock. A clean
        # workspace per shape re-inits the lamport state. (The official single-
        # workspace demo is fine because it serves one fixed shape repeatedly.)
        ws_kwargs = dict(
            backend=backend,
            world_size=ws,
            rank=rank,
            max_token_num=N_pad,
            hidden_dim=H,
            dtype=torch.bfloat16,
        )
        if backend == "mnnvl":
            # mnnvl exchanges multicast handles over a comm backend (defaults to
            # MPI, which we don't have); wire it to the existing torch.dist group.
            from flashinfer.comm.mnnvl import TorchDistBackend

            ws_kwargs["comm_backend"] = TorchDistBackend(group=None)
            ws_kwargs["gpus_per_node"] = ws
        self.workspace = fi_comm.create_allreduce_fusion_workspace(**ws_kwargs)
        self.input = torch.empty((N_pad, H), dtype=torch.bfloat16, device=device)
        self.residual = torch.empty((N_pad, H), dtype=torch.bfloat16, device=device)
        self.weight = torch.empty((H,), dtype=torch.bfloat16, device=device)
        self.norm_out = torch.empty_like(self.input)
        self.residual_out = torch.empty_like(self.residual)

    def load(self, input_rank, residual, weight):
        self.input.copy_(input_rank)
        self.residual.copy_(residual)
        self.weight.copy_(weight)

    def step(self):
        kwargs = dict(
            input=self.input,
            workspace=self.workspace,
            pattern=fi_comm.AllReduceFusionPattern.kARResidualRMSNorm,
            launch_with_pdl=True,
            norm_out=self.norm_out,
            residual_in=self.residual,
            residual_out=self.residual_out,
            rms_gamma=self.weight,
            rms_eps=self.eps,
        )
        if self.backend == "trtllm":
            kwargs["trigger_completion_at_end"] = True
            kwargs["fp32_acc"] = False
        fi_comm.allreduce_fusion(**kwargs)

    def free(self):
        if self.workspace is not None:
            self.workspace.destroy()
            self.workspace = None
        torch.cuda.empty_cache()


def format_table(hidden, rows, fi_backend):
    W = 80
    fi_col = f"fi_{fi_backend}"
    lines = ["", "=" * W]
    lines.append(f"Fused AllReduce + Residual + RMSNorm  |  world_size=8  hidden={hidden}  dtype=bfloat16")
    lines.append("latency in us (lower is better)")
    lines.append("-" * W)
    lines.append(
        f"{'tokens':>7} | {'hpc_ops_ht':>11} | {'hpc_ops_ll':>11} | {'nccl':>10} | "
        f"{fi_col:>11} | {'hpc/best':>9}"
    )
    lines.append("-" * W)

    def f(v, w=11):
        return f"{v:{w}.2f}" if isinstance(v, (int, float)) else f"{'n/a':>{w}}"

    for r in rows:
        ht, ll, nc, fi = (
            r.get("hpc_ops_ht"),
            r.get("hpc_ops_ll"),
            r.get("nccl"),
            r.get("flashinfer"),
        )
        # Best of the two hpc_ops kernels vs the best (fastest) baseline.
        hpc_vals = [x for x in (ht, ll) if x]
        base_vals = [x for x in (nc, fi) if x]
        best = (min(base_vals) / min(hpc_vals)) if (hpc_vals and base_vals) else None
        best_str = f"{best:8.2f}x" if best else f"{'n/a':>9}"
        lines.append(
            f"{r['token']:>7} | {f(ht)} | {f(ll)} | {f(nc, 10)} | {f(fi)} | {best_str}"
        )
    lines.append("=" * W)
    lines.append(
        f"hpc/best = min(nccl, {fi_col}) / min(hpc_ops_ht, hpc_ops_ll)  "
        "(best hpc-ops kernel vs best baseline; >1 means hpc-ops wins)"
    )
    lines.append("")
    return "\n".join(lines)


def enrich_rows(rows):
    enriched = []
    for row in rows:
        item = dict(row)
        hpc_vals = [item[k] for k in ("hpc_ops_ht", "hpc_ops_ll") if item.get(k)]
        base_vals = [item[k] for k in ("nccl", "flashinfer") if item.get(k)]
        item["hpc_best_us"] = min(hpc_vals) if hpc_vals else None
        item["baseline_best_us"] = min(base_vals) if base_vals else None
        item["hpc_best_speedup"] = (
            item["baseline_best_us"] / item["hpc_best_us"]
            if item["hpc_best_us"] and item["baseline_best_us"]
            else None
        )
        enriched.append(item)
    return enriched


def timing_name(timing):
    if isinstance(timing, bool):
        return "cuda_graph_median" if timing else "eager_median"
    return timing


def write_csv(path, rows, hidden, fi_backend, timing):
    if not path:
        return
    fieldnames = [
        "tokens",
        "hidden",
        "dtype",
        "world_size",
        "timing",
        "fi_backend",
        "hpc_ops_ht_us",
        "hpc_ops_ll_us",
        "nccl_us",
        "flashinfer_us",
        "hpc_best_us",
        "baseline_best_us",
        "hpc_best_speedup",
    ]
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "tokens": row["token"],
                    "hidden": hidden,
                    "dtype": "bfloat16",
                    "world_size": 8,
                    "timing": timing_name(timing),
                    "fi_backend": fi_backend,
                    "hpc_ops_ht_us": row.get("hpc_ops_ht"),
                    "hpc_ops_ll_us": row.get("hpc_ops_ll"),
                    "nccl_us": row.get("nccl"),
                    "flashinfer_us": row.get("flashinfer"),
                    "hpc_best_us": row.get("hpc_best_us"),
                    "baseline_best_us": row.get("baseline_best_us"),
                    "hpc_best_speedup": row.get("hpc_best_speedup"),
                }
            )


def write_jsonl(path, rows, hidden, fi_backend, timing):
    if not path:
        return
    with Path(path).open("w") as f:
        for row in rows:
            item = {
                "tokens": row["token"],
                "hidden": hidden,
                "dtype": "bfloat16",
                "world_size": 8,
                "timing": timing_name(timing),
                "fi_backend": fi_backend,
                "hpc_ops_ht_us": row.get("hpc_ops_ht"),
                "hpc_ops_ll_us": row.get("hpc_ops_ll"),
                "nccl_us": row.get("nccl"),
                "flashinfer_us": row.get("flashinfer"),
                "hpc_best_us": row.get("hpc_best_us"),
                "baseline_best_us": row.get("baseline_best_us"),
                "hpc_best_speedup": row.get("hpc_best_speedup"),
            }
            f.write(json.dumps(item) + "\n")


def extract_nvtx_rows(report_prefix, tokens, hidden, fi_backend):
    cmd = [
        "nsys",
        "stats",
        "--report",
        "nvtx_gpu_proj_trace",
        "--force-export=true",
        "-q",
        "-f",
        "json",
        str(report_prefix) + ".nsys-rep",
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    raw = json.loads(out.decode())
    data = raw[0]["data"] if isinstance(raw, list) and raw and "data" in raw[0] else raw
    samples = {}
    for entry in data:
        name = entry.get("Name", "").strip().strip('"')
        if name.startswith(":"):
            name = name[1:]
        if not name.startswith("step:"):
            continue
        label = name[len("step:"):]
        try:
            dur_us = float(entry["Projected Duration (ns)"]) / 1000.0
        except Exception:
            continue
        samples.setdefault(label, []).append(dur_us)

    results = {N: {"token": N} for N in tokens}
    for label, vals in samples.items():
        if "_N" not in label:
            continue
        key, token_text = label.rsplit("_N", 1)
        try:
            token = int(token_text)
        except ValueError:
            continue
        if token not in results:
            continue
        vals = sorted(vals)
        # Match the FusedMoE post-processing convention: drop early replay samples.
        vals = vals[2:] if len(vals) > 2 else vals
        if vals:
            results[token][key] = float(median(vals))
    return enrich_rows([results[N] for N in tokens])


def run_nsys_driver(args):
    tag = args.tag or f"allreduce_rmsnorm_{int(time.time())}"
    out_dir = Path(args.output_dir) / tag if args.output_dir else Path(__file__).resolve().parent / "log" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    report_prefix = out_dir / "allreduce_rmsnorm"
    report_file = str(report_prefix) + ".nsys-rep"
    if os.path.exists(report_file):
        os.remove(report_file)

    cmd = [
        "nsys",
        "profile",
        "-f",
        "true",
        "-o",
        str(report_prefix),
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        "--cuda-graph-trace=node",
        "-t",
        "cuda,nvtx",
        sys.executable,
        str(Path(__file__).resolve()),
        "--nsys-worker",
        "--hidden",
        str(args.hidden),
        "--tokens",
        *[str(t) for t in args.tokens],
        "--num-max-blocks",
        str(args.num_max_blocks),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--master-port",
        str(args.master_port),
        "--fi-backend",
        args.fi_backend,
    ]
    if args.no_check:
        cmd.append("--no-check")
    if args.skip:
        cmd += ["--skip", *args.skip]
    if not args.graph:
        cmd.append("--no-graph")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
            start_new_session=True,
        )
        stdout, stderr_bytes = proc.communicate(timeout=args.nsys_timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, 15)
        except Exception:
            pass
        raise RuntimeError("nsys profile timeout") from exc

    if not os.path.exists(report_file) and proc.returncode != 0:
        stderr = stderr_bytes.decode(errors="replace").strip().splitlines()
        raise RuntimeError(stderr[-1] if stderr else "nsys profile failed")

    rows = extract_nvtx_rows(report_prefix, args.tokens, args.hidden, args.fi_backend)
    print(format_table(args.hidden, rows, args.fi_backend))
    write_csv(args.csv, rows, args.hidden, args.fi_backend, "nsys_graph_nvtx_median")
    write_jsonl(args.jsonl, rows, args.hidden, args.fi_backend, "nsys_graph_nvtx_median")
    print(f"nsys report: {report_file}")
    return rows


def run_task(rank, world_size, master_port, tokens, H, num_max_blocks, warmup, iters, do_check, skip, fi_backend, use_graph, csv_path, jsonl_path, profile):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(rank)
    torch.set_default_device(device)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    eps = 1e-6
    do_nccl = "nccl" not in skip
    do_fi = "flashinfer" not in skip and HAS_FLASHINFER
    if "flashinfer" not in skip and not HAS_FLASHINFER and rank == 0:
        print(f"[warn] flashinfer unavailable, skipping it: {_FI_ERR}")

    # warm up NCCL channels so the first config isn't polluted by lazy setup
    _w = torch.ones(8192, device=device, dtype=torch.bfloat16)
    for _ in range(10):
        dist.all_reduce(_w)
    torch.cuda.synchronize()

    comm = hpc.MulticastCommunicator(rank, world_size, rank, f"hpc_bench_ws{world_size}.sock")
    if profile:
        dist.barrier()
        if rank == 0:
            torch.cuda.cudart().cudaProfilerStart()
        dist.barrier()

    # Largest token count flashinfer can serve at this hidden size.
    if fi_backend == "trtllm":
        # trtllm clamps its lamport buffer to MAX_COMM_SIZE; beyond that the
        # kernel silently corrupts / hangs. lamport bytes/token == hidden * 16.
        fi_max_tokens = 2145386496 // (H * 16)
    else:
        # mnnvl's twoshot kernel hits cudaErrorMisalignedAddress at large token
        # counts on this hidden size (sm90/H20); it is solid up to 8192.
        fi_max_tokens = 8192

    def make_inputs(N):
        N_pad = (N + world_size - 1) // world_size * world_size
        torch.manual_seed(SEED)
        input_list = [
            torch.randn((N_pad, H), dtype=torch.bfloat16, device=device) for _ in range(world_size)
        ]
        residual = torch.randn((N_pad, H), dtype=torch.bfloat16, device=device)
        weight = torch.randn((H,), dtype=torch.bfloat16, device=device)
        _, ref_norm = ref_allreduce_rmsnorm(
            [x[:N, :] for x in input_list], residual[:N, :], weight, eps
        )
        return input_list[rank], residual, weight, ref_norm

    def make_runner(key, N):
        if key == "hpc_ops_ht":
            return HpcOpsHighThroughput(comm, world_size, rank, N, H, num_max_blocks, eps, device)
        if key == "hpc_ops_ll":
            return HpcOpsLowLatency(comm, world_size, rank, N, H, num_max_blocks, eps, device)
        if key == "nccl":
            return NcclRunner(world_size, rank, N, H, num_max_blocks, eps, device)
        return FlashInferRunner(world_size, rank, N, H, num_max_blocks, eps, device, backend=fi_backend)

    def check_runner(key, r, N, ref_norm):
        if key == "hpc_ops_ht":
            check(key, r.out_x[:N, :], ref_norm, rank)
        elif key == "hpc_ops_ll":
            check(key, r.output[:N, :], ref_norm, rank)
        elif key == "nccl":
            check(key, r.buf[:N, :], ref_norm, rank)
        else:
            check(key, r.norm_out[:N, :], ref_norm, rank)

    # Benchmark one implementation fully (all token sizes) before moving to the
    # next, with a hard barrier between groups. Each implementation drives a
    # different collective stack (NVLink multicast, flashinfer IPC, NCCL);
    # keeping only one active at a time avoids cross-stack interference on the
    # NVLink fabric. Our multicast kernels run last as their cuMulticast
    # bindings persist across configs.
    kernel_order = []
    if do_fi:
        kernel_order.append("flashinfer")
    if do_nccl:
        kernel_order.append("nccl")
    kernel_order += [k for k in ("hpc_ops_ht", "hpc_ops_ll") if k not in skip]

    results = {N: {"token": N} for N in tokens}
    for key in kernel_order:
        if rank == 0:
            print(f"=== {key} ===", flush=True)
        for N in tokens:
            if key == "flashinfer" and N > fi_max_tokens:
                if rank == 0:
                    print(f"  tokens={N}: skip (>fi_{fi_backend} cap {fi_max_tokens})", flush=True)
                continue
            input_rank, residual, weight, ref_norm = make_inputs(N)
            r = make_runner(key, N)
            r.load(input_rank, residual, weight)
            comm.Barrier()

            if do_check:
                r.step()
                torch.cuda.synchronize()
                check_runner(key, r, N, ref_norm)
                r.load(input_rank, residual, weight)
                comm.Barrier()

            bench_fn = bench_us_graph if use_graph else bench_us
            results[N][key] = bench_fn(r.step, warmup, iters, nvtx_label=f"{key}_N{N}")
            r.free()
            if rank == 0:
                print(f"  tokens={N}: {results[N][key]:.2f}us", flush=True)

        # Fully drain this collective stack before starting the next one.
        torch.cuda.synchronize()
        dist.barrier()

    if rank == 0:
        rows = enrich_rows([results[N] for N in tokens])
        print(format_table(H, rows, fi_backend))
        write_csv(csv_path, rows, H, fi_backend, use_graph)
        write_jsonl(jsonl_path, rows, H, fi_backend, use_graph)

    if profile:
        dist.barrier()
        if rank == 0:
            torch.cuda.cudart().cudaProfilerStop()
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=7168)
    # figure 09 points: decode representatives + chunked-prefill long-token points.
    parser.add_argument(
        "--tokens", type=int, nargs="+",
        default=[8, 32, 128, 512, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--num-max-blocks", type=int, default=78)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--master-port", type=int, default=29570)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=["hpc_ops_ht", "hpc_ops_ll", "nccl", "flashinfer"],
    )
    parser.add_argument(
        "--fi-backend", default="mnnvl", choices=["mnnvl", "trtllm"],
        help="flashinfer backend: mnnvl (two-kernel two-shot with PDL overlap) "
             "or trtllm (single fused kernel).",
    )
    parser.add_argument(
        "--no-graph", dest="graph", action="store_false",
        help="use eager timing (includes host dispatch overhead) instead of the "
             "default CUDA-graph replay (pure device time).",
    )
    parser.add_argument(
        "--timing",
        choices=["event", "nsys"],
        default="event",
        help="event: reduced CUDA event timing; nsys: FusedMoE-style nsys/NVTX graph replay median.",
    )
    parser.add_argument("--output-dir", default="", help="Output directory for nsys reports.")
    parser.add_argument("--tag", default="", help="Subdirectory name for nsys reports.")
    parser.add_argument("--nsys-timeout", type=int, default=600)
    parser.add_argument("--csv", default="", help="Optional CSV output path written by rank 0.")
    parser.add_argument("--jsonl", default="", help="Optional JSONL output path written by rank 0.")
    parser.add_argument("--nsys-worker", action="store_true", help=argparse.SUPPRESS)
    parser.set_defaults(graph=True)
    args = parser.parse_args()

    if args.timing == "nsys" and not args.nsys_worker:
        run_nsys_driver(args)
        return

    world_size = 8
    ctx = multiprocessing.get_context("spawn")
    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=run_task,
            args=(
                rank,
                world_size,
                args.master_port,
                args.tokens,
                args.hidden,
                args.num_max_blocks,
                args.warmup,
                args.iters,
                not args.no_check,
                args.skip,
                args.fi_backend,
                args.graph,
                args.csv,
                args.jsonl,
                args.nsys_worker,
            ),
        )
        procs.append(p)
    for p in procs:
        p.start()
    codes = []
    for p in procs:
        p.join()
        codes.append(p.exitcode)
    for rank, code in enumerate(codes):
        assert code == 0, f"rank {rank} failed with exit code {code}"


if __name__ == "__main__":
    main()
