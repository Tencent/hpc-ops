# Top-K Benchmark

Benchmark the HPC-Ops exact FP32 Top-K operator against the PyTorch `torch.topk`
reference over a grid of row counts, row widths, top-k values and synthetic
score distributions.

## Overview

- Operator: `hpc.topk` (exact FP32 top-k for variable-length rows)
- Providers:
  - `hpc-ops`
  - `torch` (`torch.topk` reference)
  - `deepselect` (optional): the DeepSelect SM90 kernel from LightSeek
    tokenspeed PR 1737, vendored under `backends/deep_select.py` (MIT) with a
    thin adapter. Requires the optional `nvidia-cutlass-dsl` and
    `apache-tvm-ffi` packages; when they are missing the provider reports
    itself unavailable and the run continues without it.
- Timing unit: median microseconds per call (CUDA event timing)
- Default sweeps:
  - `--ks 512 2048`
  - `--m-list 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192` (row counts)
  - `--n-list 2k..4M` powers of two (row widths)
  - `--warmup 5 --iters 20`
- Combinations above `--max-elements` (default 9.5e9) are skipped so one sweep
  never exhausts device memory; skipped entries are printed and omitted from
  the CSV.

## Score distributions

- `gauss3`: three-cluster Gaussian mixture (per-row weights).
- `normal`: standard Gaussian.
- `spike30`: 30% of the entries pinned to one large value (heavy ties).
- `band8`: Gaussian quantized to eight discrete levels (heavy ties).
- `uniform`: uniform [0, 1).
- `coarse`: fp16-rounded Gaussian (ties at the coarse-bin granularity).
- `conc64`: 64 discrete levels (extreme tie groups).

## Usage

Full sweep (resumable; rerun the same command after an interruption):

```bash
python3 benchmark_topk.py --csv topk_latency.csv
```

Smoke test (row counts up to 256, short warmup/iters):

```bash
python3 benchmark_topk.py --smoke --providers hpc-ops --csv topk_smoke.csv
```

Value check (verifies the sorted top-k values against `torch.topk` on row 0
of every width, and on sampled rows of every shape at or above 2^30 elements):

```bash
python3 benchmark_topk.py --smoke --providers hpc-ops --check
```

## Notes

- The `hpc-ops` provider sizes its scratch through
  `hpc.topk_workspace_size` and hands the kernel a zeroed counters buffer;
  the kernels return the counters zero-filled, so the buffer is reused
  across timed iterations without a host-side reset.
- `torch.topk` reference indices are written into the same output buffer
  layout (`[m, top_k]` int32) before timing starts, so allocation costs stay
  out of the timed region for both providers.
- With `--check`, shapes at or above 2^30 elements verify sampled rows
  against `torch.topk` on device; a mismatch is recorded in the CSV and
  summarized at the end of the run.
- The `deepselect` provider compiles its CuTe DSL selectors once per
  (capacity, cluster size) during warmup (tens of seconds on the first run,
  cached afterwards). Its per-row offsets are 32-bit element indices, so
  launches with `rows * width >= 2**31` are skipped and recorded as
  `SKIP_INT32` in the CSV instead of crashing or returning wrong results.
