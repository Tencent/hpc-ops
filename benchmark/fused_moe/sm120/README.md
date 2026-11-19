# SM120 Fused MoE Benchmark

Latency of the SM120 fused MoE against the MoE backends of vLLM, on the
checkpoint's own quantization format:

| Format | hpc entry point | vLLM backends compared by default |
|---|---|---|
| FP8 with 128x128 block scales (`fp8_block`) | `fuse_moe_blockwise_fp8_sm120` | `triton`, `deep_gemm`, `marlin` |
| MXFP4 weights x MXFP8 activations (`mxfp4`) | `fuse_moe_blockwise_mxfp4_sm120` | `marlin`, `deep_gemm`, `flashinfer_cutlass` |

The defaults are the vLLM backends that run each format on SM120 with a stock
install; `--backends` takes any `--moe-backend` name. `b12x` also runs the MXFP4
format once the optional `b12x` package is installed.

## Method

Every backend except `hpc-direct` and `hpc-bf16-direct` is measured through vLLM:

1. Build a single-rank vLLM `FusedMoE` layer with the checkpoint's quant
   config (`Fp8Config` with `weight_block_size=[128, 128]`, or `Mxfp4Config`)
   and `--moe-backend <backend>`.
2. Fill the parameters with seeded random data and run
   `process_weights_after_loading`, so each backend gets its own weight layout.
3. Time `quant_method.apply`: activation quantization, both GEMMs, activation,
   and top-k reduce. Routing is excluded.

`hpc-direct` calls the SM120 entry point on pre-quantized activations, without
vLLM. `hpc-bf16-direct` uses the same entry point on BF16 activations with
`x_scale=None`, matching the contract vLLM passes to the `hpc` backend. They
show how much of the `hpc` time is the vLLM layer and activation quantization.
The backends set up their weights and inputs independently, so differences within
about 1% are not meaningful.

Timing: `--rotate` (default 8) random routings are drawn for each token count,
one call per routing is captured in a single CUDA graph, and the graph is
replayed `--iters` times per sample. The reported latency is the median of
`--reps` samples divided by the number of calls. Rotating the routing keeps small
batches from reading their expert weights out of L2, as each layer of a real
model reads its own weights.

Each (model, backend) pair runs `worker.py` in its own process. All backends of
a model run one after another on the same GPU so backend-to-backend comparisons
are not skewed by GPU-to-GPU variance. Models can still run in parallel on
different GPUs when using `--gpus`.

## Usage

Requires an SM120 GPU, vLLM with the HPC MoE backend, and hpc-ops.

```bash
# Both presets, default backends, over 4 GPUs
python benchmark/fused_moe/sm120/benchmark_fuse_moe.py --gpus 0 1 2 3

# One preset, with hpc-direct and selected token counts
python benchmark/fused_moe/sm120/benchmark_fuse_moe.py \
    --models deepseek-v4-flash-tp4 --backends hpc hpc-direct marlin \
    --tokens 1 8 64 512 4096

# Direct entry point on BF16 input (same contract as vLLM hpc)
python benchmark/fused_moe/sm120/benchmark_fuse_moe.py \
    --models deepseek-v4-flash-tp4 \
    --backends hpc hpc-bf16-direct

# Custom per-GPU shape: NAME:QUANT:EXPERTS:TOPK:HIDDEN:INTERMEDIATE[:SWIGLU_LIMIT]
python benchmark/fused_moe/sm120/benchmark_fuse_moe.py \
    --model-shape my-moe:fp8_block:64:8:4096:1024 --models my-moe
```

Presets (per-GPU shapes):

| Model | Format | Experts | Top-k | Hidden | Intermediate | SwiGLU limit |
|---|---|---:|---:|---:|---:|---:|
| `qwen3.8-flash-next-ep4` (TP4 + EP) | fp8_block | 128 | 10 | 2560 | 640 | - |
| `deepseek-v4-flash-tp4` (TP4) | mxfp4 | 256 | 6 | 4096 | 512 | 10 |

Output goes to `--out-dir` (default `fused_moe_sm120_results/`): `results.csv`,
`summary.json`, and one `.json` and `.log` per job and GPU. The script also prints
markdown tables of latency and `<backend> / hpc` ratios (above 1.00x means hpc
is faster).

A job's result file is deleted before its worker starts, and only read when
the worker exits with status 0. A worker that crashes or exits nonzero is
reported with its exit code and `.log` file, gets one error row in
`results.csv`, and is left out of the latency tables, so a rerun into the same
`--out-dir` never shows an earlier run's numbers.

## DRAM traffic (Nsight Compute)

`ncu_bandwidth.py` profiles one warmed-up call of each hpc entry point per
token count, with the cache flushed before every kernel
(`--cache-control all --clock-control none`), and compares the DRAM bytes moved
with the compulsory bytes: weights and scales of the experts the routing
touches, activations in, output out. Both entry points run on BF16 activations,
as in vLLM. ncu serializes kernels, so PDL overlap is lost and small batches
look slower than the graphed latencies.

```bash
python benchmark/fused_moe/sm120/ncu_bandwidth.py --gpu 0 --tokens 1 8 64 512 2048 8192
```

It needs `ncu` on `PATH` (or `--ncu`), prints one markdown table per model, and
writes the raw ncu CSV per point and `summary.json` to `--out-dir` (default
`fused_moe_sm120_ncu/`).

## Reference results

One RTX PRO 5000 (SM120), vLLM `0.31.1rc1.dev382+g86b1c2ef1` with the SM120 HPC
backend, default options; all backends of each model run sequentially on the
same GPU. Latency in us per MoE layer call. `b12x` is b12x 1.3.0 (the version
vLLM pins), installed with `--no-deps`. The tables are a historical run from
before the SM120 code was restructured for review, and show eight of the
fourteen default token counts. The SM120 rows of the root `README.md` take their
maximum ratios from these tables.

Rerunning with this script's defaults on hpc-ops `7efc063` and on the current
version, in one session with the GPUs swapped between two runs, the `hpc`,
`hpc-direct` and `hpc-bf16-direct` latencies of the two versions agree within
0.8% at every token count on both presets, and both are within 1.5% of these
tables.

`qwen3.8-flash-next-ep4`:

| Tokens | hpc | hpc-direct | triton | deep_gemm | marlin |
|---:|---:|---:|---:|---:|---:|
| 1 | 43.7 | 43.2 | 46.1 | 65.5 | 50.3 |
| 8 | 256.2 | 256.6 | 268.2 | 340.8 | 270.9 |
| 64 | 516.5 | 516.9 | 536.9 | 737.6 | 544.9 |
| 512 | 528.4 | 528.9 | 576.2 | 766.0 | 613.1 |
| 1024 | 583.9 | 592.2 | 721.7 | 802.8 | 1289.2 |
| 2048 | 822.4 | 827.0 | 1100.7 | 1224.4 | 2000.7 |
| 4096 | 1473.5 | 1476.5 | 2049.0 | 1869.7 | 3686.1 |
| 8192 | 2788.0 | 2786.9 | 4008.0 | 3499.0 | 7122.6 |

`deepseek-v4-flash-tp4`:

| Tokens | hpc | hpc-direct | marlin | deep_gemm | flashinfer_cutlass | b12x |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.5 | 25.8 | 30.3 | 47.9 | 72.8 | 28.2 |
| 8 | 125.6 | 126.0 | 138.4 | 247.7 | 242.6 | 180.8 |
| 64 | 534.0 | 534.1 | 579.9 | 1073.4 | 947.5 | 640.6 |
| 512 | 675.4 | 673.8 | 762.8 | 1350.7 | 1241.9 | 824.8 |
| 1024 | 686.5 | 684.9 | 918.4 | 1396.8 | 1313.9 | 822.8 |
| 2048 | 794.9 | 759.5 | 1366.1 | 1493.1 | 1450.1 | 964.3 |
| 4096 | 1253.6 | 1244.7 | 2631.4 | 1702.8 | 1718.2 | 1610.2 |
| 8192 | 2232.6 | 2184.2 | 4664.7 | 3115.8 | 3344.5 | 2820.5 |

With NCU cache flushing, the SM120 kernel moves 1.01x-1.06x the compulsory DRAM
bytes from 1 to 2048 tokens on both presets, and 1.36x-1.49x at 8192 tokens.
