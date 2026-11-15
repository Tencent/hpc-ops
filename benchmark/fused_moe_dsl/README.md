# CuTe DSL Fused MoE Benchmark

Latency of the CuTe DSL fused MoE (`hpc/cute/fuse_moe/`) against the MoE
backends of vLLM, on the checkpoint's own quantization format:

| Format | hpc entry point | vLLM backends compared by default |
|---|---|---|
| FP8 with 128x128 block scales (`fp8_block`) | `fuse_moe_blockwise_fp8_dsl` | `triton`, `deep_gemm`, `marlin` |
| MXFP4 weights x MXFP8 activations (`mxfp4`) | `fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl` | `marlin`, `deep_gemm`, `flashinfer_cutlass` |

The defaults are the vLLM backends that run each format on SM120 with a stock
install; `--backends` takes any `--moe-backend` name. `b12x` also runs the MXFP4
format once the optional `b12x` package is installed.

## Method

Every backend except `hpc-direct` is measured through vLLM:

1. Build a single-rank vLLM `FusedMoE` layer with the checkpoint's quant
   config (`Fp8Config` with `weight_block_size=[128, 128]`, or `Mxfp4Config`)
   and `--moe-backend <backend>`.
2. Fill the parameters with seeded random data and run
   `process_weights_after_loading`, so each backend gets its own weight layout.
3. Time `quant_method.apply`: activation quantization, both GEMMs, activation,
   and top-k reduce. Routing is excluded.

`hpc-direct` calls the hpc entry point on pre-quantized activations, without
vLLM. It shows how much of the `hpc` time is the vLLM layer and activation
quantization. The two set up their weights and inputs independently, so
differences within about 1% are not meaningful.

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

Requires an SM120 GPU, vLLM with the HPC MoE backend, and hpc-ops with
`nvidia-cutlass-dsl` (`pip install "hpc-ops[cute-dsl]"`).

```bash
# Both presets, default backends, over 4 GPUs
python benchmark/fused_moe_dsl/benchmark_fuse_moe_dsl.py --gpus 0 1 2 3

# One preset, with hpc-direct and selected token counts
python benchmark/fused_moe_dsl/benchmark_fuse_moe_dsl.py \
    --models deepseek-v4-flash-tp4 --backends hpc hpc-direct marlin \
    --tokens 1 8 64 512 4096

# Custom per-GPU shape: NAME:QUANT:EXPERTS:TOPK:HIDDEN:INTERMEDIATE[:SWIGLU_LIMIT]
python benchmark/fused_moe_dsl/benchmark_fuse_moe_dsl.py \
    --model-shape my-moe:fp8_block:64:8:4096:1024 --models my-moe
```

Presets (per-GPU shapes):

| Model | Format | Experts | Top-k | Hidden | Intermediate | SwiGLU limit |
|---|---|---:|---:|---:|---:|---:|
| `qwen3.8-flash-next-ep4` (TP4 + EP) | fp8_block | 128 | 10 | 2560 | 640 | - |
| `deepseek-v4-flash-tp4` (TP4) | mxfp4 | 256 | 6 | 4096 | 512 | 10 |

Output goes to `--out-dir` (default `fused_moe_dsl_results/`): `results.csv`,
`summary.json`, and one `.json` and `.log` per job and GPU. The script also prints
markdown tables of latency and `<backend> / hpc` ratios (above 1.00x means hpc
is faster).

## DRAM traffic (Nsight Compute)

`ncu_bandwidth.py` profiles one warmed-up call of each hpc entry point per
token count, with the cache flushed before every kernel
(`--cache-control all --clock-control none`), and compares the DRAM bytes moved
with the compulsory bytes: weights and scales of the experts the routing
touches, activations in, output out. The MXFP4 entry point runs on BF16
activations, as in vLLM. ncu serializes kernels, so PDL overlap is lost and
small batches look slower than the graphed latencies.

```bash
python benchmark/fused_moe_dsl/ncu_bandwidth.py --gpu 0 --tokens 1 8 64 512 2048 8192
```

It needs `ncu` on `PATH` (or `--ncu`), prints one markdown table per model, and
writes the raw ncu CSV per point and `summary.json` to `--out-dir` (default
`fused_moe_dsl_ncu/`).

## Reference results

One RTX PRO 5000 (SM120), vLLM 0.30.1rc1 with the SM120 HPC backend,
nvidia-cutlass-dsl 4.7.1, default options; all backends of each model run
sequentially on the same GPU. Latency in us per MoE layer call.
`b12x` is b12x 1.3.0 (the version vLLM pins), installed with `--no-deps` to keep
nvidia-cutlass-dsl 4.7.1; it times the same on its pinned 4.6.2.

`qwen3.8-flash-next-ep4`:

| Tokens | hpc | hpc-direct | triton | deep_gemm | marlin |
|---:|---:|---:|---:|---:|---:|
| 1 | 48.1 | 46.8 | 46.1 | 65.3 | 50.8 |
| 8 | 275.2 | 274.8 | 268.6 | 344.7 | 271.6 |
| 64 | 556.2 | 555.1 | 537.8 | 740.0 | 545.4 |
| 512 | 600.7 | 595.7 | 577.1 | 764.2 | 617.5 |
| 1024 | 726.6 | 724.6 | 733.2 | 803.0 | 1309.9 |
| 2048 | 1025.2 | 1023.7 | 1124.0 | 1220.7 | 2033.0 |
| 4096 | 1794.4 | 1791.7 | 2058.4 | 1873.2 | 3747.9 |
| 8192 | 3416.4 | 3374.8 | 4047.2 | 3508.1 | 7237.1 |

`deepseek-v4-flash-tp4`:

| Tokens | hpc | hpc-direct | marlin | deep_gemm | flashinfer_cutlass | b12x |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29.7 | 28.2 | 32.7 | 48.6 | 74.3 | 28.4 |
| 8 | 143.1 | 143.2 | 139.5 | 254.9 | 248.4 | 179.6 |
| 64 | 584.0 | 584.9 | 580.7 | 1071.5 | 977.3 | 642.4 |
| 512 | 774.2 | 773.3 | 765.2 | 1346.0 | 1279.5 | 834.6 |
| 1024 | 798.7 | 804.5 | 932.1 | 1395.5 | 1354.8 | 817.3 |
| 2048 | 919.2 | 926.5 | 1406.6 | 1503.4 | 1491.9 | 974.9 |
| 4096 | 1601.9 | 1600.5 | 2702.5 | 1724.1 | 1755.0 | 1636.9 |
| 8192 | 2807.4 | 2765.1 | 4785.6 | 3178.1 | 3415.9 | 2869.3 |
