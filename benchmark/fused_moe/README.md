# FusedMoE Benchmark

Benchmark blockwise FP8 FusedMoE kernels across HPC-Ops and optional comparison backends.

## Overview

- Operator: SM90 blockwise FP8 FusedMoE
- Providers:
  - `HPC-Ops`
  - `vLLM Triton`
  - `vLLM CUTLASS`
  - `SGLang`
- Timing unit: microseconds per operator call
- Timing mode:
  - `--timing nsys`: release-style timing using `nsys`, CUDA Graph replay, NVTX `step` ranges, and median latency from NVTX GPU projected duration. The worker does not synchronize inside each timed step.
- Default models: `qwen3-235b`, `hunyuan-v3`, `deepseek-v3`
- Default modes:
  - `TP=8 EP=1`: tensor-parallel shape, full expert set visible to the measured rank.
  - `TP=1 EP=8`: expert-parallel shape, one local EP rank is measured.

vLLM and SGLang are optional comparison providers. If a provider cannot be
imported or initialized in the local environment, the benchmark prints a warning
and continues with the remaining providers.

## Environment

For stable results, provide all three checkout roots explicitly:

```bash
export HPCOPS_ROOT=/workspace/hpc_cp_bench
export VLLM_ROOT=/workspace/vllm
export SGLANG_ROOT=/workspace/sglang
```

For the HPC-Ops provider, activation quantization uses the local
`hpc.scaled_fp8_quant` operator. Shared FP8 weight construction still prefers
SGLang's `scaled_fp8_quant`, then vLLM `_custom_ops.scaled_fp8_quant`, and
finally a local torch fallback.

## Usage

TP sweep:

```bash
python3 benchmark/fused_moe/benchmark_fuse_moe.py \
  --tp 8 --ep 1 \
  --providers hpcops vllm vllm_cutlass sglang \
  --csv fused_moe_tp8_ep1.csv \
  --jsonl fused_moe_tp8_ep1.jsonl
```

EP sweep:

```bash
python3 benchmark/fused_moe/benchmark_fuse_moe.py \
  --tp 1 --ep 8 \
  --providers hpcops vllm vllm_cutlass sglang \
  --csv fused_moe_tp1_ep8.csv \
  --jsonl fused_moe_tp1_ep8.jsonl
```

Smoke test:

```bash
python3 benchmark/fused_moe/benchmark_fuse_moe.py \
  --tp 8 --ep 1 \
  --models qwen3-235b \
  --bs 4 \
  --providers hpcops vllm \
  --gpu 0
```

Custom shape:

```bash
python3 benchmark/fused_moe/benchmark_fuse_moe.py \
  --model-shape custom:128:8:4096:1536 \
  --models custom \
  --bs 4 16 64 \
  --providers hpcops
```

The benchmark auto-discovers the local HPC-Ops build. To use a specific checkout
for an optional provider, pass the root explicitly or set the corresponding
environment variable:

```bash
python3 benchmark/fused_moe/benchmark_fuse_moe.py --vllm-root /path/to/vllm ...
python3 benchmark/fused_moe/benchmark_fuse_moe.py --sglang-root /path/to/sglang ...
```

## Modes

- `tp8_ep1`: tensor parallelism only. `intermediate_per_rank = intermediate / TP`, experts are not partitioned.
- `tp1_ep8`: expert parallelism only. `experts_per_rank = experts / EP`, routing is sampled within the local expert set.
- `avg/group`: average routed tokens per local expert group, computed as `bs * topk / experts_per_rank`.

Default batch sizes can be overridden with `--bs`, or by replacing the preset
lists with `--tp-batches` and `--ep-batches`. Timing controls are exposed as
`--warmup`, `--iters`, `--nsys-attempts`, and `--nsys-timeout`.

## Output Fields

- `model`: benchmark model shape.
- `bs`: kernel-visible token count on the measured rank.
- `backend`: provider key (`hpcops`, `vllm`, `vllm_cutlass`, `sglang`).
- `median_us`: median latency extracted from NVTX `step` ranges.
- `mean_us`: mean latency of the same samples.
- `n_samples`: number of timed samples.
- `avg_per_group`: average routed tokens per local expert group.
- `error`: provider error for this row, if any.

Default batch sizes:

| Mode | Batch sizes |
|---|---|
| `TP=8 EP=1` | `4 16 32 64 128 256 512 1024 2048 4096 8192 16384` |
| `TP=1 EP=8` | `4 8 16 32 64 128 256 512 1024 2048` |

Default model shapes:

| Model | Experts | topk | Hidden | Intermediate |
|---|---:|---:|---:|---:|
| `qwen3-235b` | 128 | 8 | 4096 | 1536 |
| `hunyuan-v3` | 192 | 8 | 4096 | 1536 |
| `deepseek-v3` | 256 | 8 | 7168 | 2048 |

## Warp decode blockwise FP8

`benchmark_warp_decode.py` measures the automatically selected decode path of
`hpc.fuse_moe_blockwise`. It covers the eight supported Qwen model/TP shapes
at batches 1–8 and uses the local SM90 build.

Run the independent-reference tests before timing:

```bash
python3 -m pytest -q tests/test_fuse_moe_warp_decode.py
python3 benchmark/fused_moe/benchmark_warp_decode.py --list-cases
python3 benchmark/fused_moe/benchmark_warp_decode.py \
  --cases all --batches 1,2,4,8 \
  --seed 20261027 --rounds 5 --iters 100 --calls-per-graph 16 \
  --output warp_decode.json
```

Inputs are prequantized E4M3 with one FP32 scale per row/group of 128.
Weights use the canonical `[E,N,K]` layout and block scales whose last
dimension is padded to a multiple of four. The measured call includes gate/up,
SiLU, intermediate activation quantization, down and expert combination.
Initial activation quantization, weight setup, router scoring, shared experts
and TP communication are excluded; routing uses EP=1.

The script checks finite, deterministic eager outputs and byte equality between
eager and CUDA Graph results. Run the independent-reference tests for numerical
correctness before timing.

Timing uses CUDA Events over repeated rounds. Each
sample contains 100 Graph replays with 16 calls per Graph by default. Weights,
inputs and routing stay fixed during timing, so repeated replay can benefit
from caching. JSON records raw samples, round order, fixture seeds and input
hashes/routing IDs, validation results, software and source hashes; the sibling
CSV contains median latency in microseconds.

## Warp decode per-tensor FP8

`benchmark_warp_decode_pertensor.py` measures the automatically selected decode
path of `hpc.fuse_moe` for Hunyuan-v3 TP8, EP1: hidden 4096, intermediate per
rank 192, 192 experts, top-k 8, batches 1–8.

```bash
python3 -m pytest -q tests/test_fuse_moe_warp_decode_pertensor.py
python3 benchmark/fused_moe/benchmark_warp_decode_pertensor.py \
  --batches 1,2,3,4,5,6,7,8 --seeds 20261028,20261029 \
  --use-bf16-mul both --rounds 7 --iters 100 --calls-per-graph 16 \
  --output warp_decode_hunyuan_pertensor.json
```

Inputs and weights are already E4M3. The existing per-tensor API uses FP32
GEMM multipliers `gate_up_scale[E_local]` and `down_scale[E_local]`, plus a
single activation quantization multiplier `act_and_mul_scale[1]`. Given
activation dequantization steps `x_scale` and `a2_scale`, these are
`x_scale * w1_scale`, `a2_scale * w2_scale`, and `1 / a2_scale`, respectively.
Both `use_bf16_mul` modes retain the existing activation rounding semantics;
down projections round per expert to BF16 before FP32 routing reduction.

The measured call includes gate/up, SiLU, activation quantization, down and
expert reduction. It excludes initial input quantization, weight setup, router
scoring, shared experts and TP communication. Timing reuses fixed weights and
routing in CUDA Graphs. The path must pass an independent reference check,
finite-output, determinism and eager/Graph consistency checks. JSON records
precision metrics, per-round latency, fixture and source hashes; CSV contains
per-seed median latency in microseconds.
