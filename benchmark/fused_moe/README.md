# FusedMoE Benchmark

Benchmark FusedMoE kernels across HPC-Ops and optional comparison backends.

## Overview

- Operator: SM90 FusedMoE
- Precision (`--dtype`):
  - `fp8` (default): per-tensor FP8 path. Providers: `HPC-Ops`, `vLLM Triton`, `vLLM CUTLASS`, `SGLang`.
  - `bf16`: bf16 path. Compared only against `SGLang` (Triton, unquantized). Providers restricted to `HPC-Ops` and `SGLang`.
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

bf16 (HPC-Ops vs SGLang only):

```bash
python3 benchmark/fused_moe/benchmark_fuse_moe.py \
  --dtype bf16 \
  --tp 8 --ep 1 \
  --csv fused_moe_bf16_tp8_ep1.csv \
  --jsonl fused_moe_bf16_tp8_ep1.jsonl
```

With `--dtype bf16`, providers default to `hpcops sglang`. Passing any other
provider (e.g. `vllm`) is rejected. The bf16 SGLang path runs the Triton
FusedMoE kernels with unquantized bf16 weights and activations.

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
- `dtype`: precision of the measured path (`fp8` or `bf16`).
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
