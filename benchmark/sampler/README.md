# Sampler Benchmark

Benchmark HPC-Ops sampler kernels against PyTorch/vLLM-style and FlashInfer paths.

## Overview

- Operator: HPC-Ops Sampler
- Providers:
  - `HPC-Ops`
  - `vLLM/PyTorch`
  - `FlashInfer`
- Timing unit: microseconds per sampler call
- Timing modes:
  - `--timing nsys`: release-style timing using `nsys`, NVTX ranges, eager sampler calls, and median latency from NVTX GPU projected duration. The worker does not synchronize inside each timed step.
  - `--timing event`: quick CUDA event timing with eager sampler calls.
- Default config: vocab size `120832`, BF16 logits, batch sizes `1,8,16,32,64,128,256,512`.

## Scenarios

- `temperature`: `Temperature Sampling`, temperature-only fast path.
- `full`: `Full Sampling`, repetition penalty + temperature + topk/topp + sampling.

## Usage

Full sweep:

```bash
python3 benchmark_sampler.py \
  --timing nsys \
  --output-dir sampler_nsys \
  --csv sampler_latency.csv \
  --jsonl sampler_latency.jsonl
```

Smoke test:

```bash
python3 benchmark_sampler.py \
  --timing event \
  --scenes temperature full \
  --batches 1 8 \
  --providers hpc torch \
  --warmup 1 \
  --iters 3
```

## Notes

If FlashInfer is unavailable or its API is incompatible with the local environment, the benchmark records the provider error and continues with the remaining providers.
