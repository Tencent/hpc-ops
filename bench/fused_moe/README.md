# FusedMoE Benchmark

This directory contains a per-tensor FP8 FusedMoE benchmark for HPC-Ops,
vLLM Triton, vLLM CUTLASS, and SGLang.

## Requirements

- NVIDIA GPU with FP8 support.
- CUDA, PyTorch, Triton, NumPy, `nvtx`, and `nsys`.
- Built HPC-Ops, vLLM, and SGLang checkouts.

Set checkout roots before running:

```bash
export HPCOPS_ROOT=/path/to/hpc-ops
export VLLM_ROOT=/path/to/vllm
export SGLANG_ROOT=/path/to/sglang
```

## Usage

Run TP mode:

```bash
python3 bench.py \
  --tp 8 --ep 1 \
  --gpu 0 \
  --backends hpcops vllm vllm_cutlass sglang
```

Run EP mode:

```bash
python3 bench.py \
  --tp 1 --ep 8 \
  --gpu 0 \
  --backends hpcops vllm vllm_cutlass sglang
```

Run a smaller smoke test:

```bash
python3 bench.py \
  --tp 8 --ep 1 \
  --models qwen3-235b \
  --bs 16 32 \
  --backends hpcops vllm_cutlass \
  --gpu 0
```

By default, outputs are written under `./log/<tag>/`. Override this with:

```bash
python3 bench.py --output-dir /path/to/output ...
```

## Defaults

Models:

| Model | Experts | topk | Hidden | Intermediate |
|---|---:|---:|---:|---:|
| `qwen3-235b` | 128 | 8 | 4096 | 1536 |
| `hunyuan-v3` | 192 | 8 | 4096 | 1536 |
| `deepseek-v3` | 256 | 8 | 7168 | 2048 |

Shape semantics:

- `bs` is the kernel-visible sequence count on the measured rank.
- `TP` partitions the intermediate dimension only, so `intermediate_per_rank = intermediate / TP`.
- `EP` partitions experts, so `experts_per_rank = experts / EP`.
- The reported `avg/group` is `bs * topk / experts_per_rank`.

For `TP=8 EP=1`, experts are not partitioned and the benchmark keeps the full expert set
visible to the measured rank:

```text
avg/group = bs * topk / experts
```

For `TP=1 EP=8`, the benchmark measures one EP rank with local experts only. Routing is
sampled within that local expert set, so:

```text
experts_per_rank = experts / 8
avg/group = bs * topk / experts_per_rank
```

The EP batch range is shorter than the TP range to cover the same per-rank operator regime at
comparable `avg/group` values.

Batch sizes:

| Mode | Batch sizes |
|---|---|
| `TP=8 EP=1` | `4 16 32 64 128 256 512 1024 2048 4096 8192 16384` |
| `TP=1 EP=8` | `4 8 16 32 64 128 256 512 1024 2048` |
