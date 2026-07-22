# Attention Decode Benchmark

This directory contains the benchmark entries used to reproduce the dynamic scheduling results for Attention Decode FP8 and BF16.

## Scenario Names

Shared by FP8 and BF16:

- `uniform_512`: `64x512`
- `uniform_4096`: `64x4K`
- `skewed_mix`: `32x128+32x4K`
- `skewed_extreme`: `1x16K+15x64`
- `one_64k_7x4k`, `one_64k_15x4k`, `one_64k_31x4k`: `1x64K+7/15/31x4K`
- `one_128k_31x4k`: `1x128K+31x4K`
- `two_32k_30x4k`: `2x32K+30x4K`

`AxB` means `A` decode requests with KV length `B`; `AxB+CxD` means mixed KV lengths in the same batch.

## Timing Modes

Both benches support:

- `--timing event`: quick CUDA event timing around CUDA Graph replay.
- `--timing nsys`: release-style timing aligned with FusedMoE, using `nsys`, NVTX `step`, CUDA Graph replay, and median latency.

Latency is reported in microseconds per operator call.

## FP8

- Operator: SM90 Attention Decode FP8
- Quantization modes:
  - `qkpertoken_perhead_vperhead`
  - `qpertoken_perhead_kvpertensor`
- Comparison: static split-k vs dynamic task map
- Default config: GQA `KV/Q heads=1/8`, `head_dim=128`, `block_size=64`

Full sweep with the FusedMoE-aligned `nsys` timing path:

```bash
python3 benchmark/attention_decode/bench_attention_decode_fp8.py \
  --timing nsys \
  --output-dir attention_decode_nsys \
  --csv attention_decode_fp8.csv \
  --jsonl attention_decode_fp8.jsonl
```

Fast smoke test with CUDA event timing:

```bash
python3 benchmark/attention_decode/bench_attention_decode_fp8.py \
  --cases uniform_512 skewed_extreme \
  --quant-types qpertoken_perhead_kvpertensor \
  --warmup 1 \
  --iters 3
```

Enable correctness comparison between static and dynamic paths with `--check`.

## BF16

- Operator: SM90 Attention Decode BF16
- Comparison methods (`--methods`):
  - `static`: HPC static split-k
  - `dynamic`: HPC dynamic task map
  - `flashinfer`: FlashInfer paged decode (`block_size=64`)
  - `flashattn`: FlashAttention with KV cache (`block_size=256`)
- Default config: GQA `KV/Q heads=1/8`, `head_dim=128`, HPC `block_size=64`
- CSV speedups: `speedup_vs_static`, `speedup_vs_flashattn`, `speedup_vs_flashinfer` (`baseline / dynamic`)

Full sweep with the FusedMoE-aligned `nsys` timing path:

```bash
python3 benchmark/attention_decode/bench_attention_decode_bf16.py \
  --timing nsys \
  --output-dir attention_decode_nsys \
  --csv attention_decode_bf16.csv \
  --jsonl attention_decode_bf16.jsonl
```

Fast smoke test with CUDA event timing:

```bash
python3 benchmark/attention_decode/bench_attention_decode_bf16.py \
  --cases uniform_512 skewed_extreme \
  --methods static dynamic flashinfer flashattn \
  --warmup 1 \
  --iters 3
```

Enable correctness comparison between static and dynamic paths with `--check`.
