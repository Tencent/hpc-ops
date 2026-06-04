# Attention Decode FP8 Benchmark

This directory contains the benchmark entry used to reproduce the dynamic scheduling results for Attention Decode FP8.

## Figure Mapping

- Operator: SM90 Attention Decode FP8
- Quantization modes:
  - `qkpertoken_perhead_vperhead`
  - `qpertoken_perhead_kvpertensor`
- Comparison: static split-k vs dynamic task map
- Timing unit: microseconds per operator call
- Timing modes:
  - `--timing event`: quick CUDA event timing around CUDA Graph replay.
  - `--timing nsys`: release-style timing aligned with FusedMoE, using `nsys`, NVTX `step`, CUDA Graph replay, and median latency.
- Default config: GQA `KV/Q heads=1/8`, `head_dim=128`, `block_size=64`

## Scenario Names

- `uniform_512`: `64x512`
- `uniform_4096`: `64x4K`
- `skewed_mix`: `32x128+32x4K`
- `skewed_extreme`: `1x16K+15x64`
- `one_64k_7x4k`, `one_64k_15x4k`, `one_64k_31x4k`: `1x64K+7/15/31x4K`
- `one_128k_31x4k`: `1x128K+31x4K`
- `two_32k_30x4k`: `2x32K+30x4K`

`AxB` means `A` decode requests with KV length `B`; `AxB+CxD` means mixed KV lengths in the same batch.

## Reproduction Commands

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
