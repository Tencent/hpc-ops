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
- Comparison methods (`--methods`):
  - `static`: HPC static split-k (`qpertoken_perhead` + `kvpertensor`)
  - `dynamic`: HPC dynamic task map + split-k combine (same quant)
  - `flashinfer`: FlashInfer paged decode FP8 (`qkvpertensor` scales, `block_size=64`)
  - `flashattn`: FlashAttention-3 paged decode FP8 (`qkvpertensor` descales, `block_size=256`)
- Default config: GQA `KV/Q heads=1/8`, `head_dim=128`, HPC `block_size=64`
- CSV speedups: `speedup_vs_static`, `speedup_vs_flashattn`, `speedup_vs_flashinfer` (`baseline / dynamic`)

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
  --methods static dynamic flashinfer flashattn \
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
  - `flashattn`: FlashAttention-3 with KV cache (`block_size=256`)
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

Qwen3.5-9B BF16 decode (`KV/Q heads=4/16`, `head_dim=256`):

```bash
python3 benchmark/attention_decode/bench_attention_decode_bf16.py \
  --num-head-kv 4 \
  --num-head-q 16 \
  --head-dim 256 \
  --methods static dynamic flashinfer flashattn \
  --flashinfer-use-tensor-cores \
  --warmup 10 \
  --iters 100 \
  --check \
  --csv attention_decode_bf16_dim256.csv \
  --jsonl attention_decode_bf16_dim256.jsonl
```

The BF16 benchmark uses `block_size=64` for HPC and FlashInfer. FlashAttention-3 uses
`block_size=256`, its minimum paged-KV block size. Cache allocation, page-table construction,
FlashInfer planning, and CUDA Graph capture are outside the timed region.

The Qwen3.5 command enables FlashInfer's tensor-core GQA path. FA3 uses its split-K heuristic and
automatic GQA packing by default; `--flashattn-num-splits` and `--flashattn-pack-gqa` expose both
knobs for provider tuning. CSV/JSONL rows record these provider settings.

Dynamic timing uses a prebuilt task map by default and therefore measures kernel execution and
split-K combine. Pass `--include-taskmap` to include `assign_attention_decode_task` in the timed
region. CSV/JSONL rows record this boundary, the GPU and provider versions, head configuration,
page sizes, warmup count, iteration count, and all nine scenario names.

For BF16, `--check` compares dynamic and each requested external provider with static using the
same logical Q/K/V data (`atol=0.016`, `rtol=1e-5`). The FlashAttention-3 cache is repacked to its
larger page size before timing.
