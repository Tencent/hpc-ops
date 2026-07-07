# RoPE + Norm + Store KV Benchmark

This directory benchmarks RoPE + optional QK norm + KV cache store paths.

## Figure Mapping

- Operator (HPC backend, BF16): `hpc.rope_norm_store_kv`
- Operator (HPC backend, FP8): `hpc.rope_norm_store_kv_fp8`
- Operator (FlashInfer backend, BF16 path): `apply_rope_with_cos_sin_cache_inplace + append_paged_kv_cache`
- Operator (FlashInfer backend, FP8 path): `rope_quantize_fp8_append_paged_kv_cache`
- Hardware expectation:
  - `--backend hpc`: SM90 class GPU (H100/H20)
  - `--backend flashinfer`: non-SM90 GPUs are supported by FlashInfer stack (for example SM89/SM120)
  - `--backend auto` (default): picks `hpc` on SM90 (if importable), otherwise picks `flashinfer`
- Default dtype:
  - BF16 when `--fp8` is not set
  - FP8 path when `--fp8` is set
- Default head config: `--heads 64,8,128` (`num_q_heads,num_kv_heads,qk_head_dim`)
- Default request sweep: `--num-req 16 64 256`
- Default mode sweep: `--modes prefill,decode0,decode1`
- Default norm sweep: `--norm-policies 0,1,2`
- Default quant sweep (FP8 only): `--quant-policies 1,2`
- Timing mode: CUDA event timing with median latency in microseconds
- Default samples: `--warmup 10 --iters 50`

## Recommended Reproduction Commands

Run from repository root:

```bash
cd benchmark/rope_norm_store_kv

# Auto backend selection (recommended first run)
python3 bench_rope_norm_store_kv.py \
  --backend auto \
  --num-req 16 64 256 \
  --heads 64,8,128
```

Force FlashInfer path (recommended for non-SM90 GPUs):

```bash
python3 bench_rope_norm_store_kv.py \
  --backend flashinfer \
  --modes prefill,decode0,decode1 \
  --num-req 16 64 256 \
  --heads 64,8,128
```

FP8 sweep:

```bash
python3 bench_rope_norm_store_kv.py \
  --backend flashinfer \
  --fp8 \
  --quant-policies 1,2 \
  --num-req 16 64 \
  --heads 64,8,128
```

## Profiling Notes

Torch profiler trace:

```bash
python3 bench_rope_norm_store_kv.py \
  --backend flashinfer \
  --profile \
  --profile-trace-dir ./rope_profile_trace
```

Then open the trace with TensorBoard:

```bash
tensorboard --logdir ./rope_profile_trace
```

NVTX range annotation (for Nsight Systems / Nsight Compute filtering):

```bash
python3 bench_rope_norm_store_kv.py --backend flashinfer --nvtx
```

Example Nsight Systems command:

```bash
nsys profile -t cuda,nvtx,osrt -o rope_kv_report \
  python3 bench_rope_norm_store_kv.py --backend flashinfer --nvtx --iters 100 --warmup 20
```

## Output Fields

Printed table columns:

- `mode`: `prefill` or `dec(mtp=0/1)`
- `num_req`: request count in the batch
- `norm`: `qk_norm_policy`
  - `0`: no norm
  - `1`: RoPE then RMSNorm
  - `2`: RMSNorm then RoPE
- `quant`: FP8 quant policy (`1`/`2`) or `None` in BF16 mode
- `us`: median latency per call in microseconds

## Compatibility Notes

- If runtime reports `no kernel image is available for execution on the device`, the selected backend does not have a matching kernel image for your GPU architecture.
- For non-SM90 GPUs, prefer `--backend flashinfer`.
- `--backend hpc` requires a compatible SM90 build of HPC kernels.

## Troubleshooting

- `backend=hpc requested but hpc package is not available`
  - Install/build `hpc-ops` extension in the current environment.
- `backend=flashinfer requested but flashinfer package is not available`
  - Install FlashInfer in the current environment.
- Empty/invalid `--modes`
  - Use a subset of: `prefill,decode0,decode1`.

