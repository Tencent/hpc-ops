# Fused AllReduce + RMSNorm Benchmark

This directory reproduces the AllReduce + Residual + RMSNorm latency.

## Figure Mapping

- Operator: `RMSNorm(AllReduce(x) + residual, weight)`
- Hardware expectation: single 8-GPU SM90/H20 node with NVLink/NVSwitch
- Default dtype: BF16
- Default hidden size: `7168`
- Supported hidden sizes: `4096`, `5120`, `7168`
- Timing mode: CUDA Graph replay with per-step median latency by default
- Default samples: `--warmup 5 --iters 50 --rounds 3`

The underlying HPC-Ops kernels enforce:

```cpp
TORCH_CHECK(hidden_size == 4096 || hidden_size == 5120 || hidden_size == 7168,
            "unsupported hidden_size");
```

Passing any other `--hidden` value is rejected by the benchmark before workers
are launched, so users see the supported shape list directly instead of a lower
level kernel error.

## Recommended Reproduction Command

Run from the repository root:

```bash
cd benchmark/fuse_allreduce_rmsorm/
python3 bench_allreduce_rmsnorm.py \
  --hidden 7168 \
  --tokens 8 32 128 512 4096 8192 16384 32768 \
  --fi-backend mnnvl \
  --csv allreduce_rmsnorm.csv \
  --jsonl allreduce_rmsnorm.jsonl
```

The benchmark spawns 8 local worker processes itself, so `torchrun` is not required.
The default timing path is aligned with the FusedMoE replay methodology at the
CUDA Graph level: warmup, graph capture, replay warmup, then per-step median
latency. Each measured graph replay is preceded by a rank-level synchronize and
barrier. This keeps collective kernels in lockstep so peer launch jitter is not
counted as device latency. The benchmark repeats timing for several rounds and
reports the best round median, while printing all round medians in the log. This
reduces sensitivity to occasional OS scheduling or fabric noise. Use `--no-graph`
for eager event timing. Nsight Systems profiling is not enabled by default because
this benchmark launches 8 local collective worker processes.

If a provider is not explicitly listed in `--skip` but fails to import, initialize,
or run in the current environment, the benchmark prints a warning and skips that
provider instead of aborting the whole sweep. This is useful when FlashInfer or
NCCL/HPC-Ops dependencies are not available in a local reproduction environment.

## Output Fields

- `hpc_ops_ht_us`: latency of `fuse_allreduce_rmsnorm_high_throughout`
- `hpc_ops_ll_us`: latency of `fuse_allreduce_rmsnorm_low_latency`
- `nccl_us`: NCCL AllReduce + fused add/RMSNorm baseline
- `flashinfer_us`: FlashInfer baseline, if available
- `hpc_best_us`: `min(hpc_ops_ht_us, hpc_ops_ll_us)`
- `baseline_best_us`: `min(nccl_us, flashinfer_us)`
- `hpc_best_speedup`: `baseline_best_us / hpc_best_us`

Use `--no-check` to skip correctness checks.
