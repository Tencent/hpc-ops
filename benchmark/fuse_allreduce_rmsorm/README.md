# Fused AllReduce + RMSNorm Benchmark

This directory reproduces the AllReduce + Residual + RMSNorm latency data used by the open-source performance figures.

## Figure Mapping

- Operator: `RMSNorm(AllReduce(x) + residual, weight)`
- Hardware expectation: single 8-GPU SM90/H20 node with NVLink/NVSwitch
- Default dtype: BF16
- Default hidden size: `7168`
- Timing mode: CUDA Graph replay with per-step median latency by default

## Recommended Reproduction Command

Run from the repository root:

```bash
python3 benchmark/fuse_allreduce_rmsorm/bench_allreduce_rmsnorm.py \
  --hidden 7168 \
  --tokens 8 32 128 512 4096 8192 16384 32768 \
  --fi-backend mnnvl \
  --csv allreduce_rmsnorm.csv \
  --jsonl allreduce_rmsnorm.jsonl
```

The benchmark spawns 8 local worker processes itself, so `torchrun` is not required.
The default timing path is aligned with the FusedMoE replay methodology at the
CUDA Graph level: warmup, graph capture, replay warmup, then per-step median
latency. Use `--no-graph` for eager event timing. An experimental
`--timing nsys` path is available, but Nsight Systems can be unstable with this
8-process collective worker in some environments.

## Output Fields

- `hpc_ops_ht_us`: latency of `fuse_allreduce_rmsnorm_high_throughout`
- `hpc_ops_ll_us`: latency of `fuse_allreduce_rmsnorm_low_latency`
- `nccl_us`: NCCL AllReduce + fused add/RMSNorm baseline
- `flashinfer_us`: FlashInfer baseline, if available
- `hpc_best_us`: `min(hpc_ops_ht_us, hpc_ops_ll_us)`
- `baseline_best_us`: `min(nccl_us, flashinfer_us)`
- `hpc_best_speedup`: `baseline_best_us / hpc_best_us`

Use `--no-check` to skip correctness checks.
