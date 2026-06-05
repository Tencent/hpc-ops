# HPC-Ops Benchmarks

This directory contains standalone benchmarks of Route GEMM.

## Route GEMM BF16xFP32

- Script: `benchmark_gemm_bf16xfp32.py`
- Operator: HPC-Ops BF16xFP32 GEMM compared with `FP32(cuBLAS)` and `TF32(cuBLAS)`
- Default shape sweep: `N=192`, `K=4096`, `M=2,4,8,16,48,96,208,512,1024,2048,4096`
- Hardware expectation: NVIDIA SM90/H20 class GPU

Recommended command:

```bash
python3 benchmarks/benchmark_gemm_bf16xfp32.py \
  --n 192 \
  --k 4096 \
  --m-list 2,4,8,16,48,96,208,512,1024,2048,4096 \
  --csv route_gemm_bf16xfp32.csv \
  --jsonl route_gemm_bf16xfp32.jsonl
```

The benchmark prints a latency table and records accuracy metrics against an FP64 reference. The machine-readable output includes provider latency, max absolute error, mean absolute error, and speedup of `hpc-ops-bf16xfp32` relative to the cuBLAS baselines.
