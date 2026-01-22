# HPC-Ops
HPC-Ops is a **production-grade, high-performance, and easy-to-use** operator library for LLM inference, developed by the Tencent Hunyuan AI Infra team.

## Key Features
- **SOTA Performance & Production-Proven**: Deeply optimized kernels tailored for NVIDIA H20 GPUs, delivering SOTA performance with up to 2.22x speedup. Powering large-scale production inference in Tencent.
- **Easy to Integrate**: A clean API designed for seamless integration into popular inference frameworks like vLLM and SGLang.
- **Rich Precision Support**: Native support for multiple data types including BF16 and FP8 with different quantization schemes.
- **A Modern CUDA Tutorial**: Hands-on examples of building SOTA kernels with CuTe and CUTLASS in just hundreds of lines.

## Performance

### Maximum observed speedup per operator

| Operator         | Baseline                           | Prefill      | Decode |
|:----------------:|:----------------------------------:|:------------:|:------:|
| Attention (bf16) | FlashInfer, FA2, FA3, TensorRT-LLM | 1.33x        | 2.22x  |
| Attention (fp8)  | FlashInfer, FA3, TensorRT-LLM      | 1.12x        | 2.0x   |
| FusedMoE (fp8)   | TensorRT-LLM, vLLM                 | 1.49x        | 1.14x  |
| GroupGEMM (fp8)  | DeepGEMM                           | 1.1x         | 1.88x  |

*We focus on maximum speedup to highlight the optimization potential, as performance varies substantially across cases.*


## Supporting Kernels

### Attention
- Decode, Prefill: Optimized kernels for all attention phases, including paged attention.

### Grouped GEMM
- Quantized Grouped GEMM: FP8 weights with block-wise or per-tensor scaling

### Fused MoE
- Quantized Fused MoE: FP8 expert weights with block-wise or per-tensor scaling

## Quick Start

### Requirements
- NVIDIA SM90 architecture GPU
- Python 3.8 or higher
- Compilers with C++17 support
- CUDA Toolkit: CUDA 12.3 or higher

*You can set up the environment by installing the modules listed in requirements-dev.txt.*

### Install from Source
```bash
git clone https://github.com/Tencent/hpc-ops.git
cd hpc-ops

# build packages
make wheel
python3 -m pip install dist/*.whl   
```

### Basic Usage
Example: GroupGEMM fp8 kernel usage
```python
import torch
import hpc

num_tokens = 1024
num_group, n, k = 8, 4096, 4096
x = torch.randn((num_tokens, k), dtype=torch.float, device="cuda").to(torch.float8_e4m3fn)
w = torch.randn((num_group, n, k), dtype=torch.float, device="cuda").to(torch.float8_e4m3fn)
scale = torch.full((num_group,), 1.0, dtype=torch.float, device="cuda")
num_tokens_per_group = torch.full((num_group,), 8, dtype=torch.int32, device="cuda")
cu_num_tokens_per_group = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), num_tokens_per_group]), dim=0).to(torch.int32)

output = hpc.group_gemm_pertensor_fp8(
    x, w, num_tokens_per_group, cu_num_tokens_per_group, scale,
)
```

*For the usage of other operators, please refer to the corresponding test files in the tests/ directory.*

## Roadmap

- **Sparse Attention Kernels**: Optimized for long-context LLMs, these kernels boost throughput for memory-bound workloads.
- **Extended Quantization Support**: Flexible strategies (4bit/8bit mixed-precision included)  kernel optimizations for quantized attention and GEMM which balance speed and accuracy.
- **Compute-Communication Boundary-Breaking Kernels**: Overlapped computation and inter-GPU communication logic to  minimizes overhead in multi-node/multi-GPU distributed inference.

We welcome targeted, high-impact contributions—whether it’s fixing edge-case kernel bugs, or submitting optimizations for niche LLM inference scenarios, your PRs will help refine this toolkit for production use.

⭐ **Star this repo** to follow our progress.
We’re continuously improving performance to make your LLM inference faster and more efficient.
More improvements are on the way.
