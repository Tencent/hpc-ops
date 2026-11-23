# AMD Triton backend

The AMD backend uses FlagTree's Triton compiler and preserves the public
`hpc.*` / `torch.ops.hpc.*` interfaces. Its kernels use FP32 arithmetic,
reductions, and indexed memory operations without FP8, WMMA, or MFMA.

| Public interface | File | Kernel / coverage |
|---|---|---|
| `rope_norm_store_kv` | `hpc/_triton/rope.py` | `_rope_kernel`: BF16, head dimension 128, Q/KV heads 8/1 and 64/8; prefill, decode, MTP; norm policies 0/1/2; optional Q/K/V outputs, paged cache and tail clearing |
| `reduce` | `hpc/_triton/moe.py` | `_reduce_kernel`: BF16 with FP32 accumulation, masked routes, optional shared expert output, K up to 128 |
| `fuse_ihc_pre`, `fuse_ihc_head` | `hpc/_triton/ihc.py` | `_pre_head_kernel`: four channels, hidden sizes 4096/6144, FP32 projection, optional fused RMSNorm with either rounding policy |
| `fuse_ihc_post` | `hpc/_triton/ihc.py` | `_post_kernel`: four channels, hidden sizes 4096/6144 |

Python launchers validate inputs and use the input GPU's current stream. Kernels
compile for that device on first use; no native AMD extension is built or loaded.
`hpc/_triton/_schemas.py` preserves all operator schemas and records a `TODO(amd)`
for every deferred entry. Tests check schema parity against the CUDA registrations.

## Install and use

Use ROCm-enabled PyTorch and an AMD build of
[FlagTree 0.7](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-amd)
with Triton 3.6 and TLE support. FlagTree supplies the `triton` module; ensure
Python imports that module from FlagTree when other Triton distributions coexist.

```bash
make triton
HPC_BACKEND=triton python setup.py bdist_wheel
python -m pip install dist/*triton*.whl
```

Plain `make` selects Triton on ROCm PyTorch. `make hip` and `HPC_BACKEND=hip`
are compatibility aliases. AMD wheels contain Python sources and declare the
FlagTree dependency; they need no CMake, HIP compiler, or fixed GPU architecture.
CUDA builds retain the SM-specific native modules.

`import hpc` registers operators without creating a GPU context and reports
`hpc.__backend__ == "triton"`. Public calls use `cuda` devices, following
PyTorch's ROCm API. `hpc.__built_json__` reports the backend and compiler versions.

## Optional TLE path

`HPC_TRITON_USE_TLE=1` selects `tle.extract_tile` for channel extraction in iHC
pre/head. The default uses ordinary Triton: TLE did not consistently improve
performance in the RDNA3 evaluation. The 6144-wide TLE path uses 16 waves to
limit register spills. RoPE, MoE reduction and iHC post use ordinary Triton in
both modes.

```bash
HPC_TRITON_USE_TLE=1 python your_application.py
```

## Deferred interfaces

Unsupported calls raise `NotImplementedError` with a `TODO(amd)` reason:

- FP8 data paths, including the FP8 RoPE sibling.
- Matrix kernels in Attention, GEMM, Group GEMM, STEM, and Fused MoE.
- NVIDIA multicast communication and its communicator.
- Other kernels awaiting a port: iHC post/pre fusion, sampler, TopK, STEM TPD,
  scale reformatting, and task scheduling.

Coverage is per kernel and interface. A file containing FP8 or matrix kernels
does not exclude its independent BF16 or ordinary arithmetic kernels.

## Validation

Validated on Radeon PRO W7900D (`gfx1100`, RDNA3), with ROCm PyTorch 2.13 /
ROCm 7.2 and FlagTree `0.7.0+amd3.6`. Other AMD architectures require their own
runtime validation.

```bash
HIP_VISIBLE_DEVICES=0,1 python -m pytest -q \
  tests/test_amd.py tests/test_iHC.py tests/test_rope.py tests/test_version.py \
  -k 'not post_pre and not fp8'
HPC_TRITON_USE_TLE=1 python -m pytest -q tests/test_iHC.py -k 'not post_pre'
```

The AMD tests cover both iHC channel extraction paths and RMSNorm rounding
policies, masked routes and odd widths, optional RoPE outputs, padded requests,
cache strides, streams, graph replay, multiple GPUs, fake tensors,
`torch.compile`, import without GPU initialization, schema parity, and TODO
errors. The existing compute-sanitizer target is NVIDIA-specific.
