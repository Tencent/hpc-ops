# AMD HIP backend

The first batch targets ordinary GPU multiply/add, reductions, and indexed memory
access. Kernels use no FP8, WMMA, or MFMA instructions. They were run on Radeon PRO
W7900D (`gfx1100`, RDNA3), using ROCm PyTorch 2.13 / ROCm 7.2. Other AMD devices
require building for their own target and running the tests; portability of the
source does not establish validation on another device.

| Public interface | HIP file | Kernel / coverage |
|---|---|---|
| `rope_norm_store_kv` | `src/rope/amd/rope.hip` | `rope_norm_store_kv_kernel`: BF16, head dimension 128, Q/KV heads 8/1 and 64/8; prefill, decode, MTP; norm policies 0/1/2; optional output buffers and KV tail clearing |
| `reduce` | `src/fuse_moe/amd/reduce.hip` | `reduce_kernel`: BF16 with FP32 accumulation, masked routes, optional shared expert output, K up to 128 |
| `fuse_ihc_pre`, `fuse_ihc_head` | `src/iHC/amd/iHC.hip` | `fuse_ihc_pre_or_head_kernel`: four channels, hidden sizes 4096/6144, FP32 projection, optional fused RMSNorm |
| `fuse_ihc_post` | `src/iHC/amd/iHC.hip` | `fuse_ihc_post_kernel`: four channels, hidden sizes 4096/6144 |

Each operator's `amd/entry.cc` validates tensors, selects the input GPU/current
stream, and registers the existing public schema. Private `amd/*.h` declarations
avoid importing the CUDA headers' NVIDIA types. Device helpers in
`src/utils/amd/device.cuh` support wave32 and wave64 without a hardcoded wave size.
Ordinary stream ordering replaces CUDA PDL.

## Build and use

Use a ROCm-enabled PyTorch environment with a HIP compiler and CMake >= 3.26:

```bash
HPC_HIP_ARCH=gfx1100 make hip
# Install a wheel built from this checkout:
HPC_BACKEND=hip HPC_HIP_ARCH=gfx1100 python setup.py bdist_wheel
python -m pip install dist/*hip*.whl
```

Plain `make` detects HIP from `torch.version.hip`. `HPC_HIP_ARCH` accepts a
comma/semicolon-separated target list; when omitted, setup detects visible AMD
devices. `HPC_BUILD_JOBS` controls build parallelism (default 4).

The HIP module is `hpc/_C_hip.abi3.so`. `import hpc` loads it without creating a GPU
context and reports `hpc.__backend__ == "hip"`. Public calls still use `cuda`
devices, following PyTorch's ROCm API. CUDA builds retain their SM-specific modules.

## Deferred interfaces

`src/C/amd/unsupported.cc` preserves the other operator schemas and records a
`TODO(amd)` for every deferred entry. Calls raise `NotImplementedError` with a
reason. This includes:

- FP8 data paths, including the FP8 RoPE sibling in the same operator.
- Matrix kernels in Attention, GEMM, Group GEMM, STEM, and Fused MoE.
- NVIDIA multicast communication and its communicator.
- Non-FP8 kernels deferred from this batch: iHC post/pre fusion, sampler, TopK,
  STEM TPD, scale reformatting, and task scheduling. These are not excluded by
  their file names or by belonging to an FP8 pipeline.

An implemented interface can use fewer or different kernels than its CUDA
counterpart, provided its numerical behavior and output contract are preserved.

## Validation

```bash
python -m pytest -q tests/test_amd.py
python -m pytest -q tests/test_iHC.py tests/test_rope.py -k 'not post_pre and not fp8'
```

The focused tests include FP32 reference comparisons, both iHC RMSNorm rounding
policies, masked routes and odd widths, optional RoPE outputs, padded requests,
stream/graph replay, multiple GPUs, fake tensors, schema parity, and explicit TODO
errors. The existing compute-sanitizer target is NVIDIA-specific and does not
validate HIP kernels. Performance tuning and cross-device benchmarking remain
future work.
