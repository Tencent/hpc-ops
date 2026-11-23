# AMD Triton backend

The AMD backend uses FlagTree's Triton compiler and preserves the public
`hpc.*` / `torch.ops.hpc.*` interfaces. Its kernels use FP32 arithmetic,
reductions, and indexed memory operations without FP8, WMMA, or MFMA.

| Public interface | File | Kernel / coverage |
|---|---|---|
| `rope_norm_store_kv` | `src/rope/amd/rope.py` | `_rope_kernel`: BF16, head dimension 128, Q/KV heads 8/1 and 64/8; prefill, decode, MTP; norm policies 0/1/2; optional Q/K/V outputs, paged cache and tail clearing |
| `reduce` | `src/fuse_moe/amd/reduce.py` | `_reduce_kernel`: BF16 with FP32 accumulation, masked routes, optional shared expert output, K up to 128 |
| `fuse_ihc_pre`, `fuse_ihc_head` | `src/iHC/amd/iHC.py` | `_pre_head_kernel`: four channels, hidden sizes 4096/6144, FP32 projection, optional fused RMSNorm with either rounding policy |
| `fuse_ihc_post` | `src/iHC/amd/iHC.py` | `_post_kernel`: four channels, hidden sizes 4096/6144 |
| `fuse_ihc_post_pre` | `src/iHC/amd/iHC.py` | `_pre_head_kernel` with fused post: BF16 intermediate output, optional RMSNorm with either rounding policy |
| `fused_sampler_temperature_sample` | `src/sampler/amd/sampler.py` | `_temperature_kernel`, `_temperature_finish`: FP32/BF16 logits, temperature, draft mask, Gumbel-max and token tie breaking |
| `fused_sampler` | `src/sampler/amd/sampler.py` | `_prepare_kernel`, `_sample_kernel`: block-local TopK and candidate merge, repetition penalty, temperature, softmax policies, TopP, Gumbel-max and atomic penalty writeback |
| `topk_filtered` (`hpc.topk`) | `src/topk/amd/topk.py` | `_select_kernel` for row-local selection; `_histogram_kernel`, `_refine_kernel`, `_collect_kernel` for parallel exact radix selection; K=512/2048, live rows and ragged lengths |
| `stem_tpd` | `src/stem/amd/stem_tpd.py` | `_stem_tpd_kernel`: BF16 threshold selection, prompt-based budget, chunked prefill and fixed retention patterns |
| `reformat_x_scale` | `src/group_gemm/amd/reformat.py` | `_group_offsets`, `_reformat_kernel`: FP32 grouped transpose and compaction with aligned padding |

Each operator's `src/<operator>/amd/entry.py` owns its schemas, registration,
input validation, output allocation and launch configuration. The kernel files
listed above contain only Triton device functions. Shared validation lives in
`src/utils/amd/validation.py`; `src/C/amd/__init__.py` checks the compiler,
initializes the backend and registers version information. Deferred interfaces
retain their schemas and `TODO(amd)` errors in `src/C/amd/unsupported.py`.
Tests check schema parity against the CUDA registrations.

`setup.py` installs these sources into one private package: `hpc._amd` comes from
`src/C/amd/`, `hpc._amd.<operator>` from each operator's `amd/` directory, and
`hpc._amd.utils` from `src/utils/amd/`. There is no source-directory search or
manual import-path mutation at runtime. Development follows `make triton` and
imports the built package from `build/lib.*`; a standard editable installation
also provides the package mappings.

`hpc/<operator>.py` retains the public API and existing fake implementations, while
`hpc/__init__.py` selects and loads the backend before importing those wrappers.
Python launchers use the input GPU's current stream. Kernels compile for that
device on first use; no native AMD extension is built or loaded.

## Install and use

Use ROCm-enabled PyTorch and an AMD build of
[FlagTree 0.7](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-amd)
with Triton 3.6 and TLE support. FlagTree supplies the `triton` module; ensure
Python imports that module from FlagTree when other Triton distributions coexist.

```bash
make triton
HPC_BACKEND=triton python -m pip install . --no-build-isolation
# To build a wheel instead:
HPC_BACKEND=triton python setup.py bdist_wheel
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
performance in the RDNA3 evaluation. The 6144-wide TLE and fused post/pre paths use 16 waves to
limit register spills. The sampler also offers TLE exclusive prefix sums for TopP.
RoPE, MoE reduction, TopK, scale reformatting, STEM TPD and iHC post use ordinary
Triton in both modes.

```bash
HPC_TRITON_USE_TLE=1 python your_application.py
```

## Deferred interfaces

Unsupported calls raise `NotImplementedError` with a `TODO(amd)` reason:

- FP8 data paths, including the FP8 RoPE sibling.
- Matrix kernels in Attention, GEMM, Group GEMM, STEM, and Fused MoE.
- NVIDIA multicast communication and its communicator.
- Attention decode task scheduling, which remains tied to the deferred attention kernels.

Coverage is per kernel and interface. A file containing FP8 or matrix kernels
does not exclude its independent BF16 or ordinary arithmetic kernels.

## Validation

Validated on Radeon PRO W7900D (`gfx1100`, RDNA3), with ROCm PyTorch 2.13 /
ROCm 7.2 and FlagTree `0.7.0+amd3.6`. Other AMD architectures require their own
runtime validation.

```bash
HIP_VISIBLE_DEVICES=0,1 python -m pytest -q \
  tests/test_amd.py tests/test_amd_portable.py tests/test_iHC.py \
  tests/test_rope.py tests/test_sampler.py tests/test_topk.py tests/test_version.py \
  -k 'not fp8'
HPC_TRITON_USE_TLE=1 python -m pytest -q tests/test_iHC.py tests/test_sampler.py tests/test_amd_portable.py
```

The AMD tests cover both iHC channel extraction paths and RMSNorm rounding
policies, masked routes and odd widths, optional RoPE outputs, padded requests,
cache strides, streams, graph replay, multiple GPUs, import without GPU
initialization, schema parity, and TODO
errors. The existing compute-sanitizer target is NVIDIA-specific. TopK tests that assert
the CUDA split-workspace layout are skipped on AMD; the functional selection,
minimum/recommended scratch, graph replay and counter reuse checks still run.

Reproduce graph latency measurements and reference checks with:

```bash
HIP_VISIBLE_DEVICES=0 python benchmark/amd/benchmark_portable.py --output /tmp/amd-latency.json
```

The benchmark compares iHC fusion with composing the two existing interfaces,
sampling with equivalent PyTorch expressions, and exact TopK with PyTorch's
unordered TopK. Performance varies with batch size and row width.

## Selection and sampling contracts

TopK selects exact FP32 keys, permits any valid subset at a tied boundary,
and leaves inactive rows and unused output slots untouched. Large rows use four
bytewise radix passes across multiple workgroups. The recommended workspace is
bounded by 64 simultaneously processed rows. A minimum-sized buffer selects the
row-local fallback; workspace queries report this backend's own layout. The
three queries are host helpers, not additional GPU kernels. Caller-provided
counters remain zero and scratch contents need no initialization.

STEM TPD includes every finite BF16 value at its selected threshold, so ties can
exceed the row budget. Initial, recent-window and diagonal retention are applied
after threshold selection. Invalid request rows and padded columns are zero.

Sampler vocabulary support remains 120832. External Gumbel noise supports direct
reference comparisons. Internal Philox noise uses a per-device GPU counter,
shared by both sampling interfaces, and advances on graph replay. Its sequence
is backend-specific and does not reproduce cuRAND bit for bit. High counter bits
participate in the RNG key. Concurrent penalty updates use atomic word writes;
partial boundary words are handled without accessing beyond the tensor view.

Per-row temperatures on the temperature-only path must remain positive. Eager
calls validate them before launching; captured graphs retain this input contract
without a device-to-host validation read during capture.
