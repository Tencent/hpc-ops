# hpc-ops

## Overview

hpc-ops is a CUDA operator library for LLM inference. Python wrappers expose
operators through `torch.ops.hpc.*`; the package contains one extension module
per SM architecture it was built for, and loads the one matching the device.

The usual execution path is:

```text
hpc/<op>.py
  -> torch.ops.hpc.<name>
  -> src/<op>/entry.cc          validation, allocation, launch config
  -> HPC_ARCH_DISPATCH          at the launch site
  -> src/<op>/sm<arch>/<kernel>.cu
  -> CUDA kernel
```

Keep these boundaries intact:

- Python wrappers define the user-facing API and fake implementation.
- `entry.cc` registers operators, validates inputs, allocates
  outputs/workspaces, chooses launch configuration, and dispatches to the
  architecture at the point where it launches. There is no separate dispatch
  layer and no `host.cc`.
- `.cu` and `.cuh` files contain kernels and device-side implementation.
- Architecture-specific code lives in `sm<arch>/`; code that works on every
  supported architecture lives at the operator root. Different architectures may
  use entirely different algorithms below the common `*_async` signature.

Architecture handling has two stages:

1. CMake is configured once per architecture and compiles only the sources that
   architecture selects into `_C_sm<arch>.abi3.so`.
2. `hpc/__init__.py` detects the device at import time and loads the matching
   module; `entry.cc` then confirms the match before launching.

Dispatch is exact. Architecture-specific cubins such as `sm_90a`, `sm_100a`, and
`sm_103a` are not assumed to be forward compatible, so a module is never used on
an architecture it was not built for.

## Typical development workflow

This file is a repository-development guide. End-user installation and operator
examples belong in `README.md`; here, "usage" means building, testing, debugging,
and locating the code to modify.

Use the shortest command that covers the change:

```bash
make                         # build a module for the local GPU architecture
make sm90                    # wheel for sm90 only
SM_ARCH=90,100 make          # build two modules, sm90 and sm100
make test                    # run the test suite
python3 -m pytest -v tests/test_<op>.py
make sanitizer               # run tests under compute-sanitizer checks
make format                  # format Python and C++/CUDA sources
make format-check            # check formatting and lint
```

For a normal operator change:

1. Start at `hpc/<op>.py` and its corresponding `tests/test_<op>.py`.
2. Follow the registered name into `src/<op>/entry.cc`.
3. Read its `HPC_ARCH_DISPATCH` pairs to see which architectures it dispatches
   to, and which directory the target architecture selects.
4. Make validation and launch-configuration changes in `entry.cc`; make device
   changes in `.cu`/`.cuh`.
5. Build the smallest relevant architecture set and run the focused test.
6. Run formatting checks; use `make sanitizer` for synchronization, bounds, or
   race-sensitive kernel changes.

`make test` is not fully self-contained on every machine: the communicator and
allreduce tests need several processes and can hang on a single-GPU or otherwise
restricted setup. Prefer focused `pytest` runs while iterating.

To inspect a built package:

```python
import hpc

print(hpc.__loaded_arch__)   # architecture of the module in use
print(hpc.__built_json__)    # version, git hash, sm-arch, toolchain versions
```

## Repository layout

```text
hpc/<op>.py                       Python wrapper(s); one file may expose several operators
hpc/__init__.py                   module loading, operator discovery and re-export
src/<op>/entry.cc                 registration, host logic, and the dispatch at each launch site
src/<op>/<kernel>.h               *_async declarations: the cross-architecture contract
src/<op>/sm90/<kernel>.cu         sm90 kernels and the sm90 *_async definitions
src/<op>/sm90/<kernel>.cuh        kernel templates and inline device helpers
src/<op>/sm90/<variant>/<kernel>.cu   an alternative sm90 implementation path
src/<op>/sm100/...                an independent architecture implementation
src/<op>/<kernel>.{cu,cuh}        kernels that work on every architecture, at the operator root
tests/test_<op>.py                correctness and API tests
benchmark/                        performance reproduction and comparison
```

Only two things justify a directory level: `sm<N>/` marks architecture-specific
code, and a named subdirectory of the operator root marks a feature with its own
`*_async` contract (`attention/decode/`). Everything else — including a variant
implementation strategy — is a file name, or a directory below the `sm<arch>/`
that chose it. Whether code is architecture-specific is therefore always readable
off the path: it is, exactly when the path contains an `sm<N>/` component.

There is no `entry.h`: nothing outside `entry.cc` calls the host entry point, and
the signature the architectures share is the `*_async` one in the header at the
operator root.

Architecture-agnostic code that is not an operator implementation stays at the
top level:

- `src/utils/`: shared headers and host helpers;
- `src/C/`: Python extension module glue;
- `src/communicator/`: host-side IPC without kernels.

Do not infer the current operator catalog, or any operator's architecture
coverage, from this guide. Both change as operators are added and ported;
discover them from `hpc/`, `src/`, and `tests/` — the directory tree and the
`HPC_ARCH_DISPATCH` pairs are the authority. Names used below are examples of
structure or dispatch rules only, and a wrapper file name need not match its
operator directory.

### Large operators with sub-parts

Large operators may give each feature its own subdirectory, each selecting its
architecture independently, while keeping one `entry.cc` at the operator root:

```text
src/attention/entry.cc
src/attention/prefill/prefill.h        prefill's *_async contract
src/attention/prefill/sm90/*.cu
src/attention/decode/decode.h          decode's *_async contract
src/attention/decode/sm90/*.cu
```

The single `entry.cc` holds the host code for every group and gives each group's
launch sites their own `HPC_ARCH_DISPATCH` pairs, so one feature can gain an
architecture while another stays behind.

A subdirectory of the operator root is a *feature* split: it exists because the
two halves have separate `*_async` contracts and can reach different
architectures. Do not create one for anything else — in particular, a header at
the operator root belongs directly at the root, never in a subdirectory that
merely mirrors the layout of some `sm<arch>/` tree below it.

### Alternative implementations within one architecture

An architecture may hold several functionally equivalent implementations that
take different paths. Those get a subdirectory *inside* `sm<arch>/`, because what
distinguishes them is internal to that architecture:

```text
src/fuse_moe/fuse_moe.h              the default path's *_async contract
src/fuse_moe/fuse_moe_cp_async.h     the cp.async path's *_async contract
src/fuse_moe/sm90/*.cu               sm90, default path
src/fuse_moe/sm90/cp_async/*.cu      sm90, cp.async path
```

Both headers sit at the operator root: they are the cross-architecture contract,
and another architecture implementing the cp.async path would implement that same
signature. The variant name appears in the file name, not in a directory level of
its own.

The rule that decides this: a path component says either "this is
architecture-specific" (`sm<N>/`) or "this is a feature with its own contract"
(`decode/`). An implementation strategy is neither, so it lives below the
`sm<arch>/` that chose it.

### Several operators in one directory

One Python file and one `entry.cc` may expose several related operators, and one
`entry.cc` may hold several namespaces. Their kernels can also be split into
feature subdirectories. Do not assume a one-to-one relationship between a file, a
registered operator, and a kernel.

## Architecture source selection

A source file belongs to a module when its path says so. For the architecture
being built:

1. no `sm<N>/` component in the path — shared code, always compiled;
2. `sm<TARGET>/` — this architecture's own code;
3. `sm<FALLBACK>/` — only where `sm<TARGET>/` does not exist beside it (the
   configured source fallback, see below);
4. any other `sm<N>/` — another architecture's code, not compiled.

Rule 4 is what makes the directory layout the visibility boundary: in an
`sm<TARGET>` module, another architecture's `src/<op>/sm<N>/` is not merely
unreachable, it was never compiled in.

Consequently, shared code must not include a header from an `sm<N>/` directory,
and one `sm<N>/` must not include another. Put genuinely shared code in
`src/utils/` or at the operator root, or keep the architecture implementations
separate.

Nothing in the build enforces this. Source selection decides which `.cu`/`.cc`
are compiled, not which headers are reachable: every `sm<N>/` header is on disk
in every build and the repository root is on the include path, so a shared source
that includes a header-only `sm<N>/` header compiles and links cleanly into the
wrong architecture's module, emitting that architecture's device code under
another's gencode. `-Wl,--no-undefined` catches only a missing *definition*,
which such an include does not produce. Check the rule by reading the path.

### Source fallback

Source compatibility does not imply binary compatibility, and a successful
compile does not imply a working kernel. `HPC_ARCH_SOURCE_FALLBACK_<arch>` in
`CMakeLists.txt` names the architecture an absent `sm<arch>/` inherits from; the
inherited sources are recompiled with the target's own gencode. A real
`sm<arch>/` directory always takes precedence, per operator subdirectory, so an
operator can be ported one part at a time.

Adding a link is a claim about hardware, and it has to be tested on that
hardware. A tree written for one architecture can compile cleanly with another's
gencode and then fail at launch — wgmma shapes, TMA descriptors and cluster sizes
valid on one generation are not on the next. "It compiles" is the weakest
possible evidence here, so a link belongs in the table only between architectures
that share an instruction set, and only after the inherited kernels have run on
the new hardware. Read `CMakeLists.txt` for the links that currently exist.

An implementation that is not portable to every known architecture lives in
`sm<arch>/` for each architecture it supports; an architecture no operator
directory covers raises "not implemented" when that operator is called. Where a
fallback link applies, an operator with the source architecture's directory but
not the target's is compiled into the target module with the target's own
gencode. Duplicating it into a directory of its own would be two copies to keep
in step for no gain — the fallback is what the link is for. `entry.cc` still
needs the `HPC_ARCH_DISPATCH` pair for the target: the fallback decides which
sources compile, not which architectures dispatch.

## Architecture boundaries

A module contains exactly one architecture, so implementations need no
architecture namespace. `hpc::gemm::gemm_bf16xfp32_async` means "the
implementation in this module", and which one that is was decided at compile time
by the directory the source came from.

`HPC_TARGET_ARCH` is defined for every source, so a shared implementation can
branch on small architecture differences. Use separate `sm<arch>/` trees when the
implementations substantially differ.

The only cross-architecture contract for an operator is the `*_async` signature in
the header at the operator root, e.g. `src/gemm/gemm.h`. Anything else the host
needs before launching but only an architecture can answer — a tile shape that
decides a workspace extent, a scheduling plan — goes in that same header as its
own declaration, answered per architecture. Headers inside `sm<arch>/` are private
to that architecture and may differ.

### Cross-operator calls

Cross-operator calls need no architecture qualification:

```cpp
// Inside hpc::fuse_moe, in src/fuse_moe/sm90/fuse_moe.cu:
activation::act_mul_and_quant_async(...);
```

But caller and callee must be buildable for the same architectures: the sm90
`fuse_moe` above needs `activation` in that module, whether from `sm90/` or
from shared code at the operator root. This dependency is caught by the linker
rather than declared manually; see "Unresolved symbols".

## Registration and dispatch

`src/utils/utils.h` and `src/utils/utils.cc` implement the architecture layer:
runtime architecture detection, the dispatch macros, and the
unsupported-architecture error.

Declare the `*_async` contract at the operator root, once — every architecture
implements the same signature in its own `sm<arch>/`:

The `widget` operator below is invented for this section; it is not in the tree.

```cpp
// src/widget/widget.h
namespace hpc {
namespace widget {
bool widget_async(void *y_ptr, const void *x_ptr, int m, int n, float scale,
                  cudaStream_t stream);
}  // namespace widget
}  // namespace hpc
```

Then put the host logic and the registration in `entry.cc`, dispatching where it
launches:

```cpp
torch::Tensor widget_entry(const torch::Tensor &x, double scale) {
  // validation, output allocation and launch configuration: shared by every
  // architecture, so it is written once and compiled into every module.
  TORCH_CHECK(x.is_contiguous(), "x tensor must be contiguous");
  torch::Tensor y = torch::empty({m, n}, x.options());

  bool launched = false;
  HPC_ARCH_DISPATCH("widget",
      90, launched = widget_async(y_ptr, x_ptr, m, n, scale, stream),
      100, launched = widget_async(y_ptr, x_ptr, m, n, scale, stream));

  TORCH_CHECK(launched, "widget launch failed!");
  return y;
}

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def("widget(Tensor x, float scale) -> Tensor");
  m.impl("widget", torch::kCUDA, &hpc::widget::widget_entry);
}
```

The two pairs say the operator is implemented for sm90 and sm100 and nothing
else. A real operator's pairs are whatever its `sm<arch>/` directories support.

Each architecture repeats the call expression rather than sharing one, because
the architectures may not call the same `*_async` — the pair is `<arch>, <expr>`,
and the expr is free to differ.

`HPC_ARCH_DISPATCH` is one chain: if nothing launched, it raises before any host
code after it runs. A function with launches in separate branches gets one
`HPC_ARCH_DISPATCH` per branch — each is a complete decision on its own.

An operator whose kernels compile for every architecture in `KNOWN_ARCHS` has
nothing to choose between: its `*_async` lives at the operator root and every
module has it, so `entry.cc` calls the async directly. Such an operator has no
`sm<N>/` directory at all, which is how you recognise one from the tree.

Only the pair matching `HPC_TARGET_ARCH` expands to a call; the others expand to
an unevaluated `sizeof`, so a module never references an implementation it did
not compile, and the shared host code above the chain does not collect
unused-variable warnings in the builds that dispatch elsewhere. The architecture
numbers in the `HPC_ARCH_DISPATCH` call are the operator's declared support
matrix, sitting where they can be read and reviewed rather than inferred from
the directory tree.

### Error semantics

Registration is unconditional within a loaded module: importing `hpc` and looking
up `torch.ops.hpc.<name>` works for every operator the module registers, even one
with no implementation for the device. That failure surfaces when the operator is
called:

```text
hpc::gemm_bf16xfp32 does not run on sm103: implemented for 90,
loaded module built for sm103.
```

Rebuilding cannot fix this — an implementation has to be added, or the device is
not one the operator supports. The same error covers a module loaded through
`HPC_SM_ARCH` that the device does not match.

A missing module is the other failure, and it is reported at import:

```text
hpc has no operator module for this sm103 device; this install covers [90].
Rebuild including it, e.g. 'SM_ARCH=103 make'.
```

`import hpc` raises rather than degrading, because everything past the import
depends on the module: the operator schemas come from it, so
`torch.library.register_fake` in the wrappers has nothing to attach to without
it. Use `torch.library.register_fake` directly in wrappers — no indirection is
needed, since the schema is always registered by the time a wrapper is imported.

### Unresolved symbols

The architecture lists in `entry.cc` are written by hand, and a wrong one is not
a compile error: the shared host code compiles fine, and only the link discovers
that the implementation is absent. `-Wl,--no-undefined` in `CMakeLists.txt` makes
that a link failure naming the symbol and the line that referenced it:

```text
src/gemm/entry.cc:142: undefined reference to
  `hpc::gemm::gemm_bf16xfp32_async(void*, ...)'
```

Without it a MODULE library would link happily and fail at import with nothing
but "could not load this library". When this fires, check:

1. does `src/<op>/sm<arch>/` exist for the architecture being built? If not, the
   `HPC_ARCH_DISPATCH` pair for that architecture must go, so the error says
   "does not run on sm<arch>" instead;
2. is the call inside a dispatch at all? A bare call in `entry.cc` is compiled
   into every module, whether or not the callee was built;
3. does every subdirectory of the operator cover that architecture?
4. is the callee of every cross-operator call built for it?

## Building and packaging

`SM_ARCH` selects target architectures. Without it, `make` uses the local GPU
architectures, read from `nvidia-smi` in `setup.py`, while `make wheel` covers
every architecture in `KNOWN_ARCHS` — a wheel is a release artifact, so it
defaults to the full set rather than to whatever GPU built it. Each architecture
produces its own `_C_sm<arch>.abi3.so`; several of them ship in one package.

```bash
make                              # module for local architectures (dev)
SM_ARCH=90,100 make               # two modules, sm90 and sm100
make wheel                        # wheel covering every known architecture
make sm90                         # wheel for sm90 only
SM_ARCH=90,100,103 make wheel     # one wheel, three modules
```

`SM_ARCH` from the environment or the command line overrides the `wheel` default,
which is how `make sm<arch>` narrows it to one architecture.

The architecture list lives in three places that must agree: `KNOWN_ARCHS` in the
`Makefile` (which derives the `make sm<arch>` targets and the `make wheel`
default), `HPC_KNOWN_ARCHS` in `CMakeLists.txt` (which rejects an unknown
`HPC_TARGET_ARCH`), and `KNOWN_ARCHS` in `setup.py` (which validates `SM_ARCH`
and decides the wheel suffix). An unknown architecture is rejected by `setup.py`
before any build starts, so a partial package is never produced.

CMake is configured once per architecture, into `build/sm<arch>`, so switching
targets reuses prior objects. The setuptools `build/lib.*` output contains the
modules from the latest build, which tests import; modules from an earlier build
with a different `SM_ARCH` are removed, so the loaded module always belongs to
the current one.

Other useful targets:

```bash
make test
make sanitizer
make clean
```

### Module selection at import

`hpc/__init__.py` detects the device architecture with
`torch.cuda.get_device_capability()` and loads `_C_sm<arch>.abi3.so` for exactly
that number. A single-module install is no exception: an sm90 module on an sm103
machine is the wrong module, and reporting that at import time beats crashing
inside a kernel launch.

`HPC_SM_ARCH` overrides the detection, for a build machine or a CPU-only run that
needs a module without a matching GPU. It is also the only way to load a module
the device does not match, which `src/utils/utils.cc` reports if a dispatched
operator is then called.

### Wheel architecture suffix

Wheels covering only part of `KNOWN_ARCHS` carry a sorted local version suffix
such as `.sm90` or `.sm90.103`. A wheel covering every known architecture omits
the suffix. Keep the naming logic in `setup.py` derived from `KNOWN_ARCHS`;
examples in documentation must not become a second source of truth.

## Modifying the repository

### Add an operator

1. Add the Python wrapper in `hpc/<op>.py`, with `torch.library.register_fake`
   for each operator that needs a fake implementation.
2. Add `src/<op>/<kernel>.h` with the `*_async` declarations — the signature every
   architecture will implement.
3. Add `src/<op>/entry.cc`: the host logic, the `TORCH_LIBRARY_FRAGMENT`, and at
   each launch site one `HPC_ARCH_DISPATCH` with a pair per architecture you
   implement. If the kernels compile for every known architecture, call the
   async directly.
4. Implement the kernels: at the operator root if the code works on every
   architecture in `KNOWN_ARCHS`, in `sm<arch>/` otherwise — one directory per
   architecture it supports.
5. Add focused correctness and fake-tensor tests under `tests/`.
6. Build relevant architectures, run focused tests, then format checks.

No CMake source-list edit is normally needed because sources are discovered from
the directory structure.

### Modify an existing operator

First classify the change:

- **Python API or schema:** update the wrapper, fake implementation, `m.def`, the
  host code in `entry.cc`, and tests.
- **Validation, allocation, or launch policy:** it is shared across architectures,
  so it lives in `entry.cc`. Only move something into `sm<arch>/` when the
  architectures genuinely disagree about it.
- **Kernel implementation:** update only the architecture directories that should
  change. Do not assume siblings share internals.
- **Portable behavior:** keep it at the operator root only when it really works
  on every architecture in `KNOWN_ARCHS`.
- **Architecture-specific behavior:** add or modify `sm<arch>/`; avoid growing
  large `HPC_TARGET_ARCH` branches in shared code.
- **Architecture support:** the `HPC_ARCH_DISPATCH` pairs in `entry.cc` change
  together with the directories.
- **Cross-operator call:** verify that caller and callee cover the same
  architectures.

When changing a shared signature, search every implementation before editing.
Build at least the architectures whose source selection differs; a single local
build may not compile all changed paths.

### Add architecture-specific support

For an operator already present:

1. Determine whether the shared implementation at the operator root covers the
   architecture. If it does, only the `entry.cc` lists need it.
2. If not, create `src/<op>/sm<arch>/` implementing the `*_async` declarations
   from the operator root.
3. Preserve those signatures; they are the cross-architecture contract.
4. Add an `HPC_ARCH_DISPATCH` pair for `<arch>` at every launch site in `entry.cc`.
5. Build with `SM_ARCH=<arch>` and run focused tests.
6. When the operator has sub-parts, verify every subdirectory covers it, or the
   link fails.

### Add a new architecture to the build system

1. Add the architecture number to `KNOWN_ARCHS` in the `Makefile`, `setup.py`,
   and `HPC_KNOWN_ARCHS` in `CMakeLists.txt`.
2. Add an `HPC_ARCH_EVAL_<arch>` block in `src/utils/utils.h`.
3. If required by its instruction set, add the architecture to
   `HPC_ARCH_SPECIFIC_ARCHS` in `CMakeLists.txt` for the `a` gencode suffix.
4. Add a source fallback link only after running the inherited kernels on the new
   hardware. Compiling is not evidence; see "Source fallback".
5. Add an `sm<arch>/` directory for every operator that should support it, and
   extend that operator's `entry.cc` lists. An operator whose shared code at the
   root already covers the architecture only needs the `entry.cc` lines.
6. Build it and a multi-architecture set, then check import, module selection,
   dispatch errors, and tests.

## Style and repository conventions

- Run `make format` before committing; `make format-check` must pass.
- Python uses Black with a 100-column limit.
- C++ and CUDA use the repository `.clang-format` and cpplint configuration.
- Project includes are relative to the repository root:
  `#include "src/utils/utils.cuh"`.
- Include guards mirror paths. For example, `src/group_gemm/sm90/config.h` uses
  `SRC_GROUP_GEMM_SM90_CONFIG_H_`.
- Keep `entry.cc` to registration, host logic and dispatch.
- Do not include across an architecture boundary: shared code must not reach into
  `sm<arch>/`, and one `sm<arch>/` must not reach into another.
- Keep tests focused on public behavior; use benchmarks for performance claims.
- `3rd/cutlass` is vendored, not a submodule. Refresh it with
  `3rd/update-cutlass.sh`.
