"""Top-level CuTeDSL kernel package.

Pure-Python CuTeDSL kernel implementations live here, sibling to the ``hpc``
package rather than under it. Kernels are JIT-compiled by ``cutlass`` at
runtime and do not depend on the ``hpc`` C++ extension (``_C_sm*.abi3.so``).

Public entry points are still surfaced through ``hpc`` wrappers (e.g.
``hpc.attention_with_kvcache_prefill_fp8_packed_cutedsl`` in
``hpc/attention.py``), which import the kernels from here on demand:

    from dsl.attention import (
        Fp8PagedPrefillConfig,
        attention_prefill_fp8_cutedsl,
    )
"""
