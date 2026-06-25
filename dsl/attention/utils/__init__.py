"""Cross-variant shared building blocks for CuTeDSL attention kernels.

Modules:
    fmha_helpers: tile scheduler, mask helpers, softmax accumulators.
        Dtype-agnostic — usable across dense/sparse and fp8/bf16/fp16.

This package exists alongside the per-variant subpackages (e.g.
``fp8_prefill_packed_qk_pertoken_v_perhead``) and is intentionally NOT
re-exported by ``dsl.attention.__init__``: callers should reach the
public entry points (``Fp8PagedPrefillConfig``, etc.) via the variant
package, and only the variants themselves should depend on
``utils.fmha_helpers``.
"""
