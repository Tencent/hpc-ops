"""CuTeDSL attention kernels.

Per-variant modules live alongside this file (e.g.
``fp8_prefill_packed_qk_pertoken_v_perhead``, each currently a single
``.py`` file but may grow into a sub-package as variants get more
complex). Each variant module re-exports its entry function and config
class through this aggregator so wrappers in ``hpc/attention.py`` import
via:

    from dsl.attention import (
        Fp8PagedPrefillConfig,
        attention_prefill_fp8_cutedsl,
    )

Cross-variant shared building blocks (TMA helpers, tile scheduler, mask
utilities) live under the ``utils`` subpackage and are intentionally NOT
re-exported here — variants depend on ``utils.*`` directly.
"""

from .fp8_prefill_packed_qk_pertoken_v_perhead import (
    DEFAULT_P_SCALE,
    Fp8PagedPrefillConfig,
    Fp8PagedPrefillCuteDSL,
    attention_prefill_fp8_cutedsl,
)
from .fp8_prefill_packed_q_pertoken_kv_pertensor import (
    Fp8PagedPrefillKvPertensorConfig,
    Fp8PagedPrefillKvPertensorCuteDSL,
    attention_prefill_fp8_cutedsl_q_pertoken_kv_pertensor,
)

__all__ = [
    "DEFAULT_P_SCALE",
    "Fp8PagedPrefillConfig",
    "Fp8PagedPrefillCuteDSL",
    "attention_prefill_fp8_cutedsl",
    "Fp8PagedPrefillKvPertensorConfig",
    "Fp8PagedPrefillKvPertensorCuteDSL",
    "attention_prefill_fp8_cutedsl_q_pertoken_kv_pertensor",
]
