from enum import IntEnum
from typing import Optional, Tuple, Union

import torch
from torch import Tensor


class SoftmaxPolicy(IntEnum):
    """Where (if anywhere) the kernel should run softmax.

    The fused sampler supports softmax exactly ``0`` or ``1`` times per step:

    * ``NONE``        — no softmax; topk / Gumbel-max operate directly on logits.
    * ``BEFORE_TOPK`` — "joint topk-topp": softmax1 over the full vocab (after
      rep-penalty / temperature), topk operates on probabilities, topp
      is computed directly from the softmax1 output.
    * ``AFTER_TOPK``  — "topk-first": topk operates on logits, then a small
      softmax2 is applied over the surviving top-K; topp operates on softmax2.

    Enabling topp requires ``softmax_policy != NONE`` (probabilities must exist
    somewhere for cumulative-sum truncation to be meaningful). The Python
    wrapper enforces this with a ``TORCH_CHECK`` inside the C++ entry.
    """

    NONE = 0
    BEFORE_TOPK = 1
    AFTER_TOPK = 2


def _to_tensor_scalar_tuple(x) -> Tuple[Optional[Tensor], Union[int, float]]:
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.float:
            return (x, 0.0)
        elif x.dtype == torch.int32 or x.dtype == torch.int64:
            return (x, 0)
        else:
            raise ValueError(f"Unsupported dtype {x.dtype}")
    else:
        return (None, x)


def fused_sampler(
    logits: Tensor,
    *,
    penalty_mask: Optional[Tensor] = None,
    slot_id: Optional[Tensor] = None,
    repetition_penalty: Union[Tensor, float] = 0.0,
    temperature: Union[Tensor, float] = 0.0,
    softmax_policy: SoftmaxPolicy = SoftmaxPolicy.NONE,
    topk: Union[Tensor, int] = 0,
    topp: Union[Tensor, float] = 0.0,
    max_topk: int = 32,
    gumbel_noise: Optional[Tensor] = None,
    draft_token_ids: Optional[Tensor] = None,
    seed: int = 0,
) -> Tensor:
    """2-kernel fused sampler.

    Pipeline inside the kernel::

        repetition_penalty -> temperature -> [softmax1] ->
        topk -> [softmax2] -> topp -> Gumbel-max sampling -> penalty writeback

    Every feature except the final Gumbel-max is optional. At minimum, pass
    just ``logits`` to get a per-batch random sample restricted to the
    ``max_topk`` (32/64) highest-logit candidates (the kernel internally draws
    Gumbel(0) noise via curand). Note this is NOT a full-vocabulary argmax /
    softmax sample: tokens outside the top-``max_topk`` can never be drawn.

    Args:
        logits: Compact ``[B, V]`` tensor, ``float32`` or ``bfloat16``. The
            first dim ``B`` is the effective batch count — rows are not
            gathered via ``slot_id``; only ``penalty_mask`` is.
        penalty_mask: Optional ``[MAX_BS, ceil(V/8)]`` ``uint8`` bit-mask
            (each bit = one vocab token). ``MAX_BS >= slot_id.size(0)``; only
            the rows named by ``slot_id`` are touched. After sampling, the
            kernel sets the bit for the sampled token (``atomicOr``) in
            ``penalty_mask[slot_id[b], token_id[b]]``.
        slot_id: Optional ``[B]`` ``int32`` row index into ``penalty_mask``.
            ``penalty_mask`` and ``slot_id`` must be provided together or not
            at all.
        repetition_penalty: Scalar float or ``[B]`` float32 tensor. ``0`` (or
            no tensor) disables. When enabled, requires ``penalty_mask`` /
            ``slot_id`` to be provided.
        temperature: Scalar float or ``[B]`` float32 tensor. Scalar ``0``
            disables. When a tensor is given, every element must be strictly
            ``> 0`` (the temperature fast-path kernel has no in-kernel ``t<=0``
            guard; a zero/negative entry would produce inf/NaN scores).
        softmax_policy: ``SoftmaxPolicy`` enum. See the enum docstring.
        topk: Scalar int or ``[B]`` int32/int64 tensor. Must be ``<= max_topk``
            per batch. ``0`` (or unset) does NOT widen sampling to the full
            vocab: it just means "do not tighten below ``max_topk``", so
            sampling still happens within the top-``max_topk`` (32/64)
            candidates. Tokens outside that set can never be sampled.
        topp: Scalar float or ``[B]`` float32 tensor. Requires topk to be
            enabled and ``softmax_policy != NONE``. ``0`` disables topp.
        max_topk: Compile-time topk upper bound. Must be ``32`` or ``64``.
        gumbel_noise: Optional ``[B, V]`` ``float32`` Gumbel(0) noise
            (e.g. ``u = torch.rand_like(logits.float()).clamp_min_(1e-20);
            gumbel = -(-u.log()).log()``). Same convention as
            ``fused_sampler_temperature``. The kernel adds it directly to the
            score (``score = value + gumbel_noise``). If provided, sampling is
            bit-reproducible against a PyTorch reference implementation. If
            ``None``, the kernel samples internally via curand
            (non-deterministic).
        draft_token_ids: Optional ``[B]`` ``int64`` tensor (one entry per
            logits row). For each row ``b``: if ``draft_token_ids[b] != -1``,
            the post-temperature logit at ``logits[b, draft_token_ids[b]]`` is
            treated as ``-inf`` for scoring (the original tensor is NOT
            modified). Entries with ``-1`` leave the row unmasked. Currently
            supported only on the temperature-only fast path; if the call
            would otherwise dispatch to the heavy ``fused_sampler`` kernel, a
            ``ValueError`` is raised.

    Returns:
        ``token_ids: Tensor[B, 1] int32``. The sampled token for each row.

    Note:
        The temperature-only fast path (and ``fused_sampler_temperature``) uses
        a per-device shared workspace. Do NOT invoke this sampler concurrently
        on multiple streams of the same device — use one stream per device or
        serialize the calls.
    """
    if isinstance(softmax_policy, int) and not isinstance(softmax_policy, SoftmaxPolicy):
        softmax_policy = SoftmaxPolicy(softmax_policy)

    if max_topk not in (32, 64):
        raise ValueError(f"fused_sampler: max_topk must be 32 or 64, got {max_topk}.")

    # Fast-path: temperature-only → lighter fused_sampler_temperature_sample
    # kernel (skips the cluster-cooperative topk machinery). Detected when every
    # other feature is disabled (mirrors the C++ entry's disabled-flag rules):
    # no penalty_mask/slot_id, rep_penalty/topp/topk scalar 0, softmax_policy
    # NONE, and temperature is a tensor or a positive scalar. gumbel_noise and
    # draft_token_ids are forwarded only on this fast path.
    def _is_scalar_zero(x):
        return (not isinstance(x, Tensor)) and float(x) == 0.0

    _temp_is_tensor = isinstance(temperature, Tensor)
    _temp_is_positive_scalar = (not _temp_is_tensor) and float(temperature) > 0.0
    _fast_path = (
        penalty_mask is None
        and slot_id is None
        and _is_scalar_zero(repetition_penalty)
        and _is_scalar_zero(topp)
        and (not isinstance(topk, Tensor))
        and int(topk) == 0
        and softmax_policy == SoftmaxPolicy.NONE
        and (_temp_is_tensor or _temp_is_positive_scalar)
    )
    if _fast_path:
        temp_tensor, temp_scalar = _to_tensor_scalar_tuple(temperature)
        return torch.ops.hpc.fused_sampler_temperature_sample(
            logits,
            temp_tensor,
            float(temp_scalar),
            gumbel_noise,
            draft_token_ids,
            seed,
        )

    if draft_token_ids is not None:
        raise ValueError(
            "draft_token_ids currently requires the temperature-only fast path. "
            "Disable the other sampler features "
            "(penalty_mask/slot_id/repetition_penalty/topk/topp/softmax_policy) "
            "to use draft-mask sampling."
        )

    return torch.ops.hpc.fused_sampler(
        logits,
        penalty_mask,
        slot_id,
        *_to_tensor_scalar_tuple(repetition_penalty),
        *_to_tensor_scalar_tuple(temperature),
        int(softmax_policy),
        *_to_tensor_scalar_tuple(topk),
        *_to_tensor_scalar_tuple(topp),
        max_topk,
        gumbel_noise,
        seed,
    )
