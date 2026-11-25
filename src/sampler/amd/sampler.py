# Copyright (C) 2026 Tencent.

import threading

import torch
import triton
import triton.language as tl
from triton.experimental.tle import language as tle

from ..topk.topk import _ordered


# Per-device counter reserving one Philox range per launch, so the same seed
# gives fresh noise. It lives on the GPU, so a captured graph advances on replay.
@triton.jit
def _reserve_rng(STATE, OFFSET, count):
    previous = tl.atomic_add(STATE, count.to(tl.int64), sem="relaxed")
    tl.store(OFFSET, previous)


# Gumbel(0) noise, -log(-log(U)) with the inner log clamped so U -> 1 cannot
# poison the score. The counter's high bits enter the key, the low bits the draw.
@triton.jit
def _noise(NOISE, OFFSET, seed, row, cols, WIDTH: tl.constexpr, EXTERNAL: tl.constexpr):
    index = row.to(tl.int64) * WIDTH + cols
    if EXTERNAL:
        noise = tl.load(NOISE + index)
    else:
        counter = (tl.load(OFFSET) + index).to(tl.uint64)
        key = seed.to(tl.uint64) ^ (counter >> 32)
        uniform = 1.0 - tl.rand(key, counter.to(tl.uint32))
        noise = -tl.log(tl.maximum(-tl.log(uniform), 1e-20))
    return noise


# Temperature-only path: each tile reduces to its best Gumbel score and token,
# and _temperature_finish picks across tiles. Equal scores take the lower token.
@triton.jit
def _temperature_kernel(
    X,
    TEMPERATURE,
    NOISE,
    DRAFT,
    OFFSET,
    SCORES,
    TOKENS,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_TEMP: tl.constexpr,
    EXTERNAL: tl.constexpr,
    HAS_DRAFT: tl.constexpr,
    temperature,
    seed,
):
    tile, row = tl.program_id(0), tl.program_id(1)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    valid = cols < WIDTH
    x = tl.load(X + row.to(tl.int64) * STRIDE + cols, valid, other=float("-inf")).to(tl.float32)
    t = tl.load(TEMPERATURE + row) if HAS_TEMP else temperature
    score = tl.div_rn(x, t)
    if HAS_DRAFT:
        draft = tl.load(DRAFT + row)
        score = tl.where(cols == draft, float("-inf"), score)
    if EXTERNAL:
        noise = tl.load(NOISE + row.to(tl.int64) * WIDTH + cols, valid, other=0)
    else:
        noise = _noise(NOISE, OFFSET, seed, row, cols, WIDTH, False)
    score = tl.where(valid, score + noise, float("-inf"))
    best = tl.max(score, 0)
    token = tl.min(tl.where(valid & (score == best), cols, 2147483647), 0)
    tl.store(SCORES + row * TILES + tile, best)
    tl.store(TOKENS + row * TILES + tile, token)


@triton.jit
def _temperature_finish(SCORES, TOKENS, Y, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    score = tl.load(SCORES + row * TILES + cols, cols < TILES, other=float("-inf"))
    token = tl.load(TOKENS + row * TILES + cols, cols < TILES, other=2147483647)
    best = tl.max(score, 0)
    winner = tl.min(tl.where(score == best, token, 2147483647), 0)
    tl.store(Y + row, winner)


# Pass one: repetition penalty, temperature, this tile's TopK candidates and,
# for the full-vocab softmax policy, its max and exponential sum.
@triton.jit
def _prepare_kernel(
    X,
    CANDIDATES,
    PENALTY,
    SLOTS,
    RP,
    TEMP,
    MAXIMA,
    SUMS,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    PEN_STRIDE: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    K: tl.constexpr,
    HAS_PENALTY: tl.constexpr,
    HAS_RP: tl.constexpr,
    HAS_TEMP: tl.constexpr,
    NEED_SOFTMAX: tl.constexpr,
    rp_value,
    temperature,
):
    tile, row = tl.program_id(0), tl.program_id(1)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    valid = cols < WIDTH
    x = tl.load(X + row.to(tl.int64) * STRIDE + cols, valid, other=float("-inf")).to(tl.float32)
    if HAS_PENALTY:
        slot = tl.load(SLOTS + row)
        bits = tl.load(PENALTY + slot.to(tl.int64) * PEN_STRIDE + cols // 8, valid, other=0)
        seen = (bits & (1 << (cols % 8))) != 0
        rp = tl.load(RP + row) if HAS_RP else rp_value
        adjusted = tl.where(x > 0, x * tl.div_rn(1.0, rp), x * rp)
        x = tl.where(seen & (rp > 0), adjusted, x)
    t = tl.load(TEMP + row) if HAS_TEMP else temperature
    x = tl.where(t > 0, tl.div_rn(x, t), x)
    # Key in the high half, complemented column in the low half: one TopK on the
    # pair orders by value and breaks ties toward the smaller token. 0 loses to all.
    pairs = (_ordered(x).to(tl.uint64) << 32) | (0xFFFFFFFF - cols.to(tl.uint32)).to(tl.uint64)
    pairs = tl.where(valid, pairs, 0)
    candidates = tl.topk(pairs, K)
    tl.store(CANDIDATES + (row * TILES + tile) * K + tl.arange(0, K), candidates)
    if NEED_SOFTMAX:
        maximum = tl.max(x, 0)
        total = tl.sum(tl.where(valid & (maximum != float("-inf")), tl.exp(x - maximum), 0), 0)
        tl.store(MAXIMA + row * TILES + tile, maximum)
        tl.store(SUMS + row * TILES + tile, total)


# Pass two: merge the tiles' candidates, then softmax (policy 1 over the whole
# vocabulary, policy 2 over the candidates), TopP, Gumbel-max and the penalty bit.
@triton.jit
def _sample_kernel(
    CANDIDATES,
    MAXIMA,
    SUMS,
    TOPK,
    TOPP,
    NOISE,
    OFFSET,
    PENALTY,
    SLOTS,
    RP,
    Y,
    WIDTH: tl.constexpr,
    K: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_CANDIDATES: tl.constexpr,
    POLICY: tl.constexpr,
    HAS_K: tl.constexpr,
    HAS_P: tl.constexpr,
    EXTERNAL: tl.constexpr,
    HAS_PENALTY: tl.constexpr,
    HAS_RP: tl.constexpr,
    PEN_STRIDE: tl.constexpr,
    PEN_BYTES: tl.constexpr,
    topk_value,
    topp_value,
    rp_value,
    seed,
):
    row = tl.program_id(0)
    positions = tl.arange(0, K)
    candidate_cols = tl.arange(0, BLOCK_CANDIDATES)
    pairs = tl.load(
        CANDIDATES + row * TILES * K + candidate_cols, candidate_cols < TILES * K, other=0
    )
    pairs = tl.topk(pairs, K)
    tokens = (0xFFFFFFFF - pairs.to(tl.uint32)).to(tl.int32)
    ordered = (pairs >> 32).to(tl.uint32)
    bits = tl.where(ordered & 0x80000000 != 0, ordered ^ 0x80000000, ~ordered)
    values = bits.to(tl.float32, bitcast=True)
    requested = tl.load(TOPK + row).to(tl.int32) if HAS_K else topk_value
    effective = tl.where(requested > 0, tl.minimum(requested, K), K)
    valid = positions < effective
    if POLICY == 1:
        tiles = tl.arange(0, BLOCK_TILES)
        maxima = tl.load(MAXIMA + row * TILES + tiles, tiles < TILES, other=float("-inf"))
        sums = tl.load(SUMS + row * TILES + tiles, tiles < TILES, other=0)
        maximum = tl.max(maxima, 0)
        denominator = tl.sum(tl.where(tiles < TILES, sums * tl.exp(maxima - maximum), 0), 0)
        probabilities = tl.where(
            valid & (denominator > 0), tl.exp(values - maximum) / denominator, 0
        )
    elif POLICY == 2:
        maximum = tl.max(tl.where(valid, values, float("-inf")), 0)
        probabilities = tl.where(valid, tl.exp(values - maximum), 0)
        denominator = tl.sum(probabilities, 0)
        probabilities = tl.where(denominator > 0, probabilities / denominator, 0)
    else:
        probabilities = tl.full((K,), 0, tl.float32)
    # Candidates are ordered, so the exclusive prefix decides TopP; the first one
    # is always kept, which keeps the set non-empty for any threshold.
    exclusive, _ = tle.cumsum(probabilities, 0)
    topp = tl.load(TOPP + row) if HAS_P else topp_value
    keep = valid & ((topp <= 0) | (positions == 0) | (exclusive < topp))
    if POLICY != 0:
        values = tl.where(probabilities > 0, tl.log(probabilities), float("-inf"))
    score = values + _noise(NOISE, OFFSET, seed, row, tokens, WIDTH, EXTERNAL)
    score = tl.where(keep, score, float("-inf"))
    maximum_score = tl.max(score, 0)
    token = tl.min(tl.where(keep & (score == maximum_score), tokens, 2147483647), 0)
    token = tl.where(token == 2147483647, 0, token)
    tl.store(Y + row, token)
    # The bit is set with one atomic OR on its aligned 32-bit word. A word that
    # crosses either end of the tensor is skipped here and left to _penalty_edges.
    if HAS_PENALTY:
        rp = tl.load(RP + row) if HAS_RP else rp_value
        slot = tl.load(SLOTS + row)
        byte = slot.to(tl.int64) * PEN_STRIDE + token // 8
        base = PENALTY.to(tl.uint64)
        absolute = base + byte.to(tl.uint64)
        word = absolute & ~tl.full((), 3, tl.uint64)
        bits = (1 << (token % 8)).to(tl.uint32) << ((absolute % 4).to(tl.uint32) * 8)
        address = word.to(tl.pointer_type(tl.uint32))
        inside = (word >= base) & (word + 4 <= base + PEN_BYTES)
        tl.atomic_or(address, bits, (rp > 0) & inside, sem="relaxed")


# Head and tail bytes of an unaligned mask, folded across rows in one workgroup
# so each partial word is read and written once without touching its neighbours.
@triton.jit
def _penalty_edges(
    PENALTY,
    SLOTS,
    TOKENS,
    RP,
    BATCH: tl.constexpr,
    BLOCK: tl.constexpr,
    PEN_STRIDE: tl.constexpr,
    PEN_BYTES: tl.constexpr,
    HEAD_BYTES: tl.constexpr,
    TAIL_START: tl.constexpr,
    HAS_RP: tl.constexpr,
    rp_value,
):
    rows = tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + rows, rows < BATCH, other=-1)
    token = tl.load(TOKENS + rows, rows < BATCH, other=0)
    rp = tl.load(RP + rows, rows < BATCH, other=0) if HAS_RP else rp_value
    byte = slot.to(tl.int64) * PEN_STRIDE + token // 8
    bits = (1 << (token % 8)).to(tl.uint32)
    for index in tl.static_range(HEAD_BYTES + PEN_BYTES - TAIL_START):
        address = index if index < HEAD_BYTES else TAIL_START + index - HEAD_BYTES
        additions = tl.where((rows < BATCH) & (rp > 0) & (byte == address), bits, 0)
        combined = tl.reduce(additions, 0, _or)
        previous = tl.load(PENALTY + address).to(tl.uint32)
        tl.store(PENALTY + address, previous | combined)


@triton.jit
def _or(first, second):
    return first | second


_BLOCK_SIZE = 4096
_rng_states = {}
_rng_lock = threading.Lock()


def reserve_rng_async(reference, external):
    if external:
        return None
    device = reference.device
    with torch.cuda.device(device):
        with _rng_lock:
            if device not in _rng_states:
                state = torch.zeros((), device=device, dtype=torch.int64)
                ready = torch.cuda.Event()
                ready.record()
                _rng_states[device] = state, ready
            state, ready = _rng_states[device]
        torch.cuda.current_stream(device).wait_event(ready)
        offset = torch.empty((), device=device, dtype=torch.int64)
        _reserve_rng[(1,)](state, offset, reference.numel(), num_warps=1)
    return offset


def sampler_workspace_shape(logits):
    return logits.shape[0], triton.cdiv(logits.shape[1], _BLOCK_SIZE)


def temperature_sample_async(
    logits,
    temperature,
    temperature_val,
    gumbel_noise,
    draft_token_ids,
    seed,
    offset,
    scores,
    tokens,
    output,
):
    batch, width = logits.shape
    if not batch:
        return
    with torch.cuda.device(logits.device):
        tiles = triton.cdiv(width, _BLOCK_SIZE)
        _temperature_kernel[(tiles, batch)](
            logits,
            temperature,
            gumbel_noise,
            draft_token_ids,
            offset,
            scores,
            tokens,
            width,
            logits.stride(0),
            tiles,
            _BLOCK_SIZE,
            temperature is not None,
            gumbel_noise is not None,
            draft_token_ids is not None,
            temperature_val,
            seed,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _temperature_finish[(batch,)](
            scores, tokens, output, tiles, triton.next_power_of_2(tiles), num_warps=1
        )


def fused_sampler_async(
    logits,
    penalty_mask,
    slot_id,
    repetition_penalty,
    repetition_penalty_val,
    temperature,
    temperature_val,
    softmax_policy,
    topk,
    topk_val,
    topp,
    topp_val,
    max_topk,
    gumbel_noise,
    seed,
    offset,
    candidates,
    maxima,
    sums,
    output,
):
    batch, width = logits.shape
    if not batch:
        return
    with torch.cuda.device(logits.device):
        tiles = triton.cdiv(width, _BLOCK_SIZE)
        pen_stride = penalty_mask.shape[1] if penalty_mask is not None else 0
        pen_bytes = penalty_mask.numel() if penalty_mask is not None else 0
        _prepare_kernel[(tiles, batch)](
            logits,
            candidates,
            penalty_mask,
            slot_id,
            repetition_penalty,
            temperature,
            maxima,
            sums,
            width,
            logits.stride(0),
            pen_stride,
            tiles,
            _BLOCK_SIZE,
            max_topk,
            penalty_mask is not None,
            repetition_penalty is not None,
            temperature is not None,
            softmax_policy == 1,
            repetition_penalty_val,
            temperature_val,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _sample_kernel[(batch,)](
            candidates,
            maxima,
            sums,
            topk,
            topp,
            gumbel_noise,
            offset,
            penalty_mask,
            slot_id,
            repetition_penalty,
            output,
            width,
            max_topk,
            tiles,
            triton.next_power_of_2(tiles),
            triton.next_power_of_2(tiles * max_topk),
            softmax_policy,
            topk is not None,
            topp is not None,
            gumbel_noise is not None,
            penalty_mask is not None,
            repetition_penalty is not None,
            pen_stride,
            pen_bytes,
            topk_val,
            topp_val,
            repetition_penalty_val,
            seed,
            num_warps=4,
            enable_fp_fusion=False,
        )
        alignment = penalty_mask.data_ptr() % 4 if penalty_mask is not None else 0
        head_bytes = (4 - alignment) % 4
        tail_start = pen_bytes - (alignment + pen_bytes) % 4
        if penalty_mask is not None and (head_bytes or tail_start < pen_bytes):
            _penalty_edges[(1,)](
                penalty_mask,
                slot_id,
                output,
                repetition_penalty,
                batch,
                triton.next_power_of_2(batch),
                pen_stride,
                pen_bytes,
                head_bytes,
                tail_start,
                repetition_penalty is not None,
                repetition_penalty_val,
                num_warps=4,
            )
