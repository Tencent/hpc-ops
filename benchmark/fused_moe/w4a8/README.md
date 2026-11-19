# H20 W4A8 routed MoE

`hpc.fuse_moe_w4a8` runs the complete native SM90 routed expert FFN:
GPU routing, group128 FP8 activation quantization, two W4A8 GEMMs,
SiLU/multiply, and weighted Top-K reduction. It supports E256/H2048/I512,
BF16 input/output, and 1–8 selected experts per token. Router/Top-K selection,
shared experts and communication are outside this operator and benchmark.

```python
import hpc
from hpc.w4a8 import prepare_w4a8_weights

# Once per checkpoint, outside forward and CUDA Graph capture.
w13 = prepare_w4a8_weights(raw_w13.cuda(), raw_s13.cuda())
w2 = prepare_w4a8_weights(raw_w2.cuda(), raw_s2.cuda())
# x: BF16 [M,2048]; ids: int32 [M,K]; probabilities: FP32 [M,K].
y = hpc.fuse_moe_w4a8(x, *w13, *w2, ids, probabilities, output=None)
```

Preparation accepts original E2M1 weights `[E,N,K/2]` and E8M0 group32 scales
`[E,N,K/32]`. Strict conversion requires each expert's exponent span to be at
most 11; it does not requantize weights. It returns packed int32
`[E,K/64,8*N]`, relative uint8 `[E,K/32,N]`, and FP32 `[E]` common scales.
For minimum exponent `emin`, relative exponents are `raw_exp - emin + 1`
and common scales are `2**(emin - 122)`. Pass these scales directly to the
native operator; no additional scale conversion is required.
Packed tensors use a private kernel layout. Regenerate them with the matching
`prepare_w4a8_weights` after updating the kernel; older packed scale caches are
not interchangeable, even when their tensor shapes match.

Forward inputs must be contiguous on the same CUDA device. IDs must be in
`[0,256)`; forward does not read them back to the CPU for validation. Input and
packed weight/relative-scale storage must be 16-byte aligned. An optional output
must be contiguous BF16 `[M,2048]`, 16-byte aligned, and not overlap inputs.
Routing metadata is rebuilt on the GPU, including when values change during
CUDA Graph replay. Runtime and preparation require no installed Humming package.

`bench_moe_e2e.py` compares the complete native operator with external Humming.
`bench_humming_baseline.py` and `common.py` retain the frozen baseline,
reference, timing and error helpers. The control environment is SGLang
`af98fcd545f90eaaf4b26d20bfb42e3ec1d91020` with Humming 0.1.18 (`bb9f595d3`).
Use the built hpc package and pinned SGLang `python/` directory on `PYTHONPATH`:

```bash
python3 -m pytest -q tests/test_w4a8_prepare.py tests/test_fuse_moe_w4a8.py
python3 benchmark/fused_moe/w4a8/bench_moe_e2e.py \
  --fixture /path/to/canonical-e256.pt --output native-moe.json --cold
```

Keep the original fixture SHA256
`50823fdf999ff9250137da24f028af68a7eac3c8e4d1c6289e5d79860182f859`.
The default comparison uses M=1/4/8/16/64/256/1024 and 3 rounds: warm Graphs
contain 4 calls and replay 13 times; cold Graphs contain 1 call and replay
20 times per round after 256 MiB eviction, with eviction excluded from timing.
Compare `event_graph_median` and `cold_graph.median`; eager fields remain in
this frozen harness but are not used for the Graph performance comparison.
Cold-cache timing is an eviction-based approximation. Weight preparation,
fixture loading, JIT and warmup are excluded; each backend computes its own
full intermediate chain. The separate optimization artifacts retain the frozen
Graph-only harness with a native preparation adapter.

The independent reference decodes original MXFP4, emulates group128 E4M3
quantization with `max(amax / 448, 1e-30)`, and uses FP32 dots/SiLU/sum with
BF16 stage boundaries. Every result must be finite with relative L2 below
0.035. Performance acceptance still requires the same paired warm/cold Graph
runs; changing the implementation or preparation API alone establishes no gain.
