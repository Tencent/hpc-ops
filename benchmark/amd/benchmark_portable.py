# Copyright (C) 2026 Tencent.

"""Graph latency and reference checks for the portable AMD operators."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from triton.testing import do_bench_cudagraph

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(next(_root.glob("build/lib.*"), _root)))

import hpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.version.hip:
        parser.error("ROCm PyTorch is required")
    torch.manual_seed(42)
    report = {
        "device": torch.cuda.get_device_name(),
        "build": json.loads(hpc.__built_json__),
        "tle": os.getenv("HPC_TRITON_USE_TLE", "0") == "1",
        "cases": [],
    }

    def measure(name, function, reference=None, check=None):
        actual = function()
        if check is not None:
            check(actual)
        elif reference is not None:
            exact = isinstance(actual, torch.Tensor) and not actual.is_floating_point()
            torch.testing.assert_close(
                actual, reference(), atol=0 if exact else 3e-2, rtol=0 if exact else 1e-2
            )
        result = {"case": name, "triton_us": do_bench_cudagraph(function, rep=args.rep_ms) * 1000}
        if reference is not None:
            result["reference_us"] = do_bench_cudagraph(reference, rep=args.rep_ms) * 1000
        report["cases"].append(result)
        print(json.dumps(result), flush=True)

    for batch, hidden in ((1, 4096), (128, 4096), (128, 6144)):
        x = torch.randn((batch, hidden), device="cuda").bfloat16()
        residual = torch.randn((batch, 4, hidden), device="cuda").bfloat16()
        gates = torch.rand((batch, 4), device="cuda")
        w = torch.randn((8, 4 * hidden), device="cuda") * 0.02
        scale, base = torch.ones(2, device="cuda"), torch.zeros(8, device="cuda")

        def fused():
            return hpc.fuse_ihc_post_pre(x, residual, gates, w, scale, base)

        def composed():
            middle = hpc.fuse_ihc_post(x, residual, gates)
            y, post = hpc.fuse_ihc_pre(middle, w, scale, base)
            return middle, y, post

        measure(f"ihc_post_pre_B{batch}_D{hidden}", fused, composed)

    for batch in (1, 128):
        logits = torch.randn((batch, 120832), device="cuda")
        noise = -torch.log(-torch.log(torch.rand_like(logits).clamp_min_(1e-20)))

        def torch_sample():
            values, indices = torch.topk(logits, 32, dim=-1)
            winner = (values + noise.gather(1, indices)).argmax(-1, keepdim=True)
            return indices.gather(1, winner).int()

        measure(
            f"sampler_B{batch}_V120832_K32",
            lambda: hpc.fused_sampler(logits, topk=32, gumbel_noise=noise),
            torch_sample,
        )
        measure(
            f"temperature_B{batch}_V120832",
            lambda: hpc.fused_sampler(logits, temperature=0.7, gumbel_noise=noise),
            lambda: (logits / 0.7 + noise).argmax(-1, keepdim=True).int(),
        )

    for batch in (1, 8, 128):
        values = torch.randn((batch, 65536), device="cuda")
        lengths = torch.full((batch,), values.shape[1], device="cuda", dtype=torch.int32)
        live = torch.tensor([batch], device="cuda", dtype=torch.int32)
        output = torch.empty((batch, 2048), device="cuda", dtype=torch.int32)

        def selection_check(indices):
            selected = values.gather(1, indices.long()).sort().values
            expected = values.topk(2048).values.sort().values
            torch.testing.assert_close(selected, expected, atol=0, rtol=0)

        measure(
            f"topk_B{batch}_N65536_K2048",
            lambda: hpc.topk(values, lengths, output, live),
            lambda: torch.topk(values, 2048, sorted=False),
            selection_check,
        )

    xscale = torch.randn((256 * 1280, 32), device="cuda")
    lengths = torch.full((256,), 30, device="cuda", dtype=torch.int32)
    cu = torch.arange(257, device="cuda", dtype=torch.int32) * 1280
    reformatted = torch.empty((32, xscale.shape[0]), device="cuda")
    expected = xscale.reshape(256, 1280, 32)[:, :30].transpose(1, 2)

    def reformat_check(output):
        actual = output[:, : 256 * 32].reshape(32, 256, 32).permute(1, 0, 2)[:, :, :30]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    measure(
        "reformat_G256_capacity1280_D32_live30",
        lambda: hpc.reformat_x_scale(xscale, lengths, cu, 30, reformatted),
        check=reformat_check,
    )
    scores = torch.randn((1, 4, 3, 1024), device="cuda").bfloat16()
    qlen = torch.tensor([384], device="cuda", dtype=torch.int32)
    klen = torch.tensor([1024 * 128], device="cuda", dtype=torch.int32)
    measure("stem_tpd_B1_H4_Q3_K1024", lambda: hpc.stem_tpd(scores, qlen, klen, klen))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
