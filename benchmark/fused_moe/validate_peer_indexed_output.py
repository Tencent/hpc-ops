#!/usr/bin/env python3
"""Validate the deterministic output fixture used by the serving A/B."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--expected-token-text", required=True)
    parser.add_argument("--expected-token-id", type=int)
    args = parser.parse_args()

    result = json.loads(Path(args.file).read_text())
    expected_tokens = args.batch * args.output_len
    if result.get("completed") != args.batch:
        raise SystemExit(f"completed={result.get('completed')} does not match batch={args.batch}")
    errors = result.get("errors") or []
    if any(errors):
        raise SystemExit(f"benchmark contains request errors: {errors}")
    if result.get("total_output_tokens") != expected_tokens:
        raise SystemExit(
            f"total_output_tokens={result.get('total_output_tokens')} does not match "
            f"expected={expected_tokens}"
        )

    expected_text = args.expected_token_text * args.output_len
    texts = result.get("generated_texts") or []
    bad_texts = [index for index, value in enumerate(texts) if value != expected_text]
    if len(texts) != args.batch or bad_texts:
        raise SystemExit(
            f"forced output mismatch: outputs={len(texts)}, "
            f"batch={args.batch}, bad_requests={bad_texts}"
        )

    output_ids = result.get("debug_output_ids")
    if args.expected_token_id is not None and output_ids is not None:
        bad_ids = [
            index
            for index, token_ids in enumerate(output_ids)
            if len(token_ids) != args.output_len
            or any(token_id != args.expected_token_id for token_id in token_ids)
        ]
        if len(output_ids) != args.batch or bad_ids:
            raise SystemExit(
                f"forced output id mismatch: outputs={len(output_ids)}, "
                f"batch={args.batch}, bad_requests={bad_ids}"
            )

    print(f"validated {args.file}: {args.batch} requests x {args.output_len} tokens")


if __name__ == "__main__":
    main()
