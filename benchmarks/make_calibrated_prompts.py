#!/usr/bin/env python3
"""Generate prompt files whose chat-template token count matches a target.

The same prompt text is used for Firefly and vLLM. Calibration is done against
the HuggingFace chat template (what vLLM applies); Firefly reports its own
usage token counts in the benchmark output, so any fixed template difference is
visible in the result JSON instead of being hidden.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


FILLER_SNIPPETS = [
    "The scheduler batches active requests, allocates KV cache blocks, and sends decode steps to CUDA kernels.",
    "A useful benchmark should vary prompt length, output length, concurrency, and arrival rate.",
    "For correctness testing, include multilingual text, code snippets, arithmetic, and summaries.",
    "Streaming time-to-first-token and inter-chunk latency reveal serving behavior better than end-to-end time alone.",
    "Paged KV cache blocks decouple logical sequence positions from physical GPU memory addresses.",
    "The model should preserve the original dtype of weights unless a deliberate optimization converts them.",
    "Prefix cache hits are dangerous when token ranges and KV block ranges no longer describe the same boundary.",
    "CUDA graph capture can improve steady-state latency, but failed captures must not leave invalid handles behind.",
]


def calibrated_prompt(tokenizer, target_total: int, rng: random.Random, max_attempts: int = 120) -> str:
    """Return user content so that apply_chat_template(...) tokenizes to exactly target_total."""
    best: tuple[int, str] | None = None
    for attempt in range(max_attempts):
        snippet = rng.choice(FILLER_SNIPPETS)
        repeats = max(1, target_total // 25)
        content = "Answer the following question precisely.\n\n" + (snippet + " ") * repeats
        # Pad or trim with single-token words until the total matches the target.
        for _ in range(4000):
            total = token_count(tokenizer, content)
            if total == target_total:
                return content
            if best is None or abs(total - target_total) < abs(best[0] - target_total):
                best = (total, content)
            if total < target_total:
                content = content + " " + rng.choice(["token", "benchmark", "example", "model", "test"])
            else:
                content = content.rstrip()[: max(1, len(content.rstrip()) - 8)]
    if best is None:
        raise RuntimeError("failed to calibrate prompt")
    return best[1]


def token_count(tokenizer, content: str) -> int:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Model directory containing tokenizer files.")
    parser.add_argument("--model", required=True, help="Short model name used in JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 1024, 2048])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for target in args.lengths:
        path = out_dir / f"{args.model}_in{target}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for index in range(args.num_samples):
                content = calibrated_prompt(tokenizer, target, rng)
                actual = token_count(tokenizer, content)
                if actual != target:
                    print(f"WARNING {args.model} in{target} sample {index}: got {actual}", file=sys.stderr)
                handle.write(
                    json.dumps(
                        {
                            "id": f"{args.model}-in{target}-{index}",
                            "messages": [{"role": "user", "content": content}],
                            "approx_input_tokens": actual,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"wrote {path} samples={args.num_samples} target_tokens={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
