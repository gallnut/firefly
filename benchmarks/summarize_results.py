#!/usr/bin/env python3
"""Regenerate the committed benchmark summary from local raw results.

Raw per-request JSONs and server logs live under benchmarks/results/ and are
gitignored (they contain machine-specific config); this tool aggregates them
into docs/benchmarks.md, which is the public, self-contained snapshot.

Usage:
  python benchmarks/summarize_results.py [--output docs/benchmarks.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ("qwen3", "qwen35")
INPUTS = (128, 1024, 2048)
OUTPUTS = (32, 128)
CONCURRENCIES = (1, 4, 8)


def load(engine: str, model: str, input_len: int, output_len: int, concurrency: int) -> dict | None:
    path = RESULTS / engine / f"{model}_in{input_len}_out{output_len}_c{concurrency}.json"
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)["summary"]


def row(model: str, input_len: int, output_len: int, concurrency: int) -> str:
    a = load("firefly", model, input_len, output_len, concurrency)
    b = load("vllm", model, input_len, output_len, concurrency)
    if a is None or b is None:
        return f"| {model} | {input_len} | {output_len} | {concurrency} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    return (
        f"| {model} | {input_len} | {output_len} | {concurrency} | "
        f"{a['ttft_s']['mean']*1000:.0f} | {b['ttft_s']['mean']*1000:.0f} | "
        f"{a['tpot_s']['mean']*1000:.2f} | {b['tpot_s']['mean']*1000:.2f} | "
        f"{a['request_throughput_rps']:.2f} | {b['request_throughput_rps']:.2f} | "
        f"{a['completion_token_throughput_tps']:.0f} | {b['completion_token_throughput_tps']:.0f} |"
    )


def int8_row(input_len: int, output_len: int, concurrency: int) -> str:
    a = load("firefly", "qwen3", input_len, output_len, concurrency)
    b = load("firefly-int8", "qwen3", input_len, output_len, concurrency)
    if a is None or b is None:
        return f"| {input_len} | {output_len} | {concurrency} | n/a | n/a | n/a | n/a |"
    return (
        f"| {input_len} | {output_len} | {concurrency} | "
        f"{a['ttft_s']['mean']*1000:.0f} / {b['ttft_s']['mean']*1000:.0f} | "
        f"{b['ttft_s']['mean']/a['ttft_s']['mean']:.2f} | "
        f"{a['tpot_s']['mean']*1000:.2f} / {b['tpot_s']['mean']*1000:.2f} | "
        f"{b['tpot_s']['mean']/a['tpot_s']['mean']:.2f} | "
        f"{b['completion_token_throughput_tps']/a['completion_token_throughput_tps']:.2f} |"
    )


def accuracy_rows() -> list[str]:
    stats_path = RESULTS / "int8-accuracy" / "stats.json"
    if not stats_path.exists():
        return []
    stats = json.loads(stats_path.read_text())
    rows = []
    for item in stats:
        rows.append(
            f"| {item['input_tokens']} | {item['first_token_match']}/{item['first_token_total']} | "
            f"{item['exact_token_match']*100:.1f}% | {item['char_similarity_mean']:.2f} | "
            f"{item['fully_identical']}/{item['fully_identical_total']} |"
        )
    return rows


def operator_rows() -> list[str]:
    path = RESULTS / "int8_operator_tradeoff.txt"
    if not path.exists():
        return []
    rows = []
    for raw in path.read_text().splitlines():
        parts = dict(item.split("=", 1) for item in raw.split() if "=" in item)
        rows.append(
            f"| {parts['batch']} | {parts['context']} | {parts['max_abs']} | {parts['relative_l2']} | "
            f"{parts['cosine']} | {parts['bf16_ms']} | {parts['int8_ms']} | {parts['speedup']} | "
            f"{parts['memory_reduction']} |"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=ROOT.parent / "docs" / "benchmarks.md")
    parser.add_argument("--date", default="2026-08-11")
    args = parser.parse_args()

    lines: list[str] = []
    lines.append(f"# Firefly vs vLLM — Full Performance Matrix ({args.date})")
    lines.append("")
    lines.append("## Environment and method")
    lines.append("")
    lines.append("- GPU: NVIDIA GeForce RTX 4060 Laptop (8 GiB), driver 610.57.04")
    lines.append("- Firefly: Release build (CUDA 13.3 / GCC 16, SM89, FlashInfer 0.6.14), `firefly_server`")
    lines.append("- vLLM: 0.26.0, OpenAI-compatible server, FlashAttention backend, torch.compile + CUDA graphs enabled")
    lines.append("- Both engines: same local model files (`qwen3`, `qwen35`), bf16 weights, `--max-prefill-chunk-size`/`--max-num-batched-tokens` = 768, `gpu_memory_utilization` = 0.75, greedy (temperature 0), `ignore_eos`")
    lines.append("- 32 measured requests per cell, 2 warmup requests; same prompt files (calibrated so the HF chat template tokenizes to the target input length); seed 20260811; streaming API")
    lines.append("- Firefly Qwen3.5 runs use `FIREFLY_QWEN35_GDN_PREFILL_BACKEND=chunk64`; the default fused GDN prefill path hangs on 2048-token inputs (single request > 600 s) and is a known open issue in this build")
    lines.append("- Qwen3.5 template note: Firefly's manual chat template reports 4 fewer prompt tokens than the HF template used by vLLM (e.g. 2044 vs 2048); reported input counts are per-engine usage values")
    lines.append("- Raw per-request JSONs are kept locally under `benchmarks/results/` (gitignored); this file is the public snapshot.")
    lines.append("")
    lines.append("## Firefly vs vLLM (TTFT / TPOT / throughput)")
    lines.append("")
    lines.append("| model | input | output | concurrency | FF TTFT ms | vLLM TTFT ms | FF TPOT ms | vLLM TPOT ms | FF req/s | vLLM req/s | FF out tok/s | vLLM out tok/s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        for input_len in INPUTS:
            for output_len in OUTPUTS:
                for concurrency in CONCURRENCIES:
                    lines.append(row(model, input_len, output_len, concurrency))
    lines.append("")
    lines.append("## INT8 KV cache vs model KV cache (Firefly, Qwen3-0.6B serving)")
    lines.append("")
    lines.append("| input | output | concurrency | TTFT model/int8 ms | TTFT ratio int8/model | TPOT model/int8 ms | TPOT ratio | out tok/s ratio int8/model |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for input_len in INPUTS:
        for output_len in OUTPUTS:
            for concurrency in CONCURRENCIES:
                lines.append(int8_row(input_len, output_len, concurrency))
    lines.append("")
    lines.append("## INT8 KV operator tradeoff (paged decode, Qwen3 shapes: 16 Q heads / 8 KV heads / head_dim 128)")
    lines.append("")
    lines.append("Source: `tests/quantized_kv_tradeoff_test.cu` (Release build). BF16 path = FlashInfer paged decode; INT8 path = native quantized paged decode. Accuracy compares INT8-dequantized attention output against the BF16 reference on identical synthetic K/V.")
    lines.append("")
    lines.append("| batch | context | max_abs | relative L2 | cosine | BF16 ms | INT8 ms | speedup | KV memory reduction |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    lines.extend(operator_rows())
    lines.append("")
    lines.append("## INT8 KV generation-level accuracy (greedy, 64 output tokens, 8 prompts)")
    lines.append("")
    lines.append("| input | first-token match | exact token match (over 64 tokens) | char similarity | fully identical |")
    lines.append("|---|---|---|---|---|")
    acc_rows = accuracy_rows()
    if acc_rows:
        lines.extend(acc_rows)
    else:
        lines.append("_Raw INT8 accuracy responses are kept locally (gitignored); run the accuracy pass and `stats.json` generator to reproduce._")
    lines.append("")
    lines.append("Small per-step attention error (relative L2 ~0.3%, cosine >= 0.99999) is amplified by autoregressive greedy decoding, so long outputs diverge even though the first token usually matches.")
    lines.append("")
    report = Path(args.output)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
