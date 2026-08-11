# Firefly vs vLLM — Full Performance Matrix (2026-08-11)

## Environment and method

- GPU: NVIDIA GeForce RTX 4060 Laptop (8 GiB), driver 610.57.04
- Firefly: Release build (CUDA 13.3 / GCC 16, SM89, FlashInfer 0.6.14), `firefly_server`
- vLLM: 0.26.0, OpenAI-compatible server, FlashAttention backend, torch.compile + CUDA graphs enabled
- Both engines: same local model files (`qwen3`, `qwen35`), bf16 weights, `--max-prefill-chunk-size`/`--max-num-batched-tokens` = 768, `gpu_memory_utilization` = 0.75, greedy (temperature 0), `ignore_eos`
- 32 measured requests per cell, 2 warmup requests; same prompt files (calibrated so the HF chat template tokenizes to the target input length); seed 20260811; streaming API
- Firefly Qwen3.5 runs use `FIREFLY_QWEN35_GDN_PREFILL_BACKEND=chunk64`; the default fused GDN prefill path hangs on 2048-token inputs (single request > 600 s) and is a known open issue in this build
- Qwen3.5 template note: Firefly's manual chat template reports 4 fewer prompt tokens than the HF template used by vLLM (e.g. 2044 vs 2048); reported input counts are per-engine usage values
- Raw per-request JSONs are kept locally under `benchmarks/results/` (gitignored); this file is the public snapshot.

## Firefly vs vLLM (TTFT / TPOT / throughput)

| model | input | output | concurrency | FF TTFT ms | vLLM TTFT ms | FF TPOT ms | vLLM TPOT ms | FF req/s | vLLM req/s | FF out tok/s | vLLM out tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3 | 128 | 32 | 1 | 29 | 10 | 5.32 | 5.58 | 5.15 | 5.45 | 165 | 174 |
| qwen3 | 128 | 32 | 4 | 45 | 16 | 5.65 | 6.10 | 18.18 | 19.38 | 582 | 620 |
| qwen3 | 128 | 32 | 8 | 59 | 21 | 5.95 | 6.14 | 32.81 | 37.57 | 1050 | 1202 |
| qwen3 | 128 | 128 | 1 | 29 | 11 | 5.48 | 5.60 | 1.38 | 1.39 | 176 | 177 |
| qwen3 | 128 | 128 | 4 | 45 | 18 | 5.92 | 6.15 | 5.02 | 5.00 | 643 | 640 |
| qwen3 | 128 | 128 | 8 | 59 | 21 | 6.31 | 6.30 | 9.29 | 9.72 | 1190 | 1244 |
| qwen3 | 1024 | 32 | 1 | 67 | 17 | 5.72 | 6.06 | 4.09 | 4.87 | 131 | 156 |
| qwen3 | 1024 | 32 | 4 | 191 | 22 | 7.32 | 7.61 | 9.57 | 15.44 | 306 | 494 |
| qwen3 | 1024 | 32 | 8 | 356 | 29 | 9.22 | 8.54 | 12.46 | 27.05 | 399 | 866 |
| qwen3 | 1024 | 128 | 1 | 67 | 14 | 5.90 | 6.08 | 1.22 | 1.27 | 157 | 163 |
| qwen3 | 1024 | 128 | 4 | 191 | 22 | 7.63 | 7.68 | 3.45 | 4.00 | 441 | 512 |
| qwen3 | 1024 | 128 | 8 | 355 | 30 | 9.65 | 8.66 | 5.06 | 7.06 | 647 | 904 |
| qwen3 | 2048 | 32 | 1 | 109 | 27 | 6.38 | 6.47 | 3.26 | 4.40 | 104 | 141 |
| qwen3 | 2048 | 32 | 4 | 634 | 28 | 9.31 | 9.36 | 4.34 | 12.52 | 139 | 400 |
| qwen3 | 2048 | 32 | 8 | 753 | 38 | 13.00 | 11.53 | 6.92 | 20.03 | 221 | 641 |
| qwen3 | 2048 | 128 | 1 | 117 | 17 | 6.43 | 6.51 | 1.07 | 1.19 | 137 | 152 |
| qwen3 | 2048 | 128 | 4 | 409 | 31 | 9.51 | 9.41 | 2.47 | 3.26 | 317 | 417 |
| qwen3 | 2048 | 128 | 8 | 746 | 39 | 13.46 | 11.57 | 3.26 | 5.29 | 417 | 677 |
| qwen35 | 128 | 32 | 1 | 35 | 19 | 6.77 | 6.87 | 4.08 | 4.31 | 131 | 138 |
| qwen35 | 128 | 32 | 4 | 53 | 44 | 7.49 | 7.62 | 14.01 | 14.26 | 448 | 456 |
| qwen35 | 128 | 32 | 8 | 82 | 56 | 8.28 | 8.65 | 23.58 | 24.57 | 755 | 786 |
| qwen35 | 128 | 128 | 1 | 35 | 18 | 6.95 | 6.88 | 1.09 | 1.12 | 140 | 144 |
| qwen35 | 128 | 128 | 4 | 53 | 44 | 7.68 | 7.61 | 3.89 | 3.96 | 498 | 506 |
| qwen35 | 128 | 128 | 8 | 81 | 71 | 8.51 | 8.39 | 6.88 | 7.02 | 881 | 899 |
| qwen35 | 1024 | 32 | 1 | 84 | 62 | 6.78 | 6.92 | 3.40 | 3.62 | 109 | 116 |
| qwen35 | 1024 | 32 | 4 | 271 | 127 | 7.65 | 10.09 | 7.86 | 9.05 | 252 | 290 |
| qwen35 | 1024 | 32 | 8 | 539 | 172 | 8.64 | 15.18 | 9.91 | 12.26 | 317 | 392 |
| qwen35 | 1024 | 128 | 1 | 95 | 63 | 6.95 | 6.94 | 1.02 | 1.06 | 131 | 136 |
| qwen35 | 1024 | 128 | 4 | 278 | 129 | 7.85 | 8.35 | 3.14 | 3.36 | 402 | 430 |
| qwen35 | 1024 | 128 | 8 | 539 | 173 | 8.86 | 10.28 | 4.81 | 5.37 | 615 | 688 |
| qwen35 | 2048 | 32 | 1 | 159 | 115 | 6.82 | 6.97 | 2.70 | 3.02 | 86 | 97 |
| qwen35 | 2048 | 32 | 4 | 540 | 185 | 7.85 | 14.55 | 5.10 | 6.25 | 163 | 200 |
| qwen35 | 2048 | 32 | 8 | 1067 | 277 | 9.04 | 24.55 | 5.94 | 7.56 | 190 | 242 |
| qwen35 | 2048 | 128 | 1 | 159 | 114 | 7.00 | 6.99 | 0.95 | 1.00 | 122 | 128 |
| qwen35 | 2048 | 128 | 4 | 540 | 189 | 8.05 | 9.56 | 2.56 | 2.84 | 328 | 364 |
| qwen35 | 2048 | 128 | 8 | 1066 | 277 | 9.28 | 12.87 | 3.56 | 4.14 | 456 | 530 |

## INT8 KV cache vs model KV cache (Firefly, Qwen3-0.6B serving)

| input | output | concurrency | TTFT model/int8 ms | TTFT ratio int8/model | TPOT model/int8 ms | TPOT ratio | out tok/s ratio int8/model |
|---|---|---|---|---|---|---|---|
| 128 | 32 | 1 | 29 / 29 | 1.02 | 5.32 / 6.51 | 1.22 | 0.84 |
| 128 | 32 | 4 | 45 / 46 | 1.03 | 5.65 / 7.88 | 1.39 | 0.76 |
| 128 | 32 | 8 | 59 / 62 | 1.05 | 5.95 / 9.41 | 1.58 | 0.69 |
| 128 | 128 | 1 | 29 / 29 | 1.02 | 5.48 / 6.83 | 1.25 | 0.81 |
| 128 | 128 | 4 | 45 / 46 | 1.02 | 5.92 / 8.26 | 1.40 | 0.73 |
| 128 | 128 | 8 | 59 / 62 | 1.05 | 6.31 / 9.85 | 1.56 | 0.66 |
| 1024 | 32 | 1 | 67 / 68 | 1.02 | 5.72 / 7.03 | 1.23 | 0.85 |
| 1024 | 32 | 4 | 191 / 201 | 1.06 | 7.32 / 9.05 | 1.24 | 0.87 |
| 1024 | 32 | 8 | 356 / 383 | 1.08 | 9.22 / 11.38 | 1.23 | 0.87 |
| 1024 | 128 | 1 | 67 / 68 | 1.02 | 5.90 / 7.21 | 1.22 | 0.83 |
| 1024 | 128 | 4 | 191 / 201 | 1.05 | 7.63 / 9.33 | 1.22 | 0.84 |
| 1024 | 128 | 8 | 355 / 381 | 1.07 | 9.65 / 11.73 | 1.21 | 0.85 |
| 2048 | 32 | 1 | 109 / 122 | 1.12 | 6.38 / 7.23 | 1.13 | 0.89 |
| 2048 | 32 | 4 | 634 / 420 | 0.66 | 9.31 / 9.91 | 1.06 | 1.27 |
| 2048 | 32 | 8 | 753 / 817 | 1.09 | 13.00 / 13.04 | 1.00 | 0.95 |
| 2048 | 128 | 1 | 117 / 122 | 1.04 | 6.43 / 7.41 | 1.15 | 0.88 |
| 2048 | 128 | 4 | 409 / 419 | 1.03 | 9.51 / 10.19 | 1.07 | 0.94 |
| 2048 | 128 | 8 | 746 / 817 | 1.10 | 13.46 / 13.45 | 1.00 | 0.97 |

## INT8 KV operator tradeoff (paged decode, Qwen3 shapes: 16 Q heads / 8 KV heads / head_dim 128)

Source: `tests/quantized_kv_tradeoff_test.cu` (Release build). BF16 path = FlashInfer paged decode; INT8 path = native quantized paged decode. Accuracy compares INT8-dequantized attention output against the BF16 reference on identical synthetic K/V.

| batch | context | max_abs | relative L2 | cosine | BF16 ms | INT8 ms | speedup | KV memory reduction |
|---|---|---|---|---|---|---|---|---|
| 1 | 128 | 0.000488 | 0.004621 | 0.999993 | 0.008407 | 0.020669 | 0.406753 | 48.437500% |
| 1 | 1024 | 0.000244 | 0.003388 | 0.999996 | 0.010860 | 0.034949 | 0.310724 | 48.437500% |
| 1 | 2048 | 0.000244 | 0.003754 | 0.999996 | 0.015713 | 0.038897 | 0.403974 | 48.437500% |
| 1 | 4096 | 0.000244 | 0.003536 | 0.999996 | 0.024335 | 0.047421 | 0.513172 | 48.437500% |
| 8 | 128 | 0.000488 | 0.004034 | 0.999994 | 0.014817 | 0.023224 | 0.638007 | 48.437500% |
| 8 | 1024 | 0.000244 | 0.003290 | 0.999996 | 0.046751 | 0.072412 | 0.645620 | 48.437500% |
| 8 | 2048 | 0.000244 | 0.003343 | 0.999996 | 0.276280 | 0.134871 | 2.048477 | 48.437500% |
| 8 | 4096 | 0.000244 | 0.003309 | 0.999997 | 0.543375 | 0.303923 | 1.787871 | 48.437500% |

## INT8 KV generation-level accuracy (greedy, 64 output tokens, 8 prompts)

| input | first-token match | exact token match (over 64 tokens) | char similarity | fully identical |
|---|---|---|---|---|
| 128 | 8/8 | 21.0% | 0.52 | 0/8 |
| 1024 | 6/8 | 32.9% | 0.44 | 0/8 |

Small per-step attention error (relative L2 ~0.3%, cosine >= 0.99999) is amplified by autoregressive greedy decoding, so long outputs diverge even though the first token usually matches.

