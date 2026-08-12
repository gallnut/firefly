# Firefly: C++23 & CUDA LLM Inference Engine

[![C++23](https://img.shields.io/badge/C++-23-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B23)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *"Where giants carry a Torch, we are but a Firefly—a single, self-illuminating spark lighting up the depths of LLM inference from scratch."*

Firefly is a lightweight, high-performance Large Language Model (LLM) inference framework built entirely from scratch using modern C++23 and CUDA. 

Designed without the overhead of heavy deep learning frameworks, Firefly strips away the bloat to focus on what matters: raw performance, elegant memory management, and highly optimized custom kernels. Currently, Firefly is tailored for the Qwen architecture (Qwen2/Qwen3/Qwen3.5) and provides a robust gRPC serving interface.

## ✨ Motivation

Firefly is a learning-driven project. Instead of cloning a production framework, it reimplements the whole LLM inference
pipeline from scratch (weight loading, KV cache, scheduling, CUDA kernels) to understand the engineering trade-offs
behind mature solutions like vLLM and FlashAttention.

* Pure C++/CUDA implementation with no PyTorch dependency and a minimal dependency tree (CUTLASS, FlashInfer and gRPC
  are managed through CPM).
* Modern C++23 throughout (`std::expected`, `std::mdspan`).

## 🧠 Architecture

Firefly is organized by responsibility:

- **model layer** defines models and orchestrates forward passes (Qwen2/3/3.5). It contains no CUDA kernels.
- **kernels layer** holds all operator implementations: attention (FlashInfer backend, hand-written WMMA tiled prefill,
  paged decode fallback, quantized attention), linear attention (Gated Delta Net / causal convolution), transformer ops
  (RMSNorm, RoPE, SwiGLU, ...), cache and sampling.
- **execution / scheduler layer** implements continuous batching, chunked prefill, mixed prefill/decode scheduling and
  CUDA Graph decode. The gRPC layer is decoupled from the engine through input/output queues.
- **memory management** covers the paged KV cache and block allocator, prefix cache, model weight pool, and
  mmap + pinned-memory safetensors loading.

Key mechanisms:

- **Mixed batching**: prefill and decode run in the same ragged forward, which lowers TTFT under staggered load;
  decode-only steps execute through pre-captured CUDA Graphs.
- **Batched ragged prefill**: following vLLM, a single FlashInfer batch prefill is planned and launched using per-row
  `q_indptr` / `kv_indptr`.
- **Paged KV cache + prefix cache**: block-level allocation avoids fragmentation and shared prompt prefixes reuse KV
  blocks across turns.
- **KV cache quantization**: custom Int8 quantization (per-token scale) with matching dequantization operators.

## 📊 Current Status

Supported models: Qwen2 / Qwen3 / Qwen3.5 (hybrid linear attention with Gated Delta Net).

DSpark speculative decoding is available when a trained drafter implementing the `SpeculativeProposer` capability is
supplied with `--draft-model qwen35_dspark` (or `FIREFLY_SPECULATIVE_MODEL=qwen35_dspark`). Target models opt into
transactional recurrent-state replay separately, so the generic runner is not coupled to Qwen3.5. When speculative
decoding is disabled, hidden-layer capture, proposer state, context storage, and scheduler lookahead are not created.
When enabled, ordinary prefill/decode, FlashInfer, CUDA Graph, KV quantization, prefix cache, and mixed-batch paths keep
their existing execution routes and feed the same target-context side channel. Confidence pruning uses
`FIREFLY_SPECULATIVE_CONFIDENCE_THRESHOLD=0.0..1.0`; `FIREFLY_SPECULATIVE_DRAFT_TOKENS` overrides the drafter block
size. The legacy `FIREFLY_DSPARK_*` variables remain accepted. Multi-request batches currently use the regular decode
runner until token-level speculative batching is enabled.

Correctness is checked against vLLM under identical configurations. Measured on RTX 4060 Laptop 8GB (Release build,
FlashInfer backend, Qwen3-0.6B, BF16):

- Single request (128-token input): TTFT ≈29 ms, TPOT ≈5.3 ms (decode through CUDA Graph).
- 8 concurrent requests (128-token input): TPOT ≈6.0-6.3 ms, ≈33 req/s and ~1.0-1.2k output tokens/s.
- Long input (2048 tokens) + 8 concurrent: TTFT ≈750 ms / TPOT ≈13 ms on Qwen3; Qwen3.5 TPOT ≈9 ms.

vs vLLM 0.26 (same model files, same 768-token prefill granularity, same concurrency, 32 requests per cell):

- Decode (TPOT) is competitive or faster — Qwen3.5 at 2048-in / c8 reaches 9.0 ms vs vLLM's 24.6 ms.
- Short-input throughput is within ~5-15% of vLLM; long-input prefill (TTFT) and throughput still trail, mostly due
  to chunked-prefill scheduling/kernel maturity.
- Full 90-config matrix and methodology: `docs/benchmarks.md`; raw per-request JSONs stay local (gitignored).

Int8 KV cache quantization tradeoff (Qwen3):

- KV memory drops by ~48%; the native Int8 paged-decode attention is ~2× faster at batch=8 with 2K-4K context, but
  slower than BF16 for short contexts / low batch due to quantize/dequantize overhead.
- Attention output error is small (relative L2 ≈0.3%, cosine ≥0.99999); greedy decoding still amplifies it, so long
  generations diverge even though first tokens usually match.

Highlights:

- Hand-written C++ safetensors parser with mmap + pinned memory + async DMA loading.
- Hand-written attention kernels coexist with FlashInfer so the gap between both can be studied.
- Linear-attention operators are extracted into the kernels layer for reuse by future models.
- Clear layering: the model layer only orchestrates; operators live in the kernels layer.

Known limitations (stated honestly):

- Hand-written attention still trails FlashInfer in some cases; FlashInfer is the default production backend.
- Quantization currently covers the KV cache (Int8) only; weight W4A16/AWQ is not implemented.
- No tensor parallelism / distributed support; model support is focused on the Qwen family.
- Qwen3.5's default fused Gated Delta Net prefill currently hangs on long (2048+) inputs; set
  `FIREFLY_QWEN35_GDN_PREFILL_BACKEND=chunk64` as a workaround until the fused path is fixed.

## 🛠️ Getting Started
Prerequisites

- CMake: >= 3.35

- CUDA Toolkit: >= 12.0. Firefly builds for the local GPU by default; use the standard
  `-DCMAKE_CUDA_ARCHITECTURES=<arch>` option for cross-compilation or release builds.

- Compiler: A C++23 compatible compiler (e.g., GCC 13+, Clang 16+).

- gRPC & Protobuf: Required for the server interface.

Build Instructions

Firefly uses CPM (CMake Package Manager) to handle dependencies automatically.

```bash
# Clone the repository
git clone [https://github.com/yourusername/firefly.git](https://github.com/yourusername/firefly.git)
cd firefly

# Configure and compile
cmake -S . -B build
cmake --build build -j$(nproc)
```

FlashInfer is downloaded by CPM when enabled. Use `-DFIREFLY_USE_FLASHINFER=OFF` for Firefly's native attention path,
`-DFIREFLY_FLASHINFER_SOURCE_DIR=/path/to/flashinfer` when developing against a local FlashInfer checkout, and
`-DFIREFLY_BUILD_SERVER=OFF` when gRPC and Protobuf are not needed.

## 🚀 Usage

1.Start the Server

Run the firefly_server executable, pointing it to the directory containing your model's config.json, tokenizer.json, and model.safetensors files.
```bash
./build/bin/firefly_server /path/to/qwen3/model_dir
```

Runtime options:

- `--max-prefill-chunk-size <size>` configures the prefill chunking limit.
- `--log-level <trace|debug|info|warn|error|critical>` configures the minimum log level.
- `--log-color <auto|always|never>` controls ANSI colors. `auto` is the default and disables colors when redirected.
- `--log-detail` includes the thread ID, source location, and function name.

The same logging settings can be provided through `FIREFLY_LOG_LEVEL`, `FIREFLY_LOG_COLOR`, and
`FIREFLY_LOG_DETAIL`. Setting `NO_COLOR` also disables colors when automatic color detection is used.

2.Client Examples

Generate the Python gRPC stubs and run the provided examples:
```bash
cd example
python -m grpc_tools.protoc -I../proto --python_out=. --grpc_python_out=. ../proto/firefly.proto

# Run a streaming request (natively supports parsing <think> tags for CoT reasoning)
python test_grpc_stream.py
```

3.Benchmarking

Use the built-in benchmark tool to test the engine's throughput and latency.
```bash
python benchmarks/benchmark_serving.py -n 50 -c 8 --synthetic \
  --min-input-tokens 64 --max-input-tokens 256 \
  --min-output-tokens 128 --max-output-tokens 256 --model qwen3
```

For reproducible Firefly-vs-vLLM matrices (calibrated prompts, same prefill granularity, per-cell JSON results), use
`benchmarks/run_perf_matrix.py` with `--firefly-server`, `--firefly-python`, `--vllm-bin`, `--model-dirs`, etc. pointing
at your environments; the 2026-08-11 summary report is `docs/benchmarks.md`, and raw JSONs are written to
`benchmarks/results/` (gitignored).

Calibrate the prompt files first (token counts match the target input lengths under the HF chat template):
```bash
python benchmarks/make_calibrated_prompts.py \
  --model-dir /path/to/qwen3 --model qwen3 \
  --output-dir benchmarks/results/prompts
```

## 🤝 Contributing

Firefly is built by developers, for developers. If you are passionate about C++, CUDA, and the deep engineering behind LLMs, your contributions are highly welcome!

Feel free to open an issue to discuss a feature, report a bug, or submit a Pull Request. Please ensure your C++ code follows the .clang-format guidelines provided in the repository.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Not all lights need to be a torch. Keep coding, keep shining.*
