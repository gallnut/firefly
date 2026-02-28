# Firefly: C++23 & CUDA LLM Inference Engine

[![C++23](https://img.shields.io/badge/C++-23-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B23)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *"Where giants carry a Torch, we are but a Firefly—a single, self-illuminating spark lighting up the depths of LLM inference from scratch."*

Firefly is a lightweight, high-performance Large Language Model (LLM) inference framework built entirely from scratch using modern C++23 and CUDA. 

Designed without the overhead of heavy deep learning frameworks, Firefly strips away the bloat to focus on what matters: raw performance, elegant memory management, and highly optimized custom kernels. Currently, Firefly is tailored for the Qwen architecture (Qwen2/Qwen3) and provides a robust gRPC serving interface.

## ✨ Why Firefly?

Unlike traditional frameworks, Firefly is built for developers who want to understand and control every byte of memory and every CUDA cycle. 

* **No Bloatware:** A pure C++/CUDA implementation. No PyTorch, no massive dependency trees.
* **Modern C++23 Elegance:** Leverages cutting-edge C++ features like `std::expected` for monadic error handling and `std::mdspan` for multi-dimensional, zero-overhead tensor views.
* **Hardware-Squeezing Performance:** Direct integration with **NVIDIA CUTLASS** and custom WMMA Tensor Core kernels to push your GPU to its limits.

## 🧠 Architecture & Key Features

### 1. Advanced Attention Mechanisms
* **FlashAttention (Prefill):** Implemented using WMMA (Warp Matrix Multiply Accumulate) to maximize Tensor Core utilization during the compute-heavy prefill phase.
* **PagedAttention (Decode):** A custom non-contiguous KV-cache memory manager that eliminates memory fragmentation and allows for highly efficient continuous batching.

### 2. Smart Memory Management
* **Prefix Caching via Radix Tree:** Firefly implements a thread-safe Radix Tree (`src/radix_tree.cc`) to cache and share KV-blocks across requests that share common prompt prefixes, dramatically reducing TTFT (Time To First Token) for multi-turn chats or system prompts.
* **Zero-Copy Weight Loading:** Uses memory-mapped files and CUDA pinned memory (`PinnedMappedFile`) to stream `.safetensors` weights directly to the GPU, bypassing unnecessary CPU RAM bottlenecks.
* **Custom Memory Pooling:** Features a dedicated `ModelWeightPool` and `BlockAllocator` to avoid runtime `cudaMalloc` overheads.

### 3. Continuous Batching & Graph Execution
* **Iteration-Level Scheduling:** The scheduler evaluates the queue at every token generation step, dynamically admitting new requests to maximize throughput.
* **CUDA Graphs:** The decode phase utilizes pre-captured CUDA Graphs (`cudaGraph_t`) to eliminate CPU launch overhead, ensuring microseconds-level latency per token.

## 📂 Project Structure

```text
firefly/
├── include/firefly/
│   ├── hal/         # Hardware Abstraction Layer (RAII wrappers for CUDA Streams, Events, Graphs, Memory)
│   ├── mm/          # Memory Management (Allocators, Mapped Files, KV-Block Pools)
│   ├── model/       # Model Definitions & Configurations (Qwen2/3)
│   └── ...          # Core structures: Tensor, Engine, Scheduler, RadixTree
├── src/
│   ├── kernels/     # Pure CUDA kernels (Attention, RoPE, RMSNorm, SwiGLU, Sampling)
│   ├── model/       # Forward pass implementations
│   └── ...          # System logic implementations
├── proto/           # gRPC definitions for ChatCompletion API
├── benchmarks/      # Python benchmarking scripts (Throughput, TTFT, TPOT)
└── example/         # Python gRPC client examples (Unary & Streaming)
```

## 🛠️ Getting Started
Prerequisites

- CMake: >= 3.35

- CUDA Toolkit: >= 12.0 (Targeting compute capability sm_89 by default. Adjust in .clangd and cmake/Deps.cmake if needed).

- Compiler: A C++23 compatible compiler (e.g., GCC 13+, Clang 16+).

- gRPC & Protobuf: Required for the server interface.

Build Instructions

Firefly uses CPM (CMake Package Manager) to handle dependencies automatically.

```bash
# Clone the repository
git clone [https://github.com/yourusername/firefly.git](https://github.com/yourusername/firefly.git)
cd firefly

# Create build directory
mkdir build && cd build

# Configure and compile
cmake ..
make -j$(nproc)
```

## 🚀 Usage

1.Start the Server

Run the firefly_server executable, pointing it to the directory containing your model's config.json, tokenizer.json, and model.safetensors files.
```bash
./bin/firefly_server /path/to/qwen3/model_dir
```

Optional argument: --max-prefill-chunk-size <size> to configure the prefill chunking limit.

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

cd benchmarks
python benchmark_serving.py -c 8 -n 50 --model qwen3 --max-tokens 256
```

## 🗺️ Roadmap

    [x] Base C++23 / CUDA Abstractions

    [x] Zero-copy Safetensors Loader

    [x] Paged KV Cache & Block Allocator

    [x] Radix Tree Prefix Caching

    [x] Continuous Batching & Chunked Prefill

    [x] Hand-tuned Custom CUDA Kernels (FlashAttention, RoPE, etc.)

    [ ] Quantization: Implement W4A16 / AWQ kernels to fit 7B models in 8GB VRAM.

    [ ] Distributed: Tensor Parallelism support via NCCL.

    [ ] Architectures: Support Llama-3 and Mistral architectures.

## 🤝 Contributing

Firefly is built by developers, for developers. If you are passionate about C++, CUDA, and the deep engineering behind LLMs, your contributions are highly welcome!

Feel free to open an issue to discuss a feature, report a bug, or submit a Pull Request. Please ensure your C++ code follows the .clang-format guidelines provided in the repository.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Not all lights need to be a torch. Keep coding, keep shining.*