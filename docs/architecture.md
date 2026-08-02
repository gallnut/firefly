# Firefly Architecture

Firefly separates model semantics, execution policy, request scheduling, service transport, and device kernels. A directory is not a
module by itself: each module owns a specific kind of state and may only depend on layers below it.

## Dependency direction

```text
service -> execution -> scheduler
                    \-> model -> storage
                             \-> kernels
execution ---------------------> kernels

model/scheduler/kernels -> core -> device
model ------------------> storage -> core/device
```

Dependencies must not point upward. In particular:

- Models do not own request queues, prefix-cache policy, CUDA graphs, or RPC concepts.
- Scheduling does not launch kernels or inspect model weights.
- Kernels do not know requests, models, schedulers, or service protocols.
- Serving does not allocate KV blocks or build device batches.
- Backend-specific code is selected through an operator dispatch interface, not from model implementations.

## Modules

### `core`

Own the stable data vocabulary shared across the engine: `Tensor`, `TensorView`, `DType`, and the logical `Device`
tag. Core contains no model, scheduling, storage-format, or service concepts. Tensor allocation is implemented through
the lower-level device allocator rather than embedding CUDA APIs in its public contract.

### `device`

Own CUDA resource lifetimes and host/device allocation: streams, events, graphs, pinned-memory registration, error
handling, and allocators. `device` is infrastructure, not an inference runtime; it must not know models, requests,
operators, file formats, or execution policy.

### `storage`

Own persistence and file-format concerns: mapped files and safetensors parsing. Storage may expose tensors backed by
mapped data, but it does not own model interpretation or GPU execution policy.

### `kernels`

Own only kernels required by Firefly's LLM inference path. This is not a general-purpose operator library. Interfaces
are grouped by inference responsibility: attention, KV cache, transformer primitives, sampling, quantization, and
parallel communication. Model-specific fusion belongs in that model family's kernel subdirectory; tensor/expert
parallel fusion and collectives belong in a parallel-strategy subdirectory.

Vendor implementations such as FlashAttention or FlashInfer are adapters behind Firefly-owned contracts and may be
replaced incrementally. Framework runtimes such as PyTorch are forbidden production dependencies: they hide execution
and memory behavior, substantially increase the dependency surface, and work against Firefly's educational purpose.
Reference correctness paths must be implemented in C++/CUDA tests or small standalone kernels.

Headers live under `include/firefly/kernels`; implementation-only CUDA helpers live under `kernels/detail`, while an
operator-specific adapter contract lives under that operator's `detail` directory. Detail headers may only be included
by their owning kernel implementations. `src` contains implementations only. A backend may expose a reusable workspace
object, but it must not own requests or global engine state.

CUDA streams are owned by execution workers through `device::Stream` and passed explicitly in `device::Context` to
Tensor allocation, model forward, and every kernel launch. Process-global or thread-local default streams are
forbidden. A context retains the underlying stream lifetime so asynchronous Tensor allocation and release cannot
outlive the stream handle. The current engine has one model-execution worker; adding concurrent model workers requires
moving cuBLAS handles and attention backend workspaces into per-context resources before enabling parallel launches.

### `model`

Own model configuration parsing, architecture registration, weight mapping, weight storage, tokenizer assets, and
neural-network topology. The tokenizer belongs here because it is loaded from a model package and forms part of that
model's text/token contract; it is not part of the tensor forward graph. The `model` root contains only cross-model
contracts and infrastructure. Each concrete model family lives in its own directory and namespace, such as
`model/qwen`, so architecture-specific layers, weight names, and registration cannot leak into shared interfaces.

Adding DeepSeek or Kimi must not require edits to the scheduler or server. MoE routing, MLA projections, hybrid
attention, and architecture-specific normalization belong here, while their optimized primitive operations belong in
`kernels`.

### `execution`

Own device batches, KV storage, memory planning, CUDA graphs, model runners, sampling, and output production. This is
where optimization policies are composed:

- weight and KV quantization formats;
- tensor, pipeline, expert, and data parallel execution;
- speculative decoding draft/verify runners;
- eager versus CUDA-graph execution;
- dense versus sparse operator selection.

Execution policies consume model capabilities and kernel interfaces. They must not be encoded as Qwen-specific flags.

### `scheduler`

Own sequence lifecycle, admission, batching policy, physical block ownership, and prefix reuse policy. A `Sequence`
is scheduler state for one generation path, not a service request or transport DTO. `PrefixCache` indexes token
prefixes to block IDs; it is distinct from the GPU KV tensors owned by execution. Block eviction is a scheduler policy
even though the evicted IDs refer to execution storage.

Future speculative scheduling and preemption belong here, while draft-model execution remains in `execution`.

### `service`

Own transport adapters, sessions, protocol request parsing, cancellation, and result dispatch. Protocol-neutral chat
template selection belongs to the model package; service only converts a transport request into that interface. The
executable entry point only parses process options and assembles these components.

## Extension contracts

### New model architecture

1. Add an architecture-specific config parser and model implementation under `model/<family>` and its matching namespace.
2. Let `src/model/<family>/CMakeLists.txt` own the family's complete source list and register its object target with
   `firefly_register_model_module(target, registrar)`. Do not edit a shared model source manifest.
3. Compose reusable layers and call operator interfaces from `kernels`.
4. Declare cache and execution capabilities rather than branching in the engine by architecture name.
5. Register architectures through the family's registrar; CMake generates the builtin registry call table.
6. Add weight-loading and numerical parity tests.

### New optimization backend

1. Implement the backend under the relevant inference domain, such as `kernels/attention/<backend>`.
2. Register capabilities such as dtype, head dimension, layout, sparsity, and graph safety.
3. Let execution policy select the backend from capabilities and user configuration.
4. Keep a reference path and add numerical parity plus performance benchmarks.

### Quantized KV cache

KV format metadata and storage allocation belong to `execution`; quantize/append/dequantize attention operations
belong to `kernels`; memory accounting and block admission belong to `scheduler`. Model topology must not contain KV
append or quantization kernels.

### Profiling instrumentation

NVTX ranges use `FIREFLY_NVTX_PUSH` and `FIREFLY_NVTX_POP` from `firefly/execution/trace.h`. They compile to no
operations by default. Configure with `-DFIREFLY_ENABLE_NVTX=ON` only for profiler builds; direct NVTX calls in runtime
code are forbidden.

## Known migration work

The module layout is being migrated incrementally while preserving benchmark and accuracy behavior. Remaining major
items are tracked explicitly instead of hidden behind directory renames:

- split `execution/engine.cc` into memory planning, prefill, decode graph, decode fallback, and orchestration units;
- split the paged-attention reference kernels from attention dispatch and backend selection;
- replace process-global attention backend state with an execution configuration and backend registry;
- replace the fixed common `ModelConfig` cache assumptions with model capability and cache-layout descriptors;
- split protocol-neutral chat templating from the gRPC adapter;
- create independent CMake targets that enforce the dependency direction above.
