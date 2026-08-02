#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import os
import random
import statistics
import sys
import time
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), "../example"))

grpc: Any = None
firefly_pb2: Any = None
firefly_pb2_grpc: Any = None
aiohttp: Any = None


def ensure_grpc_modules() -> None:
    global grpc, firefly_pb2, firefly_pb2_grpc

    if grpc is None:
        try:
            import grpc as grpc_module
        except ImportError as exc:
            raise RuntimeError("Python package grpcio is required to run the benchmark.") from exc
        grpc = grpc_module

    if firefly_pb2 is None or firefly_pb2_grpc is None:
        try:
            import firefly_pb2 as pb2_module
            import firefly_pb2_grpc as pb2_grpc_module
        except ImportError as exc:
            raise RuntimeError(
                "Please generate python grpc stubs first: "
                "python -m grpc_tools.protoc -Iproto --python_out=example "
                "--grpc_python_out=example proto/firefly.proto"
            ) from exc
        firefly_pb2 = pb2_module
        firefly_pb2_grpc = pb2_grpc_module


def ensure_aiohttp_module() -> None:
    global aiohttp
    if aiohttp is None:
        try:
            import aiohttp as aiohttp_module
        except ImportError as exc:
            raise RuntimeError("Python package aiohttp is required for --backend openai.") from exc
        aiohttp = aiohttp_module


BUILTIN_PROMPTS = [
    "Write a C++ program that implements quicksort, explain the edge cases, and include a small test in main.",
    "用中文解释 transformer attention 中 KV cache 的作用，并举一个 decode 阶段的例子。",
    "Summarize the tradeoffs between chunked prefill and normal prefill in an LLM serving engine.",
    "Given a list of request latencies, design a small Python function that reports p50, p90, and p99.",
    "Translate this sentence to English and explain any ambiguous wording: 这个 benchmark 不能只测一个固定输入。",
    "Solve step by step: if a GPU kernel processes 64 tokens per block and there are 3073 tokens, how many blocks are needed?",
    "Review this API behavior: streaming responses return deltas without usage. What metrics can still be measured?",
    "Create a concise incident report template for a CUDA invalid argument failure during model serving.",
    "Explain the difference between BF16 and FP16 for model weights, focusing on numerical range.",
    "Write a short SQL query and schema for storing benchmark request latency and token counts.",
    "A user reports that generated answers are fluent but wrong after prefix cache hits. List likely causes.",
    "Compare greedy decoding and temperature sampling for a chat model, including when each is useful.",
]

SYNTHETIC_SNIPPETS = [
    "The service receives mixed chat workloads with short factual questions, long coding prompts, and requests that include prior context.",
    "The scheduler batches active requests, allocates KV cache blocks, and sends decode steps to CUDA kernels.",
    "A useful benchmark should vary prompt length, output length, concurrency, and arrival rate instead of replaying one fixed input.",
    "When streaming is enabled, time to first token, inter-chunk latency, and end-to-end latency all tell different parts of the story.",
    "For correctness testing, include multilingual text, code snippets, arithmetic, summaries, and conversations with system instructions.",
    "The model should preserve the original dtype of weights unless a deliberate optimization converts them.",
    "Prefix cache hits are dangerous when token ranges and KV block ranges no longer describe the same boundary.",
    "CUDA graph capture can improve steady-state latency, but failed captures must not leave invalid graph handles behind.",
]


@dataclass
class PromptSample:
    sample_id: str
    messages: list[dict[str, str]]
    approx_input_tokens: int
    input_chars: int


@dataclass
class RequestSpec:
    request_id: int
    sample: PromptSample
    max_tokens: int


@dataclass
class RequestResult:
    request_id: int
    prompt_id: str
    success: bool
    error: str = ""
    grpc_code: str = ""
    latency_s: float = 0.0
    ttft_s: float | None = None
    tpot_s: float | None = None
    mean_inter_chunk_s: float | None = None
    output_chunks: int = 0
    output_chars: int = 0
    reasoning_chars: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    token_count_source: str = "estimated"
    max_tokens: int = 0
    response_text: str | None = None


def estimate_tokens(text: str) -> int:
    if not text:
        return 0

    cjk_chars = 0
    ascii_chars = 0
    other_chars = 0
    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            cjk_chars += 1
        elif code < 128:
            ascii_chars += 1
        elif not ch.isspace():
            other_chars += 1

    return max(1, int(cjk_chars + other_chars * 0.7 + ascii_chars / 4.0))


def messages_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)


def make_sample(sample_id: str, messages: list[dict[str, str]]) -> PromptSample:
    text = messages_text(messages)
    return PromptSample(
        sample_id=sample_id,
        messages=messages,
        approx_input_tokens=estimate_tokens(text),
        input_chars=len(text),
    )


def role_from_value(value: str | None) -> str:
    value = (value or "user").lower()
    if value in {"human", "user"}:
        return "user"
    if value in {"gpt", "assistant", "bot"}:
        return "assistant"
    if value in {"system"}:
        return "system"
    return "user"


def normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if not isinstance(raw_messages, list):
        return messages

    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = role_from_value(item.get("role") or item.get("from"))
        content = item.get("content")
        if content is None:
            content = item.get("value")
        if content is None:
            content = item.get("text")
        content = str(content or "").strip()
        if content:
            messages.append({"role": role, "content": content})

    while messages and messages[-1]["role"] == "assistant":
        messages.pop()
    return messages


def sample_from_object(obj: Any, index: int) -> PromptSample | None:
    if isinstance(obj, str):
        prompt = obj.strip()
        if not prompt:
            return None
        return make_sample(str(index), [{"role": "user", "content": prompt}])

    if not isinstance(obj, dict):
        return None

    sample_id = str(obj.get("id") or obj.get("sample_id") or index)

    messages = normalize_messages(obj.get("messages"))
    if not messages:
        messages = normalize_messages(obj.get("conversations"))

    if not messages:
        instruction = str(obj.get("instruction") or "").strip()
        input_text = str(obj.get("input") or "").strip()
        if instruction and input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = str(
                obj.get("prompt")
                or obj.get("text")
                or obj.get("question")
                or obj.get("query")
                or ""
            ).strip()
        if prompt:
            messages = [{"role": "user", "content": prompt}]

    if not messages:
        return None
    return make_sample(sample_id, messages)


def load_prompt_file(path: str, max_samples: int) -> list[PromptSample]:
    samples: list[PromptSample] = []
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".jsonl"):
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                sample = sample_from_object(json.loads(line), line_no)
                if sample:
                    samples.append(sample)
                if max_samples and len(samples) >= max_samples:
                    break
        elif path.endswith(".json"):
            data = json.load(handle)
            if isinstance(data, dict):
                data = data.get("data") or data.get("samples") or data.get("prompts") or [data]
            if not isinstance(data, list):
                raise ValueError("JSON prompt file must be a list or an object with data/samples/prompts.")
            for index, item in enumerate(data):
                sample = sample_from_object(item, index)
                if sample:
                    samples.append(sample)
                if max_samples and len(samples) >= max_samples:
                    break
        else:
            text = handle.read()
            chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
            if len(chunks) <= 1:
                chunks = [line.strip() for line in text.splitlines() if line.strip()]
            for index, chunk in enumerate(chunks):
                sample = sample_from_object(chunk, index)
                if sample:
                    samples.append(sample)
                if max_samples and len(samples) >= max_samples:
                    break

    if not samples:
        raise ValueError(f"No usable prompts found in {path}.")
    return samples


def make_builtin_samples() -> list[PromptSample]:
    return [
        make_sample(str(index), [{"role": "user", "content": prompt}])
        for index, prompt in enumerate(BUILTIN_PROMPTS)
    ]


def make_synthetic_samples(args: argparse.Namespace, rng: random.Random) -> list[PromptSample]:
    samples: list[PromptSample] = []
    sample_count = max(args.num_requests, args.concurrency, 1)
    for index in range(sample_count):
        target_tokens = rng.randint(args.min_input_tokens, args.max_input_tokens)
        generation_target = max(1, int(target_tokens * args.synthetic_token_scale))
        prompt = rng.choice(["Answer:", "Explain:", "Analyze:", "Write:"]) + "\n"
        while estimate_tokens(prompt) < generation_target:
            remaining = generation_target - estimate_tokens(prompt)
            snippet = rng.choice(SYNTHETIC_SNIPPETS)
            if estimate_tokens(snippet) > remaining:
                words = snippet.split()
                word_budget = max(1, min(len(words), remaining * 4 // 6))
                start = rng.randint(0, max(0, len(words) - word_budget))
                snippet = " ".join(words[start : start + word_budget])
            prompt += snippet + "\n"
        samples.append(make_sample(f"synthetic-{index}", [{"role": "user", "content": prompt}]))
    return samples


def load_samples(args: argparse.Namespace, rng: random.Random) -> tuple[str, list[PromptSample]]:
    if args.prompts_file:
        samples = load_prompt_file(args.prompts_file, args.max_samples)
        return f"file:{args.prompts_file}", samples
    if args.synthetic:
        return "synthetic", make_synthetic_samples(args, rng)
    if args.prompt:
        return "single-prompt", [
            make_sample("prompt", [{"role": "user", "content": args.prompt}])
        ]
    return "builtin-mixed", make_builtin_samples()


def build_specs(
    args: argparse.Namespace,
    samples: list[PromptSample],
    rng: random.Random,
) -> list[RequestSpec]:
    specs: list[RequestSpec] = []
    for request_id in range(args.num_requests):
        if args.sequential_prompts:
            sample = samples[request_id % len(samples)]
        else:
            sample = rng.choice(samples)

        max_tokens = args.max_tokens
        if args.min_output_tokens is not None or args.max_output_tokens is not None:
            low = args.min_output_tokens if args.min_output_tokens is not None else args.max_tokens
            high = args.max_output_tokens if args.max_output_tokens is not None else args.max_tokens
            max_tokens = rng.randint(low, high)

        specs.append(RequestSpec(request_id=request_id, sample=sample, max_tokens=max_tokens))
    return specs


def build_request(args: argparse.Namespace, spec: RequestSpec) -> Any:
    request = firefly_pb2.ChatCompletionRequest(model=args.model, max_tokens=spec.max_tokens)
    if args.system_prompt:
        msg = request.messages.add()
        msg.role = "system"
        msg.content = args.system_prompt
    for source in spec.sample.messages:
        msg = request.messages.add()
        msg.role = source["role"]
        msg.content = source["content"]
    return request


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "min": min(values),
        "mean": statistics.mean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def response_piece_from_stream(response: Any) -> tuple[str, int]:
    pieces: list[str] = []
    reasoning_chars = 0
    for choice in response.choices:
        reasoning = choice.delta.reasoning_content or ""
        content = choice.delta.content or ""
        if reasoning:
            reasoning_chars += len(reasoning)
            pieces.append(reasoning)
        if content:
            pieces.append(content)
    return "".join(pieces), reasoning_chars


def response_text_from_unary(response: Any) -> tuple[str, int]:
    pieces: list[str] = []
    reasoning_chars = 0
    for choice in response.choices:
        reasoning = choice.reasoning_content or ""
        content = choice.message.content or ""
        if reasoning:
            reasoning_chars += len(reasoning)
            pieces.append(reasoning)
        if content:
            pieces.append(content)
    return "".join(pieces), reasoning_chars


def usage_counts(response: Any) -> tuple[int, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return None
    return prompt_tokens, completion_tokens


def build_openai_payload(args: argparse.Namespace, spec: RequestSpec) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.extend(spec.sample.messages)
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": spec.max_tokens,
        "temperature": 0.0,
        "stream": args.stream,
        "ignore_eos": True,
    }
    if args.stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def openai_piece(payload: dict[str, Any]) -> tuple[str, int]:
    pieces: list[str] = []
    reasoning_chars = 0
    for choice in payload.get("choices") or []:
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning_content") or ""
        content = delta.get("content") or ""
        if reasoning:
            reasoning_chars += len(reasoning)
            pieces.append(reasoning)
        if content:
            pieces.append(content)
    return "".join(pieces), reasoning_chars


def openai_usage(payload: dict[str, Any]) -> tuple[int, int] | None:
    usage = payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return None
    return prompt_tokens, completion_tokens


async def run_openai_stream_request(
    session: Any,
    args: argparse.Namespace,
    spec: RequestSpec,
) -> RequestResult:
    started = time.perf_counter()
    first_piece_at: float | None = None
    piece_times: list[float] = []
    output_parts: list[str] = []
    reasoning_chars = 0
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None

    try:
        timeout = aiohttp.ClientTimeout(total=args.timeout)
        async with session.post(
            f"{args.base_url.rstrip('/')}/chat/completions",
            json=build_openai_payload(args, spec),
            timeout=timeout,
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body[:1000]}")

            buffer = b""
            async for chunk in response.content.iter_any():
                buffer += chunk
                while b"\n\n" in buffer:
                    event, buffer = buffer.split(b"\n\n", 1)
                    for line in event.splitlines():
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == b"[DONE]":
                            continue
                        payload = json.loads(data)
                        usage = openai_usage(payload)
                        if usage is not None:
                            usage_prompt_tokens, usage_completion_tokens = usage
                        piece, piece_reasoning_chars = openai_piece(payload)
                        if not piece:
                            continue
                        now = time.perf_counter()
                        if first_piece_at is None:
                            first_piece_at = now
                        piece_times.append(now)
                        output_parts.append(piece)
                        reasoning_chars += piece_reasoning_chars

        ended = time.perf_counter()
        text = "".join(output_parts)
        completion_tokens = usage_completion_tokens or estimate_tokens(text)
        prompt_tokens = usage_prompt_tokens or spec.sample.approx_input_tokens
        ttft_s = first_piece_at - started if first_piece_at is not None else None
        tpot_s = None
        if ttft_s is not None and completion_tokens > 1:
            tpot_s = max(0.0, ended - started - ttft_s) / (completion_tokens - 1)
        inter_chunks = [piece_times[index] - piece_times[index - 1] for index in range(1, len(piece_times))]
        return RequestResult(
            request_id=spec.request_id,
            prompt_id=spec.sample.sample_id,
            success=True,
            latency_s=ended - started,
            ttft_s=ttft_s,
            tpot_s=tpot_s,
            mean_inter_chunk_s=statistics.mean(inter_chunks) if inter_chunks else None,
            output_chunks=len(piece_times),
            output_chars=len(text),
            reasoning_chars=reasoning_chars,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            token_count_source="usage" if usage_completion_tokens is not None else "estimated",
            max_tokens=spec.max_tokens,
            response_text=text if args.save_responses else None,
        )
    except Exception as exc:
        return RequestResult(
            request_id=spec.request_id,
            prompt_id=spec.sample.sample_id,
            success=False,
            error=str(exc),
            grpc_code=type(exc).__name__,
            latency_s=time.perf_counter() - started,
            prompt_tokens=spec.sample.approx_input_tokens,
            max_tokens=spec.max_tokens,
        )


async def run_openai_unary_request(
    session: Any,
    args: argparse.Namespace,
    spec: RequestSpec,
) -> RequestResult:
    started = time.perf_counter()
    try:
        timeout = aiohttp.ClientTimeout(total=args.timeout)
        async with session.post(
            f"{args.base_url.rstrip('/')}/chat/completions",
            json=build_openai_payload(args, spec),
            timeout=timeout,
        ) as response:
            payload = await response.json()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {payload}")
        ended = time.perf_counter()
        message = (payload.get("choices") or [{}])[0].get("message") or {}
        reasoning = message.get("reasoning_content") or ""
        content = message.get("content") or ""
        text = reasoning + content
        usage = openai_usage(payload)
        prompt_tokens, completion_tokens = usage or (spec.sample.approx_input_tokens, estimate_tokens(text))
        return RequestResult(
            request_id=spec.request_id,
            prompt_id=spec.sample.sample_id,
            success=True,
            latency_s=ended - started,
            output_chunks=1 if text else 0,
            output_chars=len(text),
            reasoning_chars=len(reasoning),
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            token_count_source="usage" if usage is not None else "estimated",
            max_tokens=spec.max_tokens,
            response_text=text if args.save_responses else None,
        )
    except Exception as exc:
        return RequestResult(
            request_id=spec.request_id,
            prompt_id=spec.sample.sample_id,
            success=False,
            error=str(exc),
            grpc_code=type(exc).__name__,
            latency_s=time.perf_counter() - started,
            prompt_tokens=spec.sample.approx_input_tokens,
            max_tokens=spec.max_tokens,
        )


async def run_stream_request(
    stub: Any,
    args: argparse.Namespace,
    spec: RequestSpec,
) -> RequestResult:
    request = build_request(args, spec)
    started = time.perf_counter()
    first_piece_at: float | None = None
    piece_times: list[float] = []
    output_parts: list[str] = []
    output_chars = 0
    reasoning_chars = 0
    output_chunks = 0
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None

    try:
        call = stub.ChatCompletionStream(request, timeout=args.timeout)
        async for response in call:
            usage = usage_counts(response)
            if usage is not None:
                usage_prompt_tokens, usage_completion_tokens = usage

            piece, piece_reasoning_chars = response_piece_from_stream(response)
            if not piece:
                continue
            now = time.perf_counter()
            if first_piece_at is None:
                first_piece_at = now
            piece_times.append(now)
            output_parts.append(piece)
            output_chars += len(piece)
            reasoning_chars += piece_reasoning_chars
            output_chunks += 1

        ended = time.perf_counter()
        text = "".join(output_parts)
        completion_tokens = usage_completion_tokens or estimate_tokens(text)
        prompt_tokens = usage_prompt_tokens or spec.sample.approx_input_tokens
        token_source = "usage" if usage_completion_tokens is not None else "estimated"
        ttft_s = first_piece_at - started if first_piece_at is not None else None
        tpot_s = None
        if ttft_s is not None and completion_tokens > 1:
            tpot_s = max(0.0, ended - started - ttft_s) / (completion_tokens - 1)
        inter_chunks = [
            piece_times[index] - piece_times[index - 1]
            for index in range(1, len(piece_times))
        ]

        return RequestResult(
            request_id=spec.request_id,
            prompt_id=spec.sample.sample_id,
            success=True,
            latency_s=ended - started,
            ttft_s=ttft_s,
            tpot_s=tpot_s,
            mean_inter_chunk_s=statistics.mean(inter_chunks) if inter_chunks else None,
            output_chunks=output_chunks,
            output_chars=output_chars,
            reasoning_chars=reasoning_chars,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            token_count_source=token_source,
            max_tokens=spec.max_tokens,
            response_text=text if args.save_responses else None,
        )
    except grpc.RpcError as exc:
        return RequestResult(
            request_id=spec.request_id,
            prompt_id=spec.sample.sample_id,
            success=False,
            error=exc.details() or str(exc),
            grpc_code=str(exc.code()),
            latency_s=time.perf_counter() - started,
            prompt_tokens=spec.sample.approx_input_tokens,
            max_tokens=spec.max_tokens,
        )
    except Exception as exc:
        return RequestResult(
            request_id=spec.request_id,
            prompt_id=spec.sample.sample_id,
            success=False,
            error=str(exc),
            grpc_code=type(exc).__name__,
            latency_s=time.perf_counter() - started,
            prompt_tokens=spec.sample.approx_input_tokens,
            max_tokens=spec.max_tokens,
        )


async def run_unary_request(
    stub: Any,
    args: argparse.Namespace,
    spec: RequestSpec,
) -> RequestResult:
    request = build_request(args, spec)
    started = time.perf_counter()
    try:
        response = await stub.ChatCompletion(request, timeout=args.timeout)
        ended = time.perf_counter()
        text, reasoning_chars = response_text_from_unary(response)

        usage = usage_counts(response)
        if usage is not None:
            prompt_tokens, completion_tokens = usage
            token_source = "usage"
        else:
            prompt_tokens = spec.sample.approx_input_tokens
            completion_tokens = estimate_tokens(text)
            token_source = "estimated"

        return RequestResult(
            request_id=spec.request_id,
            prompt_id=spec.sample.sample_id,
            success=True,
            latency_s=ended - started,
            output_chunks=1 if text else 0,
            output_chars=len(text),
            reasoning_chars=reasoning_chars,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            token_count_source=token_source,
            max_tokens=spec.max_tokens,
            response_text=text if args.save_responses else None,
        )
    except grpc.RpcError as exc:
        return RequestResult(
            request_id=spec.request_id,
            prompt_id=spec.sample.sample_id,
            success=False,
            error=exc.details() or str(exc),
            grpc_code=str(exc.code()),
            latency_s=time.perf_counter() - started,
            prompt_tokens=spec.sample.approx_input_tokens,
            max_tokens=spec.max_tokens,
        )
    except Exception as exc:
        return RequestResult(
            request_id=spec.request_id,
            prompt_id=spec.sample.sample_id,
            success=False,
            error=str(exc),
            grpc_code=type(exc).__name__,
            latency_s=time.perf_counter() - started,
            prompt_tokens=spec.sample.approx_input_tokens,
            max_tokens=spec.max_tokens,
        )


async def run_one(stub: Any, args: argparse.Namespace, spec: RequestSpec) -> RequestResult:
    if args.backend == "openai":
        if args.stream:
            return await run_openai_stream_request(stub, args, spec)
        return await run_openai_unary_request(stub, args, spec)
    if args.stream:
        return await run_stream_request(stub, args, spec)
    return await run_unary_request(stub, args, spec)


async def run_warmup(stub: Any, args: argparse.Namespace, specs: list[RequestSpec]) -> list[RequestResult]:
    failures: list[RequestResult] = []
    if args.warmup <= 0:
        return failures
    for index in range(args.warmup):
        warm_spec = specs[index % len(specs)]
        warm_max_tokens = min(warm_spec.max_tokens, args.warmup_max_tokens)
        warm_sample = warm_spec.sample
        if args.synthetic and args.warmup_input_tokens > 0:
            rng = random.Random(args.seed + 1_000_003 + index)
            saved_min = args.min_input_tokens
            saved_max = args.max_input_tokens
            try:
                args.min_input_tokens = args.warmup_input_tokens
                args.max_input_tokens = args.warmup_input_tokens
                warm_sample = make_synthetic_samples(args, rng)[0]
            finally:
                args.min_input_tokens = saved_min
                args.max_input_tokens = saved_max
        warm_spec = RequestSpec(
            request_id=warm_spec.request_id,
            sample=warm_sample,
            max_tokens=warm_max_tokens,
        )
        result = await run_one(stub, args, warm_spec)
        if not result.success:
            failures.append(result)
            print(f"Warmup {index} failed: {result.grpc_code} {result.error}")
    return failures


async def run_benchmark(args: argparse.Namespace, specs: list[RequestSpec]) -> tuple[list[RequestResult], float]:
    async def run_with_client(client: Any) -> tuple[list[RequestResult], float]:
        warmup_failures = await run_warmup(client, args, specs)
        if warmup_failures and not args.continue_on_warmup_failure:
            first = warmup_failures[0]
            raise RuntimeError(
                "warmup failed; aborting benchmark. "
                f"First failure: {first.grpc_code} {first.error}"
            )

        semaphore = asyncio.Semaphore(args.concurrency)
        completed = 0

        async def guarded_run(spec: RequestSpec) -> RequestResult:
            nonlocal completed
            async with semaphore:
                result = await run_one(client, args, spec)
            completed += 1
            if args.progress_interval and completed % args.progress_interval == 0:
                print(f"completed {completed}/{len(specs)}")
            return result

        started = time.perf_counter()
        tasks: list[asyncio.Task[RequestResult]] = []
        for index, spec in enumerate(specs):
            if args.request_rate > 0 and index > 0:
                target_time = started + index / args.request_rate
                await asyncio.sleep(max(0.0, target_time - time.perf_counter()))
            tasks.append(asyncio.create_task(guarded_run(spec)))

        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - started
        return results, elapsed

    if args.backend == "openai":
        ensure_aiohttp_module()
        connector = aiohttp.TCPConnector(limit=max(args.concurrency, 1))
        async with aiohttp.ClientSession(connector=connector) as session:
            try:
                timeout = aiohttp.ClientTimeout(total=args.connect_timeout)
                async with session.get(f"{args.base_url.rstrip('/')}/models", timeout=timeout) as response:
                    if response.status != 200:
                        raise RuntimeError(f"OpenAI server readiness check failed: HTTP {response.status}")
            except Exception as exc:
                raise RuntimeError(f"OpenAI server is not ready at {args.base_url}: {exc}") from exc
            return await run_with_client(session)

    ensure_grpc_modules()
    address = args.address or f"{args.host}:{args.port}"
    options = [
        ("grpc.max_receive_message_length", args.max_message_mb * 1024 * 1024),
        ("grpc.max_send_message_length", args.max_message_mb * 1024 * 1024),
    ]
    async with grpc.aio.insecure_channel(address, options=options) as channel:
        await asyncio.wait_for(channel.channel_ready(), timeout=args.connect_timeout)
        stub = firefly_pb2_grpc.InferenceServiceStub(channel)
        return await run_with_client(stub)


def summarize(results: list[RequestResult], elapsed_s: float) -> dict[str, Any]:
    successful = [result for result in results if result.success]
    failed = [result for result in results if not result.success]

    completion_tokens = sum(result.completion_tokens for result in successful)
    prompt_tokens = sum(result.prompt_tokens for result in successful)
    output_chars = sum(result.output_chars for result in successful)
    token_sources = Counter(result.token_count_source for result in successful)

    summary = {
        "requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "elapsed_s": elapsed_s,
        "request_throughput_rps": len(successful) / elapsed_s if elapsed_s else 0.0,
        "completion_token_throughput_tps": completion_tokens / elapsed_s if elapsed_s else 0.0,
        "total_token_throughput_tps": (prompt_tokens + completion_tokens) / elapsed_s if elapsed_s else 0.0,
        "output_char_throughput_cps": output_chars / elapsed_s if elapsed_s else 0.0,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "output_chars": output_chars,
        "token_count_sources": dict(token_sources),
        "latency_s": distribution([result.latency_s for result in successful]),
        "ttft_s": distribution([result.ttft_s for result in successful if result.ttft_s is not None]),
        "tpot_s": distribution([result.tpot_s for result in successful if result.tpot_s is not None]),
        "inter_chunk_s": distribution(
            [
                result.mean_inter_chunk_s
                for result in successful
                if result.mean_inter_chunk_s is not None
            ]
        ),
        "input_tokens": distribution([float(result.prompt_tokens) for result in successful]),
        "completion_tokens_per_request": distribution(
            [float(result.completion_tokens) for result in successful]
        ),
        "failures": dict(Counter(f"{result.grpc_code}: {result.error}" for result in failed)),
    }
    return summary


def fmt_seconds_dist(dist: dict[str, float], scale: float = 1.0, unit: str = "s") -> str:
    if not dist:
        return "n/a"
    return (
        f"mean={dist['mean'] * scale:.2f}{unit} "
        f"p50={dist['p50'] * scale:.2f}{unit} "
        f"p90={dist['p90'] * scale:.2f}{unit} "
        f"p95={dist['p95'] * scale:.2f}{unit} "
        f"p99={dist['p99'] * scale:.2f}{unit}"
    )


def fmt_count_dist(dist: dict[str, float]) -> str:
    if not dist:
        return "n/a"
    return (
        f"mean={dist['mean']:.1f} "
        f"p50={dist['p50']:.1f} "
        f"p90={dist['p90']:.1f} "
        f"p95={dist['p95']:.1f} "
        f"p99={dist['p99']:.1f}"
    )


def output_tokens_desc(args: argparse.Namespace) -> str:
    if args.min_output_tokens is None and args.max_output_tokens is None:
        return str(args.max_tokens)
    low = args.min_output_tokens if args.min_output_tokens is not None else args.max_tokens
    high = args.max_output_tokens if args.max_output_tokens is not None else args.max_tokens
    return f"{low}..{high}"


def print_dry_run(
    args: argparse.Namespace,
    source_name: str,
    samples: list[PromptSample],
    specs: list[RequestSpec],
) -> None:
    input_dist = distribution([float(spec.sample.approx_input_tokens) for spec in specs])
    output_dist = distribution([float(spec.max_tokens) for spec in specs])
    unique_prompts = len({spec.sample.sample_id for spec in specs})

    print("\n--- Benchmark Dry Run ---")
    print(f"Prompt source:       {source_name} ({len(samples)} loaded samples)")
    print(f"Requests:            {len(specs)}")
    print(f"Unique prompts used: {unique_prompts}")
    print(f"Concurrency:         {args.concurrency}")
    print(f"Mode:                {'stream' if args.stream else 'unary'}")
    print(f"Output max tokens:   {output_tokens_desc(args)}")
    print(f"Input tokens/req:    {fmt_count_dist(input_dist)}")
    print(f"Max tokens/req:      {fmt_count_dist(output_dist)}")
    print("\nSample requests:")
    for spec in specs[: min(5, len(specs))]:
        text = messages_text(spec.sample.messages).replace("\n", " ")
        if len(text) > 120:
            text = text[:117] + "..."
        print(
            f"  #{spec.request_id}: prompt={spec.sample.sample_id} "
            f"input~{spec.sample.approx_input_tokens} max_tokens={spec.max_tokens} | {text}"
        )


def print_summary(args: argparse.Namespace, source_name: str, samples: list[PromptSample], summary: dict[str, Any]) -> None:
    address = args.base_url if args.backend == "openai" else (args.address or f"{args.host}:{args.port}")
    token_sources = set(summary["token_count_sources"])
    if token_sources == {"usage"}:
        token_note = "actual"
    elif token_sources == {"estimated"}:
        token_note = "estimated"
    else:
        token_note = "estimated/actual"

    backend_name = "OpenAI HTTP" if args.backend == "openai" else "Firefly gRPC"
    print(f"\n--- {backend_name} Serving Benchmark ---")
    print(f"Target:              {address}")
    print(f"Model:               {args.model}")
    print(f"Mode:                {'stream' if args.stream else 'unary'}")
    print(f"Prompt source:       {source_name} ({len(samples)} samples)")
    print(f"Requests:            {summary['requests']}")
    print(f"Concurrency:         {args.concurrency}")
    if args.request_rate > 0:
        print(f"Request rate:        {args.request_rate:.2f} req/s")
    print(f"Output max tokens:   {output_tokens_desc(args)}")

    print("\n--- Throughput ---")
    print(f"Successful:          {summary['successful_requests']}")
    print(f"Failed:              {summary['failed_requests']}")
    print(f"Elapsed:             {summary['elapsed_s']:.2f} s")
    print(f"Requests/s:          {summary['request_throughput_rps']:.2f}")
    print(f"Completion toks/s:   {summary['completion_token_throughput_tps']:.2f} ({token_note})")
    print(f"Total toks/s:        {summary['total_token_throughput_tps']:.2f} ({token_note})")
    if args.stream and summary["tpot_s"]:
        mean_tpot = summary["tpot_s"]["mean"]
        if mean_tpot > 0:
            print(f"Decode active est.:  {args.concurrency / mean_tpot:.2f} toks/s (concurrency / mean TPOT)")
    print(f"Output chars/s:      {summary['output_char_throughput_cps']:.2f}")

    print("\n--- Latency ---")
    print(f"E2E latency:         {fmt_seconds_dist(summary['latency_s'])}")
    if args.stream:
        print(f"TTFT:                {fmt_seconds_dist(summary['ttft_s'], 1000.0, 'ms')}")
        print(f"TPOT:                {fmt_seconds_dist(summary['tpot_s'], 1000.0, 'ms')}")
        print(f"Inter chunk:         {fmt_seconds_dist(summary['inter_chunk_s'], 1000.0, 'ms')}")

    print("\n--- Size ---")
    print(f"Input tokens/req:    {fmt_count_dist(summary['input_tokens'])}")
    print(f"Output tokens/req:   {fmt_count_dist(summary['completion_tokens_per_request'])}")

    if summary["failures"]:
        print("\n--- Failures ---")
        for message, count in list(summary["failures"].items())[:5]:
            print(f"{count}x {message}")


def write_json(path: str, args: argparse.Namespace, source_name: str, summary: dict[str, Any], results: list[RequestResult]) -> None:
    payload = {
        "config": {key: value for key, value in vars(args).items()},
        "prompt_source": source_name,
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Firefly gRPC or OpenAI-compatible serving with varied prompts and concurrency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="localhost", help="Server host.")
    parser.add_argument("--port", type=int, default=50051, help="Server port.")
    parser.add_argument("--address", help="Override host/port with an address such as localhost:50051.")
    parser.add_argument("--backend", choices=["grpc", "openai"], default="grpc", help="Serving protocol backend.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible API base URL.")
    parser.add_argument("--model", default="qwen3", help="Model name sent in requests.")
    parser.add_argument("-n", "--num-requests", type=int, default=64, help="Total measured requests.")
    parser.add_argument("-c", "--concurrency", type=int, default=8, help="Maximum in-flight requests.")
    parser.add_argument("--request-rate", type=float, default=0.0, help="Open-loop arrival rate. 0 means submit as fast as concurrency allows.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Default max_tokens per request.")
    parser.add_argument("--min-output-tokens", type=int, help="Randomize max_tokens lower bound.")
    parser.add_argument("--max-output-tokens", type=int, help="Randomize max_tokens upper bound.")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True, help="Use streaming RPC.")
    parser.add_argument("--prompt", help="Use one explicit prompt. By default, a mixed built-in prompt set is used.")
    parser.add_argument("--prompts-file", "--dataset", dest="prompts_file", help="Prompt dataset: jsonl/json/txt.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples loaded from prompt dataset. 0 means no limit.")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic prompts with varied approximate input lengths.")
    parser.add_argument("--min-input-tokens", type=int, default=64, help="Synthetic prompt lower bound in approximate tokens.")
    parser.add_argument("--max-input-tokens", type=int, default=512, help="Synthetic prompt upper bound in approximate tokens.")
    parser.add_argument(
        "--synthetic-token-scale",
        type=float,
        default=1.0,
        help="Scale synthetic prompt generation target to compensate for tokenizer/estimator drift.",
    )
    parser.add_argument("--sequential-prompts", action="store_true", help="Cycle prompts in file order instead of random sampling.")
    parser.add_argument("--system-prompt", help="Optional system message prepended to every request.")
    parser.add_argument("--warmup", type=int, default=2, help="Sequential warmup requests excluded from metrics.")
    parser.add_argument(
        "--warmup-max-tokens",
        type=int,
        default=16,
        help="Maximum generated tokens for warmup requests.",
    )
    parser.add_argument(
        "--warmup-input-tokens",
        type=int,
        default=128,
        help="Synthetic prompt target used for warmup. Set 0 to reuse measured prompts.",
    )
    parser.add_argument(
        "--continue-on-warmup-failure",
        action="store_true",
        help="Continue measured requests even if warmup fails.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    parser.add_argument("--connect-timeout", type=float, default=10.0, help="Channel connection timeout in seconds.")
    parser.add_argument("--max-message-mb", type=int, default=64, help="gRPC send/receive message size limit in MiB.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for prompt/output sampling.")
    parser.add_argument("--progress-interval", type=int, default=0, help="Print progress every N completed requests. 0 disables progress.")
    parser.add_argument("--save-responses", action="store_true", help="Store response text in --output-json.")
    parser.add_argument("--output-json", help="Write detailed benchmark results to this JSON file.")
    parser.add_argument("--dry-run", action="store_true", help="Print the sampled request mix without contacting the server.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be > 0.")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0.")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0.")
    if args.request_rate < 0:
        raise ValueError("--request-rate must be >= 0.")
    if args.min_input_tokens <= 0 or args.max_input_tokens <= 0:
        raise ValueError("--min-input-tokens and --max-input-tokens must be > 0.")
    if args.min_input_tokens > args.max_input_tokens:
        raise ValueError("--min-input-tokens must be <= --max-input-tokens.")
    if args.warmup_input_tokens < 0:
        raise ValueError("--warmup-input-tokens must be >= 0.")
    if args.synthetic_token_scale <= 0:
        raise ValueError("--synthetic-token-scale must be > 0.")
    if args.min_output_tokens is not None and args.min_output_tokens <= 0:
        raise ValueError("--min-output-tokens must be > 0.")
    if args.max_output_tokens is not None and args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be > 0.")
    if args.min_output_tokens is not None or args.max_output_tokens is not None:
        low = args.min_output_tokens if args.min_output_tokens is not None else args.max_tokens
        high = args.max_output_tokens if args.max_output_tokens is not None else args.max_tokens
        if low > high:
            raise ValueError("Effective output token range is invalid; lower bound is greater than upper bound.")
    prompt_sources = sum(bool(value) for value in [args.prompt, args.prompts_file, args.synthetic])
    if prompt_sources > 1:
        raise ValueError("Use only one of --prompt, --prompts-file/--dataset, or --synthetic.")


async def async_main() -> None:
    args = parse_args()
    validate_args(args)

    rng = random.Random(args.seed)
    source_name, samples = load_samples(args, rng)
    specs = build_specs(args, samples, rng)

    if args.dry_run:
        print_dry_run(args, source_name, samples, specs)
        return

    print(
        f"Starting benchmark: {args.num_requests} requests, "
        f"concurrency={args.concurrency}, prompts={source_name}, "
        f"mode={'stream' if args.stream else 'unary'}"
    )

    results, elapsed_s = await run_benchmark(args, specs)
    summary = summarize(results, elapsed_s)
    print_summary(args, source_name, samples, summary)

    if args.output_json:
        write_json(args.output_json, args, source_name, summary, results)
        print(f"\nWrote JSON results to {args.output_json}")


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:
        print(f"Benchmark failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
