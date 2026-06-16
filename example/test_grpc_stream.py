#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

grpc: Any = None
firefly_pb2: Any = None
firefly_pb2_grpc: Any = None


def ensure_grpc_modules() -> None:
    global grpc, firefly_pb2, firefly_pb2_grpc

    if grpc is None:
        try:
            import grpc as grpc_module
        except ImportError as exc:
            raise RuntimeError("Python package grpcio is required to run this client.") from exc
        grpc = grpc_module

    if firefly_pb2 is None or firefly_pb2_grpc is None:
        try:
            import firefly_pb2 as pb2_module
            import firefly_pb2_grpc as pb2_grpc_module
        except ImportError as exc:
            raise RuntimeError(
                "Please generate python grpc stubs first: "
                "python -m grpc_tools.protoc -I../proto --python_out=. "
                "--grpc_python_out=. ../proto/firefly.proto"
            ) from exc
        firefly_pb2 = pb2_module
        firefly_pb2_grpc = pb2_grpc_module


DEFAULT_PROMPT = (
    "Write a C++ program to implement quicksort, and please demonstrate your reasoning process step by step."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one Firefly streaming chat request and print deltas as they arrive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="localhost", help="Server host.")
    parser.add_argument("--port", type=int, default=50051, help="Server port.")
    parser.add_argument("--address", help="Override host/port with an address such as localhost:50051.")
    parser.add_argument("--model", default="qwen3", help="Model name sent in the request.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="max_tokens sent in the request.")
    parser.add_argument("--prompt", help="Prompt text. If omitted, stdin is used when piped.")
    parser.add_argument("--prompt-file", help="Read prompt text from a file.")
    parser.add_argument("--system-prompt", help="Optional system message prepended to the request.")
    parser.add_argument("--timeout", type=float, default=120.0, help="RPC timeout in seconds.")
    parser.add_argument("--no-color", action="store_true", help="Disable gray coloring for reasoning_content.")
    parser.add_argument("--stats", action="store_true", help="Print TTFT and total latency after completion.")
    return parser.parse_args()


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return DEFAULT_PROMPT


def build_request(args: argparse.Namespace, prompt: str) -> Any:
    request = firefly_pb2.ChatCompletionRequest(model=args.model, max_tokens=args.max_tokens)
    if args.system_prompt:
        msg = request.messages.add()
        msg.role = "system"
        msg.content = args.system_prompt
    msg = request.messages.add()
    msg.role = "user"
    msg.content = prompt
    return request


def write_delta(text: str, gray: bool) -> None:
    if not text:
        return
    if gray:
        sys.stdout.write("\033[90m" + text + "\033[0m")
    else:
        sys.stdout.write(text)
    sys.stdout.flush()


def run() -> None:
    args = parse_args()
    prompt = read_prompt(args)
    if not prompt:
        print("Prompt is empty.")
        sys.exit(1)

    try:
        ensure_grpc_modules()
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    address = args.address or f"{args.host}:{args.port}"
    with grpc.insecure_channel(address) as channel:
        stub = firefly_pb2_grpc.InferenceServiceStub(channel)
        request = build_request(args, prompt)

        print(f"Sending streaming request to {address}...\n")

        start = time.perf_counter()
        first_token_at = None
        chunks = 0
        chars = 0

        try:
            responses = stub.ChatCompletionStream(request, timeout=args.timeout)
            for response in responses:
                for choice in response.choices:
                    reasoning = choice.delta.reasoning_content
                    content = choice.delta.content
                    if reasoning or content:
                        now = time.perf_counter()
                        if first_token_at is None:
                            first_token_at = now
                        chunks += 1
                    if reasoning:
                        chars += len(reasoning)
                        write_delta(reasoning, gray=not args.no_color)
                    if content:
                        chars += len(content)
                        write_delta(content, gray=False)
            total_s = time.perf_counter() - start
            print()
            if args.stats:
                ttft_s = first_token_at - start if first_token_at is not None else 0.0
                print(f"\nTTFT: {ttft_s:.3f}s | total: {total_s:.3f}s | chunks: {chunks} | chars: {chars}")
        except grpc.RpcError as exc:
            print(f"\nRPC failed: {exc.code()} - {exc.details()}")
            sys.exit(1)


if __name__ == "__main__":
    run()
