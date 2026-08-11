#!/usr/bin/env python3
"""Run a reproducible Firefly vs vLLM performance matrix.

The script starts a serving backend for each model, runs
benchmark_serving.py over calibrated prompts, and stores per-cell JSON results
plus server logs under benchmarks/results/ (gitignored).

Everything machine-specific is configurable; sensible defaults assume the
usual repository layout. Examples:

  # Firefly only (uses build/bin/firefly_server, .venv/bin/python by default)
  python benchmarks/run_perf_matrix.py --engine firefly

  # vLLM matrix
  python benchmarks/run_perf_matrix.py --engine vllm

  # Custom environments and model directories
  python benchmarks/run_perf_matrix.py --engine firefly \\
      --firefly-server /path/to/release/firefly_server \\
      --firefly-python /path/to/venv/bin/python \\
      --model-dirs qwen3=/models/qwen3-0.6b

Environment variables FIREFLY_SERVER_BIN, FIREFLY_PYTHON, VLLM_BIN and
VLLM_PYTHON are honored as defaults for the corresponding flags.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "benchmarks" / "results"
PREPROMPTS = RESULTS_ROOT / "prompts"


@dataclass
class Config:
    model_dirs: dict[str, Path]
    firefly_server: str
    firefly_python: str
    vllm_bin: str
    vllm_python: str
    firefly_port: int = 50051
    vllm_port: int = 8000
    prefill_chunk: int = 768
    gpu_memory_utilization: float = 0.75
    results_root: Path = RESULTS_ROOT
    pre_prompts: Path = PREPROMPTS


def wait_for_port(host: str, port: int, timeout_s: float = 240.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            time.sleep(1.0)
    return False


def wait_for_http(url: str, timeout_s: float = 240.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(2.0)
    return False


def firefly_command(cfg: Config, model: str) -> list[str]:
    return [
        cfg.firefly_server,
        str(cfg.model_dirs[model]),
        "--max-prefill-chunk-size",
        str(cfg.prefill_chunk),
        "--log-level",
        "warn",
        "--log-color",
        "never",
    ]


def vllm_command(cfg: Config, model: str) -> list[str]:
    return [
        cfg.vllm_bin,
        "serve",
        str(cfg.model_dirs[model]),
        "--served-model-name",
        model,
        "--port",
        str(cfg.vllm_port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--enable-chunked-prefill",
        "--max-num-batched-tokens",
        str(cfg.prefill_chunk),
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
        "--disable-log-stats",
    ]


def benchmark_command(
    cfg: Config,
    engine: str,
    model: str,
    input_len: int,
    output_len: int,
    concurrency: int,
    num_requests: int,
    output_json: Path,
    seed: int,
) -> list[str]:
    prompt_file = cfg.pre_prompts / f"{model}_in{input_len}.jsonl"
    client_python = cfg.firefly_python if engine != "vllm" else cfg.vllm_python
    common = [
        client_python,
        str(ROOT / "benchmarks/benchmark_serving.py"),
        "--model",
        model,
        "-n",
        str(num_requests),
        "-c",
        str(concurrency),
        "--prompts-file",
        str(prompt_file),
        "--max-tokens",
        str(output_len),
        "--min-output-tokens",
        str(output_len),
        "--max-output-tokens",
        str(output_len),
        "--stream",
        "--warmup",
        "2",
        "--warmup-max-tokens",
        "16",
        "--seed",
        str(seed),
        "--timeout",
        "600",
        "--output-json",
        str(output_json),
    ]
    if engine == "vllm":
        return common + ["--backend", "openai", "--base-url", f"http://127.0.0.1:{cfg.vllm_port}/v1"]
    return common + ["--backend", "grpc", "--port", str(cfg.firefly_port)]


def run_matrix(
    cfg: Config,
    engine: str,
    models: list[str],
    inputs: list[int],
    outputs: list[int],
    concurrencies: list[int],
    num_requests: int,
    seed: int,
    smoke: bool,
    resume: bool,
) -> int:
    result_dir = cfg.results_root / engine
    result_dir.mkdir(parents=True, exist_ok=True)
    cfg.pre_prompts.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("FIREFLY_GPU_MEMORY_UTILIZATION", str(cfg.gpu_memory_utilization))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    if engine == "firefly-int8":
        env["FIREFLY_KV_CACHE_DTYPE"] = "int8"
    if engine in ("firefly", "firefly-int8"):
        # The default fused GDN prefill backend hangs on long Qwen3.5 inputs
        # (single 2048-token request exceeds 600 s). chunk64 is the working
        # path and is therefore used for the comparison.
        env["FIREFLY_QWEN35_GDN_PREFILL_BACKEND"] = "chunk64"

    for model in models:
        server = None
        log_path = result_dir / f"server-{model}.log"
        try:
            if engine == "vllm":
                cmd = vllm_command(cfg, model)
                log = open(log_path, "w")
                server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if server.poll() is not None:
                    print(f"vLLM exited early; log: {log_path}", file=sys.stderr)
                    return 1
                ready = wait_for_http(f"http://127.0.0.1:{cfg.vllm_port}/v1/models")
            else:
                cmd = firefly_command(cfg, model)
                log = open(log_path, "w")
                server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if server.poll() is not None:
                    print(f"Firefly exited early; log: {log_path}", file=sys.stderr)
                    return 1
                ready = wait_for_port("127.0.0.1", cfg.firefly_port)
            if not ready:
                print(f"server failed to become ready for {model}; log: {log_path}", file=sys.stderr)
                return 1
            time.sleep(3)

            for input_len in inputs:
                for output_len in outputs:
                    for concurrency in concurrencies:
                        name = f"{model}_in{input_len}_out{output_len}_c{concurrency}"
                        output_json = result_dir / f"{name}.json"
                        if output_json.exists():
                            if resume:
                                print(f"[{engine}] {name} already done, skipping", flush=True)
                                continue
                            output_json.unlink()
                        cmd = benchmark_command(cfg, engine, model, input_len, output_len, concurrency,
                                                num_requests, output_json, seed)
                        print(f"[{engine}] {name} n={num_requests}", flush=True)
                        started = time.monotonic()
                        proc = subprocess.run(cmd, cwd=ROOT, env=env)
                        elapsed = time.monotonic() - started
                        if proc.returncode != 0:
                            print(f"  FAILED rc={proc.returncode} ({elapsed:.0f}s)", file=sys.stderr)
                            return 1
                        with output_json.open() as handle:
                            summary = json.load(handle)["summary"]
                        print(
                            f"  ok {elapsed:.0f}s req/s={summary['request_throughput_rps']:.2f} "
                            f"tok/s={summary['completion_token_throughput_tps']:.1f} "
                            f"TTFT_mean_ms={summary['ttft_s']['mean']*1000:.1f} "
                            f"TPOT_mean_ms={summary['tpot_s']['mean']*1000:.2f}",
                            flush=True,
                        )
                        if smoke:
                            return 0
        finally:
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)
    return 0


def parse_model_dirs(values: list[str] | None) -> dict[str, Path]:
    defaults = {"qwen3": ROOT / "qwen3", "qwen35": ROOT / "qwen35"}
    if not values:
        return defaults
    result: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--model-dirs expects name=path, got: {item}")
        name, path = item.split("=", 1)
        result[name.strip()] = Path(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["firefly", "vllm", "firefly-int8"], required=True)
    parser.add_argument("--models", nargs="+", default=["qwen3", "qwen35"])
    parser.add_argument("--model-dirs", nargs="+", help="name=path pairs, e.g. qwen3=/models/qwen3")
    parser.add_argument("--inputs", nargs="+", type=int, default=[128, 1024, 2048])
    parser.add_argument("--outputs", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("-n", "--num-requests", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--smoke", action="store_true", help="Run one config per model and stop.")
    parser.add_argument("--resume", action="store_true", help="Skip configs whose result JSON already exists.")
    parser.add_argument("--firefly-server", default=os.environ.get("FIREFLY_SERVER_BIN",
                                                                  str(ROOT / "build/bin/firefly_server")))
    parser.add_argument("--firefly-python", default=os.environ.get("FIREFLY_PYTHON",
                                                                   str(ROOT / ".venv/bin/python")))
    parser.add_argument("--vllm-bin", default=os.environ.get("VLLM_BIN", str(ROOT / ".venv-vllm/bin/vllm")))
    parser.add_argument("--vllm-python", default=os.environ.get("VLLM_PYTHON",
                                                                str(ROOT / ".venv-vllm/bin/python")))
    parser.add_argument("--firefly-port", type=int, default=50051)
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--prefill-chunk", type=int, default=768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()

    if args.smoke:
        args.inputs = [128]
        args.outputs = [32]
        args.concurrency = [1]
        args.num_requests = 4

    cfg = Config(
        model_dirs=parse_model_dirs(args.model_dirs),
        firefly_server=args.firefly_server,
        firefly_python=args.firefly_python,
        vllm_bin=args.vllm_bin,
        vllm_python=args.vllm_python,
        firefly_port=args.firefly_port,
        vllm_port=args.vllm_port,
        prefill_chunk=args.prefill_chunk,
        gpu_memory_utilization=args.gpu_memory_utilization,
        results_root=args.results_dir,
        pre_prompts=args.results_dir / "prompts",
    )
    return run_matrix(cfg, args.engine, args.models, args.inputs, args.outputs, args.concurrency,
                      args.num_requests, args.seed, args.smoke, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
