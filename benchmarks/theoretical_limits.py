#!/usr/bin/env python3
"""Reproducible latency lower bounds for Qwen3-0.6B on RTX 4060 Laptop.

The script intentionally computes lower bounds on latency, not performance
predictions.  A lower latency number is an upper bound on achievable TPS.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    hidden: int = 1024
    intermediate: int = 3072
    layers: int = 28
    query_heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    vocab: int = 151_936
    bytes_per_element: int = 2

    @property
    def transformer_linear_parameters(self) -> int:
        return self.layers * sum(k * n for _, k, n in linear_shapes(self))

    @property
    def lm_head_parameters(self) -> int:
        return self.hidden * self.vocab

    @property
    def active_linear_parameters(self) -> int:
        return self.transformer_linear_parameters + self.lm_head_parameters


@dataclass(frozen=True)
class Hardware:
    sms: int = 24
    max_sm_clock_hz: float = 3.105e9
    reported_sm_clock_hz: float = 2.250e9
    memory_clock_hz: float = 8.001e9
    memory_bus_bits: int = 128
    l2_bytes: int = 33_554_432

    @property
    def dram_bytes_per_second(self) -> float:
        return 2 * self.memory_clock_hz * self.memory_bus_bits / 8

    def peaks(self, clock_hz: float) -> dict[str, float]:
        return {
            "fp32_fma_flops": self.sms * clock_hz * 256,
            "fp32_results": self.sms * clock_hz * 128,
            "fp32_max_results": self.sms * clock_hz * 64,
            "bf16_tensor_flops": self.sms * clock_hz * 512,
            "int8_tensor_ops": self.sms * clock_hz * 2048,
            "int4_tensor_ops": self.sms * clock_hz * 4096,
            "sfu_results": self.sms * clock_hz * 16,
            "barrier_arrivals": self.sms * clock_hz * 16,
        }


MODEL = Model()
HARDWARE = Hardware()
PREFILL_CHUNK = 4096
DECODE_SPLIT = 256


def linear_shapes(model: Model = MODEL) -> tuple[tuple[str, int, int], ...]:
    return (
        ("q_proj", model.hidden, model.query_heads * model.head_dim),
        ("k_proj", model.hidden, model.kv_heads * model.head_dim),
        ("v_proj", model.hidden, model.kv_heads * model.head_dim),
        ("o_proj", model.query_heads * model.head_dim, model.hidden),
        ("gate_proj", model.hidden, model.intermediate),
        ("up_proj", model.hidden, model.intermediate),
        ("down_proj", model.intermediate, model.hidden),
    )


def chunks(prompt_tokens: int, chunk_size: int = PREFILL_CHUNK) -> list[tuple[int, int]]:
    result = []
    past = 0
    while past < prompt_tokens:
        size = min(chunk_size, prompt_tokens - past)
        result.append((past, size))
        past += size
    return result


def causal_edges(past: int, query_tokens: int) -> int:
    return query_tokens * past + query_tokens * (query_tokens + 1) // 2


def useful_attention_flops(edges: int, model: Model = MODEL) -> int:
    return 4 * model.layers * model.query_heads * model.head_dim * edges


def optimal_attention_exp_calls(edges: int, model: Model = MODEL) -> int:
    return model.layers * model.query_heads * edges


def current_prefill_attention_counts(prompt_tokens: int, model: Model = MODEL) -> dict[str, int]:
    edges = prompt_tokens * (prompt_tokens + 1) // 2
    scores = model.layers * model.query_heads * edges
    first_size = min(prompt_tokens, PREFILL_CHUNK)
    first_edges = causal_edges(0, first_size)
    later_edges = edges - first_edges
    first_scores = model.layers * model.query_heads * first_edges
    later_scores = model.layers * model.query_heads * later_edges
    return {
        "edges": edges,
        "scores": scores,
        "fp32_ops": 1023 * first_scores + 1279 * later_scores,
        "max_ops": 128 * scores,
        "exp_calls": 256 * scores,
        "barrier_arrivals": 32 * scores,
        "global_kv_bytes": model.layers
        * model.query_heads
        * (first_edges * 2 * model.head_dim * 2 + later_edges * 2 * model.head_dim),
        "logical_metadata_bytes": 12 * later_scores,
    }


def kv_bytes_per_token(bits: int, model: Model = MODEL) -> int:
    if bits == 16:
        return model.layers * model.kv_heads * model.head_dim * 2 * 2
    packed_kv = model.layers * model.kv_heads * model.head_dim * 2 * bits // 8
    scales = model.layers * model.kv_heads * 2 * 4
    return packed_kv + scales


def decode_useful_counts(past_tokens: int, model: Model = MODEL) -> dict[str, int]:
    visible = past_tokens + 1
    return {
        "linear_flops": 2 * model.active_linear_parameters,
        "attention_flops": 4 * model.layers * model.query_heads * model.head_dim * visible,
        "attention_exp_calls": model.layers * model.query_heads * visible,
        "argmax_comparisons": model.vocab - 1,
    }


def current_int8_decode_attention_counts(past_tokens: int, model: Model = MODEL) -> dict[str, int]:
    visible = past_tokens + 1
    splits = math.ceil(visible / DECODE_SPLIT)
    full_tiles = visible // 16
    tail_tokens = visible % 16
    issued_int8_ops_per_layer = (
        full_tiles * model.kv_heads * 4 * 16 * 8 * 32 * 2
        + tail_tokens * model.kv_heads * 2 * model.head_dim * 2
    )
    main_exp_per_layer = 2048 * visible - 1024 * splits
    reduce_exp_per_layer = 2064 * splits
    scratch_payload_per_layer = model.query_heads * splits * (2 * 4 + model.head_dim * 4)
    scratch_global_bytes_per_layer = scratch_payload_per_layer + 16_576 * splits
    return {
        "visible": visible,
        "splits": splits,
        "issued_int8_ops": model.layers * issued_int8_ops_per_layer,
        "exp_calls": model.layers * (main_exp_per_layer + reduce_exp_per_layer),
        "scratch_global_bytes": model.layers * scratch_global_bytes_per_layer,
    }


def linear_stage_floor(rows: int, include_lm_head: bool, clock_hz: float) -> float:
    peaks = HARDWARE.peaks(clock_hz)
    total = 0.0
    for _, k_size, n_size in linear_shapes():
        operations = 2 * rows * k_size * n_size
        weight_bytes = 2 * k_size * n_size
        total += MODEL.layers * max(
            operations / peaks["bf16_tensor_flops"],
            weight_bytes / HARDWARE.dram_bytes_per_second,
        )
    if include_lm_head:
        operations = 2 * MODEL.hidden * MODEL.vocab
        weight_bytes = 2 * MODEL.hidden * MODEL.vocab
        total += max(
            operations / peaks["bf16_tensor_flops"],
            weight_bytes / HARDWARE.dram_bytes_per_second,
        )
    return total


def optimal_elementwise_sfu_floor(tokens: int, clock_hz: float) -> float:
    peaks = HARDWARE.peaks(clock_hz)
    swiglu_calls = MODEL.layers * tokens * MODEL.intermediate
    rms_calls = tokens * (1 + MODEL.layers * (MODEL.query_heads + MODEL.kv_heads + 1) + MODEL.layers - 1) + 1
    return (swiglu_calls + rms_calls) / peaks["sfu_results"]


def current_elementwise_sfu_floor(tokens: int, clock_hz: float) -> float:
    peaks = HARDWARE.peaks(clock_hz)
    rope_calls = MODEL.layers * tokens * (MODEL.query_heads + MODEL.kv_heads) * (MODEL.head_dim // 2) * 4
    return optimal_elementwise_sfu_floor(tokens, clock_hz) + rope_calls / peaks["sfu_results"]


def current_prefill_fixed_global_bytes(prompt_tokens: int, kv_bits: int, model: Model = MODEL) -> int:
    query_width = model.query_heads * model.head_dim
    kv_width = model.kv_heads * model.head_dim
    linear_io = 45_056
    q_norm = 8 * query_width
    k_norm = 8 * kv_width
    rope = 4 * (query_width + kv_width)
    if kv_bits == 8:
        kv_append = model.kv_heads * model.head_dim * 10 + model.kv_heads * 2 * 4
    else:
        kv_append = model.kv_heads * model.head_dim * 8
    attention_output = 2 * query_width
    post_attention_norm = 12 * model.hidden
    swiglu = 6 * model.intermediate
    next_norm = 12 * model.hidden
    last_add = 6 * model.hidden
    per_layer = linear_io + q_norm + k_norm + rope + kv_append + attention_output + post_attention_norm + swiglu
    per_token = model.layers * per_layer + (model.layers - 1) * next_norm + last_add
    embedding_and_initial_norm = 4 + 4 * model.hidden + 8 * model.hidden
    final_chunk_size = chunks(prompt_tokens)[-1][1]
    last_hidden_copy = 0 if final_chunk_size == 1 else 4 * model.hidden
    final_norm = 8 * model.hidden
    lm_head_io = 2 * model.hidden + 2 * model.vocab
    argmax_io = 2 * model.vocab + 4
    return prompt_tokens * (per_token + embedding_and_initial_norm) + last_hidden_copy + final_norm + lm_head_io + argmax_io


def current_decode_fixed_global_bytes(kv_bits: int, model: Model = MODEL) -> int:
    query_width = model.query_heads * model.head_dim
    kv_width = model.kv_heads * model.head_dim
    linear_io = 45_056
    q_norm = 8 * query_width
    k_norm = 8 * kv_width
    rope = 4 * (query_width + kv_width)
    if kv_bits == 8:
        kv_append = model.kv_heads * model.head_dim * 10 + model.kv_heads * 2 * 4
        q_quantize = 2 * query_width + model.kv_heads * 2 * model.head_dim + model.query_heads * 4
    else:
        kv_append = model.kv_heads * model.head_dim * 8
        q_quantize = 0
    attention_io = q_quantize + 2 * query_width
    post_attention_norm = 12 * model.hidden
    swiglu = 6 * model.intermediate
    next_norm = 12 * model.hidden
    last_add = 6 * model.hidden
    per_layer = linear_io + q_norm + k_norm + rope + kv_append + attention_io + post_attention_norm + swiglu
    transformer = model.layers * per_layer + (model.layers - 1) * next_norm + last_add
    embedding_and_initial_norm = 4 + 4 * model.hidden + 8 * model.hidden
    final_norm = 8 * model.hidden
    lm_head_io = 2 * model.hidden + 2 * model.vocab
    argmax_io = 2 * model.vocab + 4
    return transformer + embedding_and_initial_norm + final_norm + lm_head_io + argmax_io


def ttft_bounds(prompt_tokens: int, kv_bits: int, clock_hz: float) -> dict[str, float]:
    peaks = HARDWARE.peaks(clock_hz)
    edges = prompt_tokens * (prompt_tokens + 1) // 2
    transformer_flops = 2 * prompt_tokens * MODEL.transformer_linear_parameters
    lm_head_flops = 2 * MODEL.lm_head_parameters
    attention_flops = useful_attention_flops(edges)
    ideal_exp = optimal_attention_exp_calls(edges)
    transformer_weight_bytes = 2 * MODEL.transformer_linear_parameters
    lm_head_bytes = 2 * MODEL.lm_head_parameters
    embedding_bytes = prompt_tokens * MODEL.hidden * 2
    kv_write_bytes = prompt_tokens * kv_bytes_per_token(kv_bits)
    physical_hbm_bytes = max(
        0,
        transformer_weight_bytes
        + lm_head_bytes
        + embedding_bytes
        + kv_write_bytes
        - HARDWARE.l2_bytes,
    )
    physical = max(
        (transformer_flops + lm_head_flops + attention_flops) / peaks["bf16_tensor_flops"],
        physical_hbm_bytes / HARDWARE.dram_bytes_per_second,
        ideal_exp / peaks["sfu_results"],
    )

    optimal_linear = linear_stage_floor(prompt_tokens, True, clock_hz)
    optimal_attention = max(
        attention_flops / peaks["bf16_tensor_flops"],
        ideal_exp / peaks["sfu_results"],
    )
    optimal_dag = max(0.0, optimal_linear - HARDWARE.l2_bytes / HARDWARE.dram_bytes_per_second)
    kv_append = kv_write_bytes / HARDWARE.dram_bytes_per_second
    optimal_dag += optimal_attention + kv_append + optimal_elementwise_sfu_floor(prompt_tokens, clock_hz)

    current_linear = sum(
        linear_stage_floor(size, index == len(chunks(prompt_tokens)) - 1, clock_hz)
        for index, (_, size) in enumerate(chunks(prompt_tokens))
    )
    current_linear = max(
        0.0,
        current_linear - len(chunks(prompt_tokens)) * HARDWARE.l2_bytes / HARDWARE.dram_bytes_per_second,
    )
    current_counts = current_prefill_attention_counts(prompt_tokens)
    current_attention = max(
        current_counts["fp32_ops"] / peaks["fp32_fma_flops"],
        current_counts["max_ops"] / peaks["fp32_max_results"],
        current_counts["exp_calls"] / peaks["sfu_results"],
        current_counts["barrier_arrivals"] / peaks["barrier_arrivals"],
    )
    current = current_linear + current_attention + kv_append + current_elementwise_sfu_floor(prompt_tokens, clock_hz)
    return {
        "physical": physical,
        "optimal_dag": optimal_dag,
        "current": current,
        "useful_flops": transformer_flops + lm_head_flops + attention_flops,
        "current_global_kv_bytes": current_counts["global_kv_bytes"],
    }


def decode_bounds(past_tokens: int, kv_bits: int, clock_hz: float) -> dict[str, float]:
    peaks = HARDWARE.peaks(clock_hz)
    visible = past_tokens + 1
    weight_bytes = 2 * MODEL.active_linear_parameters
    kv_read_bytes = past_tokens * kv_bytes_per_token(kv_bits)
    steady_kv_write_bytes = kv_bytes_per_token(kv_bits)
    hbm_bytes = max(0, weight_bytes + kv_read_bytes - HARDWARE.l2_bytes) + steady_kv_write_bytes
    useful = decode_useful_counts(past_tokens)
    compute_floor = (
        useful["linear_flops"] / peaks["bf16_tensor_flops"]
        + useful["attention_flops"] / peaks["bf16_tensor_flops"]
    )
    physical = max(hbm_bytes / HARDWARE.dram_bytes_per_second, compute_floor)
    no_persistent_hbm = (weight_bytes + kv_read_bytes + steady_kv_write_bytes) / HARDWARE.dram_bytes_per_second

    optimal_linear = max(
        0.0,
        linear_stage_floor(1, True, clock_hz) - HARDWARE.l2_bytes / HARDWARE.dram_bytes_per_second,
    )
    optimal_attention = max(
        kv_read_bytes / HARDWARE.dram_bytes_per_second,
        useful["attention_flops"] / peaks["bf16_tensor_flops"],
        useful["attention_exp_calls"] / peaks["sfu_results"],
    )
    kv_append = steady_kv_write_bytes / HARDWARE.dram_bytes_per_second
    optimal_dag = optimal_linear + optimal_attention + kv_append + optimal_elementwise_sfu_floor(1, clock_hz)

    current_ideal_cache = optimal_dag
    current_no_persistent = no_persistent_hbm
    issued_int8_ops = 0
    current_exp_calls = 0
    scratch_global_bytes = 0
    if kv_bits == 8:
        counts = current_int8_decode_attention_counts(past_tokens)
        issued_int8_ops = counts["issued_int8_ops"]
        current_exp_calls = counts["exp_calls"]
        scratch_global_bytes = counts["scratch_global_bytes"]
        attention_resource_floor = max(
            kv_read_bytes / HARDWARE.dram_bytes_per_second,
            issued_int8_ops / peaks["int8_tensor_ops"],
            current_exp_calls / peaks["sfu_results"],
        )
        current_ideal_cache = optimal_linear + attention_resource_floor + kv_append
        current_ideal_cache += current_elementwise_sfu_floor(1, clock_hz)
        current_no_persistent = linear_stage_floor(1, True, clock_hz) + attention_resource_floor
        current_no_persistent += kv_append + current_elementwise_sfu_floor(1, clock_hz)
    return {
        "physical": physical,
        "optimal_dag": optimal_dag,
        "no_persistent_hbm": no_persistent_hbm,
        "current_ideal_cache": current_ideal_cache,
        "current_no_persistent": current_no_persistent,
        "hbm_bytes": hbm_bytes,
        "issued_int8_ops": issued_int8_ops,
        "current_exp_calls": current_exp_calls,
        "scratch_global_bytes": scratch_global_bytes,
    }


def milliseconds(seconds: float) -> str:
    return f"{seconds * 1000:.3f}"


def tps(seconds: float) -> str:
    return f"{1 / seconds:.1f}"


def print_hardware() -> None:
    print("| Resource | 3.105 GHz physical ceiling | 2.250 GHz reported clock |")
    print("|---|---:|---:|")
    max_peaks = HARDWARE.peaks(HARDWARE.max_sm_clock_hz)
    reported_peaks = HARDWARE.peaks(HARDWARE.reported_sm_clock_hz)
    rows = (
        ("FP32 FMA", "fp32_fma_flops", "TFLOP/s"),
        ("BF16 tensor, dense, FP32 accumulate", "bf16_tensor_flops", "TFLOP/s"),
        ("INT8 tensor, dense", "int8_tensor_ops", "TOPS"),
        ("INT4 tensor, dense", "int4_tensor_ops", "TOPS"),
        ("SFU approximate results", "sfu_results", "Tresult/s"),
    )
    for label, key, unit in rows:
        print(f"| {label} | {max_peaks[key] / 1e12:.3f} {unit} | {reported_peaks[key] / 1e12:.3f} {unit} |")
    print(f"| GDDR6 | {HARDWARE.dram_bytes_per_second / 1e9:.3f} GB/s | {HARDWARE.dram_bytes_per_second / 1e9:.3f} GB/s |")


def print_ttft(clock_hz: float) -> None:
    print("| Prompt | Chunks | Useful work | Physical absolute | Optimal DAG | Firefly instruction floor | Current attention K/V element payload |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for prompt in (64, 1024, 2048, 4096, 8192, 16_384):
        values = ttft_bounds(prompt, 8, clock_hz)
        print(
            f"| {prompt:,} | {len(chunks(prompt))} | {values['useful_flops'] / 1e12:.3f} TFLOP "
            f"| {milliseconds(values['physical'])} ms | {milliseconds(values['optimal_dag'])} ms "
            f"| {milliseconds(values['current'])} ms | {values['current_global_kv_bytes'] / 1e12:.3f} TB |"
        )


def print_tpot(clock_hz: float) -> None:
    print("| Past | BF16 physical latency / TPS | INT8 physical latency / TPS | INT4 physical latency / TPS | INT8 optimal DAG | Firefly INT8 published-resource floor, right-sized splits |")
    print("|---:|---:|---:|---:|---:|---:|")
    for past in (0, 1024, 4096, 8192, 16_384, 32_768):
        bf16 = decode_bounds(past, 16, clock_hz)
        int8 = decode_bounds(past, 8, clock_hz)
        int4 = decode_bounds(past, 4, clock_hz)
        print(
            f"| {past:,} | {milliseconds(bf16['physical'])} ms / {tps(bf16['physical'])} "
            f"| {milliseconds(int8['physical'])} ms / {tps(int8['physical'])} "
            f"| {milliseconds(int4['physical'])} ms / {tps(int4['physical'])} "
            f"| {milliseconds(int8['optimal_dag'])} ms / {tps(int8['optimal_dag'])} "
            f"| {milliseconds(int8['current_ideal_cache'])} ms / {tps(int8['current_ideal_cache'])} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clock", choices=("max", "reported"), default="max")
    parser.add_argument("--section", choices=("all", "hardware", "ttft", "tpot"), default="all")
    args = parser.parse_args()
    clock_hz = HARDWARE.max_sm_clock_hz if args.clock == "max" else HARDWARE.reported_sm_clock_hz
    if args.section in ("all", "hardware"):
        print_hardware()
    if args.section in ("all", "ttft"):
        if args.section == "all":
            print()
        print_ttft(clock_hz)
    if args.section in ("all", "tpot"):
        if args.section == "all":
            print()
        print_tpot(clock_hz)


if __name__ == "__main__":
    main()
