#!/usr/bin/env python3

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
INCLUDE_PATTERN = re.compile(r'#include\s+"firefly/([^/"]+)')
MODEL_INCLUDE_PATTERN = re.compile(r'#include\s+"firefly/model/([^/"]+)/')
RULES = {
    "device": {"storage", "kernels", "model", "scheduler", "execution", "service"},
    "core": {"storage", "kernels", "model", "scheduler", "execution", "service"},
    "storage": {"kernels", "model", "scheduler", "execution", "service"},
    "kernels": {"storage", "model", "execution", "scheduler", "service"},
    "model": {"execution", "scheduler", "service"},
    "scheduler": {"storage", "kernels", "model", "execution", "service"},
    "execution": {"storage", "service"},
}

LEGACY_PATHS = {"hal", "internal", "io", "mm"}
FORBIDDEN_RUNTIME_MARKERS = {"FIREFLY_USE_TORCH", "libtorch", "#include <torch/", "#include <ATen/"}
FORBIDDEN_STREAM_STATE = {"get_default_stream", "set_default_stream", "thread_local cudaStream_t"}
FORBIDDEN_DIRECT_NVTX = {"nvtxRangePush", "nvtxRangePop", "#include <nvtx3/"}
FORBIDDEN_CMAKE_ASSUMPTIONS = {
    ".venv",
    "site-packages",
    "EXECUTABLE_OUTPUT_PATH",
    "LIBRARY_OUTPUT_PATH",
    "${PROJECT_SOURCE_DIR}/bin",
    "${PROJECT_SOURCE_DIR}/lib",
}


def module_files(module: str):
    roots = [ROOT / "include" / "firefly" / module, ROOT / "src" / module]
    for root in roots:
        if root.exists():
            yield from (path for path in root.rglob("*") if path.suffix in {".h", ".cc", ".cu", ".cuh"})


def main() -> int:
    violations = []

    for path in (ROOT / "src").rglob("*"):
        if path.suffix in {".h", ".hpp", ".cuh"}:
            violations.append(f"{path.relative_to(ROOT)}: headers must live under include/firefly")

    for legacy in LEGACY_PATHS:
        for root in (ROOT / "include" / "firefly", ROOT / "src"):
            if (root.joinpath(legacy).exists()):
                violations.append(f"{root.joinpath(legacy).relative_to(ROOT)}: legacy module must not be recreated")

    for owner, forbidden in RULES.items():
        for path in module_files(owner):
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                match = INCLUDE_PATTERN.search(line)
                if match and match.group(1) in forbidden:
                    relative = path.relative_to(ROOT)
                    violations.append(f"{relative}:{line_number}: {owner} must not depend on {match.group(1)}")

    model_roots = [ROOT / "include" / "firefly" / "model", ROOT / "src" / "model"]
    model_families = {
        path.name for root in model_roots if root.exists() for path in root.iterdir() if path.is_dir()
    }
    for family in model_families:
        for root in model_roots:
            family_root = root / family
            if not family_root.exists():
                continue
            for path in family_root.rglob("*"):
                if path.suffix not in {".h", ".cc", ".cu", ".cuh"}:
                    continue
                for line_number, line in enumerate(path.read_text().splitlines(), 1):
                    match = MODEL_INCLUDE_PATTERN.search(line)
                    if match and match.group(1) != family:
                        relative = path.relative_to(ROOT)
                        violations.append(
                            f"{relative}:{line_number}: model family {family} must not depend on {match.group(1)}"
                        )

    for root in (ROOT / "src", ROOT / "tests"):
        for path in root.rglob("*"):
            if path.suffix not in {".h", ".cc", ".cu", ".cuh"}:
                continue
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                if '#include "firefly/kernels.h"' in line:
                    relative = path.relative_to(ROOT)
                    violations.append(
                        f"{relative}:{line_number}: internal code must include a specific kernels/<operator>.h"
                    )
                if '#include "firefly/kernels/detail/' in line and "src/kernels/" not in path.as_posix():
                    relative = path.relative_to(ROOT)
                    violations.append(f"{relative}:{line_number}: CUDA kernel details are private to kernel implementations")
                if '#include "firefly/kernels/attention/detail/' in line and "src/kernels/attention/" not in path.as_posix():
                    relative = path.relative_to(ROOT)
                    violations.append(f"{relative}:{line_number}: attention backend details are private to attention implementations")

    for path in [ROOT / "CMakeLists.txt", ROOT / "src" / "CMakeLists.txt"]:
        content = path.read_text()
        for marker in FORBIDDEN_RUNTIME_MARKERS:
            if marker in content:
                violations.append(f"{path.relative_to(ROOT)}: forbidden framework dependency marker {marker}")

    for path in [ROOT / "CMakeLists.txt", *(ROOT / "cmake").glob("*.cmake")]:
        content = path.read_text()
        for marker in FORBIDDEN_CMAKE_ASSUMPTIONS:
            if marker in content:
                violations.append(f"{path.relative_to(ROOT)}: local build assumption is forbidden: {marker}")

    for root in [ROOT / "include", ROOT / "src"]:
        for path in root.rglob("*"):
            if path.suffix not in {".h", ".cc", ".cu", ".cuh"}:
                continue
            content = path.read_text()
            for marker in FORBIDDEN_STREAM_STATE:
                if marker in content:
                    violations.append(f"{path.relative_to(ROOT)}: implicit CUDA stream state is forbidden")
            if path != ROOT / "include" / "firefly" / "execution" / "trace.h":
                for marker in FORBIDDEN_DIRECT_NVTX:
                    if marker in content:
                        violations.append(f"{path.relative_to(ROOT)}: NVTX must use execution/trace.h macros")

    if violations:
        print("Architecture dependency violations:", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
        return 1
    print("Architecture dependency boundaries passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
