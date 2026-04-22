# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin harness for the Triton-family Gluon input fixtures."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import torch
import triton

ROOT = Path(__file__).resolve().parent
DEFAULT_FIXTURE = os.environ.get("GEAK_GLUON_INPUT_FIXTURE", "01_plain_triton_input.py")
QUICK_CASES = [4096, 65536]
FULL_CASES = [4096, 65536, 1048576]


def _fixture_path(name: str) -> Path:
    path = (ROOT / name).resolve()
    if path.parent != ROOT or not path.exists():
        raise FileNotFoundError(f"Unknown fixture: {name}")
    return path


def _load_fixture(name: str) -> tuple[Path, ModuleType]:
    path = _fixture_path(name)
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load fixture module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


def _run_correctness(module: ModuleType, cases: list[int]) -> None:
    for size in cases:
        x, y = module.make_inputs(size)
        actual = module.run_kernel(x, y)
        expected = module.reference(x, y)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def _benchmark_ms(module: ModuleType, size: int) -> float:
    x, y = module.make_inputs(size)
    ms, _min_ms, _max_ms = triton.testing.do_bench(lambda: module.run_kernel(x, y), quantiles=[0.5, 0.2, 0.8])
    return float(ms)


def _print_benchmark(module: ModuleType, cases: list[int]) -> None:
    results = {size: _benchmark_ms(module, size) for size in cases}
    median_ms = sorted(results.values())[len(results) // 2]
    print(f"GEAK_SHAPES_USED={json.dumps(cases)}")
    print(f"GEAK_BENCHMARK_RESULTS_MS={json.dumps(results, sort_keys=True)}")
    print(f"GEAK_RESULT_LATENCY_MS={median_ms:.6f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Triton-family Gluon input fixtures.")
    parser.add_argument("--kernel-file", default=DEFAULT_FIXTURE, help="Fixture file under examples/triton_gluon_inputs/.")
    parser.add_argument("--mode", choices=("compile",), default=None, help="Explicit compile-only mode.")
    parser.add_argument("--correctness", action="store_true", help="Run correctness checks.")
    parser.add_argument("--profile", action="store_true", help="Run a single medium benchmark and print GEAK latency output.")
    parser.add_argument("--benchmark", action="store_true", help="Run a quick multi-shape benchmark.")
    parser.add_argument("--full-benchmark", action="store_true", help="Run the full multi-shape benchmark.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    fixture_path, module = _load_fixture(args.kernel_file)

    if args.mode == "compile" or not any((args.correctness, args.profile, args.benchmark, args.full_benchmark)):
        print(f"COMPILE_OK={fixture_path.name}")
        return 0

    if args.correctness:
        _run_correctness(module, QUICK_CASES)
        print(f"CORRECTNESS_OK={fixture_path.name}")
        return 0

    if args.profile:
        _print_benchmark(module, [QUICK_CASES[-1]])
        return 0

    if args.benchmark:
        _print_benchmark(module, QUICK_CASES)
        return 0

    if args.full_benchmark:
        _print_benchmark(module, FULL_CASES)
        return 0

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
