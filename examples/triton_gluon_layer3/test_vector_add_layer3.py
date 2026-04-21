import argparse
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import triton


ROOT = Path(__file__).resolve().parent
AMD_EXAMPLE = ROOT / "01_vector_add_amd.py"
ALL_CASES = [4096, 98_432, 262_144, 1_048_576, 4_194_304]
COMPILE_CASE = ALL_CASES[1]
DEFAULT_BENCHMARK_ITERATIONS = 5
DEFAULT_FULL_BENCHMARK_ITERATIONS = 20


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


AMD = _load_module(AMD_EXAMPLE, "layer3_vector_add_amd")


def _require_gfx942():
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx942":
        raise RuntimeError(f"Expected hip/gfx942 target, got {target!r}")
    return target


def _pick_cases(count: int):
    if len(ALL_CASES) <= count:
        return list(ALL_CASES)
    if count == 1:
        return [ALL_CASES[len(ALL_CASES) // 2]]
    n_cases = len(ALL_CASES)
    return [ALL_CASES[round(index * (n_cases - 1) / (count - 1))] for index in range(count)]


CORRECTNESS_CASES = _pick_cases(4)
PROFILE_CASES = _pick_cases(1)
BENCHMARK_CASES = _pick_cases(3)
FULL_BENCHMARK_CASES = list(ALL_CASES)


def _make_inputs(size: int, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.randn(size, device="cuda", dtype=torch.float32)
    return x, y


def _make_profile_inputs(size: int):
    # Keep the profile path focused on the lifted kernel itself rather than
    # spending profiler time inside GPU-side random-input generation.
    torch.manual_seed(0)
    x_cpu = torch.randn(size, device="cpu", dtype=torch.float32)
    y_cpu = torch.randn(size, device="cpu", dtype=torch.float32)
    return x_cpu.to("cuda"), y_cpu.to("cuda")


def _resolve_iterations(iterations, default_value):
    if iterations is not None:
        return iterations
    env_value = os.environ.get("GEAK_BENCHMARK_ITERATIONS", "").strip()
    if env_value:
        return int(env_value)
    return default_value


def _shape_list(cases):
    return [[size] for size in cases]


def run_compile_case():
    target = _require_gfx942()
    x, y = _make_inputs(COMPILE_CASE)
    compiled = AMD.add(x, y)
    torch.cuda.synchronize()
    print("COMPILE_OK=1")
    print(f"COMPILE_TARGET={target}")
    print(f"COMPILE_SHAPE={[COMPILE_CASE]}")
    return compiled


def run_correctness_case(size: int, seed: int = 0):
    _require_gfx942()
    x, y = _make_inputs(size, seed=seed)
    output = AMD.add(x, y)
    torch.cuda.synchronize()
    torch.testing.assert_close(x + y, output, atol=0, rtol=0)


def run_correctness():
    run_compile_case()
    for index, size in enumerate(CORRECTNESS_CASES):
        run_correctness_case(size, seed=index)
        print(f"CORRECTNESS_CASE_OK={[size]}")
    print("CORRECTNESS_OK=1")
    print(f"GEAK_SHAPES_USED={_shape_list(CORRECTNESS_CASES)}")


def run_profile():
    _require_gfx942()
    for size in PROFILE_CASES:
        x, y = _make_profile_inputs(size)
        AMD.add(x, y)
    torch.cuda.synchronize()
    print("PROFILE_OK=1")
    print(f"GEAK_SHAPES_USED={_shape_list(PROFILE_CASES)}")


def _benchmark_case(size: int, iterations: int, seed: int = 0):
    _require_gfx942()
    x, y = _make_inputs(size, seed=seed)
    AMD.add(x, y)
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(lambda: AMD.add(x, y), rep=iterations)
    bytes_moved = 3 * size * x.element_size()
    gbps = bytes_moved / (ms * 1e-3) / 1e9
    return {
        "shape": [size],
        "latency_ms": ms,
        "gbps": gbps,
    }


def _emit_benchmark_result(prefix: str, records):
    avg_latency_ms = sum(record["latency_ms"] for record in records) / len(records)
    avg_gbps = sum(record["gbps"] for record in records) / len(records)

    print(f"{prefix}_OK=1")
    print(f"GEAK_SHAPES_USED={[record['shape'] for record in records]}")
    for record in records:
        print(
            f"{prefix}_CASE shape={record['shape']} "
            f"latency_ms={record['latency_ms']:.4f} gbps={record['gbps']:.4f}"
        )
    print(f"GEAK_RESULT_GBPS={avg_gbps:.4f}")
    print(f"GEAK_RESULT_LATENCY_MS={avg_latency_ms:.4f}")
    return avg_latency_ms, avg_gbps


def run_benchmark(iterations=None):
    iterations = _resolve_iterations(iterations, DEFAULT_BENCHMARK_ITERATIONS)
    records = [
        _benchmark_case(size, iterations=iterations, seed=index)
        for index, size in enumerate(BENCHMARK_CASES)
    ]
    return _emit_benchmark_result("BENCHMARK", records)


def run_full_benchmark(iterations=None):
    iterations = _resolve_iterations(iterations, DEFAULT_FULL_BENCHMARK_ITERATIONS)
    records = [
        _benchmark_case(size, iterations=iterations, seed=index)
        for index, size in enumerate(FULL_BENCHMARK_CASES)
    ]
    return _emit_benchmark_result("FULL_BENCHMARK", records)


@pytest.mark.skipif(
    triton.runtime.driver.active.get_current_target().backend != "hip"
    or triton.runtime.driver.active.get_current_target().arch != "gfx942",
    reason="Requires hip/gfx942 runtime",
)
def test_compile_layer3_vector_add():
    run_compile_case()


@pytest.mark.skipif(
    triton.runtime.driver.active.get_current_target().backend != "hip"
    or triton.runtime.driver.active.get_current_target().arch != "gfx942",
    reason="Requires hip/gfx942 runtime",
)
@pytest.mark.parametrize("size", CORRECTNESS_CASES)
def test_correctness_layer3_vector_add(size):
    run_correctness_case(size)


def _resolve_requested_mode(args):
    if args.correctness:
        return "correctness"
    if args.profile:
        return "profile"
    if args.benchmark:
        return "benchmark"
    if args.full_benchmark:
        return "full-benchmark"
    return args.mode or "all"


def main():
    parser = argparse.ArgumentParser(description="Focused Layer 3 harness for direct vector-add lift on gfx942.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--correctness", action="store_true")
    mode_group.add_argument("--profile", action="store_true")
    mode_group.add_argument("--benchmark", action="store_true")
    mode_group.add_argument("--full-benchmark", dest="full_benchmark", action="store_true")
    mode_group.add_argument(
        "--mode",
        choices=["compile", "correctness", "profile", "benchmark", "full-benchmark", "all"],
        default=None,
        help="Legacy compatibility mode. Prefer the GEAK flags instead.",
    )
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()

    mode = _resolve_requested_mode(args)

    if mode in {"compile", "all"}:
        run_compile_case()
    if mode in {"correctness", "all"}:
        run_correctness()
    if mode in {"profile", "all"}:
        run_profile()
    if mode in {"benchmark", "all"}:
        run_benchmark(iterations=args.iterations)
    if mode in {"full-benchmark", "all"}:
        run_full_benchmark(iterations=args.iterations)


if __name__ == "__main__":
    main()
