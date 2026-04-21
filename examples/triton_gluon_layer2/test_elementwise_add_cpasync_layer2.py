import argparse
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import triton


ROOT = Path(__file__).resolve().parent
AMD_EXAMPLE = ROOT / "03_elementwise_add_cpasync_amd.py"
CORRECTNESS_CASES = [(65, 129), (1000, 2000)]
COMPILE_CASE = (64, 128)
PROFILE_CASE = (128, 256)
BENCHMARK_CASE = (2048, 4096)
FULL_BENCHMARK_CASES = [(512, 1024), (1024, 2048), BENCHMARK_CASE]
DEFAULT_BENCHMARK_ITERATIONS = 5
DEFAULT_FULL_BENCHMARK_ITERATIONS = 30


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


AMD = _load_module(AMD_EXAMPLE, "layer2_elementwise_add_cpasync_amd")


def _require_gfx942():
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx942":
        raise RuntimeError(f"Expected hip/gfx942 target, got {target!r}")
    return target


def _make_inputs(xnumel, ynumel, seed=0):
    torch.manual_seed(seed)
    a = torch.randn((xnumel, ynumel), device="cuda", dtype=torch.float32)
    b = torch.randn((xnumel, ynumel), device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    return a, b, c


def _make_cpu_inputs(xnumel, ynumel, seed=0):
    torch.manual_seed(seed)
    a = torch.randn((xnumel, ynumel), dtype=torch.float32)
    b = torch.randn((xnumel, ynumel), dtype=torch.float32)
    return a, b


def _move_inputs_to_cuda(a_cpu, b_cpu):
    a = a_cpu.to("cuda")
    b = b_cpu.to("cuda")
    c = torch.empty_like(a)
    return a, b, c


def _resolve_iterations(iterations, default_value):
    if iterations is not None:
        return iterations
    env_value = os.environ.get("GEAK_BENCHMARK_ITERATIONS", "").strip()
    if env_value:
        return int(env_value)
    return default_value


def _make_smem_layout():
    return AMD.make_default_smem_layout()


def run_compile_case():
    target = _require_gfx942()
    a, b, c = _make_inputs(*COMPILE_CASE)
    compiled = AMD.elementwise_add_cpasync(a, b, c, _make_smem_layout())
    torch.cuda.synchronize()
    print("COMPILE_OK=1")
    print(f"COMPILE_TARGET={target}")
    print(f"COMPILE_SHAPE={list(COMPILE_CASE)}")
    return compiled


def run_correctness_case(xnumel, ynumel):
    _require_gfx942()
    a, b, c = _make_inputs(xnumel, ynumel)
    AMD.elementwise_add_cpasync(a, b, c, _make_smem_layout())
    torch.cuda.synchronize()
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


def run_correctness():
    run_compile_case()
    for xnumel, ynumel in CORRECTNESS_CASES:
        run_correctness_case(xnumel, ynumel)
        print(f"CORRECTNESS_CASE_OK={[xnumel, ynumel]}")
    print("CORRECTNESS_OK=1")


def run_profile():
    _require_gfx942()
    a_cpu, b_cpu = _make_cpu_inputs(*PROFILE_CASE)
    a, b, c = _move_inputs_to_cuda(a_cpu, b_cpu)
    AMD.elementwise_add_cpasync(a, b, c, _make_smem_layout())
    torch.cuda.synchronize()
    print("PROFILE_OK=1")
    print(f"GEAK_SHAPES_USED={[list(PROFILE_CASE)]}")


def _benchmark_case(xnumel, ynumel, iterations, seed=0):
    _require_gfx942()
    a, b, c = _make_inputs(xnumel, ynumel, seed=seed)
    AMD.elementwise_add_cpasync(a, b, c, _make_smem_layout())
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(lambda: AMD.elementwise_add_cpasync(a, b, c, _make_smem_layout()), rep=iterations)
    bytes_moved = 3 * c.numel() * c.element_size()
    gbps = bytes_moved / (ms * 1e-3) / 1e9
    return {
        "shape": [xnumel, ynumel],
        "latency_ms": ms,
        "gbps": gbps,
    }


def _emit_benchmark_result(prefix, records):
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
    record = _benchmark_case(*BENCHMARK_CASE, iterations=iterations)
    return _emit_benchmark_result("BENCHMARK", [record])


def run_full_benchmark(iterations=None):
    iterations = _resolve_iterations(iterations, DEFAULT_FULL_BENCHMARK_ITERATIONS)
    records = [
        _benchmark_case(xnumel, ynumel, iterations=iterations, seed=index)
        for index, (xnumel, ynumel) in enumerate(FULL_BENCHMARK_CASES)
    ]
    return _emit_benchmark_result("FULL_BENCHMARK", records)


@pytest.mark.skipif(
    triton.runtime.driver.active.get_current_target().backend != "hip"
    or triton.runtime.driver.active.get_current_target().arch != "gfx942",
    reason="Requires hip/gfx942 runtime",
)
def test_compile_layer2_cpasync():
    run_compile_case()


@pytest.mark.skipif(
    triton.runtime.driver.active.get_current_target().backend != "hip"
    or triton.runtime.driver.active.get_current_target().arch != "gfx942",
    reason="Requires hip/gfx942 runtime",
)
@pytest.mark.parametrize("xnumel, ynumel", CORRECTNESS_CASES)
def test_correctness_layer2_cpasync(xnumel, ynumel):
    run_correctness_case(xnumel, ynumel)


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
    parser = argparse.ArgumentParser(description="Focused Layer 2 harness for elementwise_add_cpasync on gfx942.")
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
