import argparse
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import triton


ROOT = Path(__file__).resolve().parent
AMD_EXAMPLE = ROOT / "05_small_mma_amd.py"
COMPILE_CASE = (64, 32, 32, False, 16, 4)
CORRECTNESS_CASES = [
    (64, 32, 32, False, 16, 4),
    (64, 256, 128, True, 64, 8),
]
PROFILE_CASE = (64, 32, 32, False, 16, 4)
BENCHMARK_CASE = (64, 256, 128, False, 64, 4)
FULL_BENCHMARK_CASES = [
    (64, 32, 32, False, 16, 4),
    (64, 256, 128, False, 64, 4),
]
DEFAULT_BENCHMARK_ITERATIONS = 5
DEFAULT_FULL_BENCHMARK_ITERATIONS = 30


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


AMD = _load_module(AMD_EXAMPLE, "layer2_small_mma_amd")


def _require_gfx942():
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx942":
        raise RuntimeError(f"Expected hip/gfx942 target, got {target!r}")
    return target


def _make_inputs(M, N, K, seed=0):
    torch.manual_seed(seed)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.randn((M, N), device="cuda", dtype=torch.float32)
    d = torch.empty_like(c)
    return a, b, c, d


def _make_cpu_inputs(M, N, K, seed=0):
    torch.manual_seed(seed)
    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)
    c = torch.randn((M, N), dtype=torch.float32)
    return a, b, c


def _move_inputs_to_cuda(a_cpu, b_cpu, c_cpu):
    a = a_cpu.to("cuda")
    b = b_cpu.to("cuda")
    c = c_cpu.to("cuda")
    d = torch.empty_like(c)
    return a, b, c, d


def _resolve_iterations(iterations, default_value):
    if iterations is not None:
        return iterations
    env_value = os.environ.get("GEAK_BENCHMARK_ITERATIONS", "").strip()
    if env_value:
        return int(env_value)
    return default_value


def _run_small_mma(M, N, K, lhs_in_reg, instr_shape_n, num_warps, seed=0):
    a, b, c, d = _make_inputs(M, N, K, seed=seed)
    AMD.small_mma(a, b, c, d, instr_shape_n, lhs_in_reg, num_warps)
    torch.cuda.synchronize()
    return a, b, c, d


def run_compile_case():
    target = _require_gfx942()
    M, N, K, lhs_in_reg, instr_shape_n, num_warps = COMPILE_CASE
    a, b, c, d = _make_inputs(M, N, K)
    compiled = AMD.small_mma(a, b, c, d, instr_shape_n, lhs_in_reg, num_warps)
    torch.cuda.synchronize()
    print("COMPILE_OK=1")
    print(f"COMPILE_TARGET={target}")
    print(f"COMPILE_SHAPE={[M, N, K]}")
    return compiled


def run_correctness_case(M, N, K, lhs_in_reg, instr_shape_n, num_warps):
    _require_gfx942()
    a, b, c, d = _run_small_mma(M, N, K, lhs_in_reg, instr_shape_n, num_warps)
    torch.testing.assert_close(a @ b + c, d, atol=1e-3, rtol=1e-1)


def run_correctness():
    run_compile_case()
    for M, N, K, lhs_in_reg, instr_shape_n, num_warps in CORRECTNESS_CASES:
        run_correctness_case(M, N, K, lhs_in_reg, instr_shape_n, num_warps)
        print(f"CORRECTNESS_CASE_OK={[M, N, K, lhs_in_reg, instr_shape_n, num_warps]}")
    print("CORRECTNESS_OK=1")


def run_profile():
    _require_gfx942()
    M, N, K, lhs_in_reg, instr_shape_n, num_warps = PROFILE_CASE
    a_cpu, b_cpu, c_cpu = _make_cpu_inputs(M, N, K)
    a, b, c, d = _move_inputs_to_cuda(a_cpu, b_cpu, c_cpu)
    AMD.small_mma(a, b, c, d, instr_shape_n, lhs_in_reg, num_warps)
    torch.cuda.synchronize()
    print("PROFILE_OK=1")
    print(f"GEAK_SHAPES_USED={[[M, N, K, lhs_in_reg, instr_shape_n, num_warps]]}")


def _benchmark_case(M, N, K, lhs_in_reg, instr_shape_n, num_warps, iterations, seed=0):
    _require_gfx942()
    a, b, c, d = _make_inputs(M, N, K, seed=seed)
    AMD.small_mma(a, b, c, d, instr_shape_n, lhs_in_reg, num_warps)
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(
        lambda: AMD.small_mma(a, b, c, d, instr_shape_n, lhs_in_reg, num_warps),
        rep=iterations,
    )
    flops = 2 * M * N * K + M * N
    tflops = flops * 1e-12 / (ms * 1e-3)
    return {
        "shape": [M, N, K, lhs_in_reg, instr_shape_n, num_warps],
        "latency_ms": ms,
        "tflops": tflops,
    }


def _emit_benchmark_result(prefix, records):
    avg_latency_ms = sum(record["latency_ms"] for record in records) / len(records)
    avg_tflops = sum(record["tflops"] for record in records) / len(records)

    print(f"{prefix}_OK=1")
    print(f"GEAK_SHAPES_USED={[record['shape'] for record in records]}")
    for record in records:
        print(
            f"{prefix}_CASE shape={record['shape']} "
            f"latency_ms={record['latency_ms']:.4f} tflops={record['tflops']:.4f}"
        )
    print(f"GEAK_RESULT_TFLOPS={avg_tflops:.4f}")
    print(f"GEAK_RESULT_LATENCY_MS={avg_latency_ms:.4f}")
    return avg_latency_ms, avg_tflops


def run_benchmark(iterations=None):
    iterations = _resolve_iterations(iterations, DEFAULT_BENCHMARK_ITERATIONS)
    record = _benchmark_case(*BENCHMARK_CASE, iterations=iterations)
    return _emit_benchmark_result("BENCHMARK", [record])


def run_full_benchmark(iterations=None):
    iterations = _resolve_iterations(iterations, DEFAULT_FULL_BENCHMARK_ITERATIONS)
    records = [
        _benchmark_case(*case, iterations=iterations, seed=index)
        for index, case in enumerate(FULL_BENCHMARK_CASES)
    ]
    return _emit_benchmark_result("FULL_BENCHMARK", records)


@pytest.mark.skipif(
    triton.runtime.driver.active.get_current_target().backend != "hip"
    or triton.runtime.driver.active.get_current_target().arch != "gfx942",
    reason="Requires hip/gfx942 runtime",
)
def test_compile_layer2_small_mma():
    run_compile_case()


@pytest.mark.skipif(
    triton.runtime.driver.active.get_current_target().backend != "hip"
    or triton.runtime.driver.active.get_current_target().arch != "gfx942",
    reason="Requires hip/gfx942 runtime",
)
@pytest.mark.parametrize("M, N, K, lhs_in_reg, instr_shape_n, num_warps", CORRECTNESS_CASES)
def test_correctness_layer2_small_mma(M, N, K, lhs_in_reg, instr_shape_n, num_warps):
    run_correctness_case(M, N, K, lhs_in_reg, instr_shape_n, num_warps)


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
    parser = argparse.ArgumentParser(description="Focused Layer 2 harness for small_mma on gfx942.")
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
