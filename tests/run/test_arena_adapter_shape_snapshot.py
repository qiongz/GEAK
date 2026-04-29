"""Tests for the bb7f1a0a Arena AB adapter instrumentation.

This branch intentionally keeps the main-branch optimizer/planner behavior
unchanged. These tests cover only evaluation-side sidecar parsing, shape
profile derivation, PreprocessContext plumbing, and resume probing.
"""

from __future__ import annotations

import json
from pathlib import Path

from minisweagent.run.orchestrator import _probe_preprocess_dir
from minisweagent.run.preprocess.benchmark_parsing import (
    discover_performance_report,
    parse_performance_report_json,
)
from minisweagent.run.preprocess.context import PreprocessContext as RawPreprocessContext
from minisweagent.run.preprocess.discovery_types import (
    SHAPE_COVERAGE_BUCKETED,
    SHAPE_COVERAGE_MULTI,
    derive_shape_coverage_profile,
)
from minisweagent.run.pipeline_types import PreprocessContext


def test_parse_performance_report_json_metadata_and_invalid_times(tmp_path: Path) -> None:
    payload = [
        {
            "test_case_id": "aiter_case",
            "shape": [1, 16, 16, 128, 2048],
            "execution_time_ms": "0.456",
            "metadata": {"B": 1, "H_Q": 16, "H_KV": 16, "D": 128, "SEQ_LEN": 2048},
        },
        {"test_case_id": "failed_case", "execution_time_ms": -1.0, "params": {"M": 32}},
        {"test_case_id": "zero_case", "execution_time_ms": 0.0, "params": {"M": 64}},
        {"case_idx": 3, "ori_time": 1.25, "opt_time": None, "params": {"input_0_shape": [1024]}},
    ]
    path = tmp_path / "performance_report.json"
    path.write_text(json.dumps(payload))

    cases = parse_performance_report_json(path)

    assert cases is not None
    assert cases[0]["ms"] == 0.456
    assert cases[0]["params"]["SEQ_LEN"] == 2048
    assert "shape" not in cases[0]["params"]
    assert cases[1]["ms"] is None
    assert cases[2]["ms"] is None
    assert cases[3]["ms"] == 1.25


def test_discover_performance_report_walks_arena_task_layout(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    task_dir = repo / "tasks" / "triton2triton" / "vllm" / "triton_scaled_mm"
    (task_dir / "source").mkdir(parents=True)
    (task_dir / "scripts").mkdir()
    (task_dir / "build").mkdir()
    kernel = task_dir / "source" / "kernel.py"
    runner = task_dir / "scripts" / "task_runner.py"
    report = task_dir / "build" / "performance_report.json"
    kernel.write_text("# kernel\n")
    runner.write_text("# runner\n")
    report.write_text(json.dumps([{"test_case_id": "perf1", "execution_time_ms": 0.1, "params": {"M": 32}}]))

    found_by_kernel = discover_performance_report(kernel_path=kernel, repo_root=repo, output_dir=tmp_path / "out")
    found_by_command = discover_performance_report(
        kernel_path=repo / "elsewhere.py",
        repo_root=repo,
        output_dir=tmp_path / "out",
        performance_command="python3 tasks/triton2triton/vllm/triton_scaled_mm/scripts/task_runner.py performance",
    )

    assert found_by_kernel is not None
    assert found_by_kernel.resolve() == report.resolve()
    assert found_by_command is not None
    assert found_by_command.resolve() == report.resolve()


def test_shape_profile_derivation_uses_dimension_indexed_shape_axes() -> None:
    same_shape = [
        {"params": {"shape": [1, 16, 16, 128, 2048]}},
        {"params": {"shape": [1, 16, 16, 128, 2048]}},
        {"params": {"shape": [1, 16, 16, 128, 2048]}},
        {"params": {"shape": [1, 16, 16, 128, 2048]}},
    ]
    varied_seq = [
        {"params": {"shape": [1, 16, 16, 128, 256]}},
        {"params": {"shape": [1, 16, 16, 128, 512]}},
        {"params": {"shape": [1, 16, 16, 128, 1024]}},
        {"params": {"shape": [1, 16, 16, 128, 2048]}},
    ]

    assert derive_shape_coverage_profile(benchmark_test_cases=same_shape) == SHAPE_COVERAGE_MULTI
    assert derive_shape_coverage_profile(benchmark_test_cases=varied_seq) == SHAPE_COVERAGE_BUCKETED


def test_preprocess_context_carries_shape_snapshot_fields(tmp_path: Path) -> None:
    snapshot = tmp_path / "benchmark_test_cases.json"
    bm = tmp_path / "baseline_metrics.json"
    profile = tmp_path / "profile.json"
    bm.write_text("{}")
    profile.write_text("{}")
    snapshot.write_text(json.dumps({"schema": "geak.benchmark_test_cases.v1", "test_cases": []}))

    ctx = {
        "kernel_path": str(tmp_path / "kernel.py"),
        "repo_root": str(tmp_path),
        "harness_path": str(tmp_path / "harness.py"),
        "benchmark_shape_count": 2,
        "benchmark_test_cases": [{"case_id": "perf1", "params": {"M": 32}, "baseline_ms": 0.1}],
        "benchmark_test_cases_path": str(snapshot),
        "shape_coverage_profile": SHAPE_COVERAGE_MULTI,
    }
    raw = RawPreprocessContext.from_preprocessor_output(ctx, tmp_path)
    typed = PreprocessContext.from_dict(raw.to_dict())

    assert typed.benchmark_shape_count == 2
    assert typed.benchmark_test_cases_path == str(snapshot)
    assert typed.shape_coverage_profile == SHAPE_COVERAGE_MULTI


def test_probe_preprocess_dir_falls_back_to_discovery_and_baseline_metrics(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    kernel = repo / "kernel.py"
    kernel.write_text("def kernel(x): return x\n")

    pp = tmp_path / "pp"
    pp.mkdir()
    (pp / "discovery.json").write_text(
        json.dumps({"kernel": {"file": str(kernel), "type": "unknown", "name": "kernel"}})
    )
    (pp / "baseline_metrics.json").write_text(
        json.dumps(
            {
                "benchmark_shape_count": 2,
                "benchmark_test_cases": [{"case_id": "perf1", "params": {"M": 32}, "baseline_ms": 0.1}],
                "shape_coverage_profile": SHAPE_COVERAGE_MULTI,
            }
        )
    )
    (pp / "benchmark_test_cases.json").write_text(
        json.dumps({"schema": "geak.benchmark_test_cases.v1", "test_cases": []})
    )

    pc = _probe_preprocess_dir(pp)

    assert pc.kernel_path == str(kernel)
    assert pc.benchmark_shape_count == 2
    assert pc.benchmark_test_cases_path == str(pp / "benchmark_test_cases.json")
    assert pc.shape_coverage_profile == SHAPE_COVERAGE_MULTI
