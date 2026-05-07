from pathlib import Path

import pytest

from minisweagent.run.postprocess.benchmark_parsing import (
    compute_best_patch,
    extract_latency_ms,
    parse_shape_latencies_ms,
)


def test_parse_shape_latencies_ms_extracts_each_shape() -> None:
    output = "\n".join(
        [
            "Benchmark mode: 3 shapes, 10 iterations each",
            "  (32,4096): 0.0503 ms",
            "  (64,4096): 0.0525 ms",
            "  (256,8192): 0.0626 ms",
            "Geomean latency: 0.0548 ms",
            "GEAK_RESULT_LATENCY_MS=0.054772",
        ]
    )

    assert parse_shape_latencies_ms(output) == {
        "(32,4096)": 0.0503,
        "(64,4096)": 0.0525,
        "(256,8192)": 0.0626,
    }


def test_named_case_latencies_are_totaled_for_baseline_objective() -> None:
    output = "case_small: 0.0566 ms\ncase_medium: 0.0558 ms\n"

    assert parse_shape_latencies_ms(output) == {
        "case_small": 0.0566,
        "case_medium": 0.0558,
    }
    assert extract_latency_ms(output) == pytest.approx(0.1124)


def test_compute_best_patch_includes_per_shape_speedups(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "fused_rms_fp8"
    patch_dir = kernel_dir / "results" / "round_1" / "dispatch-path-check"
    patch_dir.mkdir(parents=True)

    (kernel_dir / "benchmark_baseline.txt").write_text(
        "\n".join(
            [
                "Benchmark mode: 2 shapes, 10 iterations each",
                "  (32,4096): 0.0500 ms",
                "  (64,4096): 0.0600 ms",
                "Geomean latency: 0.054772 ms",
                "GEAK_RESULT_LATENCY_MS=0.054772",
            ]
        )
    )
    (patch_dir / "patch_1.patch").write_text("diff --git a/kernel.py b/kernel.py\n+pass\n")
    (patch_dir / "patch_1_test.txt").write_text(
        "\n".join(
            [
                "Benchmark mode: 2 shapes, 10 iterations each",
                "  (32,4096): 0.0400 ms",
                "  (64,4096): 0.0600 ms",
                "Geomean latency: 0.048990 ms",
                "GEAK_RESULT_LATENCY_MS=0.048990",
            ]
        )
    )

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["baseline_source"] == "benchmark_baseline.txt"
    assert result["best_patch_id"] == "patch_1"
    assert result["baseline_shape_latency_ms"] == {
        "(32,4096)": 0.05,
        "(64,4096)": 0.06,
    }
    assert result["candidate_shape_latency_ms"] == {
        "(32,4096)": 0.04,
        "(64,4096)": 0.06,
    }
    assert result["per_shape_speedups"] == {
        "(32,4096)": {
            "baseline_ms": 0.05,
            "candidate_ms": 0.04,
            "speedup": 1.25,
        },
        "(64,4096)": {
            "baseline_ms": 0.06,
            "candidate_ms": 0.06,
            "speedup": 1.0,
        },
    }


def test_compute_best_patch_reports_regression_against_true_baseline(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "generic_kernel"
    patch_dir = kernel_dir / "results" / "round_1" / "extension-l0-gluon-minimal-viability"
    patch_dir.mkdir(parents=True)
    (kernel_dir / "benchmark_baseline.txt").write_text(
        "case_small: 0.0566 ms\ncase_medium: 0.0558 ms\n"
    )
    (patch_dir / "patch_0.patch").write_text("diff --git a/kernel.py b/kernel.py\n+@gluon.jit\n")
    (patch_dir / "patch_0_test.txt").write_text(
        "case_small: 0.0642 ms\ncase_medium: 0.0634 ms\n"
    )
    (patch_dir / "patch_1.patch").write_text("diff --git a/kernel.py b/kernel.py\n+@gluon.jit\n")
    (patch_dir / "patch_1_test.txt").write_text(
        "case_small: 0.0633 ms\ncase_medium: 0.0628 ms\n"
    )

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["best_patch_id"] == "patch_1"
    assert result["baseline_source"] == "benchmark_baseline.txt"
    assert result["baseline_latency_ms"] == pytest.approx(0.1124)
    assert result["candidate_latency_ms"] == pytest.approx(0.1261)
    assert result["best_patch_speedup"] == pytest.approx(0.891356)
    assert result["improves_true_baseline"] is False
    assert result["objective"] == "total_shape_latency_ms"


def test_compute_best_patch_enforces_required_target_symbol(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "extension-l1-targeted-memory"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_extension-l1-targeted-memory.md",
        {
            "label": "extension-l1-targeted-memory",
            "required_patch_target_symbols": ["target_stage_kernel"],
        },
        "Extension L1\nTarget symbol: target_stage_kernel\n",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text("diff --git a/kernel.py b/kernel.py\n+def other_stage_kernel(): pass\n")
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.4 ms\ncase_b: 0.4 ms\n")
    (patch_dir / "patch_2.patch").write_text(
        "diff --git a/kernel.py b/kernel.py\n+def target_stage_kernel(): pass\n"
    )
    (patch_dir / "patch_2_test.txt").write_text("case_a: 0.9 ms\ncase_b: 0.9 ms\n")

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["best_patch_id"] == "patch_2"
    assert result["required_patch_target_symbols"] == ["target_stage_kernel"]


def test_compute_best_patch_rejects_plain_fallback_for_required_gluon(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "extension-l0-gluon-minimal-viability"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_extension-l0-gluon-minimal-viability.md",
        {
            "label": "extension-l0-gluon-minimal-viability",
            "required_output_dialect": "amd_gluon",
        },
        "Extension L0\n",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text("diff --git a/kernel.py b/kernel.py\n+tl.load(x)\n")
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.2 ms\ncase_b: 0.2 ms\n")
    (patch_dir / "patch_2.patch").write_text(
        "diff --git a/kernel.py b/kernel.py\n+from triton.experimental import gluon\n+@gluon.jit\n"
    )
    (patch_dir / "patch_2_test.txt").write_text("case_a: 0.8 ms\ncase_b: 0.8 ms\n")

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["best_patch_id"] == "patch_2"
    assert result["required_output_dialect"] == "amd_gluon"
    assert result["actual_output_dialect"] == "amd_gluon"
    assert result["dialect_contract_satisfied"] is True


def test_compute_best_patch_infers_gluon_extension_contract_without_metadata(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "extension-l1-gluon-buffer-load-stage"
    patch_dir.mkdir(parents=True)
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "diff --git a/kernel.py b/kernel.py\n+from triton.experimental import gluon\n+\n"
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.2 ms\ncase_b: 0.2 ms\n")
    (patch_dir / "patch_2.patch").write_text("diff --git a/kernel.py b/kernel.py\n+\n+\n")
    (patch_dir / "patch_2_test.txt").write_text("case_a: 0.1 ms\ncase_b: 0.1 ms\n")
    (patch_dir / "patch_3.patch").write_text(
        "diff --git a/kernel.py b/kernel.py\n+from triton.experimental import gluon\n+@gluon.jit\n+def kernel_gluon(x):\n+    return x\n"
    )
    (patch_dir / "patch_3_test.txt").write_text("case_a: 0.8 ms\ncase_b: 0.8 ms\n")

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["best_patch_id"] == "patch_3"
    assert result["required_output_dialect"] == "amd_gluon"
    assert result["actual_output_dialect"] == "amd_gluon"


def test_compute_best_patch_rejects_definition_only_gluon_helper(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "extension-l1-targeted-memory"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_extension-l1-targeted-memory.md",
        {
            "label": "extension-l1-targeted-memory",
            "required_output_dialect": "amd_gluon",
            "required_patch_target_symbols": ["target_stage_kernel"],
        },
        "Extension L1\nTarget symbol: target_stage_kernel\n",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "\n".join(
            [
                "diff --git a/kernel.py b/kernel.py",
                "+from triton.experimental import gluon",
                "+@gluon.jit",
                "+def target_stage_kernel_gluon(x):",
                "+    return x",
                "+# Plain Triton path still dispatches the original target_stage_kernel.",
            ]
        )
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.2 ms\ncase_b: 0.2 ms\n")
    (patch_dir / "patch_2.patch").write_text(
        "\n".join(
            [
                "diff --git a/kernel.py b/kernel.py",
                "+from triton.experimental import gluon",
                "+@gluon.jit",
                "+def target_stage_kernel_gluon(x):",
                "+    return x",
                "+target_stage_kernel = target_stage_kernel_gluon",
                "+target_stage_kernel_gluon[grid](x)",
            ]
        )
    )
    (patch_dir / "patch_2_test.txt").write_text("case_a: 0.8 ms\ncase_b: 0.8 ms\n")

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["best_patch_id"] == "patch_2"
    assert result["gluon_execution_contract_satisfied"] is True
