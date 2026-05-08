from pathlib import Path

import pytest

from minisweagent.run.postprocess.benchmark_parsing import (
    _dialect_contract_satisfied,
    _required_output_dialect,
    classify_patch_output_dialect,
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


def test_required_output_dialect_uses_layer_metadata_before_search_set() -> None:
    assert (
        _required_output_dialect(
            {
                "implementation_layer": "amd_gluon overlay",
                "extension_layer": "L0",
            },
            label="same-direction-overlay",
        )
        == "amd_gluon"
    )


def test_required_output_dialect_does_not_infer_from_search_set_alone() -> None:
    assert _required_output_dialect({"search_set": "extension"}, label="same-direction-overlay") == "any"


def test_required_output_dialect_layer_overrides_any_frontmatter() -> None:
    assert (
        _required_output_dialect(
            {
                "required_output_dialect": "any",
                "implementation_layer": "amd_gluon overlay",
                "extension_layer": "L0",
            },
            label="same-direction-overlay",
        )
        == "amd_gluon"
    )


def test_required_amd_gluon_does_not_accept_mixed_output() -> None:
    assert _dialect_contract_satisfied("amd_gluon", "mixed") is False
    assert _dialect_contract_satisfied("mixed", "mixed") is True


def test_required_plain_triton_accepts_config_only_patch() -> None:
    assert _dialect_contract_satisfied("plain_triton", "unknown") is True
    assert _dialect_contract_satisfied("plain_triton", "amd_gluon") is False
    assert _dialect_contract_satisfied("plain_triton", "mixed") is False


def test_classify_patch_output_dialect_ignores_context_lines() -> None:
    patch = "\n".join(
        [
            "diff --git a/kernel.py b/kernel.py",
            "@@",
            " import triton.language as tl",
            "+from triton.experimental import gluon",
            "+x = gl.arange(0, 16, layout=layout)",
        ]
    )
    assert classify_patch_output_dialect(patch) == "amd_gluon"


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
    (patch_dir / "patch_0.patch").write_text(
        "diff --git a/kernel.py b/kernel.py\n+from triton.experimental import gluon\n+@gluon.jit\n+def gluon_k():\n+    return 0\n+gluon_k[grid]()\n"
    )
    (patch_dir / "patch_0_test.txt").write_text(
        "case_small: 0.0642 ms\ncase_medium: 0.0634 ms\n"
    )
    (patch_dir / "patch_1.patch").write_text(
        "diff --git a/kernel.py b/kernel.py\n+from triton.experimental import gluon\n+@gluon.jit\n+def gluon_k():\n+    return 0\n+gluon_k[grid]()\n"
    )
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


def test_compute_best_patch_uses_safe_anchor_for_composition_tasks(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    base_dir = root / "results" / "round_1" / "base-best"
    patch_dir = root / "results" / "round_2" / "hybrid-dispatch"
    base_dir.mkdir(parents=True)
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_2"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_hybrid-dispatch.md",
        {
            "label": "hybrid-dispatch",
            "required_output_dialect": "mixed",
        },
        "\n".join(
            [
                "Extension layer: Hybrid",
                "Safe anchor: round_1/base-best/patch_4",
                "Comparison target: safe_anchor",
            ]
        ),
    )
    (root / "benchmark_baseline.txt").write_text("case_small: 1.0 ms\ncase_medium: 1.0 ms\n")
    (base_dir / "patch_4_test.txt").write_text("case_small: 0.8 ms\ncase_medium: 0.8 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "diff --git a/kernel.py b/kernel.py\n"
        "+from triton.experimental import gluon\n"
        "+from triton.experimental.gluon import language as gl\n"
        "+import triton\n+import triton.language as tl\n"
        "+@gluon.jit\n+def gluon_k():\n+    x = gl.load(ptr)\n"
        "+gluon_k[grid]()\n"
        "+@triton.jit\n+def triton_k():\n+    y = tl.load(ptr)\n"
    )
    (patch_dir / "patch_1_test.txt").write_text("case_small: 0.7 ms\ncase_medium: 0.7 ms\n")

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["comparison_target"] == "safe_anchor"
    assert result["safe_anchor"] == "round_1/base-best/patch_4"
    assert result["baseline_source"] == "safe_anchor:round_1/base-best/patch_4"
    assert result["baseline_latency_ms"] == pytest.approx(1.6)
    assert result["true_baseline_latency_ms"] == pytest.approx(2.0)
    assert result["candidate_latency_ms"] == pytest.approx(1.4)
    assert result["best_patch_speedup"] == pytest.approx(1.142857)


def test_compute_best_patch_rejects_safe_anchor_shape_regression(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    base_dir = root / "results" / "round_1" / "base-best"
    patch_dir = root / "results" / "round_2" / "hybrid-dispatch"
    base_dir.mkdir(parents=True)
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_2"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_hybrid-dispatch.md",
        {
            "label": "hybrid-dispatch",
            "required_output_dialect": "mixed",
        },
        "Extension layer: Hybrid\nSafe anchor: round_1/base-best/patch_4\nComparison target: safe_anchor\n",
    )
    (root / "benchmark_baseline.txt").write_text("case_small: 1.0 ms\ncase_medium: 1.0 ms\n")
    (base_dir / "patch_4_test.txt").write_text("case_small: 0.8 ms\ncase_medium: 0.8 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "diff --git a/kernel.py b/kernel.py\n"
        "+from triton.experimental import gluon\n"
        "+from triton.experimental.gluon import language as gl\n"
        "+import triton\n+import triton.language as tl\n"
        "+@gluon.jit\n+def gluon_k():\n+    x = gl.load(ptr)\n"
        "+@triton.jit\n+def triton_k():\n+    y = tl.load(ptr)\n"
    )
    (patch_dir / "patch_1_test.txt").write_text("case_small: 0.6 ms\ncase_medium: 0.9 ms\n")

    assert compute_best_patch(patch_dir) is None


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
        "diff --git a/kernel.py b/kernel.py\n+from triton.experimental import gluon\n+@gluon.jit\n+def gluon_k():\n+    return 0\n+gluon_k[grid]()\n"
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
        "diff --git a/kernel.py b/kernel.py\n"
        "+from triton.experimental import gluon\n"
        "+@gluon.jit\n"
        "+def kernel_gluon(x):\n"
        "+    return x\n"
        "+kernel_gluon[grid](x)\n"
    )
    (patch_dir / "patch_3_test.txt").write_text("case_a: 0.8 ms\ncase_b: 0.8 ms\n")

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["best_patch_id"] == "patch_3"
    assert result["required_output_dialect"] == "amd_gluon"
    assert result["actual_output_dialect"] == "amd_gluon"


def test_compute_best_patch_infers_target_symbol_from_task_body(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "extension-l1-gluon-stage1-rope"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_extension-l1-gluon-stage1-rope.md",
        {
            "label": "extension-l1-gluon-stage1-rope",
            "required_output_dialect": "amd_gluon",
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
                "+# launcher still calls target_stage_kernel",
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
    assert result["required_patch_target_symbols"] == ["target_stage_kernel"]
    assert result["gluon_execution_contract_satisfied"] is True


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


def test_compute_best_patch_rejects_definition_only_gluon_helper_without_target_symbol(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "gluon-l0-explicit-layout-attention"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_gluon-l0-explicit-layout-attention.md",
        {
            "label": "gluon-l0-explicit-layout-attention",
            "required_output_dialect": "amd_gluon",
        },
        "Extension layer: L0\nImplementation layer: amd_gluon overlay\n",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "\n".join(
            [
                "diff --git a/kernel.py b/kernel.py",
                "+from triton.experimental import gluon",
                "+from triton.experimental.gluon import language as gl",
                "+@gluon.jit",
                "+def _gluon_scale_logits(x, layout: gl.constexpr):",
                "+    offs = gl.arange(0, 128, layout=layout)",
                "+    return offs",
                "+plain_triton_kernel[grid](x)",
            ]
        )
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.2 ms\ncase_b: 0.2 ms\n")
    (patch_dir / "patch_2.patch").write_text(
        "\n".join(
            [
                "diff --git a/kernel.py b/kernel.py",
                "+from triton.experimental import gluon",
                "+from triton.experimental.gluon import language as gl",
                "+@gluon.jit",
                "+def _gluon_scale_logits(x, layout: gl.constexpr):",
                "+    offs = gl.arange(0, 128, layout=layout)",
                "+    return offs",
                "+_gluon_scale_logits[grid](x, layout)",
            ]
        )
    )
    (patch_dir / "patch_2_test.txt").write_text("case_a: 0.8 ms\ncase_b: 0.8 ms\n")

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["best_patch_id"] == "patch_2"
    assert result["gluon_execution_contract_satisfied"] is True


def test_compute_best_patch_rejects_target_named_gluon_helper_without_execution(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "extension-l0-targeted-gluon"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_extension-l0-targeted-gluon.md",
        {
            "label": "extension-l0-targeted-gluon",
            "required_output_dialect": "amd_gluon",
            "required_patch_target_symbols": ["target_stage_kernel"],
        },
        "Extension layer: L0\nTarget symbol: target_stage_kernel\n",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "\n".join(
            [
                "diff --git a/kernel.py b/kernel.py",
                "+from triton.experimental import gluon",
                "+@gluon.jit",
                "+def target_stage_kernel(x):",
                "+    return x",
                "+plain_triton_kernel[grid](x)",
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
                "+def target_stage_kernel(x):",
                "+    return x",
                "+target_stage_kernel[grid](x)",
            ]
        )
    )
    (patch_dir / "patch_2_test.txt").write_text("case_a: 0.8 ms\ncase_b: 0.8 ms\n")

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["best_patch_id"] == "patch_2"


def test_compute_best_patch_rejects_unrelated_launched_gluon_helper_for_target(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "extension-l0-targeted-gluon"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_extension-l0-targeted-gluon.md",
        {
            "label": "extension-l0-targeted-gluon",
            "required_output_dialect": "amd_gluon",
            "required_patch_target_symbols": ["target_stage_kernel"],
        },
        "Extension layer: L0\nTarget symbol: target_stage_kernel\n",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "\n".join(
            [
                "diff --git a/kernel.py b/kernel.py",
                "+from triton.experimental import gluon",
                "+@gluon.jit",
                "+def unrelated_gluon_helper(x):",
                "+    return x",
                "+unrelated_gluon_helper[grid](x)",
                "+# target_stage_kernel should stay unchanged",
            ]
        )
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.2 ms\ncase_b: 0.2 ms\n")

    assert compute_best_patch(patch_dir) is None


def test_compute_best_patch_rejects_context_only_target_binding(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "extension-l0-targeted-gluon"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_extension-l0-targeted-gluon.md",
        {
            "label": "extension-l0-targeted-gluon",
            "required_output_dialect": "amd_gluon",
            "required_patch_target_symbols": ["target_stage_kernel"],
        },
        "Extension layer: L0\nTarget symbol: target_stage_kernel\n",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "\n".join(
            [
                "diff --git a/kernel.py b/kernel.py",
                "@@",
                " def target_stage_kernel(x):",
                "     return plain_triton_kernel(x)",
                "",
                "+from triton.experimental import gluon",
                "+@gluon.jit",
                "+def unrelated_gluon_helper(x):",
                "+    return x",
                "+unrelated_gluon_helper[grid](x)",
            ]
        )
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.2 ms\ncase_b: 0.2 ms\n")

    assert compute_best_patch(patch_dir) is None


def test_compute_best_patch_rejects_same_hunk_cross_scope_gluon_launch(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "extension-l0-targeted-gluon"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_extension-l0-targeted-gluon.md",
        {
            "label": "extension-l0-targeted-gluon",
            "required_output_dialect": "amd_gluon",
            "required_patch_target_symbols": ["target_stage_kernel"],
        },
        "Extension layer: L0\nTarget symbol: target_stage_kernel\n",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "\n".join(
            [
                "diff --git a/kernel.py b/kernel.py",
                "@@",
                " def target_stage_kernel(x):",
                "+    z = x + 1",
                "     return plain_triton_kernel(z)",
                "",
                "+from triton.experimental import gluon",
                "+from triton.experimental.gluon import language as gl",
                "+@gluon.jit",
                "+def unrelated_gluon_helper(x):",
                "+    return gl.load(x)",
                "+unrelated_gluon_helper[grid](x)",
            ]
        )
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.2 ms\ncase_b: 0.2 ms\n")

    assert compute_best_patch(patch_dir) is None


def test_compute_best_patch_accepts_hunk_header_target_body_gluon_launch(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "extension-l0-targeted-gluon"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_extension-l0-targeted-gluon.md",
        {
            "label": "extension-l0-targeted-gluon",
            "required_output_dialect": "amd_gluon",
            "required_patch_target_symbols": ["target_stage_kernel"],
        },
        "Extension layer: L0\nTarget symbol: target_stage_kernel\n",
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
                "@@ def target_stage_kernel(x):",
                "     y = x + 1",
                "+    target_stage_kernel_gluon[grid](y)",
                "     return y",
            ]
        )
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.8 ms\ncase_b: 0.8 ms\n")

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["best_patch_id"] == "patch_1"


def test_compute_best_patch_rejects_gluon_marker_without_helper_or_launch(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "extension-l0-gluon-marker"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_extension-l0-gluon-marker.md",
        {
            "label": "extension-l0-gluon-marker",
            "required_output_dialect": "amd_gluon",
        },
        "Extension layer: L0\nImplementation layer: amd_gluon overlay\n",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "\n".join(
            [
                "diff --git a/kernel.py b/kernel.py",
                "+from triton.experimental import gluon",
                "+from triton.experimental.gluon import language as gl",
                "+x = gl.arange(0, 16, layout=layout)",
            ]
        )
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.2 ms\ncase_b: 0.2 ms\n")

    assert compute_best_patch(patch_dir) is None


def test_compute_best_patch_rejects_mixed_dead_gluon_helper(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "hybrid-dispatch"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_hybrid-dispatch.md",
        {
            "label": "hybrid-dispatch",
            "required_output_dialect": "mixed",
        },
        "Extension layer: Hybrid\nImplementation layer: mixed hybrid\n",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "\n".join(
            [
                "diff --git a/kernel.py b/kernel.py",
                "+from triton.experimental import gluon",
                "+from triton.experimental.gluon import language as gl",
                "+import triton.language as tl",
                "+@gluon.jit",
                "+def hybrid_gluon_helper(x):",
                "+    return gl.load(x)",
                "+y = tl.load(ptr)",
            ]
        )
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.2 ms\ncase_b: 0.2 ms\n")
    (patch_dir / "patch_2.patch").write_text(
        "\n".join(
            [
                "diff --git a/kernel.py b/kernel.py",
                "+from triton.experimental import gluon",
                "+from triton.experimental.gluon import language as gl",
                "+import triton.language as tl",
                "+@gluon.jit",
                "+def hybrid_gluon_helper(x):",
                "+    return gl.load(x)",
                "+hybrid_gluon_helper[grid](x)",
                "+y = tl.load(ptr)",
            ]
        )
    )
    (patch_dir / "patch_2_test.txt").write_text("case_a: 0.8 ms\ncase_b: 0.8 ms\n")

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["best_patch_id"] == "patch_2"
