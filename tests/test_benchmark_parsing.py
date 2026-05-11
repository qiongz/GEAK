from pathlib import Path

import pytest

from minisweagent.run.preprocess import benchmark_parsing as preprocess_benchmark_parsing
from minisweagent.run.postprocess.benchmark_parsing import (
    _dialect_contract_satisfied,
    _has_added_generic_gluon_dot,
    _has_added_plain_exception_fallback,
    _infer_forbidden_patch_target_symbols,
    _infer_required_patch_target_symbols,
    _OPTIONAL_GLUON_RESULT_METADATA,
    _patch_touches_any_target_symbol,
    _patch_touches_backup_file,
    _patch_touches_forbidden_target_symbol,
    _required_amd_gluon_static_contract_error,
    _required_output_dialect,
    _scope_escalation_violation,
    classify_gluon_api_contract,
    classify_layout_contract,
    classify_patch_output_dialect,
    compute_best_patch,
    extract_latency_ms,
    parse_shape_latencies_ms,
    rewrite_best_results,
)


def test_postprocess_does_not_add_patch_evolution_metadata_fields() -> None:
    post_fields = {key for key, _field in _OPTIONAL_GLUON_RESULT_METADATA}
    pre_fields = {key for key, _field in preprocess_benchmark_parsing._OPTIONAL_GLUON_RESULT_METADATA}

    for fields in (post_fields, pre_fields):
        assert "declared_failure_layer" in fields
        assert "changed_failure_layer" in fields
        assert "failure_layers" in fields
        assert "expected_failure_layers" in fields
        assert "patch_evolution_strategy" not in fields
        assert "patch_0_goal" not in fields
        assert "patch_1_plus_rule" not in fields


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


def test_patch_touches_backup_file_detects_bak_source_copy() -> None:
    patch = "\n".join(
        [
            "diff --git a/kernel.py.bak b/kernel.py.bak",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/kernel.py.bak",
            "+import triton.language as tl",
        ]
    )

    assert _patch_touches_backup_file(patch) is True


def test_infer_required_patch_target_symbols_from_allowed_change_components() -> None:
    body = "\n".join(
        [
            "Allowed change: Replace the scoped local expressions (tile_load_a, tile_load_b, tile_load_c) in the inner loop.",
            "Reject if: target-symbol mismatch.",
        ]
    )

    assert _infer_required_patch_target_symbols(body, {}) == ["tile_load_a", "tile_load_b", "tile_load_c"]


def test_infer_required_patch_target_symbols_from_stage_scoped_allowed_change() -> None:
    body = "Allowed change: layout specification for tensor creation and load/store in stage1 inner loop"

    assert _infer_required_patch_target_symbols(body, {}) == ["stage1"]


def test_infer_required_patch_target_symbols_ignores_non_target_parentheses() -> None:
    body = "Allowed change: specialize the shape bucket (B, H, S) without naming local target components."

    assert _infer_required_patch_target_symbols(body, {}) == []


def test_stage_scope_target_does_not_accept_other_stage_patch() -> None:
    patch = "\n".join(
        [
            "diff --git a/kernel.py b/kernel.py",
            "+@gluon.jit",
            "+def _fwd_kernel_stage2_gluon(x):",
            "+    return x",
        ]
    )

    assert _patch_touches_any_target_symbol(patch, ["stage1"]) is False


def test_infer_forbidden_patch_target_symbols_from_reject_if_and_do_not() -> None:
    body = "\n".join(
        [
            "Reject if: the main dot loop or A/B loads are modified.",
            "Do NOT attempt to convert the entire kernel to Gluon.",
        ]
    )

    assert _infer_forbidden_patch_target_symbols(body, {}) == [
        "dot_loop",
        "ab_input_loads",
        "whole_kernel_rewrite",
    ]


def test_patch_touches_forbidden_dot_loop_and_input_loads() -> None:
    patch = "\n".join(
        [
            "diff --git a/kernel.py b/kernel.py",
            "+@gluon.jit",
            "+def scaled_mm_kernel(a_ptrs, b_ptrs):",
            "+    a = gl.load(a_ptrs)",
            "+    b = gl.load(b_ptrs)",
            "+    return gl.amd.cdna3.mfma(a, b, acc)",
        ]
    )

    assert _patch_touches_forbidden_target_symbol(patch, ["dot_loop"]) == "dot_loop"
    assert _patch_touches_forbidden_target_symbol(patch, ["ab_input_loads"]) == "ab_input_loads"
    assert _patch_touches_forbidden_target_symbol(patch, ["whole_kernel_rewrite"]) == "whole_kernel_rewrite"


def test_forbidden_whole_kernel_does_not_reject_scoped_helper() -> None:
    patch = "\n".join(
        [
            "diff --git a/kernel.py b/kernel.py",
            "+@gluon.jit",
            "+def scale_epilogue_gluon(x):",
            "+    return x",
        ]
    )

    assert _patch_touches_forbidden_target_symbol(patch, ["whole_kernel_rewrite"]) is None
    assert _patch_touches_forbidden_target_symbol(patch, ["new_gluon_helper"]) == "new_gluon_helper"


def test_scope_escalation_detects_inline_task_replacing_target_kernel() -> None:
    patch = "\n".join(
        [
            "diff --git a/kernel.py b/kernel.py",
            "+from triton.experimental import gluon",
            "+@gluon.jit",
            "+def _fwd_grouped_kernel_stage1_rope_gluon(q, k):",
            "+    return q",
            "+_fwd_grouped_kernel_stage1_rope_gluon[grid](q, k)",
        ]
    )

    violation = _scope_escalation_violation(
        patch,
        {
            "allowed_execution_path": "inline_scoped_helper",
            "target_symbol": "_fwd_grouped_kernel_stage1_rope",
        },
    )

    assert violation == "scope escalation: inline_scoped_helper -> whole_jit_kernel via `_fwd_grouped_kernel_stage1_rope_gluon`"


def test_infer_forbidden_new_helper_only_scope() -> None:
    body = "Reject if: helper-only path is added next to an unchanged plain Triton path."

    assert _infer_forbidden_patch_target_symbols(body, {}) == ["new_gluon_helper"]


def test_required_amd_gluon_static_contract_rejects_generic_gluon_dot() -> None:
    patch = "\n".join(
        [
            "diff --git a/kernel.py b/kernel.py",
            "+from triton.experimental import gluon",
            "+from triton.experimental.gluon import language as gl",
            "+@gluon.jit",
            "+def kernel_gluon(q, k):",
            "+    return gl.dot(q, k)",
        ]
    )

    assert _has_added_generic_gluon_dot(patch) is True
    assert _required_amd_gluon_static_contract_error(patch, "amd_gluon") is None
    assert "generic gl.dot" in (
        _required_amd_gluon_static_contract_error(
            patch,
            "amd_gluon",
            {"performance_hypothesis": "DotOperandLayout can ensure MFMA selection"},
        )
        or ""
    )
    assert _required_amd_gluon_static_contract_error(patch, "mixed") is None


def test_required_amd_gluon_static_contract_rejects_plain_exception_fallback() -> None:
    patch = "\n".join(
        [
            "diff --git a/wrapper.py b/wrapper.py",
            "+try:",
            "+    kernel_gluon[grid](x)",
            "+except Exception:",
            "+    kernel_plain[grid](x)",
        ]
    )

    assert _has_added_plain_exception_fallback(patch) is True
    assert "plain Triton launcher" in (_required_amd_gluon_static_contract_error(patch, "amd_gluon") or "")


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


def test_gluon_api_contract_allows_only_strict_generated_tl_range() -> None:
    patch = "\n".join(
        [
            "diff --git a/kernel.py b/kernel.py",
            "+from triton.experimental import gluon",
            "+import triton.language as tl",
            "+@gluon.jit",
            "+def kernel_gluon(ptr):",
            "+    for _ in tl.range(0, 4):",
            "+        pass",
        ]
    )

    contract = classify_gluon_api_contract(patch, {"source_origin": "generated_overlay"})

    assert contract["gluon_api_contract_status"] == "allowed_tl_only"
    assert contract["forbidden_tl_symbols_seen"] == []
    assert classify_patch_output_dialect(patch) == "amd_gluon"


def test_gluon_api_contract_rejects_added_tl_where_in_generated_overlay() -> None:
    patch = "\n".join(
        [
            "diff --git a/kernel.py b/kernel.py",
            "+from triton.experimental import gluon",
            "+import triton.language as tl",
            "+@gluon.jit",
            "+def kernel_gluon(x, y, mask):",
            "+    return tl.where(mask, x, y)",
        ]
    )

    contract = classify_gluon_api_contract(patch, {"source_origin": "generated_overlay"})

    assert contract["gluon_api_contract_status"] == "leftover_tl_device_api"
    assert contract["forbidden_tl_symbols_seen"][0]["symbol"] == "tl.where"
    assert classify_patch_output_dialect(patch) == "amd_gluon"


def test_gluon_api_contract_preserves_existing_tl_where_for_production_source() -> None:
    patch = "\n".join(
        [
            "diff --git a/kernel.py b/kernel.py",
            " import triton.language as tl",
            " @gluon.jit",
            " def kernel_gluon(x, y, mask):",
            "     return tl.where(mask, x, y)",
            "+    # patch edits surrounding source without adding tl.where",
        ]
    )

    contract = classify_gluon_api_contract(
        patch,
        {"source_origin": "existing_amd_gluon_operator"},
    )

    assert contract["gluon_api_contract_status"] == "allowed_tl_only"
    assert contract["tl_symbols_seen"][0]["change_kind"] == "preserved"


def test_layout_contract_reports_runtime_layout_object() -> None:
    patch = "\n".join(
        [
            "diff --git a/kernel.py b/kernel.py",
            "+@gluon.jit",
            "+def kernel_gluon(x):",
            "+    layout = gl.BlockedLayout([1], [64], [4], [0])",
            "+    return x",
        ]
    )

    contract = classify_layout_contract(patch, {"layout_construction_policy": "host_preferred"})

    assert contract["layout_contract_status"] == "runtime_layout_object"


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


def test_rewrite_best_results_inherits_task_metadata_and_invalidates_scope_escalation(tmp_path: Path) -> None:
    from minisweagent.run.task_file import write_task_file

    sidecar = tmp_path / "round_sidecar"
    task_dir = sidecar / "tasks" / "round_1"
    result_dir = sidecar / "results" / "round_1_pair" / "gluon-l0-load-store-layout"
    task_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    write_task_file(
        task_dir / "05_gluon-l0-load-store-layout.md",
        {
            "label": "gluon-l0-load-store-layout",
            "priority": 5,
            "kernel_type": "triton",
            "required_output_dialect": "amd_gluon",
            "implementation_layer": "amd_gluon overlay",
            "extension_layer": "L0",
            "source_origin": "generated_overlay",
            "minimum_executable_unit": "inline_scoped_helper",
            "allowed_execution_path": "inline_scoped_helper",
            "target_symbol": "_fwd_grouped_kernel_stage1_rope",
            "target_component": "_fwd_grouped_kernel_stage1_rope online softmax accumulator path",
            "l0_scope_classification": "low_coupling",
        },
        "Extension layer: L0\nImplementation layer: amd_gluon overlay\n",
    )
    (result_dir / "patch_0.patch").write_text(
        "\n".join(
            [
                "diff --git a/kernel.py b/kernel.py",
                "+from triton.experimental import gluon",
                "+@gluon.jit",
                "+def _fwd_grouped_kernel_stage1_rope_gluon(q, k):",
                "+    return q",
                "+_fwd_grouped_kernel_stage1_rope_gluon[grid](q, k)",
            ]
        )
    )
    (result_dir / "patch_0_test.txt").write_text("compile failed\n")
    (result_dir / "best_results.json").write_text(
        '{"best_patch_id": null, "best_patch_file": null, "best_patch_speedup": 0.0}'
    )

    result = rewrite_best_results(result_dir)

    assert result is not None
    assert result["required_output_dialect"] == "amd_gluon"
    assert result["source_origin"] == "generated_overlay"
    assert result["allowed_execution_path"] == "inline_scoped_helper"
    assert result["scope_compliant"] is False
    assert result["gluon_execution_contract_satisfied"] is False
    assert result["gluon_evidence_summary"] == "scope_escalation"
    assert "scope escalation" in result["scope_escalation_violation"]


def test_triton_gluon_docs_expose_generic_failure_layer_recipes() -> None:
    root = Path(__file__).resolve().parents[1]
    always = (root / "skills/triton-gluon/docs/00_always_read.md").read_text()
    traits = (root / "skills/triton-gluon/docs/20_component_traits.md").read_text()
    api = (root / "skills/triton-gluon/docs/50_api_reference.md").read_text()
    real = (root / "skills/triton-gluon/docs/60_real_patterns.md").read_text()

    assert "whole_kernel_layout_map_recipe" in always
    assert "broadcast_failure_debug_recipe" in always
    assert "dot_lowering_minimal_recipe" in always
    assert "## whole_kernel_layout_map_recipe" in traits
    assert "## broadcast_failure_debug_recipe" in api
    assert "## dot_lowering_minimal_recipe" in api
    assert "## reduction_accumulator_layout_recipe" in api
    assert "Composite high-coupling kernels" in real
    assert "attention-style decode kernels" not in always


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
        "+def dispatch(use_gluon, grid):\n"
        "+    if use_gluon:\n"
        "+        return gluon_k[grid]()\n"
        "+    return triton_kernel[grid]()\n"
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
    assert result["scope_compliant"] is True
    assert result["forbidden_scope_violation"] is None


def test_compute_best_patch_marks_slower_execution_anchor_not_viable_for_l1(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "extension-l0-gluon-anchor"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_extension-l0-gluon-anchor.md",
        {
            "label": "extension-l0-gluon-anchor",
            "required_output_dialect": "amd_gluon",
            "extension_intent": "execution_anchor",
            "expected_outcome": "correctness_anchor_not_speedup",
            "not_viable_for_l1_if_slower_than_base": True,
            "overhead_source_to_record": "launch_layout_overhead",
            "minimum_executable_unit": "separate_gluon_kernel",
            "target_component": "one 1D subpath",
            "allowed_execution_path": "separate_gluon_kernel",
            "scope_infeasible_policy": "separate_kernel_if_allowed",
            "whole_kernel_required_reason": "not needed for separate kernel",
        },
        "Extension L0\nTarget component: one 1D subpath\n",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "diff --git a/kernel.py b/kernel.py\n+from triton.experimental import gluon\n+@gluon.jit\n+def kernel_gluon(x):\n+    return x\n+kernel_gluon[grid](x)\n"
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 1.2 ms\ncase_b: 1.2 ms\n")

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["extension_intent"] == "execution_anchor"
    assert result["expected_outcome"] == "correctness_anchor_not_speedup"
    assert result["target_component"] == "one 1D subpath"
    assert result["minimum_executable_unit"] == "separate_gluon_kernel"
    assert result["allowed_execution_path"] == "separate_gluon_kernel"
    assert result["scope_infeasible_policy"] == "separate_kernel_if_allowed"
    assert result["scope_infeasible_reported"] is False
    assert result["scope_compliant"] is True
    assert result["gluon_l1_anchor_viability"] == "not_viable_for_l1"
    assert result["not_viable_for_l1"] is True
    assert result["overhead_source"] == "launch_layout_overhead"


def test_compute_best_patch_rejects_forbidden_scope_patch(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "gluon-l0-scale-load-layout"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "05_gluon-l0-scale-load-layout.md",
        {
            "label": "gluon-l0-scale-load-layout",
            "required_output_dialect": "amd_gluon",
            "required_patch_target_symbols": ["scale_a", "scale_b"],
        },
        "\n".join(
            [
                "Allowed change: Convert only the `scale_a` and `scale_b` load+broadcast+multiply path.",
                "Reject if: the main dot loop or A/B loads are modified.",
                "Do NOT attempt to convert the entire kernel to Gluon.",
            ]
        ),
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "\n".join(
            [
                "diff --git a/kernel.py b/kernel.py",
                "+from triton.experimental import gluon",
                "+from triton.experimental.gluon import language as gl",
                "+@gluon.jit",
                "+def scaled_mm_kernel(a_ptrs, b_ptrs, scale_a_ptr, scale_b_ptr):",
                "+    a = gl.load(a_ptrs)",
                "+    b = gl.load(b_ptrs)",
                "+    scale_a = gl.load(scale_a_ptr)",
                "+    scale_b = gl.load(scale_b_ptr)",
                "+    return gl.amd.cdna3.mfma(a, b, acc) + scale_a + scale_b",
                "+scaled_mm_kernel[grid](a, b, scale_a, scale_b)",
            ]
        )
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.5 ms\ncase_b: 0.5 ms\n")

    assert compute_best_patch(patch_dir) is None


def test_preprocess_compute_best_patch_rejects_forbidden_scope_patch(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "gluon-l0-scale-load-layout"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "05_gluon-l0-scale-load-layout.md",
        {
            "label": "gluon-l0-scale-load-layout",
            "required_output_dialect": "amd_gluon",
            "required_patch_target_symbols": ["scale_a", "scale_b"],
        },
        "\n".join(
            [
                "Allowed change: Convert only the `scale_a` and `scale_b` load+broadcast+multiply path.",
                "Reject if: the main dot loop or A/B loads are modified.",
                "Do NOT attempt to convert the entire kernel to Gluon.",
            ]
        ),
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "\n".join(
            [
                "diff --git a/kernel.py b/kernel.py",
                "+from triton.experimental import gluon",
                "+from triton.experimental.gluon import language as gl",
                "+@gluon.jit",
                "+def scaled_mm_kernel(a_ptrs, b_ptrs, scale_a_ptr, scale_b_ptr):",
                "+    a = gl.load(a_ptrs)",
                "+    b = gl.load(b_ptrs)",
                "+    scale_a = gl.load(scale_a_ptr)",
                "+    scale_b = gl.load(scale_b_ptr)",
                "+    return gl.amd.cdna3.mfma(a, b, acc) + scale_a + scale_b",
                "+scaled_mm_kernel[grid](a, b, scale_a, scale_b)",
            ]
        )
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.5 ms\ncase_b: 0.5 ms\n")

    assert preprocess_benchmark_parsing.compute_best_patch(patch_dir) is None


def test_compute_best_patch_keeps_performance_candidate_as_evidence_not_global_block(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "extension-l0-performance-candidate"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_extension-l0-performance-candidate.md",
        {
            "label": "extension-l0-performance-candidate",
            "required_output_dialect": "amd_gluon",
            "extension_intent": "performance_candidate",
            "expected_outcome": "possible_speedup",
            "overhead_source_to_record": "conversion_overhead",
        },
        "Extension L0\n",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "diff --git a/kernel.py b/kernel.py\n+from triton.experimental import gluon\n+@gluon.jit\n+def kernel_gluon(x):\n+    return x\n+kernel_gluon[grid](x)\n"
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 1.1 ms\ncase_b: 1.1 ms\n")

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["extension_intent"] == "performance_candidate"
    assert result["gluon_l1_anchor_viability"] == "neutral_or_slow_anchor"
    assert result["not_viable_for_l1"] is False
    assert result["overhead_source"] == "conversion_overhead"


def test_compute_best_patch_plain_task_has_no_positive_gluon_execution_contract(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "plain-streamline"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "00_plain-streamline.md",
        {
            "label": "plain-streamline",
            "required_output_dialect": "plain_triton",
        },
        "Base Set task\n",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text("diff --git a/kernel.py b/kernel.py\n+import triton.language as tl\n+x = tl.arange(0, 16)\n")
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.8 ms\ncase_b: 0.8 ms\n")

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["required_output_dialect"] == "plain_triton"
    assert result["gluon_execution_contract_satisfied"] is False
    assert result["gluon_l1_anchor_viability"] == "not_applicable"


def test_preprocess_and_postprocess_preserve_gluon_anchor_metadata(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "extension-l0-gluon-anchor"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_extension-l0-gluon-anchor.md",
        {
            "label": "extension-l0-gluon-anchor",
            "required_output_dialect": "amd_gluon",
            "extension_intent": "execution_anchor",
            "expected_outcome": "correctness_anchor_not_speedup",
            "not_viable_for_l1_if_slower_than_base": True,
            "overhead_source_to_record": "launch_layout_overhead",
            "minimum_executable_unit": "separate_gluon_kernel",
            "target_component": "one 1D subpath",
            "allowed_execution_path": "separate_gluon_kernel",
            "scope_infeasible_policy": "separate_kernel_if_allowed",
            "whole_kernel_required_reason": "not needed for separate kernel",
        },
        "Extension L0\nTarget component: one 1D subpath\n",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "diff --git a/kernel.py b/kernel.py\n+from triton.experimental import gluon\n+@gluon.jit\n+def kernel_gluon(x):\n+    return x\n+kernel_gluon[grid](x)\n"
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.8 ms\ncase_b: 0.8 ms\n")

    post = compute_best_patch(patch_dir)
    pre = preprocess_benchmark_parsing.compute_best_patch(patch_dir)

    assert post is not None
    assert pre is not None
    for result in (post, pre):
        assert result["required_output_dialect"] == "amd_gluon"
        assert result["extension_intent"] == "execution_anchor"
        assert result["expected_outcome"] == "correctness_anchor_not_speedup"
        assert result["target_component"] == "one 1D subpath"
        assert result["minimum_executable_unit"] == "separate_gluon_kernel"
        assert result["allowed_execution_path"] == "separate_gluon_kernel"
        assert result["scope_infeasible_policy"] == "separate_kernel_if_allowed"
        assert result["whole_kernel_required_reason"] == "not needed for separate kernel"
        assert result["gluon_execution_contract_satisfied"] is True
        assert result["gluon_l1_anchor_viability"] == "viable_for_l1"


def test_preprocess_preserves_slower_gluon_execution_anchor_evidence(tmp_path: Path) -> None:
    root = tmp_path / "generic_kernel"
    patch_dir = root / "results" / "round_1" / "extension-l0-gluon-anchor"
    patch_dir.mkdir(parents=True)
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)

    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_extension-l0-gluon-anchor.md",
        {
            "label": "extension-l0-gluon-anchor",
            "required_output_dialect": "amd_gluon",
            "extension_intent": "execution_anchor",
            "expected_outcome": "correctness_anchor_not_speedup",
            "not_viable_for_l1_if_slower_than_base": True,
            "overhead_source_to_record": "launch_layout_overhead",
            "target_component": "one 1D subpath",
        },
        "Extension L0\nTarget component: one 1D subpath\n",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0 ms\ncase_b: 1.0 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "diff --git a/kernel.py b/kernel.py\n+from triton.experimental import gluon\n+@gluon.jit\n+def kernel_gluon(x):\n+    return x\n+kernel_gluon[grid](x)\n"
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 1.2 ms\ncase_b: 1.2 ms\n")

    result = preprocess_benchmark_parsing.compute_best_patch(patch_dir)

    assert result is not None
    assert result["best_patch_speedup"] == pytest.approx(0.833333, rel=1e-5)
    assert result["improves_true_baseline"] is False
    assert result["has_significant_shape_regression"] is True
    assert result["extension_intent"] == "execution_anchor"
    assert result["gluon_execution_contract_satisfied"] is True
    assert result["gluon_l1_anchor_viability"] == "not_viable_for_l1"
    assert result["not_viable_for_l1"] is True
    assert result["overhead_source"] == "launch_layout_overhead"


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
                    "+def dispatch(use_gluon, grid):",
                    "+    if use_gluon:",
                    "+        return hybrid_gluon_helper[grid](x)",
                    "+    return triton_kernel[grid]()",
            ]
        )
    )
    (patch_dir / "patch_2_test.txt").write_text("case_a: 0.8 ms\ncase_b: 0.8 ms\n")

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["best_patch_id"] == "patch_2"
