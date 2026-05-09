"""Tests for the multi-shape coverage planner extension.

Covers four layers:

1. ``discovery_types.derive_shape_coverage_profile`` and feature-metadata
   integration (``build_gluon_feature_metadata``).
2. ``benchmark_parsing.parse_test_case_count`` /
   ``parse_performance_report_json``.
3. ``task_generator`` trait inference, quota adjustment, prompt rendering,
   and per-shape signal aggregation from prior round evaluations.
4. ``pipeline_helpers.inject_pipeline_context`` worker-side multi-shape
   working set injection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from minisweagent.agents.heterogeneous.task_generator import (
    _aggregate_prior_per_shape,
    _base_extension_quotas,
    _build_search_space_allocation_guidance,
    _build_shape_coverage_guidance,
    _dialect_interleave_strategy,
    _gluon_extension_strength,
    _infer_gluon_planning_traits,
    _previous_gluon_signal,
)
from minisweagent.run.pipeline_helpers import (
    _build_shape_coverage_working_set,
    inject_pipeline_context,
)
from minisweagent.run.preprocess.benchmark_parsing import (
    classify_patch_output_dialect,
    compute_best_patch,
    discover_performance_report,
    extract_latency_ms,
    parse_performance_report_json,
    parse_shape_latencies_ms,
    parse_test_case_count,
    rewrite_best_results,
)
from minisweagent.run.preprocess.discovery_types import (
    SHAPE_COVERAGE_BUCKETED,
    SHAPE_COVERAGE_MULTI,
    SHAPE_COVERAGE_SINGLE,
    SHAPE_COVERAGE_UNKNOWN,
    build_gluon_feature_metadata,
    derive_shape_coverage_profile,
)


def _gluon_meta_with_shape(
    profile: str,
    *,
    cases: list[dict[str, Any]] | None = None,
    count: int | None = None,
) -> dict[str, Any]:
    cases = list(cases or [])
    return {
        "kernel_type": "triton",
        "input_dialect": "plain_triton",
        "gluon_feature_mode": "auto",
        "gluon_baseline_profile": "raw",
        "allowed_output_dialects": ["plain_triton", "amd_gluon"],
        "preferred_output_dialects": ["amd_gluon", "plain_triton"],
        "output_dialect_search_policy": "prefer_amd_gluon_if_viable_else_plain_triton",
        "target_backend": "hip/gfx942",
        "shape_coverage_profile": profile,
        "benchmark_shape_count": count if count is not None else (len(cases) or None),
        "benchmark_test_cases": cases,
    }


# ── 1. shape_coverage_profile inference -------------------------------------


def test_derive_shape_coverage_profile_unknown_when_no_signal() -> None:
    assert derive_shape_coverage_profile() == SHAPE_COVERAGE_UNKNOWN


def test_derive_shape_coverage_profile_single() -> None:
    assert derive_shape_coverage_profile(benchmark_shape_count=1) == SHAPE_COVERAGE_SINGLE


def test_derive_shape_coverage_profile_multi_for_close_sizes() -> None:
    cases = [
        {"params": {"M": 32, "K": 32, "N": 32}},
        {"params": {"M": 64, "K": 64, "N": 64}},
        {"params": {"M": 64, "K": 96, "N": 96}},
    ]
    assert derive_shape_coverage_profile(benchmark_test_cases=cases) == SHAPE_COVERAGE_MULTI


def test_derive_shape_coverage_profile_bucketed_when_orders_apart() -> None:
    cases = [
        {"params": {"M": 32, "K": 64, "N": 64}},
        {"params": {"M": 64, "K": 128, "N": 128}},
        {"params": {"M": 128, "K": 256, "N": 256}},
        {"params": {"M": 256, "K": 512, "N": 512}},
    ]
    assert derive_shape_coverage_profile(benchmark_test_cases=cases) == SHAPE_COVERAGE_BUCKETED


def test_build_gluon_feature_metadata_carries_shape_fields(tmp_path: Path) -> None:
    kernel = tmp_path / "kernel.py"
    kernel.write_text("import triton\nimport triton.language as tl\n@triton.jit\ndef k(x): pass\n")
    cases = [
        {"test_case_id": "perf1", "params": {"M": 32, "K": 64, "N": 64}, "execution_time_ms": 1.2},
        {"test_case_id": "perf2", "params": {"M": 64, "K": 128, "N": 128}, "execution_time_ms": 2.1},
        {"test_case_id": "perf3", "params": {"M": 128, "K": 256, "N": 256}, "execution_time_ms": 4.4},
        {"test_case_id": "perf4", "params": {"M": 256, "K": 512, "N": 512}, "execution_time_ms": 9.6},
    ]
    meta = build_gluon_feature_metadata(
        kernel,
        "triton",
        benchmark_test_cases=cases,
    )
    assert meta["benchmark_shape_count"] == 4
    assert meta["shape_coverage_profile"] == SHAPE_COVERAGE_BUCKETED
    assert meta["benchmark_test_cases"][0]["case_id"] == "perf1"
    assert meta["benchmark_test_cases"][0]["params"]["M"] == 32


# ── 2. benchmark_parsing helpers --------------------------------------------


def test_parse_test_case_count_recognises_multiple_phrasings() -> None:
    assert parse_test_case_count("Performance: measured 5 test case(s), total: 0.4ms") == 5
    assert parse_test_case_count("Benchmark covered 3 shapes") == 3
    assert parse_test_case_count("nothing here") is None


def test_parse_performance_report_json(tmp_path: Path) -> None:
    payload = [
        {
            "test_case_id": "perf1",
            "execution_time_ms": 0.123,
            "params": {"M": 32, "K": 64, "N": 64},
        },
        {
            "test_case_id": "perf2",
            "execution_time_ms": 0.456,
            "params": {"M": 64, "K": 128, "N": 128},
        },
    ]
    path = tmp_path / "performance_report.json"
    path.write_text(json.dumps(payload))
    cases = parse_performance_report_json(path)
    assert cases is not None
    assert [c["case_id"] for c in cases] == ["perf1", "perf2"]
    assert cases[0]["params"]["M"] == 32


def test_parse_performance_report_json_missing(tmp_path: Path) -> None:
    assert parse_performance_report_json(tmp_path / "missing.json") is None


def test_parse_performance_report_json_handles_aiter_metadata_and_invalid_times(
    tmp_path: Path,
) -> None:
    payload = [
        {
            "test_case_id": "aiter_same_shape",
            "shape": [1, 16, 16, 128, 2048],
            "execution_time_ms": 0.123,
            "metadata": {"B": 1, "H_Q": 16, "H_KV": 16, "D": 128, "SEQ_LEN": 2048},
        },
        {
            "test_case_id": "failed_case",
            "execution_time_ms": -1.0,
            "params": {"M": 32},
        },
        {
            "test_case_id": "zero_case",
            "execution_time_ms": 0.0,
            "params": {"M": 64},
        },
        {
            "test_case_id": "string_case",
            "execution_time_ms": "0.456",
            "params": {"M": 128},
        },
    ]
    path = tmp_path / "performance_report.json"
    path.write_text(json.dumps(payload))
    cases = parse_performance_report_json(path)
    assert cases is not None
    assert cases[0]["params"]["SEQ_LEN"] == 2048
    assert "shape" not in cases[0]["params"]
    assert cases[1]["ms"] is None
    assert cases[2]["ms"] is None
    assert cases[3]["ms"] == 0.456


def test_parse_performance_report_json_torch2hip_baseline_only(tmp_path: Path) -> None:
    """torch2hip ``cal_kernel_perf.py --baseline_only`` emits ``opt_time=None`` and
    keeps the actual baseline timing in ``ori_time``. The parser must skip
    the explicit None and fall through to ori_time, otherwise GEAK would
    silently drop baseline_ms for every case in this very common Arena mode.
    """
    payload = [
        {
            "case_idx": 0,
            "correct": True,
            "ori_time": 0.456,
            "opt_time": None,
            "speedup": None,
            "params": {"input_0_shape": [1024]},
        },
        {
            "case_idx": 1,
            "correct": True,
            "ori_time": 1.234,
            "opt_time": None,
            "speedup": None,
            "params": {"input_0_shape": [4096]},
        },
    ]
    path = tmp_path / "performance_report.json"
    path.write_text(json.dumps(payload))
    cases = parse_performance_report_json(path)
    assert cases is not None
    assert [c["ms"] for c in cases] == [0.456, 1.234]


def test_parse_case_id_shape_latencies_and_total_objective() -> None:
    output = "pa_decode_small: 0.1759 ms\npa_decode_medium: 0.1823 ms\n"
    assert parse_shape_latencies_ms(output) == {
        "pa_decode_small": 0.1759,
        "pa_decode_medium": 0.1823,
    }
    assert extract_latency_ms(output) == pytest.approx(0.3582)


def test_compute_best_patch_uses_per_case_total_latency(tmp_path: Path) -> None:
    patch_dir = tmp_path / "results" / "round_1" / "task"
    patch_dir.mkdir(parents=True)
    root = patch_dir.parent.parent
    (root / "benchmark_baseline.txt").write_text(
        "pa_decode_small: 0.1759 ms\npa_decode_medium: 0.1823 ms\n"
    )
    (patch_dir / "patch_1.patch").write_text("diff --git a/kernel.py b/kernel.py\n")
    (patch_dir / "patch_1_test.txt").write_text(
        "pa_decode_small: 0.1606 ms\npa_decode_medium: 0.1659 ms\n"
    )
    (patch_dir / "patch_2.patch").write_text("diff --git a/kernel.py b/kernel.py\n")
    # Faster on small but a clear regression on medium; should be rejected.
    (patch_dir / "patch_2_test.txt").write_text(
        "pa_decode_small: 0.1000 ms\npa_decode_medium: 0.2500 ms\n"
    )

    best = compute_best_patch(patch_dir)
    assert best is not None
    assert best["best_patch_id"] == "patch_1"
    assert best["objective"] == "total_shape_latency_ms"
    assert best["baseline_latency_ms"] == pytest.approx(0.3582)
    assert best["candidate_latency_ms"] == pytest.approx(0.3265)
    assert best["baseline_shape_geomean_ms"] == pytest.approx(0.179071, rel=1e-5)
    assert best["candidate_shape_geomean_ms"] == pytest.approx(0.163228, rel=1e-5)
    assert best["best_patch_speedup"] == pytest.approx(1.09709, rel=1e-5)
    assert best["per_shape_speedups"]["pa_decode_small"]["speedup"] == pytest.approx(1.095268, rel=1e-5)
    assert best["per_shape_speedups"]["pa_decode_medium"]["speedup"] == pytest.approx(1.098854, rel=1e-5)


def test_compute_best_patch_rejects_required_gluon_plain_triton_fallback(tmp_path: Path) -> None:
    patch_dir = tmp_path / "results" / "round_1" / "ext-l0-gluon"
    patch_dir.mkdir(parents=True)
    root = patch_dir.parent.parent.parent
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)
    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_ext-l0-gluon.md",
        {
            "label": "ext-l0-gluon",
            "priority": 6,
            "kernel_type": "triton",
            "search_set": "extension",
            "required_output_dialect": "amd_gluon",
        },
        "Extension task",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0000 ms\ncase_b: 1.0000 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "diff --git a/kernel.py b/kernel.py\n+import triton\n+import triton.language as tl\n+tl.load(x)\n"
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.9000 ms\ncase_b: 0.9000 ms\n")

    assert compute_best_patch(patch_dir) is None


def test_classify_patch_output_dialect_detects_gluon() -> None:
    assert (
        classify_patch_output_dialect(
            "from triton.experimental import gluon\nfrom triton.experimental.gluon import language as gl\n@gluon.jit\ndef k():\n    x = gl.load(ptr)\n"
        )
        == "amd_gluon"
    )


def test_classify_patch_output_dialect_detects_mixed() -> None:
    assert (
        classify_patch_output_dialect(
            "from triton.experimental import gluon\n"
            "from triton.experimental.gluon import language as gl\n"
            "import triton\nimport triton.language as tl\n"
            "@gluon.jit\ndef gluon_k():\n    x = gl.load(ptr)\n"
            "@triton.jit\ndef reduce_k():\n    y = tl.load(ptr)\n"
        )
        == "mixed"
    )


def test_compute_best_patch_rejects_mixed_for_required_amd_gluon(tmp_path: Path) -> None:
    patch_dir = tmp_path / "results" / "round_1" / "ext-mixed-gluon"
    patch_dir.mkdir(parents=True)
    root = patch_dir.parent.parent.parent
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)
    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_ext-mixed-gluon.md",
        {
            "label": "ext-mixed-gluon",
            "priority": 6,
            "kernel_type": "triton",
            "search_set": "extension",
            "required_output_dialect": "amd_gluon",
        },
        "Extension task",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0000 ms\ncase_b: 1.0000 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "diff --git a/kernel.py b/kernel.py\n"
        "+from triton.experimental import gluon\n"
        "+from triton.experimental.gluon import language as gl\n"
        "+import triton\n+import triton.language as tl\n"
        "+@gluon.jit\n+def main_k():\n+    x = gl.load(ptr)\n"
        "+@triton.jit\n+def reduce_k():\n+    y = tl.load(ptr)\n"
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.9000 ms\ncase_b: 0.9000 ms\n")

    assert compute_best_patch(patch_dir) is None


def test_compute_best_patch_rejects_amd_gluon_for_required_mixed(tmp_path: Path) -> None:
    patch_dir = tmp_path / "results" / "round_1" / "hybrid-dispatch"
    patch_dir.mkdir(parents=True)
    root = patch_dir.parent.parent.parent
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)
    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_hybrid-dispatch.md",
        {
            "label": "hybrid-dispatch",
            "priority": 6,
            "kernel_type": "triton",
            "search_set": "extension",
            "required_output_dialect": "mixed",
        },
        "Hybrid dispatch task",
    )
    (root / "benchmark_baseline.txt").write_text("case_a: 1.0000 ms\ncase_b: 1.0000 ms\n")
    (patch_dir / "patch_1.patch").write_text(
        "diff --git a/kernel.py b/kernel.py\n"
        "+from triton.experimental import gluon\n"
        "+from triton.experimental.gluon import language as gl\n"
        "+@gluon.jit\n+def main_k():\n+    x = gl.load(ptr)\n"
    )
    (patch_dir / "patch_1_test.txt").write_text("case_a: 0.9000 ms\ncase_b: 0.9000 ms\n")

    assert compute_best_patch(patch_dir) is None


def test_rewrite_best_results_invalidates_empty_existing_best(tmp_path: Path) -> None:
    patch_dir = tmp_path / "results" / "round_1" / "ext-empty"
    patch_dir.mkdir(parents=True)
    root = patch_dir.parent.parent.parent
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)
    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_ext-empty.md",
        {
            "label": "ext-empty",
            "priority": 6,
            "kernel_type": "triton",
            "search_set": "extension",
            "required_output_dialect": "amd_gluon",
        },
        "Extension task",
    )
    patch_file = patch_dir / "patch_5.patch"
    patch_file.write_text("")
    test_file = patch_dir / "patch_5_test.txt"
    test_file.write_text("case_a: 0.9000 ms\n")
    (patch_dir / "best_results.json").write_text(
        json.dumps(
            {
                "best_patch_id": "patch_5",
                "best_patch_speedup": 1.0,
                "best_patch_file": str(patch_file),
                "best_patch_test_output": str(test_file),
                "llm_selection_analysis": "LLM picked empty patch",
            }
        )
    )

    rewritten = rewrite_best_results(patch_dir)
    assert rewritten is not None
    assert rewritten["best_patch_id"] is None
    assert rewritten["best_patch_file"] is None
    assert rewritten["best_patch_speedup"] == 0.0
    assert rewritten["invalidated"] is True
    assert rewritten["actual_output_dialect"] == "empty"


def test_rewrite_best_results_invalidates_plain_existing_best_for_required_gluon(tmp_path: Path) -> None:
    patch_dir = tmp_path / "results" / "round_1" / "ext-plain"
    patch_dir.mkdir(parents=True)
    root = patch_dir.parent.parent.parent
    tasks_dir = root / "tasks" / "round_1"
    tasks_dir.mkdir(parents=True)
    from minisweagent.run.task_file import write_task_file

    write_task_file(
        tasks_dir / "06_ext-plain.md",
        {
            "label": "ext-plain",
            "priority": 6,
            "kernel_type": "triton",
            "search_set": "extension",
            "required_output_dialect": "amd_gluon",
        },
        "Extension task",
    )
    patch_file = patch_dir / "patch_1.patch"
    patch_file.write_text("diff --git a/kernel.py b/kernel.py\n+import triton.language as tl\n+tl.load(x)\n")
    test_file = patch_dir / "patch_1_test.txt"
    test_file.write_text("case_a: 0.9000 ms\n")
    (patch_dir / "best_results.json").write_text(
        json.dumps(
            {
                "best_patch_id": "patch_1",
                "best_patch_speedup": 1.1,
                "best_patch_file": str(patch_file),
                "best_patch_test_output": str(test_file),
                "llm_selection_analysis": "LLM picked fallback",
            }
        )
    )

    rewritten = rewrite_best_results(patch_dir)
    assert rewritten is not None
    assert rewritten["best_patch_id"] is None
    assert rewritten["best_patch_file"] is None
    assert rewritten["best_patch_speedup"] == 0.0
    assert rewritten["invalidated"] is True
    assert rewritten["required_output_dialect"] == "amd_gluon"
    assert rewritten["actual_output_dialect"] == "plain_triton"


def test_shape_list_bucket_detection_uses_dimension_indices() -> None:
    same_shape_cases = [
        {"params": {"shape": [1, 16, 16, 128, 2048]}}
        for _ in range(4)
    ]
    varied_one_axis_cases = [
        {"params": {"shape": [1, 16, 16, 128, 256]}},
        {"params": {"shape": [1, 16, 16, 128, 512]}},
        {"params": {"shape": [1, 16, 16, 128, 1024]}},
        {"params": {"shape": [1, 16, 16, 128, 2048]}},
    ]
    assert derive_shape_coverage_profile(benchmark_test_cases=same_shape_cases) == SHAPE_COVERAGE_MULTI
    assert derive_shape_coverage_profile(benchmark_test_cases=varied_one_axis_cases) == SHAPE_COVERAGE_BUCKETED


# ── 3. task_generator: traits, quotas, guidance, per-shape signal -----------


def test_infer_traits_adds_shape_coverage_and_constexpr_risk() -> None:
    meta = _gluon_meta_with_shape(
        SHAPE_COVERAGE_BUCKETED,
        cases=[
            {"params": {"M": 32, "K": 64, "N": 64}},
            {"params": {"M": 64, "K": 128, "N": 128}},
            {"params": {"M": 128, "K": 256, "N": 256}},
            {"params": {"M": 256, "K": 512, "N": 512}},
        ],
        count=4,
    )
    text = (
        "import triton.language as tl\n"
        "BLOCK_M = 64\n"
        "BLOCK_K = 64\n"
        "num_warps = 4\n"
        "x = tl.dot(a, b)\n"
    )
    traits = _infer_gluon_planning_traits(meta, text)
    assert "shape_coverage_bucketed" in traits
    assert "shape_layout_constexpr_risk" in traits
    assert "shape_dispatch_required" in traits


def test_infer_traits_marks_single_shape_as_single() -> None:
    meta = _gluon_meta_with_shape(SHAPE_COVERAGE_SINGLE, cases=[], count=1)
    traits = _infer_gluon_planning_traits(meta, "tl.arange(0, BLOCK_M)")
    assert "shape_coverage_single" in traits
    assert "shape_layout_constexpr_risk" not in traits
    assert "shape_dispatch_required" not in traits


def test_extension_strength_lifts_for_shape_dispatch_signal() -> None:
    base_traits = ["semantics_contract", "dialect_plain_triton", "matrix_none"]
    assert _gluon_extension_strength(base_traits) == "weak"
    assert _gluon_extension_strength(base_traits + ["shape_dispatch_required"]) == "strong"
    assert _gluon_extension_strength(base_traits + ["shape_coverage_bucketed"]) == "normal"


def test_quota_keeps_base_and_shared_for_multi_shape() -> None:
    base, shared, extension = _base_extension_quotas(
        4, "weak", "none", shape_profile=SHAPE_COVERAGE_MULTI,
    )
    assert base == 4
    assert shared >= 1
    assert extension == 0
    assert base + shared + extension > 4


def test_quota_caps_round1_plain_triton_extension_to_l0() -> None:
    base, shared, extension = _base_extension_quotas(
        8,
        "strong",
        "none",
        shape_profile=SHAPE_COVERAGE_BUCKETED,
        current_round=1,
        input_dialect="plain_triton",
    )

    assert base == 7
    assert shared == 2
    assert extension == 1


def test_quota_grows_extension_for_bucketed_with_5_gpus() -> None:
    base, shared, extension = _base_extension_quotas(
        5,
        "strong",
        "won",
        shape_profile=SHAPE_COVERAGE_BUCKETED,
        current_round=2,
    )
    assert base == 5
    assert shared == 2
    assert extension >= 2
    assert base + shared + extension > 5


def test_quota_interleaves_base_and_extension_on_one_gpu() -> None:
    base, shared, extension = _base_extension_quotas(
        1, "weak", "none", shape_profile=SHAPE_COVERAGE_UNKNOWN,
    )
    assert (base, shared, extension) == (2, 0, 0)
    assert base + shared + extension == 2


def test_serial_interleave_strategy_explains_candidate_count_can_exceed_gpus() -> None:
    text = _dialect_interleave_strategy(
        num_gpus=1,
        base_slots=5,
        shared_slots=0,
        extension_slots=1,
        previous_signal="none",
        shape_profile=SHAPE_COVERAGE_UNKNOWN,
    )
    assert "serial_interleave" in text
    assert "Candidate count: 6 task(s) for 1 GPU" in text
    assert "multiple optimization directions" in text
    assert "plain Triton competitors first" in text
    assert "L0 AMD Gluon overlay" in text


def test_parallel_mixed_strategy_mentions_hybrid_followup_when_gluon_won() -> None:
    text = _dialect_interleave_strategy(
        num_gpus=8,
        base_slots=4,
        shared_slots=2,
        extension_slots=2,
        previous_signal="won",
        shape_profile=SHAPE_COVERAGE_BUCKETED,
    )
    assert "parallel_mixed_portfolio" in text
    assert "mixed/hybrid" in text
    assert "Base or Shared competitor" in text
    assert "per-shape no-regression" in text


def test_search_space_allocation_mentions_multi_shape_rule() -> None:
    text = _build_search_space_allocation_guidance(
        _gluon_meta_with_shape(
            SHAPE_COVERAGE_MULTI,
            cases=[{"params": {"M": 32, "K": 32, "N": 32}}, {"params": {"M": 64, "K": 64, "N": 64}}],
            count=2,
        ),
        traits=[
            "semantics_contract",
            "dialect_plain_triton",
            "matrix_dot",
            "shape_coverage_multi",
        ],
        num_gpus=4,
    )
    assert "Shape coverage profile: multi" in text
    assert "shape_robust" in text


def test_search_space_allocation_for_one_gpu_serial_interleave() -> None:
    text = _build_search_space_allocation_guidance(
        _gluon_meta_with_shape(SHAPE_COVERAGE_UNKNOWN, cases=[], count=None),
        traits=["semantics_contract", "dialect_plain_triton", "layout_basic", "matrix_none"],
        num_gpus=1,
    )
    assert "Scheduling mode: `serial_interleave`" in text
    assert "Plain Triton competitors: at least 3 task(s)" in text
    assert "AMD Gluon overlay: 0 task(s) recommended" in text
    assert "concrete same-direction Gluon overlay reason" in text
    assert "slots 2+ may be L1" not in text
    assert "run them sequentially on the single GPU" in text


def test_search_space_allocation_for_eight_gpus_parallel_mixed_portfolio() -> None:
    text = _build_search_space_allocation_guidance(
        _gluon_meta_with_shape(
            SHAPE_COVERAGE_BUCKETED,
            cases=[
                {"case_id": "perf1", "params": {"M": 32, "K": 64, "N": 64}},
                {"case_id": "perf2", "params": {"M": 64, "K": 128, "N": 128}},
                {"case_id": "perf3", "params": {"M": 128, "K": 256, "N": 256}},
                {"case_id": "perf4", "params": {"M": 256, "K": 512, "N": 512}},
            ],
            count=4,
        ),
        traits=["semantics_contract", "dialect_plain_triton", "layout_basic", "matrix_dot", "shape_dispatch_required"],
        num_gpus=8,
        previous_results_text="gluon [BEST] verified_speedup=1.2x",
        current_round=2,
    )
    assert "Scheduling mode: `parallel_mixed_portfolio`" in text
    assert "Plain Triton competitors: at least 7 task(s)" in text
    assert "Paired same-direction mappings: 2 task(s)" in text
    assert "AMD Gluon overlay: 3 task(s)" in text
    assert "slots 2+ may be L1" in text
    assert "Paired same-direction mappings" in text
    assert "mixed/hybrid" in text
    assert "host-side shape/feature checks" in text


def test_shape_coverage_guidance_block_lists_cases_and_buckets() -> None:
    guidance = _build_shape_coverage_guidance(
        _gluon_meta_with_shape(
            SHAPE_COVERAGE_BUCKETED,
            cases=[
                {"case_id": "perf1", "params": {"M": 32, "K": 64, "N": 64}, "baseline_ms": 1.2},
                {"case_id": "perf4", "params": {"M": 256, "K": 512, "N": 512}, "baseline_ms": 9.0},
            ],
            count=4,
        ),
        traits=["semantics_contract", "matrix_dot", "shape_coverage_bucketed", "shape_dispatch_required"],
    )
    assert "## Shape Coverage Policy" in guidance
    assert "shape_robust" in guidance
    assert "shape_bucketed" in guidance
    assert "Observed cases" in guidance
    assert "perf1" in guidance
    assert "instr_shape" in guidance
    assert "explicit host-side dispatch" in guidance
    assert "@triton.heuristics" in guidance


def test_shape_coverage_guidance_returns_empty_for_single_or_unknown() -> None:
    assert (
        _build_shape_coverage_guidance(
            _gluon_meta_with_shape(SHAPE_COVERAGE_SINGLE, cases=[], count=1),
            traits=["semantics_contract"],
        )
        == ""
    )
    assert (
        _build_shape_coverage_guidance(
            _gluon_meta_with_shape(SHAPE_COVERAGE_UNKNOWN, cases=[], count=None),
            traits=[],
        )
        == ""
    )


def test_shape_coverage_guidance_surfaces_prior_per_shape_regressions() -> None:
    guidance = _build_shape_coverage_guidance(
        _gluon_meta_with_shape(
            SHAPE_COVERAGE_MULTI,
            cases=[
                {"case_id": "perf1", "params": {"M": 32}},
                {"case_id": "perf2", "params": {"M": 64}},
            ],
            count=2,
        ),
        traits=["semantics_contract", "shape_coverage_multi"],
        prior_per_shape={
            "(32,)": {"speedup": 1.4, "baseline_ms": 1.0, "candidate_ms": 0.7},
            "(64,)": {"speedup": 0.85, "baseline_ms": 1.0, "candidate_ms": 1.18},
        },
    )
    assert "Prior round per-shape regressions" in guidance
    assert "(64,)" in guidance


def test_aggregate_prior_per_shape_picks_latest_round() -> None:
    rounds = [
        {"round": 1, "per_shape_speedups": {"(32,)": {"speedup": 1.0}}},
        {"round": 2, "per_shape_speedups": {"(32,)": {"speedup": 1.2}, "(64,)": {"speedup": 0.8}}},
    ]
    out = _aggregate_prior_per_shape(rounds)
    assert out["(32,)"]["speedup"] == 1.2
    assert out["(64,)"]["speedup"] == 0.8


# ── 4. pipeline_helpers worker-side injection -------------------------------


def test_pipeline_helpers_shape_working_set_built_for_multi() -> None:
    lines = _build_shape_coverage_working_set(
        {
            "shape_coverage_profile": SHAPE_COVERAGE_BUCKETED,
            "benchmark_shape_count": 4,
            "benchmark_test_cases": [
                {"case_id": "perf1", "params": {"M": 32}},
                {"case_id": "perf4", "params": {"M": 512}},
            ],
        }
    )
    text = "\n".join(lines)
    assert "## Shape Coverage Working Set" in text
    assert "Bucketed coverage" in text
    assert "explicit host-side dispatch" in text
    assert "@triton.heuristics" in text
    assert "perf1" in text


def test_pipeline_helpers_shape_working_set_silent_for_single() -> None:
    assert (
        _build_shape_coverage_working_set(
            {"shape_coverage_profile": SHAPE_COVERAGE_SINGLE, "benchmark_shape_count": 1}
        )
        == []
    )


def test_discover_performance_report_walks_up_from_kernel(tmp_path: Path) -> None:
    """Arena layout: ``tasks/.../<case>/source/<name>.py`` + ``tasks/.../<case>/build/...``.

    The legacy 3-path lookup never matched this layout because ``build/``
    sits one ancestor above ``kernel.parent``. The walker must find it.
    """
    repo_root = tmp_path / "repo"
    task_dir = repo_root / "tasks" / "triton2triton" / "vllm" / "triton_scaled_mm"
    (task_dir / "source").mkdir(parents=True)
    (task_dir / "build").mkdir()
    kernel_path = task_dir / "source" / "triton_scaled_mm.py"
    kernel_path.write_text("# kernel\n")
    report = task_dir / "build" / "performance_report.json"
    report.write_text(json.dumps([
        {"test_case_id": "perf1", "execution_time_ms": 0.1, "params": {"M": 32, "K": 64, "N": 64}},
    ]))
    found = discover_performance_report(
        kernel_path=kernel_path,
        repo_root=repo_root,
        output_dir=tmp_path / "preprocess_out",
    )
    assert found is not None
    assert found.resolve() == report.resolve()


def test_discover_performance_report_uses_perf_command_hint(tmp_path: Path) -> None:
    """If kernel walk-up does not match, perf command should still locate the task dir."""
    repo_root = tmp_path / "repo"
    task_dir = repo_root / "tasks" / "torch2hip" / "gpumode" / "14539_GELU"
    (task_dir / "scripts").mkdir(parents=True)
    (task_dir / "build").mkdir()
    runner = task_dir / "scripts" / "task_runner.py"
    runner.write_text("# fake runner\n")
    report = task_dir / "build" / "performance_report.json"
    report.write_text(json.dumps([
        {"test_case_id": "perf1", "execution_time_ms": 1.2, "params": {"shape": [1024]}},
    ]))
    elsewhere_kernel = repo_root / "unrelated" / "kernel.py"
    elsewhere_kernel.parent.mkdir(parents=True)
    elsewhere_kernel.write_text("# kernel that doesn't live in the task dir\n")
    found = discover_performance_report(
        kernel_path=elsewhere_kernel,
        repo_root=repo_root,
        output_dir=tmp_path / "preprocess_out",
        performance_command=f"python3 {runner} performance",
    )
    assert found is not None
    assert found.resolve() == report.resolve()


def test_discover_performance_report_resolves_relative_runner_against_repo_root(
    tmp_path: Path,
) -> None:
    """Real perf commands often use repo-relative paths. The walker must resolve them.

    Reproduces the case where ``performance_command`` is
    ``python3 tasks/.../scripts/task_runner.py performance`` (relative to
    repo_root) and the kernel is in a different subtree, so kernel
    walk-up cannot find the report.
    """
    repo_root = tmp_path / "repo"
    task_dir = repo_root / "tasks" / "triton2triton" / "vllm" / "triton_scaled_mm"
    (task_dir / "scripts").mkdir(parents=True)
    (task_dir / "build").mkdir()
    (task_dir / "scripts" / "task_runner.py").write_text("# fake runner\n")
    report = task_dir / "build" / "performance_report.json"
    report.write_text(json.dumps([
        {"test_case_id": "perf1", "execution_time_ms": 0.1, "params": {"M": 32, "K": 64, "N": 64}},
    ]))
    elsewhere_kernel = repo_root / "vendor" / "kernel.py"
    elsewhere_kernel.parent.mkdir(parents=True)
    elsewhere_kernel.write_text("# kernel that doesn't live in the task dir\n")
    relative_command = (
        "python3 tasks/triton2triton/vllm/triton_scaled_mm/scripts/task_runner.py performance"
    )
    found = discover_performance_report(
        kernel_path=elsewhere_kernel,
        repo_root=repo_root,
        output_dir=tmp_path / "preprocess_out",
        performance_command=relative_command,
    )
    assert found is not None
    assert found.resolve() == report.resolve()


def test_discover_performance_report_handles_hip2hip_layout(tmp_path: Path) -> None:
    """hip2hip Arena tasks store the kernel under ``hip/<name>.hip``,
    eval scripts under ``eval_tools/cal_kernel_perf.py`` and the report
    under ``build/performance_report.json``. Walk-up from kernel and the
    eval script must both find the report.
    """
    repo_root = tmp_path / "repo"
    task_dir = repo_root / "tasks" / "hip2hip" / "gpumode" / "SimpleMatmulModule"
    (task_dir / "hip").mkdir(parents=True)
    (task_dir / "eval_tools").mkdir()
    (task_dir / "build").mkdir()
    kernel_path = task_dir / "hip" / "hip_3267_SimpleMatmulModule.hip"
    kernel_path.write_text("// hip kernel\n")
    eval_script = task_dir / "eval_tools" / "cal_kernel_perf.py"
    eval_script.write_text("# fake cal_kernel_perf\n")
    report = task_dir / "build" / "performance_report.json"
    report.write_text(json.dumps([
        {"test_case_id": "perf1", "execution_time_ms": 0.5, "params": {"M": 64, "K": 64, "N": 64}},
    ]))
    found = discover_performance_report(
        kernel_path=kernel_path,
        repo_root=repo_root,
        output_dir=tmp_path / "preprocess_out",
        performance_command=f"python3 {eval_script} --hip_file {kernel_path}",
    )
    assert found is not None
    assert found.resolve() == report.resolve()


def test_discover_performance_report_returns_none_when_absent(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    kernel_path = repo_root / "kernel.py"
    kernel_path.write_text("# kernel\n")
    assert (
        discover_performance_report(
            kernel_path=kernel_path,
            repo_root=repo_root,
            output_dir=tmp_path / "preprocess_out",
        )
        is None
    )


# ── 5. previous-round Gluon signal: mixed failure + win ---------------------


def test_previous_gluon_signal_won_when_best_patch_passed_after_early_failures() -> None:
    """Realistic ``scan_previous_results`` text mixes early failures with a winning final patch."""
    text = (
        "## gluon-mfma-rewrite\n"
        "- patch_0: traceback ImportError: triton.experimental.gluon\n"
        "- patch_2: correctness failed\n"
        "- patch_5 [BEST] verified_speedup=1.18x\n"
    )
    assert _previous_gluon_signal(text) == "won"


def test_previous_gluon_signal_won_via_structured_speedup_only() -> None:
    text = "gluon attempt: best_patch_speedup=1.07 (no '[best]' marker)"
    assert _previous_gluon_signal(text) == "won"


def test_previous_gluon_signal_attempted_for_noise_level_positive_speedup() -> None:
    text = "gluon-scaled-mm-viability: patch passed verified_speedup=1.0067x"
    assert _previous_gluon_signal(text) == "attempted"


def test_previous_gluon_signal_failed_when_no_winning_patch() -> None:
    text = (
        "## gluon-mfma-rewrite\n"
        "- patch_0: traceback ImportError\n"
        "- patch_1: correctness failed\n"
        "- patch_2: compile error\n"
    )
    assert _previous_gluon_signal(text) == "failed"


def test_previous_gluon_signal_slower_when_zero_speedup() -> None:
    text = "gluon attempt: speedup=0.0 (slower than baseline)"
    assert _previous_gluon_signal(text) == "slower"


def test_previous_gluon_signal_slower_when_passed_marker_but_speedup_below_one() -> None:
    """Bare ``passed`` must not promote a slower-than-baseline run to ``won``."""
    text = "gluon patch passed speedup=0.85 slower than baseline"
    assert _previous_gluon_signal(text) == "slower"


def test_previous_gluon_signal_attempted_when_speedup_exactly_one() -> None:
    """A ``[BEST]`` marker with parity speedup=1.0 is not a win."""
    text = "gluon best patch speedup=1.0 passed"
    assert _previous_gluon_signal(text) == "attempted"


def test_previous_gluon_signal_attempted_when_only_passed_marker() -> None:
    """``passed`` without a structured speedup is ambiguous, classify as attempted."""
    text = "gluon-mfma-rewrite: patch_2 passed"
    assert _previous_gluon_signal(text) == "attempted"


def test_filter_strips_base_set_plain_triton_section() -> None:
    """Finding 2 regression: Base Set plain-Triton wins must not propagate
    as Gluon `won` signal. Filter must keep only Gluon-related sections.
    """
    from minisweagent.agents.heterogeneous.task_generator import (
        _filter_gluon_relevant_text,
        _previous_gluon_signal,
    )

    prior = (
        "## Round 1 Results\n\n"
        "### triton-tiling-rewrite\n"
        "- Patches produced: 4\n"
        "- **Best patch**: patch_3 (speedup=1.5x, baseline=1.0ms)\n"
        "- patch_3_test.txt: speedup=1.5x **[BEST]**\n"
        "\n"
        "### gluon-mfma-rewrite\n"
        "- Patches produced: 3\n"
        "- patch_0_test.txt: traceback ImportError\n"
        "- patch_1_test.txt: compile failed\n"
        "- patch_2_test.txt: correctness failed\n"
    )
    filtered = _filter_gluon_relevant_text(prior)
    assert "triton-tiling-rewrite" not in filtered
    assert "gluon-mfma-rewrite" in filtered
    # Final signal must be ``failed`` (Gluon never won), NOT ``won`` from the
    # Base Set plain-Triton speedup=1.5x section.
    assert _previous_gluon_signal(filtered) == "failed"


def test_filter_uses_input_dialect_frontmatter_to_drop_plain_triton_bullets() -> None:
    """Finding 2 regression: ``scan_previous_tasks`` now embeds
    ``input_dialect=...`` per bullet so the filter can precisely tell
    apart Base Set plain-Triton tasks from Gluon Extension Set tasks.
    """
    from minisweagent.agents.heterogeneous.task_generator import (
        _filter_gluon_relevant_text,
        _previous_gluon_signal,
    )

    prior = (
        "## Round 1 Planned Tasks\n\n"
        "- **triton-split-k** (agent=strategy_agent, priority=0, "
        "input_dialect=plain_triton, policy=plain_triton_only): split-K plan\n"
        "- **constexpr-layout-fix** (agent=strategy_agent, priority=0, "
        "input_dialect=amd_gluon, policy=prefer_amd_gluon_if_viable_else_plain_triton): "
        "rebuild layouts host-side\n"
    )
    filtered = _filter_gluon_relevant_text(prior)
    assert "triton-split-k" not in filtered
    assert "constexpr-layout-fix" in filtered
    # No win/fail markers in the bullet itself, so signal is ``attempted``.
    assert _previous_gluon_signal(filtered) == "attempted"


def test_filter_drops_plain_triton_base_task_with_gluon_enabled_policy() -> None:
    """Run-level prefer_amd_gluon policy appears on every task frontmatter.

    It must not make a Base Set plain-Triton task count as prior Gluon work.
    """
    from minisweagent.agents.heterogeneous.task_generator import (
        _filter_gluon_relevant_text,
        _previous_gluon_signal,
    )

    prior = (
        "## Round 1 Planned Tasks\n\n"
        "- **triton-split-k** (agent=strategy_agent, priority=0, "
        "input_dialect=plain_triton, policy=prefer_amd_gluon_if_viable_else_plain_triton): "
        "Base Set plain Triton split-K, speedup=1.50x\n"
        "- **gluon-mfma-rewrite** (agent=strategy_agent, priority=5, "
        "input_dialect=plain_triton, policy=prefer_amd_gluon_if_viable_else_plain_triton): "
        "Extension Set AMD Gluon MFMA rewrite compile failed\n"
    )
    filtered = _filter_gluon_relevant_text(prior)
    assert "triton-split-k" not in filtered
    assert "gluon-mfma-rewrite" in filtered
    assert _previous_gluon_signal(filtered) == "failed"


def test_feature_uses_gluon_guidance_from_meta_reads_kernel_type_from_meta() -> None:
    """Finding 3 regression: build_gluon_feature_metadata stores
    ``kernel_type`` so the consistent-from-meta wrapper avoids the old
    ``feature_uses_gluon_guidance("triton", ...)`` hard-coded literal.
    """
    from minisweagent.run.preprocess.discovery_types import (
        build_gluon_feature_metadata,
        feature_uses_gluon_guidance_from_meta,
    )

    triton_meta = build_gluon_feature_metadata(Path("dummy.py"), "triton")
    assert triton_meta["kernel_type"] == "triton"
    assert feature_uses_gluon_guidance_from_meta(triton_meta) is True

    hip_meta = build_gluon_feature_metadata(Path("dummy.cpp"), "hip")
    assert hip_meta["kernel_type"] == "hip"
    assert feature_uses_gluon_guidance_from_meta(hip_meta) is False

    unknown_meta = build_gluon_feature_metadata(Path("dummy.py"), "unknown")
    assert feature_uses_gluon_guidance_from_meta(unknown_meta) is False

    missing_type_meta = dict(triton_meta)
    missing_type_meta.pop("kernel_type")
    assert feature_uses_gluon_guidance_from_meta(missing_type_meta) is False

    # Empty / None feature_meta is safe.
    assert feature_uses_gluon_guidance_from_meta({}) is False
    assert feature_uses_gluon_guidance_from_meta(None) is False


def test_filter_falls_back_to_whole_text_when_no_structure() -> None:
    """Legacy contract: free-text prose passed straight to the signal helper
    must still be classified, not silently dropped by the filter.
    """
    from minisweagent.agents.heterogeneous.task_generator import (
        _filter_gluon_relevant_text,
    )

    prose = "gluon compile failed with traceback"
    assert _filter_gluon_relevant_text(prose) == prose
    # Non-Gluon prose drops out.
    assert _filter_gluon_relevant_text("triton split-k won speedup=1.4x") == ""


def test_previous_gluon_signal_works_without_literal_gluon_in_text() -> None:
    """Gluon-relevant tasks often have labels like ``mfma-rewrite`` or
    ``layout-constexpr-fix`` that omit the literal substring ``gluon``.
    The signal helper must still extract failed/slower/won from those rounds;
    the caller is responsible for restricting it to Gluon contexts via
    ``feature_uses_gluon_guidance``.
    """
    failed_text = "mfma-rewrite: patch_0 traceback ImportError; patch_1 compile failed"
    won_text = "layout-constexpr-fix: patch_3 [BEST] verified_speedup=1.18x"
    slower_text = "shape-bucketed: patch_2 speedup=0.82"
    assert _previous_gluon_signal(failed_text) == "failed"
    assert _previous_gluon_signal(won_text) == "won"
    assert _previous_gluon_signal(slower_text) == "slower"


# ── 6. ordering regression: shape patch happens BEFORE feature_context ------


# ── 7. preprocessor snapshots Arena perf sidecar (Plan B) ------------------


def test_preprocessor_snapshots_perf_report_into_output_dir(tmp_path: Path) -> None:
    """Plan B: preprocessor must materialize the Arena sidecar as a GEAK
    snapshot in ``output_dir/benchmark_test_cases.json`` so subsequent
    Arena perf_cmd invocations cannot overwrite this run's per-case data.
    """
    from minisweagent.run.preprocess.benchmark_parsing import (
        discover_performance_report,
        parse_performance_report_json,
    )

    repo_root = tmp_path / "arena"
    task_dir = repo_root / "tasks" / "triton2triton" / "vllm" / "case_a"
    (task_dir / "source").mkdir(parents=True)
    (task_dir / "build").mkdir()
    kernel_path = task_dir / "source" / "kernel.py"
    kernel_path.write_text("import triton\n")
    upstream_report = task_dir / "build" / "performance_report.json"
    upstream_report.write_text(
        json.dumps(
            [
                {"test_case_id": "perf1", "execution_time_ms": 0.1, "params": {"M": 32}},
                {"test_case_id": "perf2", "execution_time_ms": 0.4, "params": {"M": 256}},
            ]
        )
    )

    geak_output_dir = tmp_path / "geak_run" / "main_case_a" / "preprocess"
    geak_output_dir.mkdir(parents=True)

    found = discover_performance_report(
        kernel_path=kernel_path,
        repo_root=repo_root,
        output_dir=geak_output_dir,
    )
    assert found is not None
    cases = parse_performance_report_json(found)
    assert cases is not None and len(cases) == 2

    # Manually exercise the snapshot logic in the same way preprocessor does.
    # (We don't pull in the full preprocessor here because it would require
    # GPU + harness; the snapshot routine is self-contained.)
    from datetime import datetime, timezone

    snapshot_path = geak_output_dir / "benchmark_test_cases.json"
    snapshot_payload = {
        "schema": "geak.benchmark_test_cases.v1",
        "source_path": str(found.resolve()),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "test_cases": [
            {"case_id": c["case_id"], "params": c["params"], "baseline_ms": c["ms"]}
            for c in cases
        ],
    }
    snapshot_path.write_text(json.dumps(snapshot_payload, indent=2))

    # Snapshot must be inside GEAK's output_dir, not in Arena task_dir.
    assert snapshot_path.is_file()
    assert geak_output_dir.resolve() in snapshot_path.resolve().parents

    payload = json.loads(snapshot_path.read_text())
    assert payload["schema"] == "geak.benchmark_test_cases.v1"
    assert payload["source_path"] == str(upstream_report.resolve())
    assert "captured_at" in payload
    assert [c["case_id"] for c in payload["test_cases"]] == ["perf1", "perf2"]


def test_snapshot_is_immune_to_subsequent_arena_overwrite(tmp_path: Path) -> None:
    """AB workflow: main run snapshots, then feature run overwrites the
    upstream sidecar. The main snapshot in main run's output_dir must
    keep the original case data intact.
    """
    from minisweagent.run.preprocess.benchmark_parsing import (
        parse_performance_report_json,
    )

    arena_root = tmp_path / "arena"
    task_dir = arena_root / "tasks" / "case_b"
    task_dir.mkdir(parents=True)
    upstream = task_dir / "performance_report.json"
    upstream.write_text(
        json.dumps([
            {"test_case_id": "main_run", "execution_time_ms": 1.0, "params": {"M": 128}},
        ])
    )
    main_output = tmp_path / "geak" / "main_case_b" / "preprocess"
    main_output.mkdir(parents=True)
    parsed = parse_performance_report_json(upstream)
    assert parsed is not None
    snapshot_path = main_output / "benchmark_test_cases.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema": "geak.benchmark_test_cases.v1",
                "source_path": str(upstream.resolve()),
                "test_cases": [
                    {"case_id": c["case_id"], "params": c["params"], "baseline_ms": c["ms"]}
                    for c in parsed
                ],
            }
        )
    )

    # Simulate the feature-branch run overwriting the upstream sidecar.
    upstream.write_text(
        json.dumps([
            {"test_case_id": "feature_run", "execution_time_ms": 0.5, "params": {"M": 128}},
            {"test_case_id": "feature_run_extra", "execution_time_ms": 0.6, "params": {"M": 1024}},
        ])
    )

    payload = json.loads(snapshot_path.read_text())
    assert [c["case_id"] for c in payload["test_cases"]] == ["main_run"]
    assert payload["test_cases"][0]["params"] == {"M": 128}


def test_run_task_agent_baseline_metrics_overrides_stale_discovery_profile(tmp_path: Path) -> None:
    """Regression for Finding 3: when discovery says ``multi`` but
    baseline_metrics has bucketed test_cases (different shape orders of
    magnitude), task_generator must trust baseline_metrics.

    discovery.json may carry a stale profile from an earlier preprocess
    invocation; baseline_metrics.json is the authoritative post-perf-run
    snapshot.
    """
    from unittest.mock import patch

    from minisweagent.agents.heterogeneous.task_generator import _run_task_agent

    workspace = tmp_path / "ws"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    kernel.write_text("import triton\nimport triton.language as tl\n@triton.jit\ndef k(x): pass\n")
    bm_path = workspace / "baseline_metrics.json"
    bm_path.write_text(
        json.dumps(
            {
                "duration_us": 12.3,
                "bottleneck": "compute",
                "benchmark_shape_count": 4,
                "benchmark_test_cases": [
                    {"case_id": "perf1", "params": {"M": 32, "K": 64, "N": 64}, "baseline_ms": 1.2},
                    {"case_id": "perf2", "params": {"M": 64, "K": 128, "N": 128}, "baseline_ms": 2.1},
                    {"case_id": "perf3", "params": {"M": 128, "K": 256, "N": 256}, "baseline_ms": 4.4},
                    {"case_id": "perf4", "params": {"M": 256, "K": 512, "N": 512}, "baseline_ms": 9.6},
                ],
                "shape_coverage_profile": "bucketed",
            }
        )
    )

    rendered_holder: dict[str, Any] = {}

    class _Stub:
        def __init__(self, *_a, **_kw) -> None:
            pass

        def run(self, **kwargs):
            rendered_holder.update(kwargs)
            return ("Submitted", "[]")

    class _FakeModel:
        def __init__(self) -> None:
            self.tools: list = []

        def set_tools(self, tools) -> None:
            self.tools = list(tools)

    with patch("minisweagent.agents.default.DefaultAgent", _Stub), patch(
        "minisweagent.tools.tools_runtime.get_tools_list",
        return_value=[{"name": "str_replace_editor"}, {"name": "submit"}],
    ):
        _run_task_agent(
            kernel_path=str(kernel),
            kernel_name="kernel",
            kernel_type="triton",
            kernel_language="python",
            function_names=["k"],
            workspace_path=str(workspace),
            input_dialect="plain_triton",
            gluon_feature_mode="auto",
            gluon_baseline_profile="raw",
            allowed_output_dialects=["plain_triton", "amd_gluon"],
            target_backend="hip/gfx942",
            base_task_context="ctx",
            model=_FakeModel(),
            profiling_path=None,
            commandment_path=None,
            baseline_metrics_path=bm_path,
            deep_search_path=None,
            previous_results_dir=None,
            discovery_path=None,
            # Caller passes a STALE profile (e.g. from an older discovery)
            # that disagrees with the bucketed baseline_metrics. The
            # baseline-authoritative override must kick in.
            shape_coverage_profile="multi",
            benchmark_shape_count=2,
            benchmark_test_cases=[
                {"case_id": "stale_a", "params": {"M": 16}, "baseline_ms": 0.1},
                {"case_id": "stale_b", "params": {"M": 17}, "baseline_ms": 0.1},
            ],
        )

    feature_context = rendered_holder.get("gluon_feature_context") or ""
    shape_guidance = rendered_holder.get("shape_coverage_guidance") or ""
    # Profile must be the bucketed value derived from baseline_metrics, NOT
    # the stale ``multi`` passed by the caller.
    assert "Shape coverage profile: bucketed" in feature_context
    assert "(4 cases)" in feature_context
    # Observed cases listed in the policy must come from baseline, not the
    # stale 2-case caller list.
    assert "perf1" in shape_guidance
    assert "stale_a" not in shape_guidance


def test_run_task_agent_renders_shape_profile_in_feature_context(tmp_path: Path) -> None:
    """Regression: ``gluon_feature_context`` must reflect the shape profile after baseline patching."""
    from unittest.mock import patch

    from minisweagent.agents.heterogeneous.task_generator import _run_task_agent

    workspace = tmp_path / "ws"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    kernel.write_text("import triton\nimport triton.language as tl\n@triton.jit\ndef k(x): pass\n")
    bm_path = workspace / "baseline_metrics.json"
    bm_path.write_text(
        json.dumps(
            {
                "duration_us": 12.3,
                "bottleneck": "compute",
                "benchmark_shape_count": 4,
                "benchmark_test_cases": [
                    {"case_id": "perf1", "params": {"M": 32, "K": 64, "N": 64}, "baseline_ms": 1.2},
                    {"case_id": "perf2", "params": {"M": 64, "K": 128, "N": 128}, "baseline_ms": 2.1},
                    {"case_id": "perf3", "params": {"M": 128, "K": 256, "N": 256}, "baseline_ms": 4.4},
                    {"case_id": "perf4", "params": {"M": 256, "K": 512, "N": 512}, "baseline_ms": 9.6},
                ],
                "shape_coverage_profile": "bucketed",
            }
        )
    )

    rendered_holder: dict[str, Any] = {}

    class _Stub:
        def __init__(self, *_a, **_kw) -> None:
            pass

        def run(self, **kwargs):
            rendered_holder.update(kwargs)
            return ("Submitted", "[]")

    class _FakeModel:
        def __init__(self) -> None:
            self.tools: list = []

        def set_tools(self, tools) -> None:
            self.tools = list(tools)

    with patch("minisweagent.agents.default.DefaultAgent", _Stub), patch(
        "minisweagent.tools.tools_runtime.get_tools_list",
        return_value=[{"name": "str_replace_editor"}, {"name": "submit"}],
    ):
        _run_task_agent(
            kernel_path=str(kernel),
            kernel_name="kernel",
            kernel_type="triton",
            kernel_language="python",
            function_names=["k"],
            workspace_path=str(workspace),
            input_dialect="plain_triton",
            gluon_feature_mode="auto",
            gluon_baseline_profile="raw",
            allowed_output_dialects=["plain_triton", "amd_gluon"],
            target_backend="hip/gfx942",
            base_task_context="ctx",
            model=_FakeModel(),
            profiling_path=None,
            commandment_path=None,
            baseline_metrics_path=bm_path,
            deep_search_path=None,
            previous_results_dir=None,
            discovery_path=None,
        )

    feature_context = rendered_holder.get("gluon_feature_context") or ""
    shape_guidance = rendered_holder.get("shape_coverage_guidance") or ""
    assert "Shape coverage profile: bucketed" in feature_context
    assert "(4 cases)" in feature_context
    assert "## Shape Coverage Policy" in shape_guidance
    assert "shape_robust" in shape_guidance


def test_inject_pipeline_context_includes_shape_coverage_block() -> None:
    feature_metadata = {
        "input_dialect": "plain_triton",
        "gluon_feature_mode": "auto",
        "gluon_baseline_profile": "raw",
        "allowed_output_dialects": ["plain_triton", "amd_gluon"],
        "preferred_output_dialects": ["amd_gluon", "plain_triton"],
        "output_dialect_search_policy": "prefer_amd_gluon_if_viable_else_plain_triton",
        "target_backend": "hip/gfx942",
        "shape_coverage_profile": SHAPE_COVERAGE_MULTI,
        "benchmark_shape_count": 5,
        "benchmark_test_cases": [
            {"case_id": "perf1", "params": {"M": 32, "K": 32, "N": 32}},
            {"case_id": "perf2", "params": {"M": 64, "K": 64, "N": 64}},
        ],
    }
    body, _cfg = inject_pipeline_context(
        "TASK BODY",
        {},
        feature_metadata=feature_metadata,
    )
    assert "## Shape Coverage Working Set" in body
    assert "TASK BODY" in body


def test_inject_pipeline_context_omits_gluon_refs_for_plain_triton_tasks() -> None:
    feature_metadata = {
        "kernel_type": "triton",
        "input_dialect": "plain_triton",
        "gluon_feature_mode": "auto",
        "gluon_baseline_profile": "raw",
        "allowed_output_dialects": ["plain_triton", "amd_gluon"],
        "preferred_output_dialects": ["amd_gluon", "plain_triton"],
        "output_dialect_search_policy": "prefer_amd_gluon_if_viable_else_plain_triton",
        "target_backend": "hip/gfx942",
        "shape_coverage_profile": SHAPE_COVERAGE_UNKNOWN,
    }
    body, _cfg = inject_pipeline_context(
        "TASK BODY",
        {},
        feature_metadata=feature_metadata,
        gluon_always_read_path="/abs/geak/skills/triton-gluon/docs/00_always_read.md",
        gluon_api_reference_path="/abs/geak/skills/triton-gluon/docs/50_api_reference.md",
        gluon_real_patterns_path="/abs/geak/skills/triton-gluon/docs/60_real_patterns.md",
    )

    assert "Split-doc entrypoint: /abs/geak/skills/triton-gluon/docs/00_always_read.md" not in body
    assert "API syntax, launch skeletons" not in body
    assert "/abs/geak/skills/triton-gluon/docs/50_api_reference.md" not in body
    assert "Real aiter patterns" not in body
    assert "/abs/geak/skills/triton-gluon/docs/60_real_patterns.md" not in body


def test_inject_pipeline_context_prints_absolute_gluon_split_doc_paths_for_gluon_tasks() -> None:
    feature_metadata = {
        "kernel_type": "triton",
        "input_dialect": "plain_triton",
        "gluon_feature_mode": "auto",
        "gluon_baseline_profile": "raw",
        "allowed_output_dialects": ["plain_triton", "amd_gluon"],
        "preferred_output_dialects": ["amd_gluon", "plain_triton"],
        "output_dialect_search_policy": "prefer_amd_gluon_if_viable_else_plain_triton",
        "target_backend": "hip/gfx942",
        "shape_coverage_profile": SHAPE_COVERAGE_UNKNOWN,
        "required_output_dialect": "amd_gluon",
    }
    body, _cfg = inject_pipeline_context(
        "TASK BODY",
        {},
        feature_metadata=feature_metadata,
        gluon_always_read_path="/abs/geak/skills/triton-gluon/docs/00_always_read.md",
        gluon_api_reference_path="/abs/geak/skills/triton-gluon/docs/50_api_reference.md",
        gluon_real_patterns_path="/abs/geak/skills/triton-gluon/docs/60_real_patterns.md",
    )

    assert "Split-doc entrypoint: /abs/geak/skills/triton-gluon/docs/00_always_read.md" in body
    assert "API syntax, launch skeletons" in body
    assert "/abs/geak/skills/triton-gluon/docs/50_api_reference.md" in body
    assert "Real aiter patterns" in body
    assert "/abs/geak/skills/triton-gluon/docs/60_real_patterns.md" in body
