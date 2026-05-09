import json
from pathlib import Path

from minisweagent.agents.heterogeneous.result_scanning import scan_single_round_results


def test_scan_results_surfaces_gluon_contract_and_anchor_viability(tmp_path: Path) -> None:
    task_dir = tmp_path / "round_1" / "ext-l0-gluon-anchor"
    task_dir.mkdir(parents=True)
    (task_dir / "patch_0.patch").write_text("diff --git a/kernel.py b/kernel.py\n+@gluon.jit\n")
    (task_dir / "patch_0_test.txt").write_text("case_small: 2.0 ms\ncase_medium: 2.0 ms\n")
    (task_dir / "best_results.json").write_text(
        json.dumps(
            {
                "best_patch_id": "patch_0",
                "best_patch_speedup": 0.04,
                "baseline_latency_ms": 0.1,
                "candidate_latency_ms": 2.5,
                "required_output_dialect": "amd_gluon",
                "actual_output_dialect": "amd_gluon",
                "dialect_contract_satisfied": True,
                "gluon_execution_contract_satisfied": True,
                "required_patch_target_symbols": ["target_stage"],
                "has_significant_shape_regression": True,
                "per_shape_speedups": {
                    "case_small": {
                        "baseline_ms": 0.05,
                        "candidate_ms": 2.0,
                        "speedup": 0.025,
                    }
                },
            }
        )
    )

    sections = scan_single_round_results(tmp_path / "round_1")
    text = "\n".join(sections)

    assert "actual_output_dialect=amd_gluon" in text
    assert "gluon_execution_contract_satisfied=True" in text
    assert "required_patch_target_symbols=['target_stage']" in text
    assert "case_small=0.0250x" in text
    assert "Gluon L1 anchor viability: not_viable_for_l1" in text
    assert "Gluon result attribution: Gluon-slower" in text


def test_scan_results_keeps_missing_shape_regression_unknown(tmp_path: Path) -> None:
    task_dir = tmp_path / "round_1" / "ext-l0-gluon-anchor"
    task_dir.mkdir(parents=True)
    (task_dir / "patch_0.patch").write_text("diff --git a/kernel.py b/kernel.py\n+@gluon.jit\n")
    (task_dir / "patch_0_test.txt").write_text("case_small: 1.0 ms\n")
    (task_dir / "best_results.json").write_text(
        json.dumps(
            {
                "best_patch_id": "patch_0",
                "best_patch_speedup": 1.05,
                "baseline_latency_ms": 1.0,
                "candidate_latency_ms": 0.95,
                "required_output_dialect": "amd_gluon",
                "actual_output_dialect": "amd_gluon",
                "dialect_contract_satisfied": True,
                "gluon_execution_contract_satisfied": True,
            }
        )
    )

    text = "\n".join(scan_single_round_results(tmp_path / "round_1"))

    assert "Shape regression: has_significant_shape_regression=unknown" in text
    assert "Gluon L1 anchor viability: unknown" in text
    assert "Gluon result attribution: unknown" in text


def test_scan_results_does_not_mark_plain_any_as_gluon_informed(tmp_path: Path) -> None:
    task_dir = tmp_path / "round_1" / "shared-plain-cleanup"
    task_dir.mkdir(parents=True)
    (task_dir / "patch_0.patch").write_text("diff --git a/kernel.py b/kernel.py\n+tl.load(x)\n")
    (task_dir / "patch_0_test.txt").write_text("case_small: 1.0 ms\n")
    (task_dir / "best_results.json").write_text(
        json.dumps(
            {
                "best_patch_id": "patch_0",
                "best_patch_speedup": 1.05,
                "baseline_latency_ms": 1.0,
                "candidate_latency_ms": 0.95,
                "required_output_dialect": "any",
                "actual_output_dialect": "plain_triton",
                "dialect_contract_satisfied": True,
                "gluon_execution_contract_satisfied": True,
                "has_significant_shape_regression": False,
            }
        )
    )

    text = "\n".join(scan_single_round_results(tmp_path / "round_1"))

    assert "gluon_execution_contract_satisfied=False" in text
    assert "Gluon result attribution: unknown" in text
    assert "Gluon-informed" not in text


def test_scan_results_prefers_explicit_anchor_viability_and_overhead_source(tmp_path: Path) -> None:
    task_dir = tmp_path / "round_1" / "ext-l0-execution-anchor"
    task_dir.mkdir(parents=True)
    (task_dir / "patch_0.patch").write_text("diff --git a/kernel.py b/kernel.py\n+@gluon.jit\n")
    (task_dir / "patch_0_test.txt").write_text("case_small: 1.2 ms\n")
    (task_dir / "best_results.json").write_text(
        json.dumps(
            {
                "best_patch_id": "patch_0",
                "best_patch_speedup": 0.91,
                "baseline_latency_ms": 1.0,
                "candidate_latency_ms": 1.1,
                "required_output_dialect": "amd_gluon",
                "actual_output_dialect": "amd_gluon",
                "dialect_contract_satisfied": True,
                "gluon_execution_contract_satisfied": True,
                "extension_intent": "execution_anchor",
                "expected_outcome": "correctness_anchor_not_speedup",
                "not_viable_for_l1_if_slower_than_base": True,
                "overhead_source_to_record": "launch_layout_overhead",
                "has_significant_shape_regression": False,
            }
        )
    )

    text = "\n".join(scan_single_round_results(tmp_path / "round_1"))

    assert "Gluon L1 anchor viability: not_viable_for_l1" in text
    assert "Gluon extension intent: execution_anchor" in text
    assert "overhead_source=launch_layout_overhead" in text


def test_scan_results_surfaces_scope_violation(tmp_path: Path) -> None:
    task_dir = tmp_path / "round_1" / "gluon-l0-scale-load-layout"
    task_dir.mkdir(parents=True)
    (task_dir / "patch_0.patch").write_text("diff --git a/kernel.py b/kernel.py\n+@gluon.jit\n")
    (task_dir / "patch_0_test.txt").write_text("case_small: 1.2 ms\n")
    (task_dir / "best_results.json").write_text(
        json.dumps(
            {
                "best_patch_id": None,
                "best_patch_speedup": 0.0,
                "baseline_latency_ms": 1.0,
                "candidate_latency_ms": 1.2,
                "required_output_dialect": "amd_gluon",
                "actual_output_dialect": "amd_gluon",
                "dialect_contract_satisfied": True,
                "gluon_execution_contract_satisfied": False,
                "scope_compliant": False,
                "forbidden_scope_violation": "dot_loop",
                "minimum_executable_unit": "separate_gluon_kernel",
                "allowed_execution_path": "separate_gluon_kernel",
                "scope_infeasible_policy": "separate_kernel_if_allowed",
                "scope_infeasible_reported": False,
            }
        )
    )

    text = "\n".join(scan_single_round_results(tmp_path / "round_1"))

    assert "Scope compliance: scope_compliant=False, forbidden_scope_violation=dot_loop" in text
    assert "minimum_executable_unit=separate_gluon_kernel" in text
    assert "allowed_execution_path=separate_gluon_kernel" in text
    assert "scope_infeasible_policy=separate_kernel_if_allowed" in text
