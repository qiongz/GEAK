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
