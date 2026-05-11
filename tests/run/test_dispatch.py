# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for task-file dispatch configuration."""

from __future__ import annotations

import pytest

from minisweagent.agents.heterogeneous.tools import _group_task_files_by_dispatch_stage, _required_gluon_tasks_completed
from minisweagent.run.gluon_doc_profiles import GLUON_DOC_PROFILE_REQUIRED_KEYS
from minisweagent.run.dispatch import run_task_batch, task_file_to_agent_task
from minisweagent.run.task_file import write_task_file
from minisweagent.tools.save_and_test import SaveAndTestContext, SaveAndTestTool


def _has_doc(paths: list[str], suffix: str) -> bool:
    return any(path.endswith(suffix) for path in paths)


def test_task_file_to_agent_task_enables_skills_for_triton(tmp_path) -> None:
    task_path = tmp_path / "triton_task.md"
    write_task_file(
        task_path,
        {
            "label": "triton-task",
            "priority": 5,
            "kernel_type": "triton",
        },
        "Optimize the Triton kernel.",
    )

    task = task_file_to_agent_task(task_path)

    assert task.config["use_skills"] is True
    assert task.config["allowed_skill_tiers"] == ["general"]


def test_task_file_to_agent_task_respects_explicit_skill_disable(tmp_path) -> None:
    task_path = tmp_path / "triton_no_skills.md"
    write_task_file(
        task_path,
        {
            "label": "triton-no-skills",
            "priority": 5,
            "kernel_type": "triton",
            "input_dialect": "amd_gluon",
            "gluon_feature_mode": "auto",
            "gluon_baseline_profile": "mi3xx",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "use_skills": False,
        },
        "Optimize the Triton kernel.",
    )

    task = task_file_to_agent_task(task_path)

    assert task.config["use_skills"] is False
    assert task.config["allowed_skill_tiers"] == ["general"]


def test_task_file_to_agent_task_keeps_mi3xx_gluon_context_inline(tmp_path) -> None:
    task_path = tmp_path / "triton_mi3xx.md"
    write_task_file(
        task_path,
        {
            "label": "triton-mi3xx",
            "priority": 5,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "amd_gluon",
            "gluon_feature_mode": "auto",
            "gluon_baseline_profile": "mi3xx",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "required_output_dialect": "amd_gluon",
            "implementation_layer": "amd_gluon overlay",
            "extension_layer": "L0",
        },
        "Optimize the Triton kernel.",
    )

    task = task_file_to_agent_task(task_path)

    assert task.config["allowed_skill_tiers"] == ["general"]
    assert "## Gluon Feature Context" in task.task


def test_plain_base_task_with_gluon_run_meta_does_not_enable_doc_gate(tmp_path) -> None:
    task_path = tmp_path / "plain_base.md"
    write_task_file(
        task_path,
        {
            "label": "base-streamline",
            "priority": 0,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "search_set": "base",
            "required_output_dialect": "plain_triton",
            "implementation_layer": "plain_triton",
            "gluon_doc_profile": "base_or_shared_gluon",
        },
        "Plain Triton competitor task\nReject if: output falls back to AMD Gluon.",
    )

    task = task_file_to_agent_task(task_path)

    assert "gluon_doc_gate_enabled" not in task.config
    assert "## Canonical Gluon References" not in task.task
    assert "## Gluon Working Set" not in task.task
    assert "## REQUIRED BEFORE EDITING OR SAVE_AND_TEST" not in task.task


def test_required_block_matches_gluon_doc_gate_even_when_run_guidance_off(tmp_path) -> None:
    task_path = tmp_path / "required_gluon_feature_off.md"
    write_task_file(
        task_path,
        {
            "label": "ext-required-feature-off",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "off",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "required_output_dialect": "amd_gluon",
            "implementation_layer": "amd_gluon overlay",
            "extension_layer": "L0",
            "gluon_doc_profile": "extension_l0_minimal",
        },
        "Extension layer: L0\nImplementation layer: amd_gluon overlay\n",
    )

    task = task_file_to_agent_task(task_path)

    assert task.config["gluon_doc_gate_enabled"] is True
    assert "## Gluon Working Set" in task.task
    assert "## REQUIRED BEFORE EDITING OR SAVE_AND_TEST" in task.task


def test_label_only_gluon_variant_gate_also_injects_required_block(tmp_path) -> None:
    task_path = tmp_path / "label_only_gluon_variant.md"
    write_task_file(
        task_path,
        {
            "label": "manual-gluon_variant-label-only",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "required_output_dialect": "any",
        },
        "Manual task that relies on its label for Gluon variant routing.",
    )

    task = task_file_to_agent_task(task_path)

    assert task.config["gluon_doc_gate_enabled"] is True
    assert "## Gluon Working Set" in task.task
    assert "## REQUIRED BEFORE EDITING OR SAVE_AND_TEST" in task.task


def test_task_file_to_agent_task_keeps_general_skill_tiers_for_raw_profile(tmp_path) -> None:
    task_path = tmp_path / "triton_raw.md"
    write_task_file(
        task_path,
        {
            "label": "triton-raw",
            "priority": 5,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "amd_gluon",
            "gluon_feature_mode": "auto",
            "gluon_baseline_profile": "raw",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
        },
        "Optimize the Triton kernel.",
    )

    task = task_file_to_agent_task(task_path)

    assert task.config["allowed_skill_tiers"] == ["general"]


def test_required_gluon_task_forces_skill_context_even_if_disabled(tmp_path) -> None:
    task_path = tmp_path / "required_gluon.md"
    write_task_file(
        task_path,
        {
            "label": "ext-l0-gluon",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "search_set": "extension",
            "required_output_dialect": "amd_gluon",
            "use_skills": False,
        },
        "Optimize using AMD Gluon.",
    )

    task = task_file_to_agent_task(task_path)

    assert task.config["use_skills"] is True
    assert "## Forced Triton-Gluon Skill Context" in task.task
    assert "from triton.experimental import gluon" in task.task
    assert "from triton import gluon" in task.task
    assert "not the supported import path" in task.task
    assert "Gluon knowledge lookup plan" in task.task
    assert "Gluon implementation plan" in task.task
    assert "Task compat_search_set: extension" in task.task
    assert "Task required_output_dialect: amd_gluon" in task.task
    assert "viewed=no" in task.task
    assert "If the implementation plan cannot satisfy the scoped path fields" in task.task
    assert "Same ABI comparison" in task.task
    assert "broad `try/except Exception` fallback" in task.task
    assert "### triton-gluon/SKILL.md" not in task.task
    assert "### triton-gluon/docs/00_always_read.md" not in task.task
    assert task.config["gluon_doc_gate_enabled"] is True
    assert any(path.endswith("skills/triton-gluon/docs/00_always_read.md") for path in task.config["gluon_doc_gate_required_paths"])
    assert any(path.endswith("skills/triton-gluon/docs/50_api_reference.md") for path in task.config["gluon_doc_gate_required_paths"])
    assert "## REQUIRED BEFORE EDITING OR SAVE_AND_TEST" in task.task
    assert "GLUON_DOC_GATE_FAILED" in task.task


def test_worker_context_includes_gluon_contract_metadata(tmp_path) -> None:
    task_path = tmp_path / "contract_metadata.md"
    write_task_file(
        task_path,
        {
            "label": "ext-l1-gluon",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "search_set": "extension",
            "required_output_dialect": "amd_gluon",
            "gluon_doc_profile": "memory_lowering",
            "required_gluon_docs": ["gluon_skill_path", "gluon_always_read_path"],
            "source_base_family": "base_hot_path_streamline",
            "plain_competitor": "triton-eliminate-redundant-ops-streamline",
            "implementation_layer": "amd_gluon overlay",
            "extension_layer": "L1",
            "required_patch_target_symbols": ["target_stage"],
            "extension_intent": "execution_anchor",
            "expected_outcome": "correctness_anchor_not_speedup",
            "overhead_source_to_record": "launch_layout_overhead",
            "minimum_executable_unit": "separate_gluon_kernel",
            "target_symbol": "target_stage",
            "target_component": "one memory subpath",
            "allowed_execution_path": "separate_gluon_kernel",
            "scope_infeasible_policy": "separate_kernel_if_allowed",
            "whole_kernel_required_reason": "not needed for separate kernel",
            "source_origin": "existing_amd_gluon_operator",
            "gluon_tl_policy": "production_source_allowed",
            "layout_construction_policy": "source_preserve",
            "execution_mode": "mixed_jit_aot",
        },
        "Extension task using AMD Gluon.",
    )

    task = task_file_to_agent_task(task_path)

    assert "Task compat_search_set: extension" in task.task
    assert "Task required_output_dialect: amd_gluon" in task.task
    assert "Task gluon_doc_profile: memory_lowering" in task.task
    assert "Task required_gluon_docs: gluon_skill_path, gluon_always_read_path" in task.task
    assert "Task source_base_family: base_hot_path_streamline" in task.task
    assert "Task plain_competitor: triton-eliminate-redundant-ops-streamline" in task.task
    assert "Task implementation_layer: amd_gluon overlay" in task.task
    assert "Task extension_layer: L1" in task.task
    assert "Task required_patch_target_symbols: target_stage" in task.task
    assert "Task extension_intent: execution_anchor" in task.task
    assert "Task expected_outcome: correctness_anchor_not_speedup" in task.task
    assert "Task overhead_source_to_record: launch_layout_overhead" in task.task
    assert "Task minimum_executable_unit: separate_gluon_kernel" in task.task
    assert "Task target_symbol: target_stage" in task.task
    assert "Task target_component: one memory subpath" in task.task
    assert "Task allowed_execution_path: separate_gluon_kernel" in task.task
    assert "Task scope_infeasible_policy: separate_kernel_if_allowed" in task.task
    assert "Task whole_kernel_required_reason: not needed for separate kernel" in task.task
    assert "Task source_origin: existing_amd_gluon_operator" in task.task
    assert "Task gluon_tl_policy: production_source_allowed" in task.task
    assert "Task layout_construction_policy: source_preserve" in task.task
    assert "Task execution_mode: mixed_jit_aot" in task.task
    assert task.config["source_origin"] == "existing_amd_gluon_operator"
    assert task.config["gluon_tl_policy"] == "production_source_allowed"
    assert task.config["layout_construction_policy"] == "source_preserve"
    assert task.config["execution_mode"] == "mixed_jit_aot"
    assert "Extension intent: `execution_anchor`" in task.task
    assert "Minimum executable unit: `separate_gluon_kernel`" in task.task
    assert "Allowed execution path: `separate_gluon_kernel`" in task.task
    assert "Scope infeasible policy: `separate_kernel_if_allowed`" in task.task


def test_worker_context_infers_local_target_components_from_allowed_change(tmp_path) -> None:
    task_path = tmp_path / "local_target_components.md"
    write_task_file(
        task_path,
        {
            "label": "gluon-l0-explicit-layout",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "required_output_dialect": "amd_gluon",
            "implementation_layer": "amd_gluon overlay",
            "extension_layer": "L0",
        },
        "\n".join(
            [
                "Extension layer: L0",
                "Implementation layer: amd_gluon overlay",
                "Allowed change: Replace the scoped local expressions (tile_load_a, tile_load_b, tile_load_c) in the inner loop.",
                "Reject if: target-symbol mismatch.",
            ]
        ),
    )

    task = task_file_to_agent_task(task_path)

    assert "Task required_patch_target_symbols: tile_load_a, tile_load_b, tile_load_c" in task.task


def test_worker_context_infers_stage_scope_from_allowed_change(tmp_path) -> None:
    task_path = tmp_path / "stage_scope.md"
    write_task_file(
        task_path,
        {
            "label": "gluon-l0-stage1-overlay",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "required_output_dialect": "amd_gluon",
            "implementation_layer": "amd_gluon overlay",
            "extension_layer": "L0",
        },
        "\n".join(
            [
                "Extension layer: L0",
                "Implementation layer: amd_gluon overlay",
                "Allowed change: layout specification for tensor creation and load/store in stage1 inner loop",
                "Reject if: target-symbol mismatch.",
            ]
        ),
    )

    task = task_file_to_agent_task(task_path)

    assert "Task required_patch_target_symbols: stage1" in task.task
    assert task.config["required_patch_target_symbols"] == ["stage1"]


def test_worker_context_infers_forbidden_scope_from_reject_if(tmp_path) -> None:
    task_path = tmp_path / "forbidden_scope.md"
    write_task_file(
        task_path,
        {
            "label": "gluon-l0-scale-load-layout",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "required_output_dialect": "amd_gluon",
            "implementation_layer": "amd_gluon overlay",
            "extension_layer": "L0",
        },
        "\n".join(
            [
                "Allowed change: Convert only the `scale_a` and `scale_b` load+broadcast+multiply path.",
                "Reject if: the main dot loop or A/B loads are modified.",
                "Do NOT attempt to convert the entire kernel to Gluon.",
            ]
        ),
    )

    task = task_file_to_agent_task(task_path)

    assert task.config["required_patch_target_symbols"] == ["scale_a", "scale_b"]
    assert task.config["forbidden_patch_target_symbols"] == [
        "dot_loop",
        "ab_input_loads",
        "whole_kernel_rewrite",
    ]


def test_worker_context_sets_matrix_contract_tag_for_mfma_task(tmp_path) -> None:
    task_path = tmp_path / "matrix_contract.md"
    write_task_file(
        task_path,
        {
            "label": "gluon-l0-explicit-layout-overlay",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["amd_gluon"],
            "required_output_dialect": "amd_gluon",
        },
        "Performance hypothesis: DotOperandLayout can ensure optimal MFMA instruction selection.\n",
    )

    task = task_file_to_agent_task(task_path)

    assert task.config["required_amd_gluon_contract_tags"] == ["matrix_lowering"]


def test_worker_context_infers_required_gluon_from_body_layer_contract(tmp_path) -> None:
    task_path = tmp_path / "body_layer_contract.md"
    write_task_file(
        task_path,
        {
            "label": "gluon-l0-explicit-layout-attention",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "required_output_dialect": "any",
            "use_skills": False,
        },
        "Extension layer: L0\nImplementation layer: amd_gluon overlay\n",
    )

    task = task_file_to_agent_task(task_path)

    assert task.config["use_skills"] is True
    assert task.config["required_output_dialect"] == "amd_gluon"
    assert "Task required_output_dialect: amd_gluon" in task.task
    assert "Task implementation_layer: amd_gluon overlay" in task.task
    assert "Task extension_layer: L0" in task.task


def test_required_gluon_docs_metadata_augments_heuristic_gate(tmp_path) -> None:
    task_path = tmp_path / "required_docs.md"
    write_task_file(
        task_path,
        {
            "label": "ext-custom-docs",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "search_set": "extension",
            "required_output_dialect": "amd_gluon",
            "target_backend": "hip/gfx942",
            "required_gluon_docs": ["gluon_skill_path", "gluon_always_read_path"],
        },
        "Extension task using AMD Gluon with many matrix mfma terms.",
    )

    task = task_file_to_agent_task(task_path)
    paths = task.config["gluon_doc_gate_required_paths"]
    assert any(path.endswith("skills/triton-gluon/SKILL.md") for path in paths)
    assert any(path.endswith("skills/triton-gluon/docs/00_always_read.md") for path in paths)
    assert any(path.endswith("skills/triton-gluon/docs/10_search_policies.md") for path in paths)
    assert any(path.endswith("skills/triton-gluon/docs/20_component_traits.md") for path in paths)
    assert any(path.endswith("skills/triton-gluon/docs/30_architecture_notes.md") for path in paths)
    assert any(path.endswith("skills/triton-gluon/docs/50_api_reference.md") for path in paths)


@pytest.mark.parametrize("profile", sorted(GLUON_DOC_PROFILE_REQUIRED_KEYS))
def test_gluon_doc_profile_mapping_covers_all_profiles(tmp_path, profile: str) -> None:
    task_path = tmp_path / f"{profile}.md"
    write_task_file(
        task_path,
        {
            "label": f"gluon-{profile}",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "required_output_dialect": "amd_gluon",
            "gluon_doc_profile": profile,
        },
        "Extension layer: L0\nImplementation layer: amd_gluon overlay\n",
    )

    task = task_file_to_agent_task(task_path)
    paths = task.config["gluon_doc_gate_required_paths"]
    for key in GLUON_DOC_PROFILE_REQUIRED_KEYS[profile]:
        suffix = {
            "gluon_skill_path": "skills/triton-gluon/SKILL.md",
            "gluon_always_read_path": "skills/triton-gluon/docs/00_always_read.md",
            "gluon_search_policies_path": "skills/triton-gluon/docs/10_search_policies.md",
            "gluon_component_traits_path": "skills/triton-gluon/docs/20_component_traits.md",
            "gluon_architecture_notes_path": "skills/triton-gluon/docs/30_architecture_notes.md",
            "gluon_api_reference_path": "skills/triton-gluon/docs/50_api_reference.md",
            "gluon_real_patterns_path": "skills/triton-gluon/docs/60_real_patterns.md",
        }[key]
        assert _has_doc(paths, suffix), f"{profile} missing {key}"


def test_explicit_required_gluon_docs_merge_with_profile_docs(tmp_path) -> None:
    task_path = tmp_path / "explicit_only_real_patterns.md"
    write_task_file(
        task_path,
        {
            "label": "matrix-explicit-doc",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "required_output_dialect": "amd_gluon",
            "gluon_doc_profile": "matrix_lowering",
            "required_gluon_docs": ["gluon_real_patterns_path"],
        },
        "Extension layer: L0\nImplementation layer: amd_gluon overlay\n",
    )

    task = task_file_to_agent_task(task_path)
    paths = task.config["gluon_doc_gate_required_paths"]
    assert _has_doc(paths, "skills/triton-gluon/SKILL.md")
    assert _has_doc(paths, "skills/triton-gluon/docs/00_always_read.md")
    assert _has_doc(paths, "skills/triton-gluon/docs/10_search_policies.md")
    assert _has_doc(paths, "skills/triton-gluon/docs/20_component_traits.md")
    assert _has_doc(paths, "skills/triton-gluon/docs/30_architecture_notes.md")
    assert _has_doc(paths, "skills/triton-gluon/docs/50_api_reference.md")
    assert _has_doc(paths, "skills/triton-gluon/docs/60_real_patterns.md")


def test_explicit_examples_and_backup_doc_keys_resolve_to_split_docs(tmp_path) -> None:
    task_path = tmp_path / "explicit_examples_backup.md"
    write_task_file(
        task_path,
        {
            "label": "ext-explicit-examples-backup",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "required_output_dialect": "amd_gluon",
            "gluon_doc_profile": "extension_l0_minimal",
            "required_gluon_docs": ["gluon_examples_doc_path", "gluon_backup_details_path"],
        },
        "Extension layer: L0\nImplementation layer: amd_gluon overlay\n",
    )

    task = task_file_to_agent_task(task_path)
    paths = task.config["gluon_doc_gate_required_paths"]
    assert _has_doc(paths, "skills/triton-gluon/docs/40_examples.md")
    assert _has_doc(paths, "skills/triton-gluon/docs/70_backup_details.md")
    assert not any(path.endswith("/gluon_examples_doc_path") for path in paths)
    assert not any(path.endswith("/gluon_backup_details_path") for path in paths)


def test_worker_gate_keeps_memory_and_matrix_implementation_docs(tmp_path) -> None:
    expected = {
        "memory_lowering": [
            "skills/triton-gluon/docs/20_component_traits.md",
            "skills/triton-gluon/docs/50_api_reference.md",
        ],
        "matrix_lowering": [
            "skills/triton-gluon/docs/20_component_traits.md",
            "skills/triton-gluon/docs/30_architecture_notes.md",
            "skills/triton-gluon/docs/50_api_reference.md",
        ],
    }
    for profile, suffixes in expected.items():
        task_path = tmp_path / f"{profile}.md"
        write_task_file(
            task_path,
            {
                "label": f"ext-{profile}",
                "priority": 6,
                "kernel_type": "triton",
                "kernel_path": str(tmp_path / "kernel.py"),
                "repo_root": str(tmp_path),
                "input_dialect": "plain_triton",
                "gluon_feature_mode": "auto",
                "allowed_output_dialects": ["plain_triton", "amd_gluon"],
                "required_output_dialect": "amd_gluon",
                "implementation_layer": "amd_gluon overlay",
                "extension_layer": "L1",
                "gluon_doc_profile": profile,
            },
            "Extension layer: L1\nImplementation layer: amd_gluon overlay\n",
        )

        task = task_file_to_agent_task(task_path)
        paths = task.config["gluon_doc_gate_required_paths"]
        for suffix in suffixes:
            assert _has_doc(paths, suffix), f"{profile} missing {suffix}"


def test_gluon_doc_gate_not_enabled_for_hip_task(tmp_path) -> None:
    task_path = tmp_path / "hip.md"
    write_task_file(
        task_path,
        {
            "label": "hip",
            "priority": 5,
            "kernel_type": "hip",
            "kernel_path": str(tmp_path / "kernel.hip"),
            "repo_root": str(tmp_path),
        },
        "Optimize the HIP kernel.",
    )

    task = task_file_to_agent_task(task_path)

    assert "gluon_doc_gate_enabled" not in task.config


def test_gluon_doc_gate_routes_aot_compile_terms_to_arch_and_api_docs(tmp_path) -> None:
    task_path = tmp_path / "aot.md"
    write_task_file(
        task_path,
        {
            "label": "ext-aot",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "search_set": "extension",
            "required_output_dialect": "amd_gluon",
        },
        "Extension task using compile_gluon signature waves_per_eu global_scratch.",
    )

    task = task_file_to_agent_task(task_path)
    paths = task.config["gluon_doc_gate_required_paths"]
    assert any(path.endswith("skills/triton-gluon/docs/30_architecture_notes.md") for path in paths)
    assert any(path.endswith("skills/triton-gluon/docs/50_api_reference.md") for path in paths)


def test_gluon_doc_gate_routes_reduction_api_terms_to_api_doc(tmp_path) -> None:
    task_path = tmp_path / "reduce.md"
    write_task_file(
        task_path,
        {
            "label": "shared-softmax-reduce",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "search_set": "shared",
            "required_output_dialect": "any",
        },
        "Composition type: shared_transplant\nShared Set task using gl.sum, gl.max and softmax reduction.",
    )

    task = task_file_to_agent_task(task_path)
    paths = task.config["gluon_doc_gate_required_paths"]
    assert any(path.endswith("skills/triton-gluon/docs/50_api_reference.md") for path in paths)


def test_gluon_doc_gate_ignores_plain_api_and_config_words(tmp_path) -> None:
    task_path = tmp_path / "plain_words.md"
    write_task_file(
        task_path,
        {
            "label": "shared-plain-words",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "required_output_dialect": "any",
        },
        "Composition type: shared_transplant\nTune max sum reshape config env for plain helper notes.",
    )

    task = task_file_to_agent_task(task_path)
    paths = task.config["gluon_doc_gate_required_paths"]
    assert not _has_doc(paths, "skills/triton-gluon/docs/50_api_reference.md")
    assert not _has_doc(paths, "skills/triton-gluon/docs/60_real_patterns.md")


def test_gluon_doc_gate_routes_translator_descriptor_terms_to_arch_and_real_docs(tmp_path) -> None:
    task_path = tmp_path / "translator.md"
    write_task_file(
        task_path,
        {
            "label": "ext-translator-tdm",
            "priority": 6,
            "kernel_type": "triton",
            "kernel_path": str(tmp_path / "kernel.py"),
            "repo_root": str(tmp_path),
            "input_dialect": "nv_gluon",
            "gluon_feature_mode": "auto",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "search_set": "extension",
            "required_output_dialect": "amd_gluon",
        },
        "Translate via triton_to_gluon translator current_target TensorDescriptor tdm path.",
    )

    task = task_file_to_agent_task(task_path)
    paths = task.config["gluon_doc_gate_required_paths"]
    assert any(path.endswith("skills/triton-gluon/docs/30_architecture_notes.md") for path in paths)
    assert any(path.endswith("skills/triton-gluon/docs/60_real_patterns.md") for path in paths)


def test_save_and_test_rejects_missing_gluon_doc_views(tmp_path) -> None:
    required = tmp_path / "skills" / "triton-gluon" / "docs" / "00_always_read.md"
    required.parent.mkdir(parents=True)
    required.write_text("# always read\n")
    tool = SaveAndTestTool()
    tool.set_context(
        SaveAndTestContext(
            cwd=str(tmp_path),
            test_command=None,
            timeout=1,
            patch_output_dir=None,
            viewed_file_paths=set(),
            gluon_doc_gate_enabled=True,
            gluon_doc_gate_required_paths=[str(required)],
        )
    )

    result = tool(description="missing docs")

    assert result["returncode"] == 1
    assert "GLUON_DOC_GATE_FAILED" in result["output"]
    assert str(required.resolve()) in result["output"]


def test_save_and_test_allows_after_required_gluon_doc_views(tmp_path) -> None:
    required = tmp_path / "skills" / "triton-gluon" / "docs" / "00_always_read.md"
    required.parent.mkdir(parents=True)
    required.write_text("# always read\n")
    tool = SaveAndTestTool()
    tool.set_context(
        SaveAndTestContext(
            cwd=str(tmp_path),
            test_command="true",
            timeout=5,
            patch_output_dir=None,
            viewed_file_paths={str(required.resolve())},
            gluon_doc_gate_enabled=True,
            gluon_doc_gate_required_paths=[str(required)],
        )
    )

    result = tool(description="docs viewed")

    assert result["returncode"] == 0
    assert "GLUON_DOC_GATE_FAILED" not in result["output"]


def test_save_and_test_accepts_worktree_view_for_base_repo_required_doc(tmp_path) -> None:
    base = tmp_path / "repo"
    worktree = tmp_path / "worktree"
    required = base / "skills" / "triton-gluon" / "docs" / "00_always_read.md"
    viewed = worktree / "skills" / "triton-gluon" / "docs" / "00_always_read.md"
    required.parent.mkdir(parents=True)
    viewed.parent.mkdir(parents=True)
    required.write_text("# base doc\n")
    viewed.write_text("# worktree doc\n")
    tool = SaveAndTestTool()
    tool.set_context(
        SaveAndTestContext(
            cwd=str(worktree),
            test_command="true",
            timeout=5,
            patch_output_dir=None,
            base_repo_path=base,
            viewed_file_paths={str(viewed)},
            gluon_doc_gate_enabled=True,
            gluon_doc_gate_required_paths=[str(required)],
        )
    )

    result = tool(description="docs viewed in worktree")

    assert result["returncode"] == 0
    assert "GLUON_DOC_GATE_FAILED" not in result["output"]


def test_save_and_test_rejects_required_gluon_plain_fallback(tmp_path) -> None:
    tool = SaveAndTestTool()
    tool._get_patch_content = lambda: "diff --git a/kernel.py b/kernel.py\n+import triton.language as tl\n+x = tl.arange(0, 16)\n"  # type: ignore[method-assign]
    tool.set_context(
        SaveAndTestContext(
            cwd=str(tmp_path),
            test_command="true",
            timeout=5,
            patch_output_dir=None,
            required_output_dialect="amd_gluon",
        )
    )

    result = tool(description="plain fallback")

    assert result["returncode"] == 1
    assert "PATCH_CONTRACT_FAILED" in result["output"]


def test_save_and_test_allows_plain_config_only_patch(tmp_path) -> None:
    tool = SaveAndTestTool()
    tool._get_patch_content = lambda: "\n".join(  # type: ignore[method-assign]
        [
            "diff --git a/config.json b/config.json",
            "--- a/config.json",
            "+++ b/config.json",
            "@@",
            '-  "num_stages": 1,',
            '+  "num_stages": 2,',
        ]
    )
    tool.set_context(
        SaveAndTestContext(
            cwd=str(tmp_path),
            test_command="true",
            timeout=5,
            patch_output_dir=None,
            required_output_dialect="plain_triton",
        )
    )

    result = tool(description="plain config tuning")

    assert result["returncode"] == 0
    assert "PATCH_CONTRACT_FAILED" not in result["output"]


def test_save_and_test_rejects_backup_file_patch(tmp_path) -> None:
    tool = SaveAndTestTool()
    tool._get_patch_content = lambda: "\n".join(  # type: ignore[method-assign]
        [
            "diff --git a/kernel.py.bak b/kernel.py.bak",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/kernel.py.bak",
            "+import triton.language as tl",
        ]
    )
    tool.set_context(
        SaveAndTestContext(
            cwd=str(tmp_path),
            test_command="true",
            timeout=5,
            patch_output_dir=None,
            required_output_dialect="plain_triton",
        )
    )

    result = tool(description="backup file")

    assert result["returncode"] == 1
    assert "backup or temporary files" in result["output"]


def test_save_and_test_rejects_forbidden_dot_loop_scope(tmp_path) -> None:
    tool = SaveAndTestTool()
    tool._get_patch_content = lambda: "\n".join(  # type: ignore[method-assign]
        [
            "diff --git a/kernel.py b/kernel.py",
            "--- a/kernel.py",
            "+++ b/kernel.py",
            "@@",
            "+from triton.experimental import gluon",
            "+from triton.experimental.gluon import language as gl",
            "+@gluon.jit",
            "+def scaled_mm_kernel(a_ptrs, b_ptrs):",
            "+    a = gl.load(a_ptrs)",
            "+    b = gl.load(b_ptrs)",
            "+    scale_a = gl.load(scale_a_ptr)",
            "+    scale_b = gl.load(scale_b_ptr)",
            "+    return gl.amd.cdna3.mfma(a, b, acc)",
        ]
    )
    tool.set_context(
        SaveAndTestContext(
            cwd=str(tmp_path),
            test_command="true",
            timeout=5,
            patch_output_dir=None,
            required_output_dialect="amd_gluon",
            required_patch_target_symbols=["scale_a", "scale_b"],
            forbidden_patch_target_symbols=["dot_loop", "ab_input_loads", "whole_kernel_rewrite"],
        )
    )

    result = tool(description="forbidden dot loop")

    assert result["returncode"] == 1
    assert "forbidden target scope" in result["output"]
    assert "Revert the forbidden change" in result["output"]
    assert "shrink or split the scoped Gluon path" in result["output"]


def test_save_and_test_rejects_helper_without_execution_with_wiring_hint(tmp_path) -> None:
    tool = SaveAndTestTool()
    tool._get_patch_content = lambda: "\n".join(  # type: ignore[method-assign]
        [
            "diff --git a/kernel.py b/kernel.py",
            "+from triton.experimental import gluon",
            "+@gluon.jit",
            "+def target_stage_gluon(x):",
            "+    return x",
        ]
    )
    tool.set_context(
        SaveAndTestContext(
            cwd=str(tmp_path),
            test_command="true",
            timeout=5,
            patch_output_dir=None,
            required_output_dialect="amd_gluon",
            required_patch_target_symbols=["target_stage_gluon"],
        )
    )

    result = tool(description="helper not executed")

    assert result["returncode"] == 1
    assert "helper wiring, launch, or measured-output feeding" in result["output"]
    assert "do not widen the target scope" in result["output"]


def test_save_and_test_allows_required_gluon_generic_dot_without_matrix_contract(tmp_path) -> None:
    tool = SaveAndTestTool()
    tool._get_patch_content = lambda: "\n".join(  # type: ignore[method-assign]
        [
            "diff --git a/kernel.py b/kernel.py",
            "+from triton.experimental import gluon",
            "+from triton.experimental.gluon import language as gl",
            "+@gluon.jit",
            "+def kernel_gluon(q, k):",
            "+    return gl.dot(q, k)",
            "+kernel_gluon[grid](q, k)",
        ]
    )
    tool.set_context(
        SaveAndTestContext(
            cwd=str(tmp_path),
            test_command="true",
            timeout=5,
            patch_output_dir=None,
            required_output_dialect="amd_gluon",
        )
    )

    result = tool(description="generic gl.dot")

    assert result["returncode"] == 0
    assert "PATCH_CONTRACT_FAILED" not in result["output"]


def test_save_and_test_rejects_required_gluon_generic_dot_for_matrix_contract(tmp_path) -> None:
    tool = SaveAndTestTool()
    tool._get_patch_content = lambda: "\n".join(  # type: ignore[method-assign]
        [
            "diff --git a/kernel.py b/kernel.py",
            "+from triton.experimental import gluon",
            "+from triton.experimental.gluon import language as gl",
            "+@gluon.jit",
            "+def kernel_gluon(q, k):",
            "+    return gl.dot(q, k)",
            "+kernel_gluon[grid](q, k)",
        ]
    )
    tool.set_context(
        SaveAndTestContext(
            cwd=str(tmp_path),
            test_command="true",
            timeout=5,
            patch_output_dir=None,
            required_output_dialect="amd_gluon",
            required_amd_gluon_contract_tags=["matrix_lowering"],
        )
    )

    result = tool(description="generic gl.dot")

    assert result["returncode"] == 1
    assert "generic gl.dot" in result["output"]


def test_save_and_test_rejects_required_gluon_plain_exception_fallback(tmp_path) -> None:
    tool = SaveAndTestTool()
    tool._get_patch_content = lambda: "\n".join(  # type: ignore[method-assign]
        [
            "diff --git a/wrapper.py b/wrapper.py",
            "+try:",
            "+    kernel_gluon[grid](x)",
            "+except Exception:",
            "+    kernel_plain[grid](x)",
        ]
    )
    tool.set_context(
        SaveAndTestContext(
            cwd=str(tmp_path),
            test_command="true",
            timeout=5,
            patch_output_dir=None,
            required_output_dialect="amd_gluon",
        )
    )

    result = tool(description="plain exception fallback")

    assert result["returncode"] == 1
    assert "plain Triton launcher" in result["output"]


def test_save_and_test_rejects_mixed_dead_gluon_helper(tmp_path) -> None:
    tool = SaveAndTestTool()
    tool._get_patch_content = lambda: "\n".join(  # type: ignore[method-assign]
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
    tool.set_context(
        SaveAndTestContext(
            cwd=str(tmp_path),
            test_command="true",
            timeout=5,
            patch_output_dir=None,
            required_output_dialect="mixed",
        )
    )

    result = tool(description="mixed dead helper")

    assert result["returncode"] == 1
    assert "PATCH_CONTRACT_FAILED" in result["output"]


def test_run_task_batch_rejects_duplicate_labels(tmp_path) -> None:
    first = tmp_path / "tasks" / "00_dup.md"
    second = tmp_path / "tasks" / "01_dup.md"
    first.parent.mkdir()
    for path in (first, second):
        write_task_file(
            path,
            {
                "label": "dup-task",
                "priority": 1,
                "kernel_type": "triton",
                "repo_root": str(tmp_path),
            },
            "Optimize.",
        )

    with pytest.raises(ValueError, match="Duplicate task labels"):
        run_task_batch([first, second], gpu_ids=[0], output_dir=tmp_path / "results", model_factory=lambda: None)


def test_required_gluon_extension_groups_into_high_stage(tmp_path) -> None:
    base_task = tmp_path / "00_base.md"
    ext_task = tmp_path / "06_ext.md"
    write_task_file(
        base_task,
        {"label": "base", "priority": 0, "kernel_type": "triton"},
        "Base task",
    )
    write_task_file(
        ext_task,
        {
            "label": "ext",
            "priority": 6,
            "kernel_type": "triton",
            "search_set": "extension",
            "required_output_dialect": "amd_gluon",
        },
        "Extension task",
    )

    stages = _group_task_files_by_dispatch_stage([base_task, ext_task])
    assert stages[0][0] == "high"
    assert {path.name for path in stages[0][1]} == {"00_base.md", "06_ext.md"}


def test_required_gluon_completion_uses_body_layer_contract(tmp_path) -> None:
    results_dir = tmp_path / "results"
    ext_task = tmp_path / "06_ext.md"
    write_task_file(
        ext_task,
        {
            "label": "ext",
            "priority": 6,
            "kernel_type": "triton",
        },
        "Extension layer: L0\nImplementation layer: amd_gluon overlay",
    )

    assert _required_gluon_tasks_completed(results_dir, [ext_task]) is False

    (results_dir / "ext").mkdir(parents=True)
    (results_dir / "ext" / "patch_0.patch").write_text("diff --git a/kernel.py b/kernel.py\n+from triton.experimental import gluon\n")
    assert _required_gluon_tasks_completed(results_dir, [ext_task]) is True


def test_required_gluon_groups_high_without_search_set(tmp_path) -> None:
    base_task = tmp_path / "00_base.md"
    ext_task = tmp_path / "06_ext.md"
    write_task_file(
        base_task,
        {"label": "base", "priority": 0, "kernel_type": "triton"},
        "Base task",
    )
    write_task_file(
        ext_task,
        {
            "label": "ext",
            "priority": 6,
            "kernel_type": "triton",
            "required_output_dialect": "amd_gluon",
        },
        "Extension layer: L0\nImplementation layer: amd_gluon overlay",
    )

    stages = _group_task_files_by_dispatch_stage([base_task, ext_task])
    assert stages[0][0] == "high"
    assert {path.name for path in stages[0][1]} == {"00_base.md", "06_ext.md"}
