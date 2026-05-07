# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for task-file dispatch configuration."""

from __future__ import annotations

from minisweagent.agents.heterogeneous.tools import _group_task_files_by_dispatch_stage
from minisweagent.run.dispatch import task_file_to_agent_task
from minisweagent.run.task_file import write_task_file
from minisweagent.tools.save_and_test import SaveAndTestContext, SaveAndTestTool


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
        },
        "Optimize the Triton kernel.",
    )

    task = task_file_to_agent_task(task_path)

    assert task.config["allowed_skill_tiers"] == ["general"]
    assert "## Gluon Feature Context" in task.task


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
    assert task.config["gluon_doc_gate_enabled"] is True
    assert any(path.endswith("skills/triton-gluon/docs/00_always_read.md") for path in task.config["gluon_doc_gate_required_paths"])
    assert any(path.endswith("skills/triton-gluon/docs/50_api_reference.md") for path in task.config["gluon_doc_gate_required_paths"])


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
