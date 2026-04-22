# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for task-file dispatch configuration."""

from __future__ import annotations

from minisweagent.run.dispatch import task_file_to_agent_task
from minisweagent.run.task_file import write_task_file


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
