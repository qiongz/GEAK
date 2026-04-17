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


def test_task_file_to_agent_task_respects_explicit_skill_disable(tmp_path) -> None:
    task_path = tmp_path / "triton_no_skills.md"
    write_task_file(
        task_path,
        {
            "label": "triton-no-skills",
            "priority": 5,
            "kernel_type": "triton",
            "use_skills": False,
        },
        "Optimize the Triton kernel.",
    )

    task = task_file_to_agent_task(task_path)

    assert task.config["use_skills"] is False
