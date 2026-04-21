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
    assert task.config["allowed_skill_tiers"] == ["general", "benchmark_safe"]


def test_task_file_to_agent_task_injects_benchmark_safe_gluon_knowledge(tmp_path) -> None:
    gluon_kb = tmp_path / "knowledge_base" / "triton_gluon_mi3xx_benchmark_safe.md"
    gluon_kb.parent.mkdir()
    gluon_kb.write_text("# benchmark-safe\n")

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
            "gluon_benchmark_safe_knowledge_path": str(gluon_kb),
        },
        "Optimize the Triton kernel.",
    )

    task = task_file_to_agent_task(task_path)

    assert task.config["allowed_skill_tiers"] == ["general", "benchmark_safe"]
    assert "## Benchmark-safe Gluon Knowledge" in task.task
    assert str(gluon_kb) in task.task


def test_task_file_to_agent_task_omits_benchmark_safe_gluon_knowledge_for_raw(tmp_path) -> None:
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
    assert "## Benchmark-safe Gluon Knowledge" not in task.task
