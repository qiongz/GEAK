"""Tests for the agent-based task generator."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.agents.heterogeneous.task_generator import (
    _SYSTEM_PROMPT,
    _build_workload_guidance,
    _extract_kernel_meta,
    _parse_llm_response,
    _run_task_agent,
    generate_tasks,
    write_task_files,
)
from minisweagent.run.task_file import read_task_file


class FakeAgentClass:
    """Stand-in for an agent class in tests."""

    pass


class FakePlanningModel:
    """Minimal model stub for task-generator unit tests."""

    def __init__(self) -> None:
        self.tools: list[dict[str, str]] = []

    def set_tools(self, tools) -> None:
        self.tools = list(tools)


def _make_kernel_kwargs(
    kernel_type: str = "triton",
) -> dict[str, Any]:
    return {
        "kernel_path": "/workspace/kernel.py",
        "kernel_name": "test_kernel",
        "kernel_type": kernel_type,
        "kernel_language": "python",
        "function_names": ["kernel_fwd"],
        "workspace_path": "/workspace",
    }


def _write_knowledge_files(workspace: Path) -> tuple[Path, Path]:
    kb_dir = workspace / "knowledge_base"
    kb_dir.mkdir()
    general_kb = kb_dir / "optimization_strategies.py"
    general_kb.write_text("# general optimization knowledge\n")
    gluon_kb = kb_dir / "triton_gluon_mi3xx_benchmark_safe.md"
    gluon_kb.write_text("# benchmark-safe gluon knowledge\n")
    return general_kb, gluon_kb


# ---- Agent submits valid JSON -> tasks produced ----


VALID_TASK_JSON = """[
    {
        "label": "evolve-inner",
        "priority": 0,
        "agent_type": "openevolve",
        "kernel_language": "python",
        "task_prompt": "Run OpenEvolve on /ws/inner.py"
    },
    {
        "label": "mem-opt",
        "priority": 10,
        "agent_type": "strategy_agent",
        "kernel_language": "python",
        "task_prompt": "Optimize memory patterns"
    }
]"""


@patch("minisweagent.agents.heterogeneous.task_generator._run_task_agent", return_value=VALID_TASK_JSON)
def test_agent_submits_valid_json(mock_agent):
    model = MagicMock()
    tasks = generate_tasks(
        base_task_context="ctx",
        agent_class=FakeAgentClass,
        model=model,
        **_make_kernel_kwargs("triton"),
    )
    assert len(tasks) == 2
    assert tasks[0].label == "evolve-inner"
    assert tasks[0].priority == 0
    assert tasks[1].label == "mem-opt"
    mock_agent.assert_called_once()


# ---- Agent fails -> RuntimeError propagates ----


@patch(
    "minisweagent.agents.heterogeneous.task_generator._run_task_agent",
    side_effect=RuntimeError("agent did not submit"),
)
def test_agent_failure_propagates(mock_agent):
    model = MagicMock()
    with pytest.raises(RuntimeError, match="agent did not submit"):
        generate_tasks(
            base_task_context="ctx",
            agent_class=FakeAgentClass,
            model=model,
            **_make_kernel_kwargs("triton"),
        )


@patch("minisweagent.tools.tools_runtime.get_tools_list", return_value=[{"name": "str_replace_editor"}, {"name": "submit"}])
@patch("minisweagent.agents.default.DefaultAgent")
def test_run_task_agent_enables_skills_for_triton(mock_default_agent, _mock_tools, tmp_path: Path):
    model = FakePlanningModel()
    mock_default_agent.return_value.run.return_value = ("Submitted", "[]")
    general_kb, _gluon_kb = _write_knowledge_files(tmp_path)

    submitted = _run_task_agent(
        kernel_path=str(tmp_path / "kernel.py"),
        kernel_name="test_kernel",
        kernel_type="triton",
        kernel_language="python",
        function_names=["kernel_fwd"],
        workspace_path=str(tmp_path),
        input_dialect="plain_triton",
        gluon_feature_mode="off",
        gluon_baseline_profile="raw",
        allowed_output_dialects=["plain_triton"],
        target_backend="hip/gfx942",
        base_task_context="ctx",
        model=model,
        profiling_path=None,
        commandment_path=None,
        baseline_metrics_path=None,
        deep_search_path=None,
        previous_results_dir=None,
        discovery_path=None,
        codebase_context_path=None,
        previous_tasks_dir=None,
        round_evaluations=None,
        current_round=1,
        num_gpus=1,
    )

    assert submitted == "[]"
    assert mock_default_agent.call_args.kwargs["use_skills"] is True
    assert mock_default_agent.call_args.kwargs["allowed_skill_tiers"] == ["general"]
    run_kwargs = mock_default_agent.return_value.run.call_args.kwargs
    assert run_kwargs["knowledge_base_path"] == str(general_kb)
    assert run_kwargs["gluon_benchmark_safe_knowledge_path"] == ""


@patch("minisweagent.tools.tools_runtime.get_tools_list", return_value=[{"name": "str_replace_editor"}, {"name": "submit"}])
@patch("minisweagent.agents.default.DefaultAgent")
def test_run_task_agent_raw_profile_omits_benchmark_safe_gluon_knowledge(
    mock_default_agent, _mock_tools, tmp_path: Path
):
    model = FakePlanningModel()
    mock_default_agent.return_value.run.return_value = ("Submitted", "[]")
    general_kb, _gluon_kb = _write_knowledge_files(tmp_path)

    submitted = _run_task_agent(
        kernel_path=str(tmp_path / "kernel.py"),
        kernel_name="test_kernel",
        kernel_type="triton",
        kernel_language="python",
        function_names=["kernel_fwd"],
        workspace_path=str(tmp_path),
        input_dialect="amd_gluon",
        gluon_feature_mode="auto",
        gluon_baseline_profile="raw",
        allowed_output_dialects=["plain_triton", "amd_gluon"],
        target_backend="hip/gfx942",
        base_task_context="ctx",
        model=model,
        profiling_path=None,
        commandment_path=None,
        baseline_metrics_path=None,
        deep_search_path=None,
        previous_results_dir=None,
        discovery_path=None,
        codebase_context_path=None,
        previous_tasks_dir=None,
        round_evaluations=None,
        current_round=1,
        num_gpus=1,
    )

    assert submitted == "[]"
    assert mock_default_agent.call_args.kwargs["allowed_skill_tiers"] == ["general"]
    run_kwargs = mock_default_agent.return_value.run.call_args.kwargs
    assert run_kwargs["knowledge_base_path"] == str(general_kb)
    assert run_kwargs["gluon_benchmark_safe_knowledge_path"] == ""


@patch("minisweagent.tools.tools_runtime.get_tools_list", return_value=[{"name": "str_replace_editor"}, {"name": "submit"}])
@patch("minisweagent.agents.default.DefaultAgent")
def test_run_task_agent_uses_benchmark_safe_skill_tiers_for_mi3xx(mock_default_agent, _mock_tools, tmp_path: Path):
    model = FakePlanningModel()
    mock_default_agent.return_value.run.return_value = ("Submitted", "[]")
    general_kb, gluon_kb = _write_knowledge_files(tmp_path)

    submitted = _run_task_agent(
        kernel_path=str(tmp_path / "kernel.py"),
        kernel_name="test_kernel",
        kernel_type="triton",
        kernel_language="python",
        function_names=["kernel_fwd"],
        workspace_path=str(tmp_path),
        input_dialect="amd_gluon",
        gluon_feature_mode="auto",
        gluon_baseline_profile="mi3xx",
        allowed_output_dialects=["plain_triton", "amd_gluon"],
        target_backend="hip/gfx942",
        base_task_context="ctx",
        model=model,
        profiling_path=None,
        commandment_path=None,
        baseline_metrics_path=None,
        deep_search_path=None,
        previous_results_dir=None,
        discovery_path=None,
        codebase_context_path=None,
        previous_tasks_dir=None,
        round_evaluations=None,
        current_round=1,
        num_gpus=1,
    )

    assert submitted == "[]"
    assert mock_default_agent.call_args.kwargs["allowed_skill_tiers"] == ["general", "benchmark_safe"]
    run_kwargs = mock_default_agent.return_value.run.call_args.kwargs
    assert run_kwargs["knowledge_base_path"] == str(general_kb)
    assert run_kwargs["gluon_benchmark_safe_knowledge_path"] == str(gluon_kb)


def test_write_task_files_marks_triton_tasks_with_skill_usage(tmp_path: Path):
    tasks = _parse_llm_response(
        '[{"label": "opt", "priority": 5, "agent_type": "strategy_agent", "task_prompt": "Do it"}]',
        FakeAgentClass,
    )
    _general_kb, gluon_kb = _write_knowledge_files(tmp_path)

    paths = write_task_files(
        tasks,
        tmp_path,
        kernel_path=str(tmp_path / "kernel.py"),
        kernel_type="triton",
        input_dialect="amd_gluon",
        gluon_feature_mode="auto",
        gluon_baseline_profile="mi3xx",
        allowed_output_dialects=["plain_triton", "amd_gluon"],
        target_backend="hip/gfx942",
    )

    meta, _body = read_task_file(paths[0])
    assert meta["kernel_type"] == "triton"
    assert meta["use_skills"] is True
    assert meta["input_dialect"] == "amd_gluon"
    assert meta["gluon_feature_mode"] == "auto"
    assert meta["gluon_baseline_profile"] == "mi3xx"
    assert meta["allowed_output_dialects"] == ["plain_triton", "amd_gluon"]
    assert meta["allowed_skill_tiers"] == ["general", "benchmark_safe"]
    assert meta["gluon_benchmark_safe_knowledge_path"] == str(gluon_kb)


def test_write_task_files_omits_benchmark_safe_gluon_knowledge_for_raw(tmp_path: Path):
    tasks = _parse_llm_response(
        '[{"label": "opt", "priority": 5, "agent_type": "strategy_agent", "task_prompt": "Do it"}]',
        FakeAgentClass,
    )
    _write_knowledge_files(tmp_path)

    paths = write_task_files(
        tasks,
        tmp_path,
        kernel_path=str(tmp_path / "kernel.py"),
        kernel_type="triton",
        input_dialect="amd_gluon",
        gluon_feature_mode="auto",
        gluon_baseline_profile="raw",
        allowed_output_dialects=["plain_triton", "amd_gluon"],
        target_backend="hip/gfx942",
    )

    meta, _body = read_task_file(paths[0])
    assert "gluon_benchmark_safe_knowledge_path" not in meta


def test_extract_kernel_meta_reinfers_unknown_gluon_type(tmp_path: Path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text(
        "from triton.experimental import gluon\n"
        "@gluon.jit\n"
        "def kernel_fwd(x):\n"
        "    return x\n"
    )

    meta = _extract_kernel_meta(
        {
            "workspace": str(tmp_path),
            "kernel": {
                "file": str(kernel),
                "name": "kernel",
                "type": "unknown",
                "functions": ["kernel_fwd"],
            },
        },
        str(kernel),
    )

    assert meta["kernel_type"] == "triton"
    assert meta["input_dialect"] == "amd_gluon"
    assert meta["gluon_feature_mode"] == "auto"


# ---- No kernels -> empty ----


def test_no_kernel_path_returns_empty():
    tasks = generate_tasks("ctx", FakeAgentClass, model=MagicMock(), kernel_path="")
    assert tasks == []


# ---- _parse_llm_response edge cases ----


def test_parse_valid_json():
    tasks = _parse_llm_response(VALID_TASK_JSON, FakeAgentClass)
    assert len(tasks) == 2
    assert tasks[0].label == "evolve-inner"


def test_parse_strategy_agent_uses_mapped_class():
    from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent

    tasks = _parse_llm_response(
        '[{"label": "opt", "priority": 5, "agent_type": "strategy_agent", "task_prompt": "Do it"}]',
        FakeAgentClass,
    )
    assert tasks[0].agent_class is StrategyInteractiveAgent


def test_parse_rejects_non_array():
    with pytest.raises(TypeError, match="Expected JSON array"):
        _parse_llm_response('{"not": "array"}', FakeAgentClass)


def test_parse_rejects_empty_task_prompt():
    with pytest.raises(ValueError, match="no valid tasks"):
        _parse_llm_response(
            '[{"label": "x", "priority": 5, "task_prompt": ""}]',
            FakeAgentClass,
        )


def test_parse_clamps_priority():
    tasks = _parse_llm_response(
        '[{"label": "x", "priority": 99, "task_prompt": "Do it"}]',
        FakeAgentClass,
    )
    assert tasks[0].priority == 15


def test_parse_code_fenced_json():
    fenced = '```json\n[{"label": "opt-1", "priority": 5, "agent_type": "strategy_agent", "kernel_language": "python", "task_prompt": "Do something"}]\n```'
    tasks = _parse_llm_response(fenced, FakeAgentClass)
    assert len(tasks) == 1
    assert tasks[0].label == "opt-1"


def test_parse_sorts_by_priority():
    json_text = """[
        {"label": "low", "priority": 15, "task_prompt": "Low priority task"},
        {"label": "high", "priority": 0, "task_prompt": "High priority task"},
        {"label": "mid", "priority": 5, "task_prompt": "Mid priority task"}
    ]"""
    tasks = _parse_llm_response(json_text, FakeAgentClass)
    assert [t.label for t in tasks] == ["high", "mid", "low"]


def test_build_workload_guidance_classifies_hip_search_as_latency_bound():
    kernel = {
        "file_path": "/workspace/rocprim/device_binary_search.hpp",
        "kernel_name": "device_binary_search",
        "kernel_type": "unknown",
    }
    baseline_metrics = {
        "kernel_name": "rocprim::detail::binary_search lower_bound",
        "bottleneck": "latency",
        "metrics": {
            "memory.hbm_bandwidth_utilization": 0.3,
            "memory.l2_hit_rate": 70.6,
        },
        "top_kernels": [
            {
                "name": "transform_kernel<binary_search<lower_bound>>",
                "bottleneck": "latency",
            }
        ],
    }

    guidance = _build_workload_guidance(kernel, baseline_metrics)

    assert "HIP backend detected." in guidance
    assert "Prefer First:" in guidance
    assert "Branchless/control-flow simplification" in guidance
    assert "Size-specialized kernel variants" in guidance
    assert "Bandwidth-maximization or generic vectorization ideas as the main strategy." in guidance
    assert "Search / pointer-chasing classifier:" in guidance


def test_build_workload_guidance_for_triton_deprioritizes_dispatch():
    kernel = {
        "file_path": "/workspace/kernel.py",
        "kernel_name": "fused_rms",
        "kernel_type": "triton",
    }
    baseline_metrics = {
        "kernel_name": "fused_rms_fp8",
        "bottleneck": "memory-bound",
        "duration_us": 12.4,
        "metrics": {
            "memory.hbm_bandwidth_utilization": 71.2,
            "memory.l2_hit_rate": 44.0,
        },
    }

    guidance = _build_workload_guidance(kernel, baseline_metrics)

    assert "Triton backend detected." in guidance
    assert "Prefer First:" in guidance
    assert "Memory-access rewrites inside the kernel body" in guidance
    assert "@triton.autotune-only config sweeps." in guidance
    assert "Python dispatch, import-routing, or wrapper-only edits" in guidance


def test_build_workload_guidance_empty_when_no_backend_and_no_metrics():
    kernel = {
        "file_path": "/workspace/kernel.txt",
        "kernel_name": "mystery",
        "kernel_type": "unknown",
    }

    assert _build_workload_guidance(kernel, {}) == ""


def test_system_prompt_deprioritizes_dispatch_path_work():
    assert "- 0: Novel algorithmic kernel rewrites" in _SYSTEM_PROMPT
    assert "- 15: Wrapper/launch-config/dispatch-only changes (lowest priority)" in _SYSTEM_PROMPT
    assert 'Generate at least 3 tasks from the "Prefer First" families' in _SYSTEM_PROMPT
    assert "leave some gpus idle" in _SYSTEM_PROMPT.lower()
    assert "Generate at least one priority-0 task that specifically checks the dispatch path" not in _SYSTEM_PROMPT
