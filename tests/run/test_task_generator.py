"""Tests for the agent-based task generator."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.agents.heterogeneous.task_generator import (
    _BASE_FAMILY_PERSISTENT,
    _BASE_FAMILY_SCALED_FUSION,
    _BASE_FAMILY_SPLIT_K,
    _BASE_FAMILY_STREAMLINE,
    _BASE_FAMILY_SWIZZLE_TILE,
    _SYSTEM_PROMPT,
    _base_triton_mandatory_families,
    _build_gluon_planning_traits_guidance,
    _build_search_space_allocation_guidance,
    _build_workload_guidance,
    _gluon_extension_strength,
    _extract_kernel_meta,
    _infer_gluon_planning_traits,
    _parse_llm_response,
    _previous_gluon_signal,
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


def _write_knowledge_files(workspace: Path) -> Path:
    kb_dir = workspace / "knowledge_base"
    kb_dir.mkdir()
    general_kb = kb_dir / "optimization_strategies.py"
    general_kb.write_text("# general optimization knowledge\n")
    return general_kb


def _gluon_feature_meta(
    input_dialect: str = "plain_triton",
    *,
    target_backend: str = "hip/gfx942",
) -> dict[str, Any]:
    return {
        "kernel_type": "triton",
        "input_dialect": input_dialect,
        "gluon_feature_mode": "auto",
        "gluon_baseline_profile": "raw",
        "allowed_output_dialects": ["plain_triton", "amd_gluon"],
        "preferred_output_dialects": ["amd_gluon", "plain_triton"],
        "output_dialect_search_policy": "prefer_amd_gluon_if_viable_else_plain_triton",
        "target_backend": target_backend,
    }


def test_infer_gluon_planning_traits_for_plain_triton_dot() -> None:
    traits = _infer_gluon_planning_traits(
        _gluon_feature_meta("plain_triton"),
        "import triton.language as tl\nx = tl.arange(0, BLOCK)\nacc = tl.dot(a, b)\n",
    )

    assert "semantics_contract" in traits
    assert "dialect_plain_triton" in traits
    assert "layout_basic" in traits
    assert "matrix_dot" in traits
    assert "memory_amd_buffer" in traits


def test_infer_gluon_planning_traits_for_source_first_scaled_descriptor() -> None:
    traits = _infer_gluon_planning_traits(
        _gluon_feature_meta("amd_gluon", target_backend="hip/gfx1250"),
        "TensorDescriptor DistributedLinearLayout reshape permute tl.dot_scaled fp8 e2m1 AMDMFMALayout instr_shape",
    )

    assert "dialect_amd_gluon" in traits
    assert "layout_source_first_required" in traits
    assert "matrix_scaled_dot" in traits
    assert "matrix_wmma_descriptor" in traits
    assert "version_sensitive" in traits


def test_build_gluon_planning_traits_guidance_includes_candidate_slots(tmp_path: Path) -> None:
    kernel = tmp_path / "kernel.py"
    kernel.write_text(
        "import triton\n"
        "import triton.language as tl\n"
        "@triton.jit\n"
        "def kernel_fwd(a, b, c):\n"
        "    acc = tl.dot(a, b)\n"
        "    tl.store(c, acc)\n"
    )

    guidance = _build_gluon_planning_traits_guidance(
        _gluon_feature_meta("plain_triton"),
        kernel_path=kernel,
        kernel_name="kernel",
        function_names=["kernel_fwd"],
        baseline_metrics={},
    )

    assert "## Gluon Planning Traits" in guidance
    assert "`dialect_plain_triton`" in guidance
    assert "`matrix_dot`" in guidance
    assert "Prefer First:" in guidance
    assert "DotOperandLayout" in guidance
    assert "Fallback candidate" in guidance


def test_build_gluon_planning_traits_guidance_for_nv_gluon_translation(tmp_path: Path) -> None:
    kernel = tmp_path / "kernel.py"
    kernel.write_text("from triton.experimental import gluon\n# nvidia tma warpgroup_mma TensorDescriptor\n")

    guidance = _build_gluon_planning_traits_guidance(
        _gluon_feature_meta("nv_gluon"),
        kernel_path=kernel,
        kernel_name="kernel",
        function_names=["kernel_fwd"],
        baseline_metrics={},
    )

    assert "`dialect_nv_gluon`" in guidance
    assert "translate NVIDIA-facing Gluon assumptions" in guidance
    assert "do not rename APIs mechanically" in guidance


def test_search_space_allocation_for_two_gpus_preserves_base_and_extension() -> None:
    guidance = _build_search_space_allocation_guidance(
        _gluon_feature_meta("plain_triton"),
        traits=["semantics_contract", "dialect_plain_triton", "layout_basic", "matrix_none"],
        num_gpus=2,
    )

    assert "## Search Space Allocation" in guidance
    assert "Base Set (plain Triton): at least 3 task(s)" in guidance
    assert "Extension Set (AMD Gluon): 1 task(s)" in guidance
    assert "Do not replace or reduce Base Set tasks with Gluon tasks" in guidance
    assert "Extension L0: minimal AMD Gluon viability" in guidance


def test_search_space_allocation_for_large_budget_keeps_full_base() -> None:
    guidance = _build_search_space_allocation_guidance(
        _gluon_feature_meta("plain_triton"),
        traits=["semantics_contract", "dialect_plain_triton", "layout_basic", "matrix_dot"],
        num_gpus=6,
    )

    assert "Gluon extension strength: strong" in guidance
    assert "Base Set (plain Triton): at least 6 task(s)" in guidance
    assert "Shared Set (Triton/Gluon common strategies): 1 task(s)" in guidance
    assert "Extension Set (AMD Gluon): 2 task(s)" in guidance
    assert "slots 2+ may be Extension L1" in guidance


def test_base_triton_mandatory_families_for_scaled_mm_latency_bucketed() -> None:
    meta = {
        **_gluon_feature_meta("plain_triton"),
        "shape_coverage_profile": "bucketed",
        "benchmark_shape_count": 5,
        "benchmark_test_cases": [{"case_id": "perf1", "params": {"M": 32, "K": 64, "N": 64}}],
    }
    families = _base_triton_mandatory_families(
        meta,
        ["semantics_contract", "dialect_plain_triton", "matrix_scaled_dot", "shape_coverage_bucketed"],
        {"bottleneck": "latency", "duration_us": 147.3},
    )

    assert families == [
        _BASE_FAMILY_SWIZZLE_TILE,
        _BASE_FAMILY_SPLIT_K,
        _BASE_FAMILY_SCALED_FUSION,
        _BASE_FAMILY_STREAMLINE,
        _BASE_FAMILY_PERSISTENT,
    ]


def test_search_space_allocation_lists_base_family_checklist() -> None:
    guidance = _build_search_space_allocation_guidance(
        {**_gluon_feature_meta("plain_triton"), "shape_coverage_profile": "bucketed"},
        traits=["semantics_contract", "dialect_plain_triton", "layout_basic", "matrix_scaled_dot"],
        num_gpus=1,
        baseline_metrics={"bottleneck": "latency", "duration_us": 147.3},
    )

    assert "Mandatory Base Set family checklist" in guidance
    assert "`base_swizzle_and_tile_schedule`" in guidance
    assert "`base_split_k_or_multipass_reduce`" in guidance
    assert "`base_scaled_dot_fusion`" in guidance
    assert "`base_hot_path_streamline`" in guidance
    assert "`base_small_matrix_persistent_or_launch_amortization`" in guidance
    assert "Base family: <family_id>" in guidance


def test_search_space_allocation_includes_evidence_anchored_composition() -> None:
    guidance = _build_search_space_allocation_guidance(
        {**_gluon_feature_meta("plain_triton"), "shape_coverage_profile": "multi"},
        traits=["semantics_contract", "dialect_plain_triton", "layout_basic", "memory_amd_buffer", "matrix_dot"],
        num_gpus=8,
        baseline_metrics={"bottleneck": "memory", "duration_us": 180.0},
        previous_results_text=(
            "### base-split-k\n"
            "- patch_8 [BEST] verified_speedup=1.85x\n"
            "### ext-l1-gluon-buffer-load\n"
            "- patch_3 verified_speedup=1.07x\n"
        ),
        previous_tasks_text="- **ext-l1-gluon-buffer-load** (input_dialect=amd_gluon): buffer load policy\n",
    )

    assert "## Evidence-Anchored Composition" in guidance
    assert "safe anchor" in guidance
    assert "`shared_transplant`" in guidance
    assert "`gluon_variant`" in guidance
    assert "`hybrid_dispatch`" in guidance
    assert "`memory_access_policy`" in guidance
    assert "Comparison target: safe_anchor" in guidance
    assert "generate at least one `shared_transplant` or `gluon_variant` task" in guidance


def test_search_space_allocation_degrades_after_gluon_failure() -> None:
    guidance = _build_search_space_allocation_guidance(
        _gluon_feature_meta("plain_triton"),
        traits=["semantics_contract", "dialect_plain_triton", "layout_basic", "matrix_dot"],
        num_gpus=6,
        previous_results_text="gluon compile failed with traceback",
    )

    assert "Previous Gluon signal: failed" in guidance
    assert "Extension Set (AMD Gluon): 1 task(s)" in guidance
    assert "Extension L0: minimal AMD Gluon viability" in guidance
    assert "layout-only, translation-only, or memory-only" in guidance


def test_gluon_extension_strength_and_previous_signal_helpers() -> None:
    assert _gluon_extension_strength(["semantics_contract", "matrix_dot"]) == "strong"
    assert _gluon_extension_strength(["semantics_contract", "layout_slice_broadcast"]) == "normal"
    assert _gluon_extension_strength(["semantics_contract", "matrix_none"]) == "weak"
    assert _previous_gluon_signal("gluon correctness failed") == "failed"
    assert _previous_gluon_signal("gluon slower performance regression") == "slower"
    assert _previous_gluon_signal("gluon [BEST] verified_speedup=1.2x") == "won"
    assert _previous_gluon_signal("gluon [BEST] verified_speedup=1.0067x") == "attempted"


def test_audit_rejects_missing_split_k_or_persistent_base_family() -> None:
    degraded = json.dumps(
        [
            {
                "label": "triton-scale-simplify-kernel-body",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nsimplify scale and masks",
            },
            {
                "label": "triton-small-tile-rewrite",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_swizzle_and_tile_schedule\nsmall tile rewrite",
            },
            {
                "label": "triton-fused-scale-dot-loop",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_scaled_dot_fusion\nfuse scale into dot",
            },
            {
                "label": "amd-gluon-l0-viability",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Extension Set task\nExtension layer: L0\nminimal gluon viability",
            },
        ]
    )

    with pytest.raises(ValueError, match="base_split_k_or_multipass_reduce"):
        _parse_llm_response(
            degraded,
            FakeAgentClass,
            required_base_families=[
                _BASE_FAMILY_SWIZZLE_TILE,
                _BASE_FAMILY_SPLIT_K,
                _BASE_FAMILY_SCALED_FUSION,
                _BASE_FAMILY_STREAMLINE,
                _BASE_FAMILY_PERSISTENT,
            ],
            expected_extension_slots=1,
        )


def test_audit_accepts_main_like_five_base_plus_gluon_l0() -> None:
    accepted = json.dumps(
        [
            {
                "label": "triton-swizzle-and-tile-optimization",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_swizzle_and_tile_schedule\noptimize tile traversal",
            },
            {
                "label": "triton-split-k-reduction",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_split_k_or_multipass_reduce\ntry split-K reduction",
            },
            {
                "label": "triton-scale-fusion-into-dot",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_scaled_dot_fusion\nfuse scale into tl.dot",
            },
            {
                "label": "triton-eliminate-redundant-ops-streamline",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nremove casts and redundant masks",
            },
            {
                "label": "triton-small-matrix-persistent-kernel",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_small_matrix_persistent_or_launch_amortization\npersistent small matrix path",
            },
            {
                "label": "amd-gluon-l0-viability",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Extension Set task\nExtension layer: L0\nminimal AMD Gluon viability",
            },
        ]
    )

    tasks = _parse_llm_response(
        accepted,
        FakeAgentClass,
        required_base_families=[
            _BASE_FAMILY_SWIZZLE_TILE,
            _BASE_FAMILY_SPLIT_K,
            _BASE_FAMILY_SCALED_FUSION,
            _BASE_FAMILY_STREAMLINE,
            _BASE_FAMILY_PERSISTENT,
        ],
        expected_extension_slots=1,
    )
    assert [task.label for task in tasks][-1] == "amd-gluon-l0-viability"


def test_parse_llm_response_infers_search_contract_for_extension() -> None:
    tasks = _parse_llm_response(
        json.dumps(
            [
                {
                    "label": "ext-l0-gluon-memory",
                    "priority": 6,
                    "agent_type": "strategy_agent",
                    "kernel_language": "python",
                    "task_prompt": "Extension Set task\nExtension layer: L0\nAttempt AMD Gluon.",
                },
                {
                    "label": "shared-anchor-transplant",
                    "priority": 5,
                    "agent_type": "strategy_agent",
                    "kernel_language": "python",
                    "task_prompt": "Shared Set task\nShared source family: base_hot_path_streamline",
                },
            ]
        ),
        FakeAgentClass,
    )

    assert tasks[0].config["search_set"] == "shared"
    assert tasks[0].config["required_output_dialect"] == "any"
    assert tasks[1].config["search_set"] == "extension"
    assert tasks[1].config["required_output_dialect"] == "amd_gluon"


def test_parse_llm_response_infers_mixed_for_hybrid_before_extension_default() -> None:
    tasks = _parse_llm_response(
        json.dumps(
            [
                {
                    "label": "ext-hybrid-shape-dispatch",
                    "priority": 6,
                    "agent_type": "strategy_agent",
                    "kernel_language": "python",
                    "task_prompt": "Extension Set task\nExtension layer: Hybrid\nComposition type: hybrid_dispatch",
                }
            ]
        ),
        FakeAgentClass,
    )

    assert tasks[0].config["search_set"] == "extension"
    assert tasks[0].config["required_output_dialect"] == "mixed"


def test_parse_llm_response_infers_memory_lowering_profile_for_l1_buffer_task() -> None:
    tasks = _parse_llm_response(
        json.dumps(
            [
                {
                    "label": "ext-l1-amd-gluon-buffer-loads",
                    "priority": 7,
                    "agent_type": "strategy_agent",
                    "kernel_language": "python",
                    "task_prompt": "Extension Set task\nExtension layer: L1\nRefine the verified Gluon anchor with buffer_load and buffer_store for KV cache memory path.",
                }
            ]
        ),
        FakeAgentClass,
    )

    cfg = tasks[0].config
    assert cfg["gluon_doc_profile"] == "memory_lowering"
    assert "gluon_component_traits_path" in cfg["required_gluon_docs"]
    assert "gluon_architecture_notes_path" in cfg["required_gluon_docs"]
    assert "gluon_api_reference_path" in cfg["required_gluon_docs"]
    assert "gluon_real_patterns_path" in cfg["required_gluon_docs"]


def test_parse_llm_response_infers_gluon_variant_from_anchor_profile() -> None:
    tasks = _parse_llm_response(
        json.dumps(
            [
                {
                    "label": "compose-gluon-variant",
                    "priority": 6,
                    "agent_type": "strategy_agent",
                    "kernel_language": "python",
                    "task_prompt": (
                        "Extension Set task\n"
                        "Composition type: gluon_variant\n"
                        "Safe anchor: round_1/base-best/patch_4\n"
                        "Source component: matrix_lowering from round_1/ext-l0/patch_2\n"
                        "Comparison target: safe_anchor\n"
                        "Allowed change: re-express safe anchor algorithm in AMD Gluon only\n"
                        "Reject if: changes launcher ABI or adds scheduler changes"
                    ),
                }
            ]
        ),
        FakeAgentClass,
    )

    cfg = tasks[0].config
    assert cfg["gluon_doc_profile"] == "gluon_variant_from_anchor"
    assert "gluon_component_traits_path" in cfg["required_gluon_docs"]
    assert "gluon_architecture_notes_path" in cfg["required_gluon_docs"]
    assert "gluon_api_reference_path" in cfg["required_gluon_docs"]
    assert "gluon_real_patterns_path" in cfg["required_gluon_docs"]


def test_parse_llm_response_infers_hybrid_dispatch_from_evidence_profile() -> None:
    tasks = _parse_llm_response(
        json.dumps(
            [
                {
                    "label": "hybrid-dispatch-from-evidence",
                    "priority": 6,
                    "agent_type": "strategy_agent",
                    "kernel_language": "python",
                    "task_prompt": (
                        "Extension Set task\n"
                        "Extension layer: Hybrid\n"
                        "Composition type: hybrid_dispatch\n"
                        "Safe anchor: round_2/base-best/patch_7\n"
                        "Source component: launch_or_dispatch_policy from per-shape evidence\n"
                        "Comparison target: safe_anchor\n"
                        "Allowed change: add host-side per-shape dispatch only\n"
                        "Reject if: removes Base path for regressed shapes"
                    ),
                }
            ]
        ),
        FakeAgentClass,
    )

    cfg = tasks[0].config
    assert cfg["gluon_doc_profile"] == "hybrid_dispatch_from_evidence"
    assert cfg["required_output_dialect"] == "mixed"
    assert "gluon_component_traits_path" in cfg["required_gluon_docs"]
    assert "gluon_architecture_notes_path" in cfg["required_gluon_docs"]
    assert "gluon_real_patterns_path" in cfg["required_gluon_docs"]


def test_extension_audit_does_not_count_shared_gluon_variant_as_extension() -> None:
    tasks = _parse_llm_response(
        json.dumps(
            [
                {
                    "label": "shared-amd-gluon-variant",
                    "priority": 4,
                    "agent_type": "strategy_agent",
                    "kernel_language": "python",
                    "task_prompt": "Shared Set task\nShared source family: base_hot_path_streamline\namd_gluon variant",
                },
                {
                    "label": "amd-gluon-l0-viability",
                    "priority": 8,
                    "agent_type": "strategy_agent",
                    "kernel_language": "python",
                    "task_prompt": "Extension Set task\nExtension layer: L0\nminimal AMD Gluon viability",
                },
            ]
        ),
        FakeAgentClass,
        expected_extension_slots=1,
    )

    assert tasks[0].config["search_set"] == "shared"
    assert tasks[1].config["search_set"] == "extension"


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
    general_kb = _write_knowledge_files(tmp_path)

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


@patch("minisweagent.tools.tools_runtime.get_tools_list", return_value=[{"name": "str_replace_editor"}, {"name": "submit"}])
@patch("minisweagent.agents.default.DefaultAgent")
def test_run_task_agent_raw_profile_keeps_single_skill_tier(
    mock_default_agent, _mock_tools, tmp_path: Path
):
    model = FakePlanningModel()
    mock_default_agent.return_value.run.return_value = ("Submitted", "[]")
    general_kb = _write_knowledge_files(tmp_path)

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


@patch("minisweagent.tools.tools_runtime.get_tools_list", return_value=[{"name": "str_replace_editor"}, {"name": "submit"}])
@patch("minisweagent.agents.default.DefaultAgent")
def test_run_task_agent_plain_triton_auto_prefers_amd_gluon_first(
    mock_default_agent, _mock_tools, tmp_path: Path
):
    model = FakePlanningModel()
    mock_default_agent.return_value.run.return_value = ("Submitted", "[]")
    general_kb = _write_knowledge_files(tmp_path)

    submitted = _run_task_agent(
        kernel_path=str(tmp_path / "kernel.py"),
        kernel_name="test_kernel",
        kernel_type="triton",
        kernel_language="python",
        function_names=["kernel_fwd"],
        workspace_path=str(tmp_path),
        input_dialect="plain_triton",
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
    assert "Preferred output dialect order: amd_gluon, plain_triton" in run_kwargs["gluon_feature_context"]
    assert "prefer_amd_gluon_if_viable_else_plain_triton" in run_kwargs["gluon_feature_context"]
    assert "after preserving the Base Set plain-Triton quota" in run_kwargs["output_dialect_guidance"]
    assert "Extension L0" in run_kwargs["output_dialect_guidance"]
    assert "Keep a plain Triton fallback path alive" in run_kwargs["output_dialect_guidance"]
    assert "## Gluon Planning Traits" in run_kwargs["gluon_planning_traits_guidance"]
    assert "`dialect_plain_triton`" in run_kwargs["gluon_planning_traits_guidance"]
    assert "Base Set (plain Triton): at least 5 task(s)" in run_kwargs["search_space_allocation_guidance"]
    assert "Extension L0: minimal AMD Gluon viability" in run_kwargs["search_space_allocation_guidance"]
    assert "Round 1 should include one L0 minimal AMD Gluon viability task" in run_kwargs["gluon_planning_traits_guidance"]
    system_prompt = mock_default_agent.call_args.kwargs["system_template"]
    assert "Knowledge lookup contract: write a Gluon knowledge lookup plan before" in system_prompt
    assert "Implementation contract: write a Gluon implementation plan before" in system_prompt
    assert "Performance hypothesis:" in system_prompt
    assert "leftover plain Triton tensor APIs" in system_prompt
    assert "required_patch_target_symbols" in system_prompt
    assert "AMD buffer ops" in system_prompt


@patch("minisweagent.tools.tools_runtime.get_tools_list", return_value=[{"name": "str_replace_editor"}, {"name": "submit"}])
@patch("minisweagent.agents.default.DefaultAgent")
def test_run_task_agent_nv_gluon_mentions_translation_before_tuning(
    mock_default_agent, _mock_tools, tmp_path: Path
):
    model = FakePlanningModel()
    mock_default_agent.return_value.run.return_value = ("Submitted", "[]")
    _write_knowledge_files(tmp_path)

    submitted = _run_task_agent(
        kernel_path=str(tmp_path / "kernel.py"),
        kernel_name="test_kernel",
        kernel_type="triton",
        kernel_language="python",
        function_names=["kernel_fwd"],
        workspace_path=str(tmp_path),
        input_dialect="nv_gluon",
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
    run_kwargs = mock_default_agent.return_value.run.call_args.kwargs
    assert "Preferred output dialect order: amd_gluon, plain_triton" in run_kwargs["gluon_feature_context"]
    assert "translate vendor-specific APIs, layouts, or memory paths" in run_kwargs["gluon_feature_context"]
    assert "translates vendor-specific APIs, layout assumptions, or memory paths" in run_kwargs["output_dialect_guidance"]


@patch("minisweagent.tools.tools_runtime.get_tools_list", return_value=[{"name": "str_replace_editor"}, {"name": "submit"}])
@patch("minisweagent.agents.default.DefaultAgent")
def test_run_task_agent_keeps_general_skill_tiers_for_mi3xx(mock_default_agent, _mock_tools, tmp_path: Path):
    model = FakePlanningModel()
    mock_default_agent.return_value.run.return_value = ("Submitted", "[]")
    general_kb = _write_knowledge_files(tmp_path)

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
    assert mock_default_agent.call_args.kwargs["allowed_skill_tiers"] == ["general"]
    run_kwargs = mock_default_agent.return_value.run.call_args.kwargs
    assert run_kwargs["knowledge_base_path"] == str(general_kb)


def test_write_task_files_marks_triton_tasks_with_skill_usage(tmp_path: Path):
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
    assert meta["preferred_output_dialects"] == ["amd_gluon", "plain_triton"]
    assert meta["output_dialect_search_policy"] == "prefer_amd_gluon_if_viable_else_plain_triton"
    assert meta["allowed_skill_tiers"] == ["general"]
    assert meta["gluon_doc_profile"] == "base_or_shared_gluon"
    assert "required_gluon_docs" in meta
    assert "gluon_skill_path" in meta["required_gluon_docs"]
    assert Path(meta["gluon_always_read_path"]).is_absolute()
    assert meta["gluon_always_read_path"].endswith("skills/triton-gluon/docs/00_always_read.md")
    assert meta["gluon_api_reference_path"].endswith("skills/triton-gluon/docs/50_api_reference.md")
    assert meta["gluon_real_patterns_path"].endswith("skills/triton-gluon/docs/60_real_patterns.md")


def test_write_task_files_records_plain_triton_gluon_preference(tmp_path: Path):
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
        input_dialect="plain_triton",
        gluon_feature_mode="auto",
        gluon_baseline_profile="raw",
        allowed_output_dialects=["plain_triton", "amd_gluon"],
        target_backend="hip/gfx942",
    )

    meta, _body = read_task_file(paths[0])
    assert meta["preferred_output_dialects"] == ["amd_gluon", "plain_triton"]
    assert meta["output_dialect_search_policy"] == "prefer_amd_gluon_if_viable_else_plain_triton"


def test_write_task_files_patches_shape_metadata_from_baseline_metrics(tmp_path: Path):
    tasks = _parse_llm_response(
        '[{"label": "opt", "priority": 5, "agent_type": "strategy_agent", "task_prompt": "Do it"}]',
        FakeAgentClass,
    )
    _write_knowledge_files(tmp_path)
    cases_path = tmp_path / "benchmark_test_cases.json"
    bm_path = tmp_path / "baseline_metrics.json"
    bm_path.write_text(
        json.dumps(
            {
                "benchmark_shape_count": 4,
                "benchmark_test_cases": [
                    {"case_id": "perf1", "params": {"M": 32, "K": 64, "N": 64}, "baseline_ms": 0.03},
                    {"case_id": "perf2", "params": {"M": 64, "K": 128, "N": 128}, "baseline_ms": 0.04},
                    {"case_id": "perf3", "params": {"M": 128, "K": 256, "N": 256}, "baseline_ms": 0.05},
                    {"case_id": "perf4", "params": {"M": 256, "K": 512, "N": 512}, "baseline_ms": 0.06},
                ],
                "benchmark_test_cases_path": str(cases_path),
                "shape_coverage_profile": "bucketed",
            }
        )
    )

    paths = write_task_files(
        tasks,
        tmp_path,
        kernel_path=str(tmp_path / "kernel.py"),
        kernel_type="triton",
        input_dialect="plain_triton",
        gluon_feature_mode="auto",
        gluon_baseline_profile="raw",
        allowed_output_dialects=["plain_triton", "amd_gluon"],
        target_backend="hip/gfx942",
        benchmark_shape_count=None,
        benchmark_test_cases=[],
        shape_coverage_profile="unknown",
        baseline_metrics=str(bm_path),
    )

    meta, _body = read_task_file(paths[0])
    assert meta["shape_coverage_profile"] == "bucketed"
    assert meta["benchmark_shape_count"] == 4
    assert [case["case_id"] for case in meta["benchmark_test_cases"]] == [
        "perf1",
        "perf2",
        "perf3",
        "perf4",
    ]
    assert meta["benchmark_test_cases_path"] == str(cases_path)


def test_write_task_files_writes_triton_search_contract_only(tmp_path: Path):
    triton_tasks = _parse_llm_response(
        json.dumps(
            [
                {
                    "label": "ext-l0-gluon-memory",
                    "priority": 6,
                    "agent_type": "strategy_agent",
                    "task_prompt": "Extension Set task\nExtension layer: L0",
                }
            ]
        ),
        FakeAgentClass,
    )
    _write_knowledge_files(tmp_path)

    triton_paths = write_task_files(
        triton_tasks,
        tmp_path / "triton",
        kernel_path=str(tmp_path / "kernel.py"),
        kernel_type="triton",
        input_dialect="plain_triton",
        gluon_feature_mode="auto",
        allowed_output_dialects=["plain_triton", "amd_gluon"],
    )
    triton_meta, _body = read_task_file(triton_paths[0])
    assert triton_meta["search_set"] == "extension"
    assert triton_meta["required_output_dialect"] == "amd_gluon"

    hip_paths = write_task_files(
        triton_tasks,
        tmp_path / "hip",
        kernel_path=str(tmp_path / "kernel.hip"),
        kernel_type="hip",
    )
    hip_meta, _body = read_task_file(hip_paths[0])
    assert "search_set" not in hip_meta
    assert "required_output_dialect" not in hip_meta


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
    assert 'If an "Output Dialect Planning Policy" block is present' in _SYSTEM_PROMPT
    assert 'If an "Evidence-Anchored Composition" block is present' in _SYSTEM_PROMPT
    assert "Composition type: base_refine | shared_transplant | gluon_variant | hybrid_dispatch" in _SYSTEM_PROMPT
    assert "leave some gpus idle" in _SYSTEM_PROMPT.lower()
    assert "Generate at least one priority-0 task that specifically checks the dispatch path" not in _SYSTEM_PROMPT
