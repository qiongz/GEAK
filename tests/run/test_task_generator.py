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
    _build_gluon_task_generation_guidance,
    _SYSTEM_PROMPT,
    _base_triton_mandatory_families,
    _build_gluon_planning_traits_guidance,
    _build_search_space_allocation_guidance,
    _build_workload_guidance,
    _gluon_extension_strength,
    _extract_kernel_meta,
    _infer_executed_route_symbols,
    _infer_gluon_planning_traits,
    _infer_required_patch_target_symbols,
    _is_gluon_extension_task,
    _is_plain_competitor_task,
    _parse_llm_response,
    _previous_gluon_signal,
    _run_task_agent,
    _task_audit_summary,
    generate_tasks,
    get_last_gluon_task_generation_diagnostics,
    write_task_files,
)
from minisweagent.agents.heterogeneous.prompts import TASKGEN_INSTANCE_TEMPLATE
from minisweagent.agents.agent_spec import AgentTask
from minisweagent.run.preprocess.discovery_types import build_gluon_feature_metadata
from minisweagent.run.gluon_doc_profiles import GLUON_DOC_PROFILE_REQUIRED_KEYS
from minisweagent.run.resource_paths import resolve_project_resource
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


def _gluon_overlay_prompt(
    layer: str = "L0",
    extra: list[str] | None = None,
    *,
    source_family: str = "base_hot_path_streamline",
    plain_competitor: str = "triton-eliminate-redundant-ops-streamline",
    target_component: str = "streamline_component",
) -> str:
    lines = [
        "Extension Set task",
        f"Extension layer: {layer}",
        "Optimization direction: memory/layout cleanup",
        f"Source Base family: {source_family}",
        f"Plain competitor: {plain_competitor}",
        "Gluon overlay reason: explicit_layout",
        "Overlay priority: Prefer",
        "Implementation layer: amd_gluon overlay",
        "Performance hypothesis: explicit layout may reduce hot-path memory/index overhead",
        "Measurement boundary: kernel_only",
        "Same ABI comparison: required",
        "Comparison target: true_baseline",
        f"Target component: {target_component}",
        f"Allowed change: one layout or memory component in `{target_component}`",
        "Minimum executable unit: inline_scoped_helper",
        "Allowed execution path: inline_scoped_helper",
        "Scope infeasible policy: shrink_or_report",
        "Reject if: non-executed Gluon or shape regression",
    ]
    if layer == "L0":
        lines.extend(
            [
                "L0 scope classification: low_coupling",
                "L0 coupling reasons: single local memory/layout component without loop-carried state",
                "Task signals: layout, memory, l0",
                "Routed doc reasons: layout signal -> 20_component_traits.md; api signal -> 50_api_reference.md",
                "Kernel family signal: generic_memory_layout",
                "Failure layers: broadcast/layout layer, memory/load-store layer",
            ]
        )
    lines.extend(extra or [])
    return "\n".join(lines)


def _existing_amd_gluon_refinement_prompt(
    *,
    task_type: str = "amd_gluon_in_dialect_refine",
    target_component: str = "load_store_buffer",
    target_symbol: str = "value_load_path",
    allowed_change: str = "optimize one load_store_buffer path only",
) -> str:
    return "\n".join(
        [
            "Existing AMD Gluon refinement task",
            "source_origin: existing_amd_gluon_operator",
            f"Task type: {task_type}",
            "Implementation layer: amd_gluon in-dialect refinement",
            "required_output_dialect: amd_gluon",
            "gluon_tl_policy: production_source_allowed",
            "layout_construction_policy: source_preserve",
            "Kernel family signal: memory_or_elementwise",
            f"Target component: {target_component}",
            f"Target symbol: {target_symbol}",
            f"Allowed change: {allowed_change} in `{target_symbol}`",
            "Failure layers: memory/load-store layer",
            "Measurement boundary: kernel_only",
            "Comparison target: true_baseline",
            "Reject if: correctness fails or any benchmark shape regresses",
        ]
    )


def test_overlay_direction_policy_lives_in_split_docs() -> None:
    repo = Path(__file__).resolve().parents[2]
    search_doc = (repo / "skills/triton-gluon/docs/10_search_policies.md").read_text()
    routing_doc = (repo / "skills/triton-gluon/docs/00_always_read.md").read_text()
    patterns_doc = (repo / "skills/triton-gluon/docs/60_real_patterns.md").read_text()
    component_doc = (repo / "skills/triton-gluon/docs/20_component_traits.md").read_text()

    assert "### Search policy: overlay_direction_vs_mechanism" in search_doc
    assert "### Search policy: existing_amd_gluon_refinement_policy" in search_doc
    assert "### Search policy: atomic_component_lattice" in search_doc
    assert "index_map" in search_doc
    assert "selection_update" in search_doc
    assert "scheduler_launch" in search_doc
    assert "Use `Optimization direction` for the shared performance or algorithmic goal" in search_doc
    assert "Base tasks are real no-regression performance candidates" in search_doc
    assert "not a synthetic placeholder" in routing_doc
    assert "Gluon-specific mechanisms" in patterns_doc
    assert "Whole-kernel L0 comparison anchor" in patterns_doc
    assert "## broadcast_heavy_whole_kernel_l0" in patterns_doc
    assert "## l0_scope_by_kernel_family" in patterns_doc
    assert "### Search policy: l0_scope_decision_before_emit" in search_doc
    assert "Branch A: local single-component smoke/probe" in search_doc
    assert "Branch B: whole-helper layout skeleton" in search_doc
    assert "Unknown or ambiguous boundary" in search_doc
    assert "l0_scope_decision_before_emit" in routing_doc
    assert "Branch A local smoke/probe L0" in patterns_doc
    assert "Branch B whole-helper skeleton L0" in patterns_doc
    assert "softmax/reduction/norm" in patterns_doc
    assert "topk/sampler/routing" in patterns_doc
    assert "overlay_direction_vs_mechanism" in routing_doc
    assert "atomic_component_lattice" in routing_doc
    assert "source contract preserved" in routing_doc
    assert "existing_amd_gluon_in_dialect_refinement" in patterns_doc
    assert "layout_parent_slice" in component_doc
    assert "matrix_operand_mfma" in component_doc


def test_slow_l0_refinement_guidance_lives_in_split_docs_without_kernel_hardcoding() -> None:
    repo = Path(__file__).resolve().parents[2]
    routing_doc = (repo / "skills/triton-gluon/docs/00_always_read.md").read_text()
    component_doc = (repo / "skills/triton-gluon/docs/20_component_traits.md").read_text()
    patterns_doc = (repo / "skills/triton-gluon/docs/60_real_patterns.md").read_text()

    assert "Slow L0 removable-overhead checklist" in patterns_doc
    for token in (
        "host_layout_construction",
        "layout_padding",
        "mask_path_overhead",
        "typed_fallback_overhead",
        "loop_invariant_overhead",
        "small_stage_launch_params",
    ):
        assert token in patterns_doc
        assert token in component_doc

    assert "Next patch decision" in routing_doc
    assert "try_one_removable_overhead:<name>" in routing_doc
    assert "stop_not_viable_for_l1" in routing_doc
    assert "Measurement boundary reconciliation" in routing_doc
    assert "Base/plain Triton tasks are not forced into this Gluon state machine" in routing_doc
    assert "Next patch decision" in patterns_doc
    assert "tiny_stage_overhead` as an umbrella label" in patterns_doc
    assert "host-side cache" in patterns_doc
    assert "num_warps=4" in patterns_doc
    assert "layout padding explicitly" in component_doc
    assert "Loop-invariant hoists are follow-up experiments" in component_doc


def test_gluon_task_generation_guidance_adds_l0_patch_evolution_without_plain_task_gate() -> None:
    guidance = _build_gluon_task_generation_guidance(
        {
            **_gluon_feature_meta("plain_triton"),
            "source_origin": "generated_overlay",
        }
    )

    assert "patch_0` establishes a real executed anchor" in guidance
    assert "try exactly one named removable overhead" in guidance
    assert "host_layout_construction" in guidance
    assert "Measurement boundary reconciliation" in guidance
    assert "Do not inject this Gluon patch-evolution ladder into `required_output_dialect=plain_triton`" in guidance
    assert "plain_subkernel_refine" in guidance


def test_existing_amd_gluon_guidance_uses_same_l0_patch_evolution_and_plain_subkernel_escape() -> None:
    guidance = _build_gluon_task_generation_guidance(
        {
            **_gluon_feature_meta("amd_gluon"),
            "source_origin": "existing_amd_gluon_operator",
        }
    )

    assert "existing-refinement tasks" in guidance
    assert "patch_0` establishes a real executed anchor" in guidance
    assert "tiny_stage_overhead` only as an umbrella label" in guidance
    assert "plain_subkernel_refine" in guidance
    assert "required_output_dialect=plain_triton" in guidance

    repo = Path(__file__).resolve().parents[2]
    routing_doc = (repo / "skills/triton-gluon/docs/00_always_read.md").read_text()
    component_doc = (repo / "skills/triton-gluon/docs/20_component_traits.md").read_text()
    patterns_doc = (repo / "skills/triton-gluon/docs/60_real_patterns.md").read_text()
    docs_text = "\n".join([routing_doc, component_doc, patterns_doc])
    for kernel_specific in ("mla_decode", "mqa_logits", "pa_decode", "arena_branch", "patch_3_test"):
        assert kernel_specific not in docs_text


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


def test_infer_required_patch_target_symbols_from_allowed_change_components() -> None:
    prompt = "\n".join(
        [
            "Extension layer: L0",
            "Allowed change: Replace the scoped local expressions (tile_load_a, tile_load_b, tile_load_c) in the inner loop.",
            "Reject if: target-symbol mismatch.",
        ]
    )

    assert _infer_required_patch_target_symbols(prompt) == ["tile_load_a", "tile_load_b", "tile_load_c"]


def test_infer_required_patch_target_symbols_from_stage_scoped_allowed_change() -> None:
    prompt = "\n".join(
        [
            "Extension layer: L0",
            "Allowed change: layout specification for tensor creation and load/store in stage1 inner loop",
            "Reject if: target-symbol mismatch.",
        ]
    )

    assert _infer_required_patch_target_symbols(prompt) == ["stage1"]


def test_infer_required_patch_target_symbols_ignores_abstract_atomic_components() -> None:
    prompt = "\n".join(
        [
            "Existing AMD Gluon refinement task",
            "Target component: state_update_softmax",
            "Allowed change: optimize one state_update_softmax path only",
        ]
    )

    assert _infer_required_patch_target_symbols(prompt) == []


def test_infer_required_patch_target_symbols_keeps_concrete_target_symbol() -> None:
    prompt = "\n".join(
        [
            "Existing AMD Gluon refinement task",
            "Target component: state_update_softmax",
            "Target symbol: paged_attention_decode_v2_gluon_dot_kernel",
            "Allowed change: adjust `attention_accumulator` only",
        ]
    )

    assert _infer_required_patch_target_symbols(prompt) == [
        "paged_attention_decode_v2_gluon_dot_kernel",
        "attention_accumulator",
    ]


def test_infer_required_patch_target_symbols_keeps_backticked_symbol_at_end_of_allowed_change() -> None:
    prompt = "\n".join(
        [
            "Task type: amd_gluon_in_dialect_refine",
            "Kernel family signal: attention_decode_kv_cache",
            "Target component: load_store_buffer",
            "Allowed change: Reorder value cache loading to overlap with QK MFMA computation in `paged_attention_decode_v2_gluon_dot_kernel`",
            "source_origin: existing_amd_gluon_operator",
            "required_output_dialect: amd_gluon",
        ]
    )

    assert _infer_required_patch_target_symbols(prompt) == [
        "paged_attention_decode_v2_gluon_dot_kernel",
    ]


def test_infer_required_patch_target_symbols_from_existing_refinement_action_prose() -> None:
    prompt = "\n".join(
        [
            "Task type: amd_gluon_in_dialect_refine",
            "Target component: reduction_accumulator",
            "Allowed change: Fuse PV MFMA accumulation directly into the scaled attention accumulator",
            "Look at `paged_attention_decode_sliding_window` for the reference pattern.",
            "Concrete changes needed in `paged_attention_decode_v2_gluon_dot_kernel`: remove the temporary accumulator.",
            "Also apply the same optimization to `paged_attention_decode_v2_gluon_large_block_dot_kernel`.",
        ]
    )

    assert _infer_required_patch_target_symbols(prompt) == [
        "paged_attention_decode_v2_gluon_dot_kernel",
        "paged_attention_decode_v2_gluon_large_block_dot_kernel",
    ]


def test_infer_required_patch_target_symbols_from_focus_on_function_instruction() -> None:
    prompt = "\n".join(
        [
            "source_origin: existing_amd_gluon_operator",
            "Task type: amd_gluon_in_dialect_refine",
            "Implementation layer: amd_gluon in-dialect refinement",
            "Target component: reduction_accumulator",
            "Allowed change: Fuse the PV MFMA zero-accumulator + add pattern.",
            "Focus on the `paged_attention_decode_v2_gluon_dot_kernel` function, specifically the PV MFMA section.",
        ]
    )

    assert _infer_required_patch_target_symbols(prompt) == [
        "paged_attention_decode_v2_gluon_dot_kernel",
    ]


def test_wrapper_shape_dispatch_splits_route_symbol_from_patch_target() -> None:
    prompt = "\n".join(
        [
            "Task type: amd_gluon_shape_dispatch_refine",
            "Target component: wrapper_shape_dispatch",
            "Target symbol: paged_attention_decode_v2_gluon_dot_kernel",
            "required_patch_target_symbols: paged_attention_decode_v2_gluon_dot_kernel, _paged_attention_decode_v2_with_dot_kernel_reshape_wrapper",
            "Allowed change: Tune KV_COMPUTE_BLOCK_SIZE and waves_per_eu in `_paged_attention_decode_v2_with_dot_kernel_reshape_wrapper`.",
        ]
    )

    assert _infer_required_patch_target_symbols(prompt) == [
        "_paged_attention_decode_v2_with_dot_kernel_reshape_wrapper",
    ]
    assert _infer_executed_route_symbols(
        prompt,
        required_patch_target_symbols=["_paged_attention_decode_v2_with_dot_kernel_reshape_wrapper"],
    ) == ["paged_attention_decode_v2_gluon_dot_kernel"]


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
    assert "use the routed Gluon matrix docs" in guidance
    assert "DotOperandLayout" not in guidance
    assert "Plain competitor" in guidance


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


def test_search_space_allocation_for_two_gpus_preserves_base_without_weak_gluon() -> None:
    guidance = _build_search_space_allocation_guidance(
        _gluon_feature_meta("plain_triton"),
        traits=["semantics_contract", "dialect_plain_triton", "layout_basic", "matrix_none"],
        num_gpus=2,
    )

    assert "## Search Space Allocation" in guidance
    assert "Plain Triton competitors: at least 3 task(s)" in guidance
    assert "AMD Gluon overlay: 0 task(s) recommended" in guidance
    assert "Keep a plain Triton competitor for every high-value direction" in guidance
    assert "Gluon overlay reason" in guidance
    assert "Performance hypothesis:" in guidance
    assert "Comparison target:" in guidance
    assert "Allowed change:" in guidance
    assert "L0 execution path choices are" in guidance
    assert "layout/broadcast micro-anchor" in guidance
    assert "attention-like composite path" in guidance
    assert "do_not_optimize_before_compile: true" in guidance
    assert "one primary component" in guidance
    assert "First-pass L0 branch rule" in guidance
    assert "default to a local single-component smoke/probe" in guidance
    assert "Final L0 overlay pair check" in guidance
    assert "Branch A signal partition" in guidance
    assert "Base/plain Triton tasks are no-regression performance candidates" in guidance
    assert "same direction has a plausible plain competitor" in guidance
    assert "Route worker docs by `gluon_doc_profile`" in guidance
    assert "Keep API-level rewrite and pass/fail patch details in the routed skills/docs" in guidance
    assert "patch_evolution_strategy" not in guidance


def test_taskgen_system_prompt_has_l0_final_submit_checklist() -> None:
    assert "Gluon contract source of truth" in _SYSTEM_PROMPT
    assert "injected policy blocks" in _SYSTEM_PROMPT
    assert "Keep L0 overlays" in _SYSTEM_PROMPT
    assert "omit or downgrade the overlay" in _SYSTEM_PROMPT
    assert "Documentation routing" in _SYSTEM_PROMPT
    assert "Plain Triton Base tasks are real no-regression" in _SYSTEM_PROMPT
    assert "synthetic anchors for Gluon" in _SYSTEM_PROMPT


def test_search_space_allocation_for_large_budget_keeps_full_base() -> None:
    guidance = _build_search_space_allocation_guidance(
        _gluon_feature_meta("plain_triton"),
        traits=["semantics_contract", "dialect_plain_triton", "layout_basic", "matrix_dot"],
        num_gpus=6,
    )

    assert "Gluon extension strength: strong" in guidance
    assert "Plain Triton competitors: at least 6 task(s)" in guidance
    assert "Paired same-direction mappings: 1 task(s)" in guidance
    assert "AMD Gluon overlay: 1 task(s)" in guidance
    assert "Round 1 plain Triton input may have at most one L0 overlay" in guidance
    assert "Gluon L0 scope classification" in guidance
    assert "local target + whole_jit_kernel" in guidance
    assert "default whole-kernel rewrite" in guidance
    assert "Target component" in guidance
    assert "same direction/component" in guidance
    assert "overlay_priority_routing" in guidance
    assert "gluon_doc_profile" in guidance
    assert "slots 2+ may be L1" not in guidance


def test_search_space_allocation_allows_l1_after_prior_gluon_win() -> None:
    guidance = _build_search_space_allocation_guidance(
        _gluon_feature_meta("plain_triton"),
        traits=["semantics_contract", "dialect_plain_triton", "layout_basic", "matrix_dot"],
        num_gpus=6,
        previous_results_text="### ext-l0-gluon\n- Best patch: patch_2 (speedup=1.12x)",
        current_round=2,
    )

    assert "Previous Gluon signal: won" in guidance
    assert "AMD Gluon overlay: 2 task(s)" in guidance
    assert "slots 2+ may be L1" in guidance


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


def test_existing_amd_gluon_operator_does_not_force_plain_base_families() -> None:
    meta = {
        **_gluon_feature_meta("amd_gluon"),
        "source_origin": "existing_amd_gluon_operator",
    }

    assert _base_triton_mandatory_families(meta, ["matrix_dot"], {"bottleneck": "latency"}) == []


def test_build_gluon_feature_metadata_infers_existing_amd_gluon_origin(tmp_path: Path) -> None:
    kernel = tmp_path / "kernel.py"
    kernel.write_text(
        "from triton.experimental import gluon\n"
        "from triton.experimental.gluon import language as gl\n"
        "@gluon.jit\n"
        "def kernel_fwd(x):\n"
        "    return x\n"
    )

    meta = build_gluon_feature_metadata(
        kernel,
        "triton",
        input_dialect="amd_gluon",
        gluon_feature_mode="auto",
    )

    assert meta["source_origin"] == "existing_amd_gluon_operator"

    plainish = tmp_path / "plainish.py"
    plainish.write_text("from triton.experimental import gluon\n# imported but no measured Gluon kernel\n")
    unknown = build_gluon_feature_metadata(
        plainish,
        "triton",
        input_dialect="amd_gluon",
        gluon_feature_mode="auto",
    )

    assert unknown["source_origin"] == "unknown"


def test_search_space_allocation_for_existing_amd_gluon_uses_refinement_taxonomy() -> None:
    guidance = _build_search_space_allocation_guidance(
        {
            **_gluon_feature_meta("amd_gluon"),
            "source_origin": "existing_amd_gluon_operator",
            "shape_coverage_profile": "multi",
        },
        traits=["semantics_contract", "dialect_amd_gluon", "layout_basic", "memory_amd_buffer", "matrix_dot"],
        num_gpus=8,
        baseline_metrics={"bottleneck": "memory", "duration_us": 180.0},
    )

    assert "AMD Gluon in-dialect refinements" in guidance
    assert "kernel_family_signal -> atomic_component_graph" in guidance
    assert "amd_gluon_in_dialect_refine" in guidance
    assert "plain_subkernel_refine" in guidance
    assert "Plain Triton competitors: at least" not in guidance
    assert "Existing AMD Gluon production operator anchors" in guidance
    assert "Existing AMD Gluon lightweight planning recommendation" in guidance
    assert "This recommendation is prompt guidance only" in guidance
    assert "do not add new task metadata such as `gluon_depth`" in guidance
    assert "Larger GPU budget does not imply maximum width" in guidance
    assert "Gluon-first portfolio" in guidance
    assert "prefer fewer tasks" in guidance
    assert "do not emit generic plain/Base tasks to fill budget" in guidance


def test_search_space_allocation_lists_base_family_checklist() -> None:
    guidance = _build_search_space_allocation_guidance(
        {**_gluon_feature_meta("plain_triton"), "shape_coverage_profile": "bucketed"},
        traits=["semantics_contract", "dialect_plain_triton", "layout_basic", "matrix_scaled_dot"],
        num_gpus=1,
        baseline_metrics={"bottleneck": "latency", "duration_us": 147.3},
    )

    assert "Mandatory plain Triton family checklist" in guidance
    assert "`base_swizzle_and_tile_schedule`" in guidance
    assert "`base_split_k_or_multipass_reduce`" in guidance
    assert "`base_scaled_dot_fusion`" in guidance
    assert "`base_hot_path_streamline`" in guidance
    assert "`base_small_matrix_persistent_or_launch_amortization`" in guidance
    assert "Base family: <family_id>" in guidance
    assert "Gluon overlay reasons by family" in guidance


def test_round1_allocation_preserves_plain_base_families_with_gluon_overlay() -> None:
    guidance = _build_search_space_allocation_guidance(
        {**_gluon_feature_meta("plain_triton"), "shape_coverage_profile": "bucketed"},
        traits=["semantics_contract", "dialect_plain_triton", "layout_basic", "memory_amd_buffer", "matrix_scaled_dot"],
        num_gpus=8,
        baseline_metrics={"bottleneck": "memory", "duration_us": 220.0},
        current_round=1,
    )

    assert "Plain Triton competitors: at least" in guidance
    assert "AMD Gluon overlay: 1 task(s)" in guidance
    assert "`base_swizzle_and_tile_schedule`" in guidance
    assert "`base_split_k_or_multipass_reduce`" in guidance
    assert "`base_scaled_dot_fusion`" in guidance
    assert "`base_hot_path_streamline`" in guidance
    assert "register-pressure reduction" in guidance
    assert "Round 1 plain Triton input may have at most one L0 overlay" in guidance
    assert "must not ask the worker to convert an entire stage/helper/kernel" in guidance


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
    assert "AMD Gluon overlay: 1 task(s)" in guidance
    assert "L0" in guidance
    assert "layout-only, translation-only, or memory-only" in guidance


def test_gluon_extension_strength_and_previous_signal_helpers() -> None:
    assert _gluon_extension_strength(["semantics_contract", "matrix_dot"]) == "strong"
    assert _gluon_extension_strength(["semantics_contract", "layout_slice_broadcast"]) == "normal"
    assert _gluon_extension_strength(["semantics_contract", "matrix_none"]) == "weak"
    assert _previous_gluon_signal("gluon correctness failed") == "failed"
    assert _previous_gluon_signal("gluon slower performance regression") == "slower"
    assert _previous_gluon_signal("gluon [BEST] verified_speedup=1.2x") == "won"
    assert _previous_gluon_signal("gluon [BEST] verified_speedup=1.0067x") == "attempted"


def test_audit_repairs_missing_split_k_or_persistent_base_family() -> None:
    degraded = json.dumps(
        [
            {
                "label": "triton-eliminate-redundant-ops-streamline",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nOptimization direction: memory/layout cleanup\nTarget component: streamline_component\nAllowed change: simplify `streamline_component`.\nsimplify scale and masks",
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
                "task_prompt": _gluon_overlay_prompt(),
            },
        ]
    )

    tasks = _parse_llm_response(
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

    labels = {task.label for task in tasks}
    assert "triton-split-k-or-multipass-reduce" in labels
    assert "triton-small-matrix-persistent-or-launch-amortization" in labels


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
                "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nOptimization direction: memory/layout cleanup\nTarget component: streamline_component\nAllowed change: simplify `streamline_component`.\nremove casts and redundant masks",
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
                "task_prompt": _gluon_overlay_prompt(),
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


def test_audit_accepts_multiple_existing_amd_gluon_refinements_without_extension_cap() -> None:
    feature_meta = {
        **_gluon_feature_meta("amd_gluon"),
        "source_origin": "existing_amd_gluon_operator",
    }
    payload = json.dumps(
        [
            {
                "label": "gluon-refine-load-store",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": _existing_amd_gluon_refinement_prompt(),
            },
            {
                "label": "gluon-refine-mfma-operand",
                "priority": 1,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": _existing_amd_gluon_refinement_prompt(
                    task_type="amd_gluon_layout_or_matrix_refine",
                    target_component="matrix_operand_mfma",
                    allowed_change="adjust one matrix_operand_mfma boundary only",
                ),
            },
        ]
    )

    tasks = _parse_llm_response(
        payload,
        FakeAgentClass,
        expected_extension_slots=0,
        feature_meta=feature_meta,
    )

    assert len(tasks) == 2
    assert all(task.config["source_origin"] == "existing_amd_gluon_operator" for task in tasks)
    assert all(not _is_gluon_extension_task(task) for task in tasks)


def test_existing_amd_gluon_feature_meta_fills_refinement_defaults() -> None:
    feature_meta = {
        **_gluon_feature_meta("amd_gluon"),
        "source_origin": "existing_amd_gluon_operator",
    }
    payload = json.dumps(
        [
            {
                "label": "gluon-refine-output-store",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "\n".join(
                    [
                        "Existing AMD Gluon refinement task",
                        "required_output_dialect: amd_gluon",
                        "Kernel family signal: memory_or_elementwise",
                        "Target component: epilogue_output_store",
                        "Target symbol: output_store_path",
                        "Allowed change: optimize one epilogue_output_store expression in `output_store_path` only",
                        "Failure layers: output store layer",
                        "Measurement boundary: kernel_only",
                        "Comparison target: true_baseline",
                        "Reject if: correctness fails or any benchmark shape regresses",
                    ]
                ),
            },
        ]
    )

    tasks = _parse_llm_response(
        payload,
        FakeAgentClass,
        expected_extension_slots=0,
        feature_meta=feature_meta,
    )
    cfg = tasks[0].config

    assert cfg["source_origin"] == "existing_amd_gluon_operator"
    assert cfg["gluon_tl_policy"] == "production_source_allowed"
    assert cfg["layout_construction_policy"] == "source_preserve"
    assert cfg["implementation_layer"] == "amd_gluon in-dialect refinement"
    assert cfg["task_type"] == "amd_gluon_in_dialect_refine"


def test_existing_amd_gluon_refinement_requires_concrete_patch_target() -> None:
    feature_meta = {
        **_gluon_feature_meta("amd_gluon"),
        "source_origin": "existing_amd_gluon_operator",
    }
    payload = json.dumps(
        [
            {
                "label": "gluon-softmax-abstract-target",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": _existing_amd_gluon_refinement_prompt(
                    target_component="state_update_softmax",
                    target_symbol="",
                    allowed_change="optimize one state_update_softmax path only",
                ).replace("Target symbol: \n", ""),
            },
        ]
    )

    with pytest.raises(ValueError, match="missing concrete patch target"):
        _parse_llm_response(
            payload,
            FakeAgentClass,
            expected_extension_slots=0,
            feature_meta=feature_meta,
        )


def test_existing_amd_gluon_portfolio_drops_generic_plain_overflow_force_l0_anchor() -> None:
    feature_meta = {
        **_gluon_feature_meta("amd_gluon"),
        "source_origin": "existing_amd_gluon_operator",
    }
    plain_tasks = [
        {
            "label": f"triton-plain-{idx}",
            "priority": 0,
            "agent_type": "strategy_agent",
            "kernel_language": "python",
            "task_prompt": "\n".join(
                [
                    "Plain Triton comparison task",
                    "Implementation layer: plain_triton",
                    "required_output_dialect: plain_triton",
                    "Base family: base_hot_path_streamline",
                    f"Optimization direction: plain comparison {idx}",
                    "Allowed change: one plain Triton comparison only",
                ]
            ),
        }
        for idx in range(4)
    ]
    payload = json.dumps(
        [
            *plain_tasks,
            {
                "label": "gluon-refine-load-store",
                "priority": 5,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": _existing_amd_gluon_refinement_prompt(),
            },
        ]
    )

    tasks = _parse_llm_response(
        payload,
        FakeAgentClass,
        expected_extension_slots=0,
        gluon_feature_mode="force_l0_anchor",
        feature_meta=feature_meta,
    )

    assert [task.label for task in tasks] == ["gluon-refine-load-store"]
    diagnostics = get_last_gluon_task_generation_diagnostics()
    dropped = diagnostics["dropped_existing_amd_plain_overflow"]
    assert len(dropped) == 4
    assert {item["label"] for item in dropped} == {f"triton-plain-{idx}" for idx in range(4)}


def test_existing_amd_gluon_portfolio_accepts_gluon_first_with_typed_optional_plain() -> None:
    feature_meta = {
        **_gluon_feature_meta("amd_gluon"),
        "source_origin": "existing_amd_gluon_operator",
    }
    payload = json.dumps(
        [
            {
                "label": "gluon-refine-load",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": _existing_amd_gluon_refinement_prompt(target_symbol="value_load_path"),
            },
            {
                "label": "gluon-refine-store",
                "priority": 1,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": _existing_amd_gluon_refinement_prompt(
                    target_component="epilogue_output_store",
                    target_symbol="output_store_path",
                    allowed_change="optimize one epilogue_output_store path only",
                ),
            },
            {
                "label": "plain-reduce-subkernel",
                "priority": 2,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "\n".join(
                    [
                        "Task type: plain_subkernel_refine",
                        "required_output_dialect: plain_triton",
                        "Implementation layer: plain_triton",
                        "Base family: base_split_k_or_multipass_reduce",
                        "Optimization direction: refine a plain reduction helper inside the production operator",
                        "Target component: reduction_accumulator",
                        "Target symbol: reduce_kernel",
                        "Allowed change: optimize `reduce_kernel` only",
                    ]
                ),
            },
            {
                "label": "shared-safe-anchor-comparison",
                "priority": 3,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "\n".join(
                    [
                        "Task type: shared_or_plain_comparison",
                        "required_output_dialect: plain_triton",
                        "Implementation layer: plain_triton",
                        "Shared source family: base_hot_path_streamline",
                        "Optimization direction: portable safe-anchor comparison for the existing AMD Gluon path",
                        "Target component: source_contract_integration",
                        "Target symbol: safe_anchor_path",
                        "Allowed change: compare `safe_anchor_path` only",
                    ]
                ),
            },
        ]
    )

    tasks = _parse_llm_response(
        payload,
        FakeAgentClass,
        expected_extension_slots=0,
        gluon_feature_mode="force_l0_anchor",
        feature_meta=feature_meta,
    )

    assert [task.label for task in tasks] == [
        "gluon-refine-load",
        "gluon-refine-store",
        "plain-reduce-subkernel",
        "shared-safe-anchor-comparison",
    ]
    assert get_last_gluon_task_generation_diagnostics() == {}


def test_existing_amd_gluon_allows_single_refinement_with_typed_plain_subkernel() -> None:
    feature_meta = {
        **_gluon_feature_meta("amd_gluon"),
        "source_origin": "existing_amd_gluon_operator",
    }
    payload = json.dumps(
        [
            {
                "label": "gluon-refine-load",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": _existing_amd_gluon_refinement_prompt(target_symbol="value_load_path"),
            },
            {
                "label": "plain-reduce-subkernel",
                "priority": 1,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "\n".join(
                    [
                        "Task type: plain_subkernel_refine",
                        "required_output_dialect: plain_triton",
                        "Implementation layer: plain_triton",
                        "Base family: base_split_k_or_multipass_reduce",
                        "Optimization direction: refine a plain reduction helper inside the production operator",
                        "Target component: reduction_accumulator",
                        "Target symbol: reduce_kernel",
                        "Allowed change: optimize `reduce_kernel` only",
                    ]
                ),
            },
        ]
    )

    tasks = _parse_llm_response(
        payload,
        FakeAgentClass,
        expected_extension_slots=0,
        gluon_feature_mode="force_l0_anchor",
        feature_meta=feature_meta,
    )

    assert [task.label for task in tasks] == ["gluon-refine-load", "plain-reduce-subkernel"]


def test_existing_amd_gluon_portfolio_rules_do_not_affect_plain_triton_tasks() -> None:
    payload = json.dumps(
        [
            {
                "label": f"triton-plain-{idx}",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "\n".join(
                    [
                        "Base Set task",
                        "Implementation layer: plain_triton",
                        "required_output_dialect: plain_triton",
                        "Base family: base_hot_path_streamline",
                        f"Optimization direction: plain Triton direction {idx}",
                        "Allowed change: one plain Triton optimization only",
                    ]
                ),
            }
            for idx in range(4)
        ]
    )

    tasks = _parse_llm_response(
        payload,
        FakeAgentClass,
        expected_extension_slots=0,
        gluon_feature_mode="force_l0_anchor",
        feature_meta=_gluon_feature_meta("plain_triton"),
    )

    assert len(tasks) == 4
    assert all(_is_plain_competitor_task(task) for task in tasks)


def test_generated_overlay_stays_strict_even_with_existing_amd_gluon_feature_meta() -> None:
    feature_meta = {
        **_gluon_feature_meta("amd_gluon"),
        "source_origin": "existing_amd_gluon_operator",
    }
    payload = json.dumps(
        [
            {
                "label": "generated-overlay",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": _gluon_overlay_prompt(extra=["source_origin: generated_overlay"]),
            },
        ]
    )

    with pytest.raises(ValueError, match="expected at most 0 AMD Gluon Extension"):
        _parse_llm_response(
            payload,
            FakeAgentClass,
            expected_extension_slots=0,
            feature_meta=feature_meta,
        )


def test_plain_subkernel_refine_is_not_gluon_extension_for_existing_source() -> None:
    payload = json.dumps(
        [
            {
                "label": "plain-reduce-subkernel",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "\n".join(
                    [
                        "Existing AMD Gluon source contains a plain helper.",
                        "source_origin: existing_amd_gluon_operator",
                        "Task type: plain_subkernel_refine",
                        "Implementation layer: plain_triton",
                        "required_output_dialect: plain_triton",
                        "Base family: base_split_k_or_multipass_reduce",
                        "Optimization direction: refine the plain reduction helper",
                        "Target component: reduction_accumulator",
                        "Allowed change: optimize one reduction_accumulator path only",
                    ]
                ),
            },
        ]
    )

    tasks = _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=0)

    assert tasks[0].config["required_output_dialect"] == "plain_triton"
    assert _is_plain_competitor_task(tasks[0])
    assert not _is_gluon_extension_task(tasks[0])


def test_audit_rejects_l0_overlay_missing_execution_boundary() -> None:
    prompt = "\n".join(
        line
        for line in _gluon_overlay_prompt().splitlines()
        if not line.startswith(("Minimum executable unit:", "Allowed execution path:", "Scope infeasible policy:"))
    )
    payload = json.dumps(
        [
            {
                "label": "triton-eliminate-redundant-ops-streamline",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nOptimization direction: memory/layout cleanup\nTarget component: streamline_component\nAllowed change: simplify `streamline_component`.\nstreamline plain Triton path",
            },
            {
                "label": "amd-gluon-l0-viability",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": prompt,
            },
        ]
    )

    tasks = _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1)
    assert [task.label for task in tasks][-1] == "amd-gluon-l0-viability"


def test_audit_failure_writes_raw_and_parsed_task_diagnostics(tmp_path: Path) -> None:
    prompt = "\n".join(
        line
        for line in _gluon_overlay_prompt(plain_competitor="shared-paired-1d-acc-and-mask").splitlines()
        if not line.startswith(("Minimum executable unit:", "Allowed execution path:", "Scope infeasible policy:"))
    )
    payload = json.dumps(
        [
            {
                "label": "shared-paired-1d-acc-and-mask",
                "priority": 5,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "\n".join(
                    [
                        "Shared Set task",
                        "Shared source family: base_hot_path_streamline",
                        "Implementation layer: paired comparison",
                        "Optimization direction: mask/accumulator cleanup",
                    ]
                ),
            },
            {
                "label": "gluon-l0-load-store-layout",
                "priority": 6,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": prompt,
            },
        ]
    )

    with pytest.raises(ValueError, match="not a plain Triton competitor task"):
        _parse_llm_response(
            payload,
            FakeAgentClass,
            expected_extension_slots=1,
            audit_diagnostics_dir=tmp_path,
        )

    dumps = list(tmp_path.glob("task_generation_audit_failed_*.json"))
    assert len(dumps) == 1
    data = json.loads(dumps[0].read_text())
    assert "gluon-l0-load-store-layout" in data["raw_submitted_json"]
    assert "Plain competitor `shared-paired-1d-acc-and-mask` is not a plain Triton competitor task" in data["error"]
    assert any("required_output_dialect=plain_triton" in hint for hint in data["repair_hints"])
    summaries = {item["label"]: item for item in data["parsed_task_summaries"]}
    assert summaries["gluon-l0-load-store-layout"]["plain_competitor"] == "shared-paired-1d-acc-and-mask"
    assert summaries["gluon-l0-load-store-layout"]["minimum_executable_unit"] == ""
    assert summaries["gluon-l0-load-store-layout"]["allowed_execution_path"] == ""
    assert summaries["shared-paired-1d-acc-and-mask"]["search_set"] == "shared"


def test_audit_rejects_l0_overlay_that_allows_and_forbids_whole_kernel() -> None:
    prompt = _gluon_overlay_prompt(
        extra=[
            "Target component: scale_epilogue_load_broadcast",
            "Whole kernel required reason: scale epilogue cannot execute independently",
            "Do NOT attempt to convert the entire kernel to Gluon.",
        ],
        target_component="scale_epilogue_load_broadcast",
    ).replace("Minimum executable unit: inline_scoped_helper", "Minimum executable unit: whole_jit_kernel").replace(
        "Allowed execution path: inline_scoped_helper", "Allowed execution path: whole_jit_kernel"
    )
    payload = json.dumps(
        [
            {
                "label": "triton-eliminate-redundant-ops-streamline",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nOptimization direction: memory/layout cleanup\nTarget component: scale_epilogue_load_broadcast\nAllowed change: simplify `scale_epilogue_load_broadcast`.\nstreamline plain Triton path",
            },
            {
                "label": "amd-gluon-l0-viability",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": prompt,
            },
        ]
    )

    with pytest.raises(ValueError, match="allows whole_jit_kernel but forbids whole-kernel rewrite"):
        _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1)


def test_audit_accepts_whole_kernel_anchor_when_declared() -> None:
    prompt = _gluon_overlay_prompt(
        extra=[
            "Target component: scale_epilogue_load_broadcast",
            "Whole kernel required reason: scale epilogue cannot feed measured output as an isolated subpath",
                "Expected failure layers: broadcast/layout layer, memory/load-store layer",
                "First patch compile goal: compile a minimal whole-kernel anchor before tuning",
                "Do not optimize before compile: true",
                "Matrix lowering required: false",
            "Reject if: MFMA or buffer_load is introduced.",
        ],
        target_component="scale_epilogue_load_broadcast",
    ).replace("Minimum executable unit: inline_scoped_helper", "Minimum executable unit: whole_jit_kernel").replace(
        "Allowed execution path: inline_scoped_helper", "Allowed execution path: whole_jit_kernel"
    )
    payload = json.dumps(
        [
            {
                "label": "triton-eliminate-redundant-ops-streamline",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nOptimization direction: memory/layout cleanup\nTarget component: scale_epilogue_load_broadcast\nAllowed change: simplify `scale_epilogue_load_broadcast`.\nstreamline plain Triton path",
            },
            {
                "label": "amd-gluon-l0-viability",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": prompt,
            },
        ]
    )

    tasks = _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1)

    assert tasks[-1].config["minimum_executable_unit"] == "whole_jit_kernel"
    assert tasks[-1].config["allowed_execution_path"] == "whole_jit_kernel"
    assert tasks[-1].config["whole_kernel_required_reason"].startswith("scale epilogue")


def test_audit_accepts_whole_kernel_anchor_with_do_not_emit_fallback_policy() -> None:
    prompt = (
        _gluon_overlay_prompt(
            extra=[
                "Target component: _fwd_kernel_stage2 1D load/store and accumulator layout",
                "Whole kernel required reason: _fwd_kernel_stage2 is a self-contained reduction kernel and the smallest viable Gluon target component",
                    "Expected failure layers: reduction/accumulator layer, memory/load-store layer",
                    "First patch compile goal: compile the reduction kernel anchor before tuning",
                    "Do not optimize before compile: true",
                    "Matrix lowering required: false",
                "Reject if: non-executed Gluon, target-symbol mismatch, or leftover plain Triton device APIs.",
            ],
            plain_competitor="triton-stage2-load-store-layout",
            target_component="_fwd_kernel_stage2",
        )
        .replace("Minimum executable unit: inline_scoped_helper", "Minimum executable unit: whole_jit_kernel")
        .replace("Allowed execution path: inline_scoped_helper", "Allowed execution path: whole_jit_kernel")
        .replace("Scope infeasible policy: shrink_or_report", "Scope infeasible policy: do_not_emit")
    )
    payload = json.dumps(
        [
            {
                "label": "triton-stage2-load-store-layout",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nOptimization direction: memory/layout cleanup\nTarget component: _fwd_kernel_stage2\nAllowed change: simplify `_fwd_kernel_stage2` loads in plain Triton.",
            },
            {
                "label": "gluon-l0-load-store-layout",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": prompt,
            },
        ]
    )

    tasks = _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1)

    assert tasks[-1].config["minimum_executable_unit"] == "whole_jit_kernel"
    assert tasks[-1].config["allowed_execution_path"] == "whole_jit_kernel"
    assert tasks[-1].config["scope_infeasible_policy"] == "do_not_emit"
    assert tasks[-1].config["required_patch_target_symbols"] == ["_fwd_kernel_stage2"]


def test_parse_l0_metadata_from_snake_case_prompt_fields() -> None:
    prompt = "\n".join(
        [
            "Shape coverage: shape_robust",
            "required_output_dialect: amd_gluon",
            "search_set: Extension",
            "source_origin: generated_overlay",
            "gluon_tl_policy: strict_generated",
            "layout_construction_policy: host_preferred",
            "l0_scope_classification: low_coupling",
            "l0_coupling_reasons: selected load path only",
            "task_signals: layout, memory, l0",
            "routed_doc_reasons: layout signal -> component traits; api signal -> api reference",
            "kernel_family_signal: generic_memory_layout",
            "failure_layers: broadcast/layout layer, memory/load-store layer",
            "minimum_executable_unit: separate_gluon_kernel",
            "allowed_execution_path: separate_gluon_kernel",
            "scope_infeasible_policy: shrink_or_report",
            "",
            "Extension layer: L0",
            "Optimization direction: reduce register pressure for selected load path",
            "Source Base family: base_hot_path_streamline",
            "Plain competitor: base-selected-load-path",
            "Gluon overlay reason: explicit_layout",
            "Overlay priority: high-confidence Consider",
            "Implementation layer: amd_gluon overlay",
            "Performance hypothesis: explicit layout may reduce register pressure",
            "Measurement boundary: kernel_only",
            "Comparison target: true_baseline",
            "Target component: selected_load_path",
            "Allowed change: convert `selected_load_path` only.",
            "Reject if: non-executed Gluon or shape regression",
        ]
    )
    payload = json.dumps(
        [
            {
                "label": "base-selected-load-path",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "\n".join(
                    [
                        "Base Set task",
                        "Base family: base_hot_path_streamline",
                        "Optimization direction: reduce register pressure for selected load path",
                        "Target component: selected_load_path",
                        "Allowed change: optimize `selected_load_path` in plain Triton.",
                    ]
                ),
            },
            {
                "label": "gluon-l0-layout-load-store",
                "priority": 6,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": prompt,
            },
        ]
    )

    tasks = _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1)
    cfg = tasks[-1].config

    assert cfg["required_output_dialect"] == "amd_gluon"
    assert cfg["search_set"] == "extension"
    assert cfg["source_origin"] == "generated_overlay"
    assert cfg["gluon_tl_policy"] == "strict_generated"
    assert cfg["layout_construction_policy"] == "host_preferred"
    assert cfg["minimum_executable_unit"] == "separate_gluon_kernel"
    assert cfg["allowed_execution_path"] == "separate_gluon_kernel"
    assert cfg["scope_infeasible_policy"] == "shrink_or_report"


def test_audit_warns_l0_overlay_when_plain_competitor_lacks_auditable_component(tmp_path: Path) -> None:
    prompt = (
        _gluon_overlay_prompt(
            plain_competitor="precompute-kv-pointers",
            target_component="_fwd_kernel_stage2",
        )
        .replace("Minimum executable unit: inline_scoped_helper", "Minimum executable unit: whole_jit_kernel")
        .replace("Allowed execution path: inline_scoped_helper", "Allowed execution path: whole_jit_kernel")
        .replace("Scope infeasible policy: shrink_or_report", "Scope infeasible policy: do_not_emit")
    )
    payload = json.dumps(
        [
            {
                "label": "precompute-kv-pointers",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nOptimization direction: memory/layout cleanup\nprecompute KV pointers in stage1.",
            },
            {
                "label": "gluon-l0-load-store-layout",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": prompt,
            },
        ]
    )

    tasks = _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1, audit_diagnostics_dir=tmp_path)
    assert [task.label for task in tasks][-1] == "gluon-l0-load-store-layout"
    assert not list(tmp_path.glob("task_generation_audit_failed_*.json"))


def test_audit_warns_l0_overlay_when_optimization_direction_differs_from_plain_competitor() -> None:
    prompt = _gluon_overlay_prompt(plain_competitor="triton-stage2-load-store-layout", target_component="_fwd_kernel_stage2")
    payload = json.dumps(
        [
            {
                "label": "triton-stage2-load-store-layout",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nOptimization direction: precompute KV pointers\nTarget component: _fwd_kernel_stage2\nAllowed change: simplify `_fwd_kernel_stage2` loads in plain Triton.",
            },
            {
                "label": "gluon-l0-load-store-layout",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": prompt,
            },
        ]
    )

    tasks = _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1)
    assert [task.label for task in tasks][-1] == "gluon-l0-load-store-layout"


def test_audit_warning_allows_direction_mismatch_without_dump(tmp_path: Path) -> None:
    prompt = _gluon_overlay_prompt(
        plain_competitor="precompute-rope-outside-loop",
        target_component="_fwd_kernel_stage2",
    ).replace(
        "Optimization direction: memory/layout cleanup",
        "Optimization direction: Explicit layout control for stage2 reduction kernel memory access",
    )
    payload = json.dumps(
        [
            {
                "label": "precompute-rope-outside-loop",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "\n".join(
                    [
                        "Base Set task",
                        "Base family: base_hot_path_streamline",
                        "Optimization direction: Restructure RoPE application and inner loop to minimize conditional branches and redundant work",
                        "Target component: rope_inner_loop",
                        "Allowed change: simplify `rope_inner_loop` in plain Triton.",
                    ]
                ),
            },
            {
                "label": "gluon-l0-stage2-layout-overlay",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": prompt,
            },
        ]
    )

    tasks = _parse_llm_response(
        payload,
        FakeAgentClass,
        expected_extension_slots=1,
        audit_diagnostics_dir=tmp_path,
    )
    assert [task.label for task in tasks][-1] == "gluon-l0-stage2-layout-overlay"
    assert not list(tmp_path.glob("task_generation_audit_failed_*.json"))


def test_audit_warning_combines_direction_mismatch_with_l0_branch_choice(tmp_path: Path) -> None:
    payload = json.dumps(
        [
            {
                "label": "inner-loop-memory-reorder",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "plain_triton",
                "task_prompt": "\n".join(
                    [
                        "Base Set task",
                        "Base family: base_hot_path_streamline",
                        "Implementation layer: plain_triton",
                        "Optimization direction: restructure inner loop memory access pattern to improve data reuse and reduce global memory transactions",
                        "Target component: stage1 inner loop",
                        "Allowed change: restructure `stage1 inner loop` memory access in plain Triton.",
                    ]
                ),
            },
            {
                "label": "gluon-l0-stage1-load-layout",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "amd_gluon",
                "implementation_layer": "amd_gluon overlay",
                "extension_layer": "L0",
                "source_base_family": "base_hot_path_streamline",
                "plain_competitor": "inner-loop-memory-reorder",
                "minimum_executable_unit": "whole_jit_kernel",
                "allowed_execution_path": "whole_jit_kernel",
                "scope_infeasible_policy": "shrink_or_report",
                "task_signals": "load_store, layout_broadcast",
                "failure_layers": "layout_construction, load_store",
                "expected_failure_layers": "layout_construction, load_store",
                "kernel_family_signal": "attention_decode",
                "first_patch_compile_goal": "compile and execute layout skeleton",
                "do_not_optimize_before_compile": True,
                "matrix_lowering_required": False,
                "task_prompt": "\n".join(
                    [
                        "Extension Set task",
                        "Extension layer: L0",
                        "Optimization direction: reduce K_Buffer memory transactions in stage1 inner loop",
                        "Source Base family: base_hot_path_streamline",
                        "Plain competitor: inner-loop-memory-reorder",
                        "Gluon overlay reason: explicit_layout",
                        "Overlay priority: high-confidence Consider",
                        "Implementation layer: amd_gluon overlay",
                        "Performance hypothesis: explicit layout may reduce K_Buffer memory transactions",
                        "Measurement boundary: kernel_only",
                        "Comparison target: true_baseline",
                        "Target component: K_Buffer load path",
                        "Allowed change: convert one `K_Buffer` load path only",
                        "Reject if: non-executed Gluon or whole-kernel rewrite",
                        "L0 scope classification: high_coupling",
                        "L0 coupling reasons: stage1 K_Buffer load path may need whole helper wiring",
                        "Minimum executable unit: whole_jit_kernel",
                        "Allowed execution path: whole_jit_kernel",
                        "Scope infeasible policy: shrink_or_report",
                        "Task signals: load_store, layout_broadcast",
                        "Failure layers: layout_construction, load_store",
                        "Expected failure layers: layout_construction, load_store",
                        "Kernel family signal: attention_decode",
                        "First patch compile goal: compile and execute layout skeleton",
                        "Do not optimize before compile: true",
                        "Matrix lowering required: false",
                    ]
                ),
            },
        ]
    )

    tasks = _parse_llm_response(
        payload,
        FakeAgentClass,
        expected_extension_slots=1,
        audit_diagnostics_dir=tmp_path,
    )
    assert [task.label for task in tasks][-1] == "gluon-l0-stage1-load-layout"
    assert not list(tmp_path.glob("task_generation_audit_failed_*.json"))


def test_audit_warning_combines_missing_anchor_with_l0_branch_choice(tmp_path: Path) -> None:
    payload = json.dumps(
        [
            {
                "label": "broad-memory-cleanup",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "plain_triton",
                "task_prompt": "\n".join(
                    [
                        "Base Set task",
                        "Base family: base_hot_path_streamline",
                        "Implementation layer: plain_triton",
                        "Optimization direction: reduce repeated memory transactions on the hot path",
                        "This broad Base task discusses a whole-loop memory cleanup but omits an auditable target.",
                    ]
                ),
            },
            {
                "label": "gluon-l0-local-load-layout",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "amd_gluon",
                "implementation_layer": "amd_gluon overlay",
                "extension_layer": "L0",
                "source_base_family": "base_hot_path_streamline",
                "plain_competitor": "broad-memory-cleanup",
                "minimum_executable_unit": "whole_jit_kernel",
                "allowed_execution_path": "whole_jit_kernel",
                "scope_infeasible_policy": "shrink_or_report",
                "task_signals": "load_store, matrix_operand",
                "failure_layers": "layout_construction",
                "expected_failure_layers": "layout_construction",
                "kernel_family_signal": "generic_memory_layout",
                "whole_kernel_required_reason": "the local load path feeds a matrix-like downstream operation",
                "first_patch_compile_goal": "compile and execute layout skeleton",
                "do_not_optimize_before_compile": True,
                "matrix_lowering_required": False,
                "task_prompt": "\n".join(
                    [
                        "Extension Set task",
                        "Extension layer: L0",
                        "Optimization direction: reduce repeated memory transactions on the hot path",
                        "Source Base family: base_hot_path_streamline",
                        "Plain competitor: broad-memory-cleanup",
                        "Gluon overlay reason: explicit_layout",
                        "Overlay priority: high-confidence Consider",
                        "Implementation layer: amd_gluon overlay",
                        "Performance hypothesis: explicit layout may reduce local memory transactions",
                        "Measurement boundary: kernel_only",
                        "Comparison target: true_baseline",
                        "Target component: local load path",
                        "Allowed change: convert one local load path only",
                        "Reject if: non-executed Gluon or whole-kernel rewrite",
                        "L0 scope classification: high_coupling",
                        "L0 coupling reasons: local load path feeds a matrix-like downstream operation",
                        "Minimum executable unit: whole_jit_kernel",
                        "Allowed execution path: whole_jit_kernel",
                        "Scope infeasible policy: shrink_or_report",
                        "Whole kernel required reason: the local load path feeds a matrix-like downstream operation",
                        "Task signals: load_store, matrix_operand",
                        "Failure layers: layout_construction",
                        "Expected failure layers: layout_construction",
                        "Kernel family signal: generic_memory_layout",
                        "First patch compile goal: compile and execute layout skeleton",
                        "Do not optimize before compile: true",
                        "Matrix lowering required: false",
                    ]
                ),
            },
        ]
    )

    tasks = _parse_llm_response(
        payload,
        FakeAgentClass,
        expected_extension_slots=1,
        audit_diagnostics_dir=tmp_path,
    )
    summary = {task.label: _task_audit_summary(task) for task in tasks}
    assert "local_target_promoted_to_whole_kernel" in summary["gluon-l0-local-load-layout"]["soft_audit_diagnostics"]
    assert not list(tmp_path.glob("task_generation_audit_failed_*.json"))


def test_audit_failure_hint_combines_missing_anchor_with_high_coupling_inline(tmp_path: Path) -> None:
    payload = json.dumps(
        [
            {
                "label": "fuse-k-buffer-loads",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "plain_triton",
                "task_prompt": "\n".join(
                    [
                        "Base Set task",
                        "Base family: base_hot_path_streamline",
                        "Implementation layer: plain_triton",
                        "Optimization direction: reduce KBuffer load transactions before dot product",
                        "Broadly fuse KBuffer loads in the inner loop without a scoped target.",
                    ]
                ),
            },
            {
                "label": "gluon-l0-kbuffer-load-layout",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "amd_gluon",
                "implementation_layer": "amd_gluon overlay",
                "extension_layer": "L0",
                "source_base_family": "base_hot_path_streamline",
                "plain_competitor": "fuse-k-buffer-loads",
                "minimum_executable_unit": "inline_scoped_helper",
                "allowed_execution_path": "inline_scoped_helper",
                "scope_infeasible_policy": "shrink_or_report",
                "task_signals": "load_store",
                "failure_layers": "layout_construction, load_store",
                "expected_failure_layers": "layout_construction, load_store",
                "kernel_family_signal": "attention_decode",
                "task_prompt": "\n".join(
                    [
                        "Extension Set task",
                        "Extension layer: L0",
                        "Optimization direction: reduce KBuffer load transactions before dot product",
                        "Source Base family: base_hot_path_streamline",
                        "Plain competitor: fuse-k-buffer-loads",
                        "Gluon overlay reason: explicit_layout",
                        "Overlay priority: high-confidence Consider",
                        "Implementation layer: amd_gluon overlay",
                        "Performance hypothesis: explicit layout may reduce KBuffer load transactions",
                        "Measurement boundary: kernel_only",
                        "Comparison target: true_baseline",
                        "Target component: KBuffer load path feeding dot product",
                        "Allowed change: convert one `KBuffer` load path only",
                        "Reject if: non-executed Gluon or whole-kernel rewrite",
                        "L0 scope classification: high_coupling",
                        "L0 coupling reasons: KBuffer load path directly feeds dot product",
                        "Minimum executable unit: inline_scoped_helper",
                        "Allowed execution path: inline_scoped_helper",
                        "Scope infeasible policy: shrink_or_report",
                        "Task signals: load_store",
                        "Failure layers: layout_construction, load_store",
                        "Expected failure layers: layout_construction, load_store",
                        "Kernel family signal: attention_decode",
                    ]
                ),
            },
        ]
    )

    tasks = _parse_llm_response(
        payload,
        FakeAgentClass,
        expected_extension_slots=1,
        audit_diagnostics_dir=tmp_path,
    )
    assert [task.label for task in tasks][-1] == "gluon-l0-kbuffer-load-layout"
    assert not list(tmp_path.glob("task_generation_audit_failed_*.json"))


def test_l0_soft_audit_diagnostics_cover_atomic_component_scope() -> None:
    task = AgentTask(
        agent_class=FakeAgentClass,
        task="\n".join(
            [
                "Extension layer: L0",
                "Optimization direction: reduce paged load traffic",
                "Source Base family: base_hot_path_streamline",
                "Plain competitor: base-paged-load",
                "Target component: K/V load path",
                "Allowed change: optimize K/V load path only",
                "L0 scope classification: high_coupling",
                "L0 coupling reasons: paged attention load path touches tl.dot, online softmax, mask broadcast, and wrapper integration",
                "Minimum executable unit: whole_jit_kernel",
                "Allowed execution path: whole_jit_kernel",
                "Scope infeasible policy: shrink_or_report",
                "Task signals: index_map, load_store, layout_broadcast, matrix_operand, reduction_accumulator",
                "Failure layers: layout_construction",
                "Expected failure layers: layout_construction",
                "Kernel family signal: attention_decode",
                "Matrix lowering required: false",
            ]
        ),
        config={
            "kernel_type": "triton",
            "required_output_dialect": "amd_gluon",
            "extension_layer": "L0",
            "implementation_layer": "amd_gluon overlay",
            "source_base_family": "base_hot_path_streamline",
            "plain_competitor": "base-paged-load",
            "minimum_executable_unit": "whole_jit_kernel",
            "allowed_execution_path": "whole_jit_kernel",
            "scope_infeasible_policy": "shrink_or_report",
        },
    )

    summary = _task_audit_summary(task)

    assert "local_target_promoted_to_whole_kernel" in summary["soft_audit_diagnostics"]
    assert "reduction_metadata_inconsistent" in summary["soft_audit_diagnostics"]
    assert "matrix_metadata_inconsistent" not in summary["soft_audit_diagnostics"]
    assert "component_bundle_too_broad" not in summary["soft_audit_diagnostics"]


def test_l0_soft_audit_ignores_blockers_and_negative_clauses_for_local_smoke() -> None:
    task = AgentTask(
        agent_class=FakeAgentClass,
        task="\n".join(
            [
                "Extension layer: L0",
                "Target component: K_Buffer load path",
                "Allowed change: convert one K_Buffer load/store path only",
                "Primary atomic component: load_store",
                "Secondary components / blockers: matrix_operand, reduction_accumulator, wrapper_integration",
                "Failure layers: load_store, layout_broadcast, matrix_operand, reduction_accumulator",
                "L0 scope classification: low_coupling",
                "Minimum executable unit: inline_scoped_helper",
                "Allowed execution path: inline_scoped_helper",
                "Reject if: MFMA or tl.dot is introduced",
                "Do NOT introduce buffer ops or whole-kernel rewrite",
                "Matrix lowering required: false",
            ]
        ),
        config={
            "kernel_type": "triton",
            "required_output_dialect": "amd_gluon",
            "extension_layer": "L0",
            "implementation_layer": "amd_gluon overlay",
            "minimum_executable_unit": "inline_scoped_helper",
            "allowed_execution_path": "inline_scoped_helper",
            "scope_infeasible_policy": "shrink_or_report",
        },
    )

    summary = _task_audit_summary(task)

    assert "component_bundle_too_broad" not in summary["soft_audit_diagnostics"]
    assert "matrix_metadata_inconsistent" not in summary["soft_audit_diagnostics"]
    assert "local_target_promoted_to_whole_kernel" not in summary["soft_audit_diagnostics"]


def test_l0_soft_audit_detects_multiple_patch_target_components() -> None:
    task = AgentTask(
        agent_class=FakeAgentClass,
        task="\n".join(
            [
                "Extension layer: L0",
                "Target component: K_Buffer load_store and layout_broadcast path",
                "Allowed change: change load_store and layout_broadcast together",
                "Primary atomic component: load_store, layout_broadcast",
                "L0 scope classification: low_coupling",
                "Minimum executable unit: inline_scoped_helper",
                "Allowed execution path: inline_scoped_helper",
            ]
        ),
        config={
            "kernel_type": "triton",
            "required_output_dialect": "amd_gluon",
            "extension_layer": "L0",
            "implementation_layer": "amd_gluon overlay",
            "minimum_executable_unit": "inline_scoped_helper",
            "allowed_execution_path": "inline_scoped_helper",
        },
    )

    summary = _task_audit_summary(task)

    assert "component_bundle_too_broad" in summary["soft_audit_diagnostics"]


def test_required_gluon_high_risk_soft_diagnostics_warn_before_dispatch() -> None:
    payload = json.dumps(
        [
            {
                "label": "base-paged-load",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "plain_triton",
                "task_prompt": "\n".join(
                    [
                        "Base Set task",
                        "Base family: base_hot_path_streamline",
                        "Optimization direction: reduce paged load traffic",
                        "Target component: K/V load path",
                        "Allowed change: optimize `K/V load path` in plain Triton.",
                    ]
                ),
            },
            {
                "label": "gluon-paged-load",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "amd_gluon",
                "implementation_layer": "amd_gluon overlay",
                "extension_layer": "L0",
                "source_base_family": "base_hot_path_streamline",
                "plain_competitor": "base-paged-load",
                "minimum_executable_unit": "whole_jit_kernel",
                "allowed_execution_path": "whole_jit_kernel",
                "scope_infeasible_policy": "shrink_or_report",
                "whole_kernel_required_reason": "paged load path is coupled to tl.dot, online softmax, mask broadcast, and wrapper integration",
                "task_signals": "index_map, load_store, layout_broadcast, matrix_operand, reduction_accumulator",
                "failure_layers": "layout_construction",
                "expected_failure_layers": "layout_construction",
                "kernel_family_signal": "attention_decode",
                "task_prompt": "\n".join(
                    [
                        "Extension task",
                        "Extension layer: L0",
                        "Optimization direction: reduce paged load traffic",
                        "Source Base family: base_hot_path_streamline",
                        "Plain competitor: base-paged-load",
                        "Gluon overlay reason: explicit_layout",
                        "Overlay priority: high-confidence Consider",
                        "Implementation layer: amd_gluon overlay",
                        "Performance hypothesis: explicit layout may help paged load coalescing",
                        "Measurement boundary: kernel_only",
                        "Comparison target: true_baseline",
                        "Target component: K/V load path",
                        "Allowed change: optimize K/V load path only",
                        "Reject if: correctness fails",
                        "L0 scope classification: high_coupling",
                        "L0 coupling reasons: paged attention load path touches tl.dot, online softmax, mask broadcast, and wrapper integration",
                        "Minimum executable unit: whole_jit_kernel",
                        "Allowed execution path: whole_jit_kernel",
                        "Scope infeasible policy: shrink_or_report",
                        "Whole kernel required reason: paged load path is coupled to tl.dot, online softmax, mask broadcast, and wrapper integration",
                        "Task signals: index_map, load_store, layout_broadcast, matrix_operand, reduction_accumulator",
                        "Failure layers: layout_construction",
                        "Expected failure layers: layout_construction",
                        "Kernel family signal: attention_decode",
                        "Matrix lowering required: false",
                        "Do not optimize before compile: true",
                        "First patch compile goal: true",
                    ]
                ),
            },
        ]
    )

    tasks = _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1)
    summary = {task.label: _task_audit_summary(task) for task in tasks}
    assert "local_target_promoted_to_whole_kernel" in summary["gluon-paged-load"]["soft_audit_diagnostics"]


def test_audit_rejects_l0_overlay_without_same_batch_plain_competitor() -> None:
    payload = json.dumps(
        [
            {
                "label": "triton-other-direction",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nstreamline plain Triton path",
            },
            {
                "label": "amd-gluon-l0-viability",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": _gluon_overlay_prompt(),
            },
        ]
    )

    with pytest.raises(ValueError, match="Plain competitor `triton-eliminate-redundant-ops-streamline`"):
        _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1)


def test_audit_warns_l0_overlay_when_plain_competitor_family_mismatches() -> None:
    payload = json.dumps(
        [
            {
                "label": "triton-eliminate-redundant-ops-streamline",
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
                "task_prompt": _gluon_overlay_prompt(),
            },
        ]
    )

    tasks = _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1)
    assert [task.label for task in tasks][-1] == "amd-gluon-l0-viability"


def test_audit_drops_l0_overlay_with_low_priority_bucket() -> None:
    payload = json.dumps(
        [
            {
                "label": "triton-eliminate-redundant-ops-streamline",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nstreamline plain Triton path",
            },
            {
                "label": "amd-gluon-l0-viability",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": _gluon_overlay_prompt().replace("Overlay priority: Prefer", "Overlay priority: Deprioritize"),
            },
        ]
    )

    tasks = _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1)

    assert [task.label for task in tasks] == ["triton-eliminate-redundant-ops-streamline"]


def test_parse_repairs_missing_family_and_drops_plain_consider_l0_overlay() -> None:
    payload = json.dumps(
        [
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
                "task_prompt": "Base Set task\nBase family: base_scaled_dot_fusion\nfuse scale into dot",
            },
            {
                "label": "triton-eliminate-redundant-ops-streamline",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nstreamline plain Triton path",
            },
            {
                "label": "triton-persistent-small-matrix",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_small_matrix_persistent_or_launch_amortization\npersistent kernel",
            },
            {
                "label": "gluon-l0-explicit-layout-overlay",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "amd_gluon",
                "task_prompt": _gluon_overlay_prompt().replace(
                    "Overlay priority: Prefer",
                    "Overlay priority: Consider",
                ),
            },
        ]
    )

    tasks = _parse_llm_response(
        payload,
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

    labels = {task.label for task in tasks}
    assert "gluon-l0-explicit-layout-overlay" not in labels
    assert "triton-swizzle-and-tile-schedule" in labels


def _mla_low_priority_l0_payload() -> str:
    overlay_prompt = (
        _gluon_overlay_prompt(
            plain_competitor="dot-accumulator-and-cast-cleanup",
            target_component="load_store for K_Buffer inner loop loads",
        )
        .replace("Overlay priority: Prefer", "Overlay priority: Consider")
        .replace("L0 scope classification: low_coupling", "L0 scope classification: high_coupling")
        .replace(
            "L0 coupling reasons: single local memory/layout component without loop-carried state",
            "L0 coupling reasons: online softmax, tl.dot paths, loop-carried state, and 2D parent layouts",
        )
        .replace("Minimum executable unit: inline_scoped_helper", "Minimum executable unit: whole_jit_kernel")
        .replace("Allowed execution path: inline_scoped_helper", "Allowed execution path: whole_jit_kernel")
    )
    overlay_prompt += "\n".join(
        [
            "",
            "Whole kernel required reason: K_Buffer loads feed tl.dot and online softmax accumulation.",
            "Expected failure layers: layout_verifier, dot_operand_layout, broadcast_slice",
            "First patch compile goal: compile the whole stage1 kernel with explicit layout",
            "Do not optimize before compile: true",
            "Matrix lowering required: true",
        ]
    )
    return json.dumps(
        [
            {
                "label": "dot-accumulator-and-cast-cleanup",
                "priority": 5,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "\n".join(
                    [
                        "Base Set task",
                        "Optimization direction: memory/layout cleanup",
                        "Base family: base_hot_path_streamline",
                        "Target component: load_store for K_Buffer inner loop loads",
                        "Allowed change: one load/store layout path for K_Buffer access in the inner loop",
                    ]
                ),
            },
            {
                "label": "gluon-l0-layout-load-store-anchor",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "amd_gluon",
                "task_prompt": overlay_prompt,
            },
        ]
    )


def test_auto_mode_records_and_drops_mla_low_priority_l0_overlay(tmp_path: Path) -> None:
    tasks = _parse_llm_response(
        _mla_low_priority_l0_payload(),
        FakeAgentClass,
        expected_extension_slots=1,
        audit_diagnostics_dir=tmp_path,
        gluon_feature_mode="auto",
    )

    assert [task.label for task in tasks] == ["dot-accumulator-and-cast-cleanup"]
    diagnostics = get_last_gluon_task_generation_diagnostics()
    dropped = diagnostics["dropped_gluon_overlays"]
    assert dropped[0]["label"] == "gluon-l0-layout-load-store-anchor"
    assert dropped[0]["overlay_priority"] == "Consider"
    assert dropped[0]["l0_scope_classification"] == "high_coupling"
    assert (tmp_path / "task_generation_gluon_diagnostics.json").exists()


def test_force_l0_anchor_mode_keeps_mla_low_priority_l0_overlay(tmp_path: Path) -> None:
    tasks = _parse_llm_response(
        _mla_low_priority_l0_payload(),
        FakeAgentClass,
        expected_extension_slots=1,
        audit_diagnostics_dir=tmp_path,
        gluon_feature_mode="force_l0_anchor",
    )

    assert [task.label for task in tasks] == [
        "dot-accumulator-and-cast-cleanup",
        "gluon-l0-layout-load-store-anchor",
    ]
    assert get_last_gluon_task_generation_diagnostics() == {}
    assert not (tmp_path / "task_generation_gluon_diagnostics.json").exists()


def test_require_viable_gluon_mode_reports_no_viable_task(tmp_path: Path) -> None:
    tasks = _parse_llm_response(
        _mla_low_priority_l0_payload(),
        FakeAgentClass,
        expected_extension_slots=1,
        audit_diagnostics_dir=tmp_path,
        gluon_feature_mode="require_viable_gluon",
    )

    assert tasks == []
    diagnostics = get_last_gluon_task_generation_diagnostics()
    assert diagnostics["no_viable_gluon_task"]["dropped_gluon_overlay_count"] == 1
    assert diagnostics["dropped_gluon_overlays"][0]["label"] == "gluon-l0-layout-load-store-anchor"


def test_plain_competitor_requires_plain_triton_contract() -> None:
    shared_with_base_family = AgentTask(
        agent_class=FakeAgentClass,
        label="shared-paired-task",
        task="Base family: base_hot_path_streamline\nPaired transplant task.",
        config={"required_output_dialect": "any", "search_set": "shared"},
    )
    plain_base = AgentTask(
        agent_class=FakeAgentClass,
        label="triton-streamline",
        task="Base family: base_hot_path_streamline\nPlain Triton competitor.",
        config={"required_output_dialect": "plain_triton", "search_set": "base"},
    )

    assert _is_plain_competitor_task(shared_with_base_family) is False
    assert _is_plain_competitor_task(plain_base) is True


def test_audit_treats_body_only_l0_overlay_as_l0_for_priority_and_binding() -> None:
    payload = json.dumps(
        [
            {
                "label": "triton-eliminate-redundant-ops-streamline",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "plain_triton",
                "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nstreamline plain Triton path",
            },
            {
                "label": "same-direction-gluon-overlay",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "amd_gluon",
                "task_prompt": _gluon_overlay_prompt().replace("Extension layer: L0\n", "").replace(
                    "Overlay priority: Prefer", "Overlay priority: Deprioritize"
                ),
            },
        ]
    )

    tasks = _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1)

    assert [task.label for task in tasks] == ["triton-eliminate-redundant-ops-streamline"]


def test_l0_overlay_binding_warns_mismatched_target_component() -> None:
    payload = json.dumps(
        [
            {
                "label": "triton-stage1-mask",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "plain_triton",
                "task_prompt": "\n".join(
                    [
                        "Base Set task",
                        "Base family: base_hot_path_streamline",
                        "Optimization direction: memory/layout cleanup",
                        "Target component: stage1_mask",
                        "Allowed change: simplify `stage1_mask` only.",
                    ]
                ),
            },
            {
                "label": "gluon-l0-stage2-anchor",
                "priority": 6,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "amd_gluon",
                "target_component": "stage2_reduce",
                "task_prompt": _gluon_overlay_prompt(
                    extra=[
                        "Target component: stage2_reduce",
                        "Allowed change: convert `stage2_reduce` only.",
                    ],
                    plain_competitor="triton-stage1-mask",
                ),
            },
        ]
    )

    tasks = _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1)
    assert [task.label for task in tasks][-1] == "gluon-l0-stage2-anchor"


def test_l0_overlay_binding_warns_same_family_different_target_scope() -> None:
    payload = json.dumps(
        [
            {
                "label": "stage1-register-pressure",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "plain_triton",
                "task_prompt": "\n".join(
                    [
                        "Base Set task",
                        "Base family: base_hot_path_streamline",
                        "Optimization direction: memory/layout cleanup",
                        "Target component: shared_load_path",
                        "Target symbol: stage1_kernel",
                        "Allowed change: simplify `shared_load_path` in stage1_kernel.",
                    ]
                ),
            },
            {
                "label": "gluon-l0-stage2-reduction",
                "priority": 6,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "amd_gluon",
                "target_symbol": "stage2_kernel",
                "target_component": "shared_load_path",
                "task_prompt": _gluon_overlay_prompt(
                    extra=[
                        "Optimization direction: reduce pointer arithmetic",
                        "Target symbol: stage2_kernel",
                        "Target component: shared_load_path",
                        "Allowed change: convert `shared_load_path` in stage2_kernel only.",
                    ],
                    plain_competitor="stage1-register-pressure",
                    target_component="shared_load_path",
                ),
            },
        ]
    )

    tasks = _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1)
    assert [task.label for task in tasks][-1] == "gluon-l0-stage2-reduction"


def test_l0_audit_allows_missing_planner_doc_routing_metadata() -> None:
    prompt = _gluon_overlay_prompt()
    for line in (
        "Task signals: layout, memory, l0",
        "Routed doc reasons: layout signal -> 20_component_traits.md; api signal -> 50_api_reference.md",
        "Kernel family signal: generic_memory_layout",
        "Failure layers: broadcast/layout layer, memory/load-store layer",
    ):
        prompt = prompt.replace(f"\n{line}", "")
    payload = json.dumps(
        [
            {
                "label": "triton-eliminate-redundant-ops-streamline",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "plain_triton",
                "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nOptimization direction: memory/layout cleanup\nTarget component: streamline_component\nAllowed change: simplify `streamline_component`.",
            },
            {
                "label": "gluon-l0-missing-routing",
                "priority": 6,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "amd_gluon",
                "task_prompt": prompt,
            },
        ]
    )

    tasks = _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1)
    assert [task.label for task in tasks][-1] == "gluon-l0-missing-routing"


def test_l0_audit_warns_high_coupling_inline_scoped_helper() -> None:
    payload = json.dumps(
        [
            {
                "label": "softmax-rescale",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "plain_triton",
                "task_prompt": "\n".join(
                    [
                        "Base Set task",
                        "Base family: base_hot_path_streamline",
                        "Optimization direction: memory/layout cleanup",
                        "Target component: online softmax accumulator path",
                        "Allowed change: online softmax accumulator path only",
                    ]
                ),
            },
            {
                "label": "gluon-l0-softmax-accumulator",
                "priority": 6,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "amd_gluon",
                "target_component": "online softmax accumulator path",
                "task_prompt": _gluon_overlay_prompt(
                    extra=[
                        "Target component: online softmax accumulator path",
                        "Allowed change: `acc`, `e_sum`, `e_max` and `tl.dot(p, v)` inside the online softmax accumulator loop",
                        "L0 scope classification: high_coupling",
                        "L0 coupling reasons: online reduction plus tl.dot plus loop-carried accumulator state",
                    ],
                    plain_competitor="softmax-rescale",
                    target_component="online softmax accumulator path",
                ),
            },
        ]
    )

    tasks = _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1)
    summary = {task.label: _task_audit_summary(task) for task in tasks}
    assert "matrix_metadata_inconsistent" in summary["gluon-l0-softmax-accumulator"]["soft_audit_diagnostics"]


def test_l0_audit_warns_whole_jit_without_compile_risk_fields() -> None:
    payload = json.dumps(
        [
            {
                "label": "base-composite-path",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "plain_triton",
                "task_prompt": "\n".join(
                    [
                        "Base Set task",
                        "Base family: base_hot_path_streamline",
                        "Optimization direction: memory/layout cleanup",
                        "Target component: composite_path",
                        "Allowed change: optimize `composite_path` in plain Triton.",
                    ]
                ),
            },
            {
                "label": "gluon-l0-composite-whole",
                "priority": 6,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "required_output_dialect": "amd_gluon",
                "target_component": "composite_path",
                "task_prompt": _gluon_overlay_prompt(
                    extra=[
                        "L0 scope classification: high_coupling",
                        "L0 coupling reasons: broadcast-heavy layout plus matrix-like and reduction/accumulator layers",
                        "Minimum executable unit: whole_jit_kernel",
                        "Allowed execution path: whole_jit_kernel",
                        "Whole kernel required reason: no smaller subpath executes independently",
                    ],
                    plain_competitor="base-composite-path",
                    target_component="composite_path",
                ).replace("Minimum executable unit: inline_scoped_helper", "Minimum executable unit: whole_jit_kernel").replace("Allowed execution path: inline_scoped_helper", "Allowed execution path: whole_jit_kernel"),
            },
        ]
    )

    tasks = _parse_llm_response(payload, FakeAgentClass, expected_extension_slots=1)
    assert [task.label for task in tasks][-1] == "gluon-l0-composite-whole"


def test_audit_warns_l1_without_anchor_contract() -> None:
    payload = json.dumps(
        [
            {
                "label": "ext-l1-gluon-mfma",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Extension Set task\nExtension layer: L1\nTry MFMA lowering.",
            }
        ]
    )

    tasks = _parse_llm_response(
        payload,
        FakeAgentClass,
        expected_extension_slots=2,
    )
    assert [task.label for task in tasks] == ["ext-l1-gluon-mfma"]


def test_audit_accepts_l1_with_anchor_contract() -> None:
    payload = json.dumps(
        [
            {
                "label": "triton-eliminate-redundant-ops-streamline",
                "priority": 0,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nOptimization direction: memory/layout cleanup\nTarget component: streamline_component\nAllowed change: simplify `streamline_component`.\nremove redundant indexing",
            },
            {
                "label": "ext-l0-gluon-anchor",
                "priority": 8,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": _gluon_overlay_prompt(),
            },
            {
                "label": "ext-l1-gluon-one-buffer-load",
                "priority": 9,
                "agent_type": "strategy_agent",
                "kernel_language": "python",
                "task_prompt": _gluon_overlay_prompt(
                    "L1",
                    [
                        "Anchor patch: ext-l0-gluon-anchor/patch_2",
                        "Anchor speedup: 1.04x",
                        "Anchor execution: true",
                        "Comparison target: anchor_patch",
                        "Allowed change: one KV buffer_load",
                        "Reject if: changes MFMA or dispatch",
                        "Target symbol: target_stage",
                    ],
                ),
            },
        ]
    )

    tasks = _parse_llm_response(
        payload,
        FakeAgentClass,
        expected_extension_slots=2,
    )

    assert [task.label for task in tasks] == [
        "triton-eliminate-redundant-ops-streamline",
        "ext-l0-gluon-anchor",
        "ext-l1-gluon-one-buffer-load",
    ]


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


def test_parse_llm_response_normalizes_contradictory_hybrid_metadata() -> None:
    tasks = _parse_llm_response(
        json.dumps(
            [
                {
                    "label": "hybrid-dispatch-from-evidence",
                    "priority": 6,
                    "agent_type": "strategy_agent",
                    "kernel_language": "python",
                    "search_set": "base",
                    "required_output_dialect": "plain_triton",
                    "task_prompt": _gluon_overlay_prompt(
                        "Hybrid",
                        [
                            "Composition type: hybrid_dispatch",
                            "Safe anchor: round_1/base-best/patch_4",
                        ],
                    ),
                }
            ]
        ),
        FakeAgentClass,
    )

    assert tasks[0].config["search_set"] == "extension"
    assert tasks[0].config["required_output_dialect"] == "mixed"


def test_parse_llm_response_normalizes_extension_plain_output_contract() -> None:
    tasks = _parse_llm_response(
        json.dumps(
            [
                {
                    "label": "ext-l0-gluon-anchor",
                    "priority": 6,
                    "agent_type": "strategy_agent",
                    "kernel_language": "python",
                    "search_set": "extension",
                    "required_output_dialect": "plain_triton",
                    "task_prompt": _gluon_overlay_prompt(),
                }
            ]
        ),
        FakeAgentClass,
    )

    assert tasks[0].config["search_set"] == "extension"
    assert tasks[0].config["required_output_dialect"] == "amd_gluon"


def test_parse_llm_response_does_not_treat_search_set_as_output_source_of_truth() -> None:
    tasks = _parse_llm_response(
        json.dumps(
            [
                {
                    "label": "plain-direction-with-legacy-metadata",
                    "priority": 6,
                    "agent_type": "strategy_agent",
                    "kernel_language": "python",
                    "search_set": "extension",
                    "required_output_dialect": "plain_triton",
                    "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nplain Triton cleanup",
                }
            ]
        ),
        FakeAgentClass,
    )

    assert tasks[0].config["search_set"] == "base"
    assert tasks[0].config["required_output_dialect"] == "plain_triton"


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
    assert "gluon_api_reference_path" in cfg["required_gluon_docs"]
    assert "gluon_real_patterns_path" in cfg["required_gluon_docs"]
    assert "gluon_architecture_notes_path" not in cfg["required_gluon_docs"]


def test_parse_llm_response_omits_gluon_doc_metadata_for_plain_base_task() -> None:
    tasks = _parse_llm_response(
        json.dumps(
            [
                {
                    "label": "epilogue-scale-streamline",
                    "priority": 0,
                    "agent_type": "strategy_agent",
                    "required_output_dialect": "plain_triton",
                    "task_prompt": "\n".join(
                        [
                            "Base Set task",
                            "Base family: base_hot_path_streamline",
                            "Implementation layer: plain_triton",
                            "Optimization direction: Streamline scale application epilogue",
                        ]
                    ),
                }
            ]
        ),
        FakeAgentClass,
    )

    cfg = tasks[0].config
    assert cfg["search_set"] == "base"
    assert cfg["required_output_dialect"] == "plain_triton"
    assert "gluon_doc_profile" not in cfg
    assert "required_gluon_docs" not in cfg


def test_unreferenced_broad_base_task_remains_valid_with_l0_overlay_anchor() -> None:
    tasks = _parse_llm_response(
        json.dumps(
            [
                {
                    "label": "broad-whole-loop-cleanup",
                    "priority": 0,
                    "agent_type": "strategy_agent",
                    "kernel_language": "python",
                    "required_output_dialect": "plain_triton",
                    "task_prompt": "\n".join(
                        [
                            "Base Set task",
                            "Base family: base_hot_path_streamline",
                            "Implementation layer: plain_triton",
                            "Optimization direction: broad whole-loop cleanup",
                            "Rewrite a broad loop-level plain Triton strategy without naming a local L0 anchor.",
                        ]
                    ),
                },
                {
                    "label": "exact-local-base",
                    "priority": 0,
                    "agent_type": "strategy_agent",
                    "kernel_language": "python",
                    "required_output_dialect": "plain_triton",
                    "task_prompt": "\n".join(
                        [
                            "Base Set task",
                            "Base family: base_hot_path_streamline",
                            "Implementation layer: plain_triton",
                            "Optimization direction: memory/layout cleanup",
                            "Target component: streamline_component",
                            "Allowed change: simplify `streamline_component` in plain Triton.",
                        ]
                    ),
                },
                {
                    "label": "ext-l0-local-layout",
                    "priority": 8,
                    "agent_type": "strategy_agent",
                    "kernel_language": "python",
                    "task_prompt": _gluon_overlay_prompt(plain_competitor="exact-local-base"),
                },
            ]
        ),
        FakeAgentClass,
        expected_extension_slots=1,
    )

    broad_cfg = tasks[0].config
    assert broad_cfg["search_set"] == "base"
    assert broad_cfg["required_output_dialect"] == "plain_triton"
    assert "gluon_doc_profile" not in broad_cfg
    assert "required_gluon_docs" not in broad_cfg
    assert "minimum_executable_unit" not in broad_cfg
    assert tasks[-1].config["search_set"] == "extension"


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
                    "label": "triton-eliminate-redundant-ops-streamline",
                    "priority": 0,
                    "agent_type": "strategy_agent",
                    "kernel_language": "python",
                    "task_prompt": "Base Set task\nBase family: base_hot_path_streamline\nOptimization direction: memory/layout cleanup\nTarget component: streamline_component\nAllowed change: simplify `streamline_component`.\nremove redundant indexing",
                },
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
                    "task_prompt": _gluon_overlay_prompt(),
                },
            ]
        ),
        FakeAgentClass,
        expected_extension_slots=1,
    )

    assert tasks[0].config["search_set"] == "base"
    assert tasks[1].config["search_set"] == "shared"
    assert tasks[2].config["search_set"] == "extension"


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
    assert "after preserving the same-direction plain Triton competitor" in run_kwargs["output_dialect_guidance"]
    assert "L0 AMD Gluon overlay" in run_kwargs["output_dialect_guidance"]
    assert "Performance hypothesis:" in run_kwargs["output_dialect_guidance"]
    assert "Keep a plain Triton fallback path alive" in run_kwargs["output_dialect_guidance"]
    assert "## Gluon Planning Traits" in run_kwargs["gluon_planning_traits_guidance"]
    assert "`dialect_plain_triton`" in run_kwargs["gluon_planning_traits_guidance"]
    assert "Plain Triton competitors: at least 3 task(s)" in run_kwargs["search_space_allocation_guidance"]
    assert "AMD Gluon overlay: 1 task(s)" in run_kwargs["search_space_allocation_guidance"]
    assert "Route worker docs by `gluon_doc_profile`" in run_kwargs["search_space_allocation_guidance"]
    assert "Round 1 may include at most one L0 minimal AMD Gluon overlay" in run_kwargs["gluon_planning_traits_guidance"]
    system_prompt = mock_default_agent.call_args.kwargs["system_template"]
    assert "Gluon contract source of truth" in system_prompt
    assert "injected policy blocks" in system_prompt
    assert "after a passing patch, the next patch" not in system_prompt
    assert "Do not create extra patch-evolution metadata fields" not in system_prompt
    assert "Required AMD Gluon tasks must attempt a real" in system_prompt
    assert "API-level rewrite details belong in routed skill docs" in system_prompt
    assert "tl.sigmoid" not in system_prompt
    assert "gl.sum" not in system_prompt


@patch("minisweagent.tools.tools_runtime.get_tools_list", return_value=[{"name": "str_replace_editor"}, {"name": "submit"}])
@patch("minisweagent.agents.default.DefaultAgent")
def test_run_task_agent_missing_legacy_kb_uses_plain_triton_fallback(
    mock_default_agent, _mock_tools, tmp_path: Path
):
    model = FakePlanningModel()
    mock_default_agent.return_value.run.return_value = ("Submitted", "[]")

    submitted = _run_task_agent(
        kernel_path=str(tmp_path / "kernel.py"),
        kernel_name="test_kernel",
        kernel_type="triton",
        kernel_language="python",
        function_names=["kernel_fwd"],
        workspace_path=str(tmp_path),
        input_dialect="plain_triton",
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
    run_kwargs = mock_default_agent.return_value.run.call_args.kwargs
    assert run_kwargs["knowledge_base_path"].endswith(
        "knowledge-base/amd-knowledge-base/layer-3-libraries/compilers/triton-on-rocm.md"
    )
    assert "triton-gluon-on-rocm.md" not in run_kwargs["knowledge_base_path"]
    assert run_kwargs["gluon_kb_path"].endswith(
        "knowledge-base/amd-knowledge-base/layer-3-libraries/compilers/triton-gluon-on-rocm.md"
    )


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


def test_write_task_files_keeps_plain_base_without_skill_usage(tmp_path: Path):
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
    assert meta["use_skills"] is False
    assert meta["input_dialect"] == "amd_gluon"
    assert meta["gluon_feature_mode"] == "auto"
    assert meta["gluon_baseline_profile"] == "mi3xx"
    assert meta["allowed_output_dialects"] == ["plain_triton", "amd_gluon"]
    assert meta["preferred_output_dialects"] == ["amd_gluon", "plain_triton"]
    assert meta["output_dialect_search_policy"] == "prefer_amd_gluon_if_viable_else_plain_triton"
    assert meta["allowed_skill_tiers"] == ["general"]
    assert "gluon_doc_profile" not in meta
    assert "required_gluon_docs" not in meta
    assert "gluon_always_read_path" not in meta
    assert "gluon_api_reference_path" not in meta
    assert "gluon_real_patterns_path" not in meta


def test_write_task_files_separates_plain_and_gluon_knowledge_paths(tmp_path: Path):
    tasks = _parse_llm_response(
        json.dumps(
            [
                {
                    "label": "softmax-rescale",
                    "priority": 0,
                    "agent_type": "strategy_agent",
                    "task_prompt": "\n".join(
                        [
                            "Base family: base_hot_path_streamline",
                            "Optimization direction: memory/layout cleanup",
                            "Target component: streamline_component",
                            "Allowed change: one layout or memory component in `streamline_component`",
                            "Reject if: output falls back to AMD Gluon.",
                        ]
                    ),
                },
                {
                    "label": "gluon-l0-layout",
                    "priority": 6,
                    "agent_type": "strategy_agent",
                    "task_prompt": _gluon_overlay_prompt(
                        plain_competitor="softmax-rescale",
                        target_component="streamline_component",
                    ),
                },
            ]
        ),
        FakeAgentClass,
    )

    paths = write_task_files(
        tasks,
        tmp_path,
        kernel_path=str(tmp_path / "kernel.py"),
        kernel_type="triton",
        input_dialect="plain_triton",
        gluon_feature_mode="auto",
        gluon_baseline_profile="mi3xx",
        allowed_output_dialects=["plain_triton", "amd_gluon"],
        target_backend="hip/gfx942",
    )

    by_label = {}
    for path in paths:
        meta, _body = read_task_file(path)
        by_label[meta["label"]] = meta

    base_meta = by_label["softmax-rescale"]
    assert base_meta["knowledge_base_path"].endswith(
        "knowledge-base/amd-knowledge-base/layer-3-libraries/compilers/triton-on-rocm.md"
    )
    assert "triton-gluon-on-rocm.md" not in base_meta["knowledge_base_path"]
    assert base_meta["use_skills"] is False
    assert "gluon_kb_path" not in base_meta
    assert "gluon_skill_path" not in base_meta

    ext_meta = by_label["gluon-l0-layout"]
    assert ext_meta["knowledge_base_path"].endswith(
        "knowledge-base/amd-knowledge-base/layer-3-libraries/compilers/triton-on-rocm.md"
    )
    assert ext_meta["gluon_kb_path"].endswith(
        "knowledge-base/amd-knowledge-base/layer-3-libraries/compilers/triton-gluon-on-rocm.md"
    )
    assert ext_meta["use_skills"] is False
    assert ext_meta["required_output_dialect"] == "amd_gluon"
    assert "gluon_always_read_path" in ext_meta
    assert "required_gluon_docs" in ext_meta


def test_parse_llm_response_augments_explicit_required_gluon_docs() -> None:
    tasks = _parse_llm_response(
        json.dumps(
            [
                {
                    "label": "matrix-l0",
                    "priority": 6,
                    "agent_type": "strategy_agent",
                    "gluon_doc_profile": "matrix_lowering",
                    "required_gluon_docs": ["gluon_real_patterns_path"],
                    "task_prompt": "Extension layer: L0\nImplementation layer: amd_gluon overlay\n",
                }
            ]
        ),
        FakeAgentClass,
    )

    docs = tasks[0].config["required_gluon_docs"]
    assert "gluon_skill_path" in docs
    assert "gluon_always_read_path" in docs
    assert "gluon_search_policies_path" in docs
    assert "gluon_component_traits_path" in docs
    assert "gluon_architecture_notes_path" in docs
    assert "gluon_api_reference_path" in docs
    assert "gluon_real_patterns_path" in docs


def test_parse_llm_response_preserves_optional_l0_metadata() -> None:
    tasks = _parse_llm_response(
        json.dumps(
            [
                {
                    "label": "gluon-l0-stage-anchor",
                    "priority": 6,
                    "agent_type": "strategy_agent",
                    "required_output_dialect": "amd_gluon",
                    "extension_intent": "execution_anchor",
                    "expected_outcome": "correctness_anchor_not_speedup",
                    "not_viable_for_l1_if_slower_than_base": True,
                    "overhead_source_to_record": "launch_layout_overhead",
                    "minimum_executable_unit": "separate_gluon_kernel",
                    "allowed_execution_path": "separate_gluon_kernel",
                    "scope_infeasible_policy": "separate_kernel_if_allowed",
                    "whole_kernel_required_reason": "not needed for separate kernel",
                    "target_symbol": "_stage_kernel",
                    "target_component": "one 1D reduction subpath",
                    "task_prompt": "\n".join(
                        [
                            "Extension layer: L0",
                            "Implementation layer: amd_gluon overlay",
                            "Target component: one 1D reduction subpath",
                            "Allowed change: Convert `_stage_kernel` only.",
                        ]
                    ),
                }
            ]
        ),
        FakeAgentClass,
    )

    cfg = tasks[0].config
    assert cfg["extension_intent"] == "execution_anchor"
    assert cfg["expected_outcome"] == "correctness_anchor_not_speedup"
    assert cfg["not_viable_for_l1_if_slower_than_base"] is True
    assert cfg["overhead_source_to_record"] == "launch_layout_overhead"
    assert cfg["minimum_executable_unit"] == "separate_gluon_kernel"
    assert cfg["allowed_execution_path"] == "separate_gluon_kernel"
    assert cfg["scope_infeasible_policy"] == "separate_kernel_if_allowed"
    assert cfg["whole_kernel_required_reason"] == "not needed for separate kernel"
    assert cfg["target_symbol"] == "_stage_kernel"
    assert cfg["target_component"] == "one 1D reduction subpath"


def test_gluon_doc_profile_mapping_covers_prompt_enum_values() -> None:
    expected = {
        "extension_l0_minimal",
        "nv_to_amd_translation",
        "memory_lowering",
        "matrix_lowering",
        "shape_bucketed_dispatch",
        "jit_aot_sensitive",
        "shared_transplant",
        "gluon_variant_from_anchor",
        "hybrid_dispatch",
        "hybrid_dispatch_from_evidence",
        "base_or_shared_gluon",
    }

    assert set(GLUON_DOC_PROFILE_REQUIRED_KEYS) == expected


def test_resolve_project_resource_finds_installed_geak_share_root(tmp_path: Path) -> None:
    installed_doc = tmp_path / "share" / "geak" / "skills" / "triton-gluon" / "docs" / "00_always_read.md"
    installed_doc.parent.mkdir(parents=True)
    installed_doc.write_text("# installed doc\n")

    resolved = resolve_project_resource(
        "skills/triton-gluon/docs/00_always_read.md",
        workspace=tmp_path / "workspace",
        extra_roots=[tmp_path / "share" / "geak"],
    )

    assert resolved == installed_doc.resolve()


def test_resolve_project_resource_finds_main_rag_knowledge_base(tmp_path: Path) -> None:
    kb_doc = (
        tmp_path
        / "mcp_tools"
        / "rag-mcp"
        / "knowledge-base"
        / "amd-knowledge-base"
        / "layer-3-libraries"
        / "compilers"
        / "triton-gluon-on-rocm.md"
    )
    kb_doc.parent.mkdir(parents=True)
    kb_doc.write_text("# source RAG doc\n")

    resolved = resolve_project_resource(
        "knowledge-base/amd-knowledge-base/layer-3-libraries/compilers/triton-gluon-on-rocm.md",
        workspace=tmp_path / "workspace",
        extra_roots=[tmp_path],
    )

    assert resolved == kb_doc.resolve()


def test_taskgen_planner_default_files_do_not_require_worker_deep_docs() -> None:
    assert "Triton-Gluon API reference" not in TASKGEN_INSTANCE_TEMPLATE
    assert "Triton-Gluon schematic examples" not in TASKGEN_INSTANCE_TEMPLATE
    assert "Triton-Gluon residual backup routing" not in TASKGEN_INSTANCE_TEMPLATE
    assert "gluon_component_traits_path" not in TASKGEN_INSTANCE_TEMPLATE
    assert "gluon_architecture_notes_path" not in TASKGEN_INSTANCE_TEMPLATE
    assert "gluon_real_patterns_path" not in TASKGEN_INSTANCE_TEMPLATE
    assert "gluon_api_reference_path" not in TASKGEN_INSTANCE_TEMPLATE
    assert "gluon_kb_path" not in TASKGEN_INSTANCE_TEMPLATE
    assert "Gluon worker docs are profile-routed" in TASKGEN_INSTANCE_TEMPLATE


def test_gluon_entry_docs_keep_hard_contract_and_profile_performance_hints() -> None:
    root = Path(__file__).resolve().parents[2]
    skill = (root / "skills" / "triton-gluon" / "SKILL.md").read_text()
    always = (root / "skills" / "triton-gluon" / "docs" / "00_always_read.md").read_text()
    component = (root / "skills" / "triton-gluon" / "docs" / "20_component_traits.md").read_text()
    api = (root / "skills" / "triton-gluon" / "docs" / "50_api_reference.md").read_text()
    real = (root / "skills" / "triton-gluon" / "docs" / "60_real_patterns.md").read_text()

    for text in (skill, always):
        assert "doc" in text.lower()
        assert "Gluon knowledge lookup plan" in text
        assert "Gluon implementation plan" in text
        assert "broad `try/except Exception` fallback" in text
        assert "one subpath/component" in text
        assert "backup" in text.lower()
    assert "When To Read This File" in component
    assert "First patch" in component
    assert "Hard-case stops" in component
    assert "When To Read This File" in api
    assert "API cookbook" in api
    assert "When To Read This File" in real
    assert "background_rag" in real
    assert "source-first" in real


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
    assert "If feature-specific planning blocks are present" in _SYSTEM_PROMPT
    assert "Evidence-Anchored" in _SYSTEM_PROMPT
    assert "Composition" in _SYSTEM_PROMPT
    assert "Gluon contract source of truth" in _SYSTEM_PROMPT
    assert "leave some gpus idle" in _SYSTEM_PROMPT.lower()
    assert "Generate at least one priority-0 task that specifically checks the dispatch path" not in _SYSTEM_PROMPT
