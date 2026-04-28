"""Unit tests for ``minisweagent.run.orchestrator``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.agents.heterogeneous.orchestrator import run_heterogeneous_orchestrator
from minisweagent.run.orchestrator import _probe_preprocess_dir, run_orchestrator


class TestRunOrchestrator:
    def test_homogeneous_raises_not_implemented(self, tmp_path: Path) -> None:
        ctx = {"output_dir": str(tmp_path)}
        with pytest.raises(NotImplementedError, match="Homogeneous mode is not supported"):
            run_orchestrator(
                preprocess_ctx=ctx,
                gpu_ids=[0],
                model=MagicMock(),
                model_factory=MagicMock(),
                heterogeneous=False,
            )

    def test_heterogeneous_delegates_to_run_heterogeneous_orchestrator(self, tmp_path: Path) -> None:
        ctx = {"output_dir": str(tmp_path / "pp")}
        model = MagicMock()
        factory = MagicMock()
        sentinel = {"status": "ok"}

        with patch(
            "minisweagent.agents.heterogeneous.orchestrator.run_heterogeneous_orchestrator",
            return_value=sentinel,
        ) as mock_hetero:
            out = run_orchestrator(
                preprocess_ctx=ctx,
                gpu_ids=[0, 1],
                model=model,
                model_factory=factory,
                heterogeneous=True,
                max_rounds=3,
                start_round=2,
            )

        assert out is sentinel
        assert mock_hetero.call_count == 1
        args, kwargs = mock_hetero.call_args
        assert args[0] is ctx
        assert args[1] == [0, 1]
        assert args[2] is model
        assert args[3] is factory
        assert args[4] == Path(ctx["output_dir"]).resolve()
        assert args[5] == 3
        assert args[6] == 2
        assert len(args) == 7

    def test_max_rounds_defaults_from_env_when_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ctx = {"output_dir": str(tmp_path)}
        monkeypatch.setenv("GEAK_MAX_ROUNDS", "7")
        with patch(
            "minisweagent.agents.heterogeneous.orchestrator.run_heterogeneous_orchestrator",
            return_value={},
        ) as mock_hetero:
            run_orchestrator(
                preprocess_ctx=ctx,
                gpu_ids=[0],
                model=MagicMock(),
                model_factory=MagicMock(),
                heterogeneous=True,
                max_rounds=None,
            )
        assert mock_hetero.call_args[0][5] == 7

    def test_output_dir_override(self, tmp_path: Path) -> None:
        ctx = {"output_dir": str(tmp_path / "ignored")}
        override = tmp_path / "override_out"
        with patch(
            "minisweagent.agents.heterogeneous.orchestrator.run_heterogeneous_orchestrator",
            return_value={},
        ) as mock_hetero:
            run_orchestrator(
                preprocess_ctx=ctx,
                gpu_ids=[0],
                model=MagicMock(),
                model_factory=MagicMock(),
                heterogeneous=True,
                output_dir=override,
            )
        assert mock_hetero.call_args[0][4] == override


class TestProbePreprocessDir:
    def test_empty_dir_uses_preprocess_as_repo_root(self, tmp_path: Path) -> None:
        pc = _probe_preprocess_dir(tmp_path)
        assert pc.kernel_path == ""
        assert pc.repo_root == str(tmp_path)
        assert pc.harness_path == ""
        assert pc.preprocess_dir == str(tmp_path)
        assert pc.discovery is None

    def test_resolved_json_sets_kernel_and_finds_git_root(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        sub = repo / "kernels"
        sub.mkdir(parents=True)
        k = sub / "k.hip"
        k.write_text("kernel")
        (repo / ".git").mkdir()

        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "resolved.json").write_text(
            json.dumps(
                {
                    "local_file_path": str(k.resolve()),
                    "local_repo_path": None,
                }
            )
        )

        pc = _probe_preprocess_dir(pp)
        assert Path(pc.kernel_path).resolve() == k.resolve()
        assert Path(pc.repo_root).resolve() == repo.resolve()

    def test_resolved_json_no_git_uses_local_repo_path(self, tmp_path: Path) -> None:
        k = tmp_path / "solo.cu"
        k.write_text("x")
        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "resolved.json").write_text(
            json.dumps(
                {
                    "local_file_path": str(k.resolve()),
                    "local_repo_path": "/workspace/myproject",
                }
            )
        )

        pc = _probe_preprocess_dir(pp)
        assert pc.repo_root == "/workspace/myproject"

    def test_testcase_selection_sets_harness_path(self, tmp_path: Path) -> None:
        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "testcase_selection.json").write_text(
            json.dumps({"harness_path": "/h/run.py"})
        )

        pc = _probe_preprocess_dir(pp)
        assert pc.harness_path == "/h/run.py"

    def test_discovery_json_loaded(self, tmp_path: Path) -> None:
        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "discovery.json").write_text(
            json.dumps(
                {
                    "kernel": {
                        "type": "triton",
                        "input_dialect": "amd_gluon",
                        "gluon_feature_mode": "auto",
                        "gluon_baseline_profile": "mi3xx",
                        "allowed_output_dialects": ["plain_triton", "amd_gluon"],
                    }
                }
            )
        )

        pc = _probe_preprocess_dir(pp)
        assert pc.discovery["kernel"]["type"] == "triton"
        assert pc.input_dialect == "amd_gluon"
        assert pc.gluon_feature_mode == "auto"
        assert pc.gluon_baseline_profile == "mi3xx"
        assert pc.preferred_output_dialects == ["amd_gluon", "plain_triton"]
        assert pc.output_dialect_search_policy == "prefer_amd_gluon_if_viable_else_plain_triton"

    def test_probe_preprocess_dir_prefers_amd_gluon_for_plain_triton_auto(self, tmp_path: Path) -> None:
        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "discovery.json").write_text(
            json.dumps(
                {
                    "kernel": {
                        "type": "triton",
                        "input_dialect": "plain_triton",
                        "gluon_feature_mode": "auto",
                        "gluon_baseline_profile": "raw",
                        "allowed_output_dialects": ["plain_triton", "amd_gluon"],
                    }
                }
            )
        )

        pc = _probe_preprocess_dir(pp)
        assert pc.preferred_output_dialects == ["amd_gluon", "plain_triton"]
        assert pc.output_dialect_search_policy == "prefer_amd_gluon_if_viable_else_plain_triton"

    def test_discovery_unknown_type_still_inferrs_gluon_feature_metadata(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        kernel = repo / "kernel.py"
        kernel.write_text(
            "from triton.experimental import gluon\n"
            "@gluon.jit\n"
            "def kernel_fwd(x):\n"
            "    return x\n"
        )

        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "resolved.json").write_text(
            json.dumps(
                {
                    "local_file_path": str(kernel.resolve()),
                    "local_repo_path": str(repo.resolve()),
                }
            )
        )
        (pp / "discovery.json").write_text(
            json.dumps(
                {
                    "kernel": {
                        "file": str(kernel.resolve()),
                        "type": "unknown",
                    }
                }
            )
        )

        pc = _probe_preprocess_dir(pp)
        assert pc.input_dialect == "amd_gluon"
        assert pc.gluon_feature_mode == "auto"
        assert pc.allowed_output_dialects == ["plain_triton", "amd_gluon"]

    def test_resume_without_resolved_json_falls_back_to_discovery_kernel_file(
        self, tmp_path: Path
    ) -> None:
        """Finding 1 regression: when resume-from-disk runs only have
        ``discovery.json`` (no ``resolved.json``), ``_probe_preprocess_dir``
        must still locate the kernel via ``discovery['kernel']['file']``
        and re-infer kernel_type from source so Gluon does not silently
        default to ``off``.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        kernel = repo / "kernel.py"
        kernel.write_text(
            "from triton.experimental import gluon\n"
            "@gluon.jit\n"
            "def kernel_fwd(x):\n"
            "    return x\n"
        )

        pp = tmp_path / "pp"
        pp.mkdir()
        # Note: NO resolved.json on disk -- only discovery.json.
        (pp / "discovery.json").write_text(
            json.dumps(
                {
                    "kernel": {
                        "file": str(kernel.resolve()),
                        "name": "kernel_fwd",
                        "type": "unknown",
                    }
                }
            )
        )

        pc = _probe_preprocess_dir(pp)
        assert pc.kernel_path == str(kernel.resolve())
        assert pc.input_dialect == "amd_gluon"
        assert pc.gluon_feature_mode == "auto"

    def test_probe_preprocess_dir_baseline_cases_override_stale_discovery(
        self, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        kernel = repo / "kernel.py"
        kernel.write_text("import triton\n")

        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "resolved.json").write_text(
            json.dumps(
                {
                    "local_file_path": str(kernel.resolve()),
                    "local_repo_path": str(repo.resolve()),
                }
            )
        )
        (pp / "discovery.json").write_text(
            json.dumps(
                {
                    "kernel": {
                        "file": str(kernel.resolve()),
                        "type": "triton",
                        "shape_coverage_profile": "multi",
                        "benchmark_shape_count": 2,
                        "benchmark_test_cases": [
                            {"case_id": "stale_a", "params": {"M": 16}},
                            {"case_id": "stale_b", "params": {"M": 17}},
                        ],
                    }
                }
            )
        )
        (pp / "baseline_metrics.json").write_text(
            json.dumps(
                {
                    "shape_coverage_profile": "bucketed",
                    "benchmark_shape_count": 4,
                    "benchmark_test_cases": [
                        {"case_id": "perf1", "params": {"M": 32}},
                        {"case_id": "perf2", "params": {"M": 64}},
                        {"case_id": "perf3", "params": {"M": 128}},
                        {"case_id": "perf4", "params": {"M": 256}},
                    ],
                }
            )
        )

        pc = _probe_preprocess_dir(pp)
        assert pc.shape_coverage_profile == "bucketed"
        assert pc.benchmark_shape_count == 4
        assert [c["case_id"] for c in pc.benchmark_test_cases or []] == [
            "perf1",
            "perf2",
            "perf3",
            "perf4",
        ]

    def test_optional_artifact_paths(self, tmp_path: Path) -> None:
        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "COMMANDMENT.md").write_text("cmd")
        (pp / "CODEBASE_CONTEXT.md").write_text("ctx")
        (pp / "baseline_metrics.json").write_text("{}")
        (pp / "profile.json").write_text("{}")

        pc = _probe_preprocess_dir(pp)
        assert pc.commandment_path == str(pp / "COMMANDMENT.md")
        assert pc.codebase_context_path == str(pp / "CODEBASE_CONTEXT.md")
        assert pc.baseline_metrics_path == str(pp / "baseline_metrics.json")
        assert pc.profiling_result_path == str(pp / "profile.json")


class _DummyModel:
    def __init__(self) -> None:
        self.tools = []


@patch("minisweagent.tools.tools_runtime.ToolRuntime", return_value=MagicMock())
@patch("minisweagent.agents.heterogeneous.orchestrator.build_tools_schema", return_value=[])
@patch("minisweagent.agents.heterogeneous.orchestrator.run_llm_steps", return_value={"status": "done"})
def test_run_heterogeneous_orchestrator_prefers_explicit_feature_context(
    mock_run_llm_steps,
    _mock_tools_schema,
    _mock_toolruntime,
    tmp_path: Path,
) -> None:
    kernel = tmp_path / "kernel.py"
    kernel.write_text("import triton\n")

    report = run_heterogeneous_orchestrator(
        preprocess_ctx={
            "kernel_path": str(kernel),
            "repo_root": str(tmp_path),
            "commandment": "Keep correctness first.",
            "input_dialect": "amd_gluon",
            "gluon_feature_mode": "auto",
            "gluon_baseline_profile": "mi3xx",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "preferred_output_dialects": ["amd_gluon", "plain_triton"],
            "output_dialect_search_policy": "prefer_amd_gluon_if_viable_else_plain_triton",
            "target_backend": "hip/gfx942",
            "discovery": {
                "workspace": str(tmp_path),
                "kernel": {
                    "file": str(kernel),
                    "name": "kernel",
                    "type": "triton",
                    "input_dialect": "plain_triton",
                    "gluon_feature_mode": "off",
                    "gluon_baseline_profile": "raw",
                    "allowed_output_dialects": ["plain_triton"],
                },
            },
        },
        gpu_ids=[0],
        model=_DummyModel(),
        model_factory=MagicMock(),
        output_dir=tmp_path,
        max_rounds=1,
        start_round=1,
    )

    assert report == {"status": "done"}
    messages = mock_run_llm_steps.call_args.args[1]
    instance_msg = messages[1]["content"]
    assert "Input dialect: amd_gluon" in instance_msg
    assert "Gluon feature mode: auto" in instance_msg
    assert "Gluon baseline profile: mi3xx" in instance_msg
    assert "Preferred output dialect order: amd_gluon, plain_triton" in instance_msg
    assert "Output-dialect search policy: prefer_amd_gluon_if_viable_else_plain_triton" in instance_msg


@patch("minisweagent.tools.tools_runtime.ToolRuntime", return_value=MagicMock())
@patch("minisweagent.agents.heterogeneous.orchestrator.build_tools_schema", return_value=[])
@patch("minisweagent.agents.heterogeneous.orchestrator.run_llm_steps", return_value={"status": "done"})
def test_run_heterogeneous_orchestrator_injects_plain_triton_gluon_preference(
    mock_run_llm_steps,
    _mock_tools_schema,
    _mock_toolruntime,
    tmp_path: Path,
) -> None:
    kernel = tmp_path / "kernel.py"
    kernel.write_text("import triton\n")

    report = run_heterogeneous_orchestrator(
        preprocess_ctx={
            "kernel_path": str(kernel),
            "repo_root": str(tmp_path),
            "commandment": "Keep correctness first.",
            "input_dialect": "plain_triton",
            "gluon_feature_mode": "auto",
            "gluon_baseline_profile": "raw",
            "allowed_output_dialects": ["plain_triton", "amd_gluon"],
            "preferred_output_dialects": ["amd_gluon", "plain_triton"],
            "output_dialect_search_policy": "prefer_amd_gluon_if_viable_else_plain_triton",
            "target_backend": "hip/gfx942",
            "discovery": {
                "workspace": str(tmp_path),
                "kernel": {
                    "file": str(kernel),
                    "name": "kernel",
                    "type": "triton",
                },
            },
        },
        gpu_ids=[0],
        model=_DummyModel(),
        model_factory=MagicMock(),
        output_dir=tmp_path,
        max_rounds=1,
        start_round=1,
    )

    assert report == {"status": "done"}
    messages = mock_run_llm_steps.call_args.args[1]
    instance_msg = messages[1]["content"]
    assert "Preferred output dialect order: amd_gluon, plain_triton" in instance_msg
    assert "Output-dialect search policy: prefer_amd_gluon_if_viable_else_plain_triton" in instance_msg
    assert "Keep a plain Triton fallback alive" in instance_msg
