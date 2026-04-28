"""Tests for DRA configuration profiles."""
from __future__ import annotations

from minisweagent.dra.config import DRAConfig


def test_deep_mode_raises_research_budget_defaults(monkeypatch):
    monkeypatch.setenv("GEAK_DRA_DEEP_MODE", "1")

    cfg = DRAConfig.from_env()

    assert cfg.deep_mode is True
    assert cfg.max_questions == 16
    assert cfg.max_blindspots == 6
    assert cfg.max_blindspot_rounds == 2
    assert cfg.retrieval_top_k == 16
    assert cfg.web_max_refinements == 3
    assert cfg.web_concurrency == 6
    assert cfg.per_question_result_budget == 16
    assert cfg.search_depth == 2
    assert cfg.read_top_k == 5
    assert cfg.read_max_length == 12000
    assert cfg.supplemental_web_enabled is False
    # Bumped from 800 to 1200 (deep) when parallel-initial-queries fanout
    # raised the average per-question web call count ~3x; see config.py.
    assert cfg.max_total_calls == 1200


def test_deep_mode_respects_explicit_env_overrides(monkeypatch):
    monkeypatch.setenv("GEAK_DRA_DEEP_MODE", "1")
    monkeypatch.setenv("GEAK_DRA_MAX_QUESTIONS", "7")
    monkeypatch.setenv("GEAK_DRA_PER_QUESTION_RESULT_BUDGET", "5")
    monkeypatch.setenv("GEAK_DRA_SEARCH_DEPTH", "3")
    monkeypatch.setenv("GEAK_DRA_READ_TOP_K", "2")
    monkeypatch.setenv("GEAK_DRA_READ_MAX_LENGTH", "4321")
    monkeypatch.setenv("GEAK_DRA_SUPPLEMENTAL_WEB_ENABLE", "1")

    cfg = DRAConfig.from_env()

    assert cfg.max_questions == 7
    assert cfg.per_question_result_budget == 5
    assert cfg.search_depth == 3
    assert cfg.read_top_k == 2
    assert cfg.read_max_length == 4321
    assert cfg.supplemental_web_enabled is True
