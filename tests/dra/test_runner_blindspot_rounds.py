"""Tests for runner internals: blindspot dedup, early-stop, budget abort."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from minisweagent.dra.runner import (
    CallBudget,
    _coerce_evidence,
    _dedup_blindspots,
    _stage5_blindspot,
)
from minisweagent.dra.schemas import BlindSpot, Facts

# ---- CallBudget ------------------------------------------------------------


def test_call_budget_aborts_at_max():
    b = CallBudget(max_total=3)
    b.record("llm")
    b.record("llm")
    assert not b.aborted
    b.record("web")
    assert b.aborted
    assert b.total == 3


def test_call_budget_summary_groups_by_kind():
    b = CallBudget(max_total=10)
    b.record("llm")
    b.record("llm")
    b.record("web")
    s = b.summary()
    assert s["llm_calls"] == 2
    assert s["web_calls"] == 1
    assert s["total"] == 3
    assert s["max_total"] == 10
    assert s["kind:llm"] == 2
    assert s["kind:web"] == 1


# ---- Blindspot dedup -------------------------------------------------------


def test_dedup_blindspots_drops_high_overlap():
    existing = [
        BlindSpot(description="register pressure causing low occupancy"),
    ]
    new = [
        BlindSpot(description="register pressure causes low occupancy on CDNA"),  # near-dup
        BlindSpot(description="kernel may be using bf16 paths instead of fp16"),
    ]
    out = _dedup_blindspots(new, existing)
    descs = [b.description for b in out]
    assert any("bf16" in d for d in descs)
    assert not any("register pressure" in d.lower() for d in descs)


def test_dedup_blindspots_keeps_all_when_no_overlap():
    existing = [BlindSpot(description="alpha beta")]
    new = [BlindSpot(description="gamma delta epsilon")]
    out = _dedup_blindspots(new, existing)
    assert len(out) == 1


def test_dedup_blindspots_avoids_dupes_within_new_batch():
    existing: list[BlindSpot] = []
    new = [
        BlindSpot(description="memory bandwidth limit"),
        BlindSpot(description="memory bandwidth limit"),  # exact dup
    ]
    out = _dedup_blindspots(new, existing)
    assert len(out) == 1


# ---- _stage5_blindspot respects budget abort -------------------------------


def test_stage5_returns_empty_when_budget_aborted():
    budget = CallBudget(max_total=1)
    budget.record("llm")  # already at max -> aborted
    fake_model = MagicMock()
    fake_model.query.side_effect = AssertionError("must not be called when aborted")

    out = _stage5_blindspot(
        fake_model,
        Facts(),
        answers=[],
        prior_blindspots=[],
        max_blindspots=5,
        round_idx=1,
        max_rounds=3,
        budget=budget,
    )
    assert out == []


def test_stage5_parses_blindspots_and_tags_round():
    budget = CallBudget(max_total=10)
    payload = {
        "blindspots": [
            {
                "description": "weak coverage of LDS path",
                "why_it_matters": "could change recommended strategy",
                "follow_up_question": "is LDS-backed heap viable on CDNA?",
            }
        ]
    }

    fake_model = MagicMock()
    fake_model.query.return_value = {"content": json.dumps(payload)}

    out = _stage5_blindspot(
        fake_model,
        Facts(),
        answers=[],
        prior_blindspots=[],
        max_blindspots=5,
        round_idx=2,
        max_rounds=3,
        budget=budget,
    )
    assert len(out) == 1
    assert out[0].round == 2
    assert "LDS" in out[0].follow_up_question
    assert budget.llm_calls == 1


# ---- _coerce_evidence: accepts legacy dicts/strings without fabricating cites --


def test_coerce_evidence_from_dicts():
    raw = [
        {
            "source_type": "web_arxiv",
            "title": "Paper X",
            "url": "https://arxiv.org/abs/9",
            "snippet": "abc",
            "score": 0.9,
        }
    ]
    out = _coerce_evidence(raw, hits=[])
    assert len(out) == 1
    assert out[0].source_type == "web_arxiv"
    assert out[0].url == "https://arxiv.org/abs/9"


def test_coerce_evidence_from_strings():
    raw = ["kb://Section A", "kb://Section B"]
    out = _coerce_evidence(raw, hits=[])
    assert [e.source_type for e in out] == ["web_search", "web_search"]
    assert out[0].title == "kb://Section A"


def test_coerce_evidence_returns_empty_when_model_omits_legacy_evidence():
    from minisweagent.dra.iterative_search import UnifiedHit

    hits = [
        UnifiedHit(title="Paper X", content="body", score=0.4, origin="web_arxiv", url="https://arxiv.org/abs/x"),
        UnifiedHit(title="Section A", content="kb body", score=0.3, origin="kb", chunk_id="amd/secA"),
    ]
    out = _coerce_evidence(raw=[], hits=hits)
    assert out == []


def test_coerce_evidence_overrides_mislabeled_source_type_from_hits():
    """LLM may pick `web_rocm_docs` for a URL that actually came from
    `web_search` (the open-websearch Google-SERP channel). The actual hit's
    origin must win.
    """
    from minisweagent.dra.iterative_search import UnifiedHit

    hits = [
        UnifiedHit(
            title="AMD Lab Notes",
            content="...",
            score=7.0,
            origin="web_search",
            url="https://gpuopen.com/learn/amd-lab-notes/",
        )
    ]
    raw = [
        {
            "source_type": "web_rocm_docs",  # WRONG label from the model
            "title": "AMD Lab Notes",
            "url": "https://gpuopen.com/learn/amd-lab-notes/",
            "snippet": "..",
        }
    ]
    out = _coerce_evidence(raw, hits)
    assert out[0].source_type == "web_search"


def test_coerce_evidence_keeps_model_label_when_url_unmatched():
    """If the model cites a URL we never fetched, fall back to its label."""
    out = _coerce_evidence(
        [
            {
                "source_type": "web_arxiv",
                "title": "T",
                "url": "https://arxiv.org/abs/9999",
            }
        ],
        hits=[],  # no matching hit
    )
    assert out[0].source_type == "web_arxiv"
