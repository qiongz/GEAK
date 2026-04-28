"""DRA: Deep Research Artifact generation for GEAK.

DRA produces Markdown artifacts that the task generator consumes:

  - deep_search.md                  -- convergent, source-pack-grounded research
  - experimental_directions.md      -- orthogonal probes that challenge
                                       the dominant thesis
  - deep_search_search_records.jsonl -- query/result trace for debugging
  - deep_search_synth_records.jsonl  -- per-answer + telemetry trace

Public API:

    from minisweagent.dra import run_dra, DRAConfig, DRAInputs

    inputs = DRAInputs(
        kernel_path=Path("path/to/kernel.py"),
        output_dir=Path("path/to/run/dir"),
        profile_path=Path("path/to/profile.json"),
        baseline_metrics_path=Path("path/to/baseline_metrics.json"),
        # ... other optional inputs
    )
    paths = run_dra(inputs)
    # paths["deep_search_md"], paths["search_records"], ...

DRA is opt-in. See `config.py` for env-var knobs.
"""

from __future__ import annotations

from minisweagent.dra.config import DRAConfig
from minisweagent.dra.evidence import DRAInputs
from minisweagent.dra.runner import run_dra

__all__ = ["DRAConfig", "DRAInputs", "run_dra"]
