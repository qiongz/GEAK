"""Fixed-mode dispatch: N identical task bodies across GPUs.

This package implements "fixed" mode from the execution plan: the same
task body is replicated across ``num_parallel`` worker slots.  Variance
across workers comes from LLM sampling alone (temperature > 0 or
trajectory seed differences).

Every worker runs the same ``OptimizationAgent`` class — only the
number of copies differs from planned mode (where each worker gets a
distinct planner-generated body).

Key entry points:
- ``run_fixed_mode`` (in ``homogeneous_agent``) -- public dispatch
  function called from ``run/unified.py::_run_fixed``.

Historical note: this package was named ``homogeneous`` when the
codebase had distinct agent classes per dispatch style.  With the
unified ``OptimizationAgent``, the directory name is retained as a
compatibility shim; new code should reach this logic through
``run_pipeline(ctx, mode="fixed")`` in ``run/unified.py`` rather than
importing from here directly.
"""
