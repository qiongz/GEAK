# Triton-Gluon Backup Details

Backup routing doc. Read this file only after `00_always_read.md` routes the
task to the primary split docs and those routed docs still do not contain the
required detail. It is not planner-default context and not an implementation
cookbook.

This file is intentionally small. Most long-form details now live in their
proper primary docs:

- planner/task allocation -> `10_search_policies.md`
- implementation traits -> `20_component_traits.md`
- runtime, target, arch, JIT/AOT -> `30_architecture_notes.md`
- concrete API, snippets, compatibility checks, debug order ->
  `50_api_reference.md`
- writing model, family differences, real operator patterns, benchmark rules ->
  `60_real_patterns.md`
- schematic examples -> `40_examples.md`

## What This File Is For

Use this file for the last routing decision, not for implementation guidance.

If the primary routed docs lack a detail:

1. Re-check the `Task routing` table in `00_always_read.md`.
2. Read the nearest primary doc listed above.
3. If the detail is still missing, stop instead of guessing.
4. Report which split-doc route was insufficient so the missing detail can be
   moved into the proper primary doc.

## Do Not Guess

If a Gluon API, layout rule, architecture support rule, or benchmark contract is
not present in the routed split docs, do not invent it from memory. Report the
missing documentation as a blocker so the primary split docs can be updated.
