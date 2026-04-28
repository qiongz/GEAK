"""Prompt templates for the DRA pipeline.

All LLM call sites in `runner.py` source their prompts from this module so
they can be tuned and diffed in one place.

The runner still expects compact JSON handoffs internally, but the prompts are
written for research quality first. Keep output contracts terse so the model
spends attention on question quality, gap analysis, and synthesis rather than
copying verbose output templates.
"""

from __future__ import annotations

import textwrap

# ---------------------------------------------------------------------------
# Shared header used at the top of every DRA prompt.
# ---------------------------------------------------------------------------

_HEADER = textwrap.dedent(
    """\
    You are a brilliant GPU kernel scientist-engineer inside the GEAK optimization
    system. Your job is to understand the exact local kernel from the source pack,
    then use mandatory web research to discover the outside knowledge needed to
    build the best possible optimized version of that kernel.

    Treat the local source pack, profile, benchmark contract, and hardware context
    as ground truth. Use web results to expand your technical imagination: known
    fast implementations, papers, hardware details, library tricks, and design
    patterns that could change the implementation plan.

    Think like a world-class engineer preparing a code-writing agent. Be concrete,
    skeptical, mechanism-driven, and willing to challenge the obvious bottleneck
    story. Return one compact JSON object with the requested keys only; no prose
    outside JSON and no code fences.
    """
)


# ---------------------------------------------------------------------------
# Stage 0: Fact extraction
# ---------------------------------------------------------------------------

FACTS_PROMPT = (
    _HEADER
    + textwrap.dedent(
        """\

        ## Task
        Read the local source pack like a kernel expert preparing to optimize it.
        Extract not only the visible facts, but the technical shape of the problem:
        what computation is being performed, why it may be slow, which constraints
        are load-bearing, and which implementation levers are visible from the
        source/profile/hardware context.

        ## Inputs
        ### Local source pack
        ```
        {source_pack}
        ```

        ### Prior round summary (may be empty)
        {previous_results}

        ### Round evaluations (may be empty)
        {round_evaluations}

        Return compact JSON with keys:
        kernel_language, kernel_backend, bottleneck_type, hot_kernels,
        benchmark_contract, correctness_constraints, prior_successes,
        prior_failures, likely_targets, notes.

        In `notes`, emphasize kernel-specific reasoning: likely hidden constraints,
        suspicious assumptions, and optimization levers that deserve web research.
        """
    )
)


# ---------------------------------------------------------------------------
# Stages 1+2: Question generation AND ranking in one call.
# Keeping these in one call (instead of two) saves a round-trip without
# losing the ranking lever -- rank_score is produced in the same response.
# ---------------------------------------------------------------------------

QUESTIONS_PROMPT = (
    _HEADER
    + textwrap.dedent(
        """\

        ## Task
        Generate the research questions a brilliant kernel scientist-engineer
        would investigate before writing the best possible optimized version
        of this exact kernel. Each question must expose a decision-changing
        knowledge gap: if the answer goes one way, GEAK tries one
        implementation family; if it goes the other, GEAK avoids that family.

        ## Question routing - REQUIRED
        For EACH question, set the boolean field needs_web:

          needs_web=false  -> answer is fully in the source pack (kernel code,
                              wrapper, build files, profile JSON, hardware
                              context). Examples: "what data layout does the
                              wrapper pass?", "is best1 declared double or
                              float?", "what block size does the launcher
                              use?", "does the harness include torch.sqrt in
                              its timing?". These get a fast no-web synth.

          needs_web=true   -> answer needs outside knowledge: papers, other
                              implementations, hardware specs, ROCm/CDNA
                              primitives, GitHub references, tuning guidance.

        Out of {max_questions} total questions, AIM for roughly 30-40% with
        needs_web=false and 60-70% with needs_web=true. Local questions are
        FOUNDATIONAL - they extract clean facts from the source pack that
        anchor every external answer. Do not skip them just because the
        answer feels obvious to you; the synthesizer benefits from explicit
        anchors. If you find yourself emitting all needs_web=true, STOP and
        re-read the source pack for facts you have not extracted yet.

        ## Search queries
        For needs_web=true questions, include 3-5 concise SERP-style queries.
        Natural-language engineer phrases, not keyword soup. Keep code
        identifiers verbatim. Include the target hardware named in the
        source pack (e.g. gfx950, MI355X) when relevant. For needs_web=false
        questions, search_queries can be empty or a 1-2 item fallback hint.

        ## Scoring
        Score each candidate (integers 0-10):
          decision_impact, actionability, kernel_relevance, novelty
        Set rank_score = decision_impact + actionability + kernel_relevance + novelty.

        ## JSON HYGIENE - this output keeps tripping the parser
        - Output exactly one JSON object with one top-level key: "questions".
        - Each question is an object with these fields and ONLY these fields:
          question, search_queries, rationale, decision_impact, actionability,
          kernel_relevance, novelty, rank_score, needs_web.
        - All string fields must be SHORT (under ~40 words each) and use
          plain ASCII double-quotes. NO markdown, NO backticks, NO triple
          backticks, NO nested code blocks anywhere in any string. If you
          want to mention a function name, just write three_nn_kernel, NOT
          `three_nn_kernel`.
        - search_queries is an array of plain strings.
        - All numeric fields are bare integers/floats, not strings.
        - No trailing comments, no preamble, no closing remarks.

        Return AT MOST {max_questions} of the highest-ranked questions across
        BOTH buckets combined.

        ## Facts
        ```json
        {facts_json}
        ```

        ## Local source pack
        {source_pack}
        """
    )
)


# ---------------------------------------------------------------------------
# Stage 4: Per-question synthesis
# ---------------------------------------------------------------------------

PER_QUESTION_SYNTH_PROMPT = (
    _HEADER
    + textwrap.dedent(
        """\

        ## Task
        Answer the research question below in a way that decides how this
        knowledge changes the optimization strategy for the local kernel.

        Use local facts and the LOCAL CODE below as authoritative for this run.
        The local code section contains the actual source files that will be
        optimized -- the kernel(s), the wrapper(s), the build harness, and
        the benchmark task runner. ALWAYS read them before answering. When
        the question can be answered directly from a local file, do that and
        CITE the file path in the ``answer`` body (for example: ``best1`` is
        declared as ``double`` in ``src/three_nn_cuda.hip``). When useful,
        quote the exact line/snippet you are reasoning from rather than
        paraphrasing it. Do not paraphrase a similar implementation from web
        material when the local code is right here.

        ## Two answer modes

        - LOCAL-ONLY mode: if the "Web-search + read material" section says
          ``(no open-search results retrieved)``, this is a local-only
          introspection question. Answer it directly from the local source
          pack with file:line citations. Keep the answer under ~150 words --
          this is a fact extraction, not a literature review. ``status``
          should be ``prefer`` (if the answer points to a concrete optimization
          opportunity), ``reject`` (if it rules one out), or ``open`` (if the
          local code reveals a missing measurement).

        - WEB-AUGMENTED mode: web material is present. Use it as external
          knowledge: implementation precedent, hardware guidance, warning, or
          counterexample. If web material conflicts with local source/profile
          facts, local facts win.

        Rank usefulness for optimization (``status`` field):
          - "prefer":       likely useful for a high-value patch
          - "deprioritize": plausible but lower priority or weakly applicable
          - "reject":       likely bad for this local kernel/profile
          - "open":         needs local measurement or source inspection
                            before GEAK should commit to anything

        Think mechanistically. Explain why the idea could help or fail on this
        exact workload/backend/hardware, not merely that it appears in a paper
        or repository. End with the implementation consequence: what GEAK
        should try, avoid, or measure because of this research.

        ## "I don't know" is a first-class answer
        If neither the local code nor the web material lets you answer the
        question concretely, your answer MUST be:
          1. Short. ONE OR TWO SENTENCES. Hard cap ~50 words.
          2. Name the exact file/function/measurement that would resolve it
             ("look at lines X-Y of src/foo.hip" or "run `rocprof --stats`
             with the .vgpr_count metric").
          3. Set ``status: "open"``.
          4. Set ``taskgen_implications`` to the concrete next step GEAK
             should take (also short).

        DO NOT pad an "I don't know" answer with generic background, related
        implementations, or "based on standard X" paraphrasing. A 30-word
        "I don't know, inspect file X" is strictly better than a 300-word
        confabulation. The runner truncates long status=open answers that
        carry no local citation, so there is no benefit to padding -- the
        truncated stub is what downstream sees anyway.

        ## Question
        {question}

        ## Local code excerpts (selected from the source pack)
        {local_code}

        ## Facts
        ```json
        {facts_json}
        ```

        ## Web-search + read material
        {kb_chunks}

        ## Prior-run context (may be empty)
        {prior_run_context}

        Return compact JSON with keys: answer, affected,
        taskgen_implications, status.

        Do not include a citation list. Mention any essential source names/URLs
        inline in `answer` only when they directly shape the implementation idea.
        """
    )
)


# ---------------------------------------------------------------------------
# Query refinement (used by iterative_search.py when first try is weak)
# ---------------------------------------------------------------------------

QUERY_REFINEMENT_PROMPT = (
    _HEADER
    + textwrap.dedent(
        """\

        ## Task
        We are doing iterative deep research. Given the current research
        direction, previous queries, and result titles, write the next query that
        would add the most useful new information.

        This is not only for weak results. If results are promising, dig deeper:
        search for source code, papers, hardware details, benchmarks, or a more
        specific variant that could improve the final kernel strategy.

        ## Hard constraints on the rewrite
        - Output concise human web-search query.
        - Prefer natural-language phrases over keyword soup.
        - Keep code identifiers verbatim (e.g. ``knn_kernel``, ``__shfl_xor``).
        - Include source intent when useful but this is not necessary: "GitHub", "paper", "arXiv",
          "ROCm docs", "CUDA implementation", "HIP implementation".
        - Include exact target hardware from the source pack when relevant,
          such as GPU name, gfx arch, microarchitecture, wavefront, LDS, VGPR.
          Do not guess hardware that is not present in the source pack.
        - Avoid asking for local repo facts; ask for external implementations,
          papers, docs, or hardware techniques.

        Pick ONE strategy:
          - swap a vague term for a specific one (e.g. "kernel" -> "knn_kernel")
          - shift to a sibling formulation (cause-and-effect -> the named pattern)
          - drop the qualifier and search for the bare concept
          - target a specific source family if needed (e.g. "GitHub ROCm point cloud KNN implementation")

        ## Original question
        {question}

        ## Queries already tried
        {tried_queries}

        ## Sample of current results (titles only)
        {weak_titles}

        Return compact JSON with key `refined_query`.
        """
    )
)


# ---------------------------------------------------------------------------
# Search-result triage (search -> deeper search -> read)
# ---------------------------------------------------------------------------

SEARCH_TRIAGE_PROMPT = (
    _HEADER
    + textwrap.dedent(
        """\

        ## Task
        Triage open-search results for one DRA research question.

        Decide:
        - which URLs are worth reading now
        - whether the current result set is sufficient
        - if not sufficient, what deeper follow-up search queries should be tried

        Be selective. Prefer official docs, arXiv HTML/abstract pages, GitHub
        source/code pages, papers, and high-signal technical posts. Avoid generic
        SEO pages, unrelated tutorials, duplicate URLs, and results that only
        match broad words.

        ## Research question
        {question}

        ## Search round
        {round_idx} of {max_rounds}

        ## Query used
        {query}

        ## Results
        {results}

        Return compact JSON with keys:
        selected_urls, sufficient, follow_up_queries, rationale.
        `selected_urls` should contain at most {read_top_k} URLs from Results.
        `follow_up_queries` should contain 0-3 short search queries, only if
        `sufficient` is false and another round remains.
        """
    )
)


# ---------------------------------------------------------------------------
# Stage 5: Blindspot / meta critique
# ---------------------------------------------------------------------------

BLINDSPOT_PROMPT = (
    _HEADER
    + textwrap.dedent(
        """\

        ## Task
        You are running a skeptical thinking-blindspot pass (round {round_idx} of
        up to {max_rounds}) over the per-question answers gathered so far. Your job
        is to find missing ideas, bad assumptions, and alternative optimization
        frames that could change GEAK's next patch. Do not merely agree with the
        current thesis.

        Look for:
          - strategy families that are underexplored
          - over-commitment to the bottleneck label
          - over-weighting of wrapper / layout changes
          - under-weighting of wrapper / layout changes when kernel work may be the trap
          - important files or dependencies that are missing
          - recommendations that conflict with each other
          - missing source-code facts needed to implement the recommendation
          - profile ambiguities that could make a "fast" idea irrelevant
          - benchmark-overfitting risk or suspicious wrapper-only wins
          - optimization families that the current answers mention but do not
            operationalize into an edit
          - cases where DRA is repeating generic GPU advice instead of reasoning
            from this exact source pack
          - algorithmic reframings that web search did not naturally surface
          - hardware-specific primitives or limits the current thesis ignored

        Each blindspot must include a follow-up question that would force sharper
        thinking. It may require web research, source inspection, profiling, or a
        small local measurement, but it should not be a request for "more sources."
        It should point toward a concrete implementation decision.

        ## Avoid repeating prior blindspots
        The blindspots below were already raised in earlier rounds. Do NOT
        emit blindspots that are semantically duplicates of these. If you have
        nothing genuinely new and important to raise, return an empty list.

        ```json
        {prior_blindspots_json}
        ```

        Return AT MOST {max_blindspots} NEW blindspots, ranked by expected ability
        to change the next patch. Prefer a diverse set over many variants of the
        same missing fact.

        ## Facts
        ```json
        {facts_json}
        ```

        ## All answers so far (first pass + any prior rounds)
        ```json
        {answers_json}
        ```

        Return compact JSON with key `blindspots`. Each item should have:
        description, why_it_matters, follow_up_question.
        """
    )
)


# ---------------------------------------------------------------------------
# Stage 7: Final artifact synthesis
#
# Stage 7 used to ask for ranked_hypotheses + taskgen_guidance + a 1200-word
# executive_summary_md in one JSON object. That payload routinely tripped the
# JSON parser (and even an LLM-driven repair pass) because Markdown with code
# fences and nested quotes is hostile to JSON-string escaping. We now split
# the work into two calls: a small structured-guidance call (cheap to parse)
# and a free-form Markdown executive-summary call (no JSON parsing at all).
# ---------------------------------------------------------------------------

FINAL_GUIDANCE_PROMPT = (
    _HEADER
    + textwrap.dedent(
        """\

        ## Task
        Produce the structured task-generator guidance block from the per-question
        answers and blindspots below. This is the handoff to an autonomous
        code-writing agent: every item must be actionable.

        Resolve contradictions across answers (the blindspots often catch them).
        Group affected files/functions. Produce ranked hypotheses (most to least
        supported) and a `taskgen_guidance` block that the task generator will
        act on directly.

        The guidance must separate:
          - high-confidence implementation bets to try first
          - second-tier ideas that are plausible but lower leverage
          - ideas to reject or avoid because mechanism/profile/source constraints
            say they are weak
          - open measurements that should be gathered before spending a patch

        For each `prefer_first` item, include enough detail that a sub-agent
        could start coding: target function/file, edit pattern, expected speedup
        mechanism, and one validation/kill criterion. Do not say "optimize
        memory" without naming the concrete edit pattern.

        IMPORTANT JSON HYGIENE:
          - Keep every string short (one or two sentences). DO NOT embed
            markdown code fences, multi-paragraph essays, or large quoted
            passages anywhere in this JSON.
          - Long-form prose belongs in the SEPARATE executive-summary call,
            not here. If you find yourself writing more than ~50 words for one
            item, move the detail to the summary call and keep the guidance item
            as a one-liner pointer.
          - All strings must use plain ASCII quotes. No nested triple-backticks.

        ## Facts
        ```json
        {facts_json}
        ```

        ## All answers (first + second pass)
        ```json
        {answers_json}
        ```

        ## Blindspots
        ```json
        {blindspots_json}
        ```

        Return compact JSON with keys: ranked_hypotheses, taskgen_guidance.
        `ranked_hypotheses` is an array of short strings.
        `taskgen_guidance` contains four arrays: prefer_first, deprioritize,
        reject, open_questions. Items may be concise markdown strings or small
        objects with fields like target / edit_pattern / expected_upside /
        kill_criteria. Prefer mechanism and implementation detail over citation
        lists.
        """
    )
)


# Free-form Markdown — NO JSON parsing on the response. This is the document
# the human / task generator reads first, so spend the model's budget here.
#
# Do NOT ask the model to re-print the structured guidance below this. The
# rendered artifact already includes a "Task-Generator Guidance" section
# right under this summary; copying the same Prefer First / Reject lists here
# is just noise. The summary's job is to give the NARRATIVE the structured
# guidance answers: why this strategy beats alternatives, where the evidence
# is strongest vs weakest, and what the operator should watch for.
FINAL_EXEC_SUMMARY_PROMPT = (
    _HEADER
    + textwrap.dedent(
        """\

        ## Task
        Write the "Task-Ready Summary" that sits at the very top of
        `deep_search.md`. This is the FIRST thing a code-writing agent or human
        reviewer reads. The artifact also includes a "Task-Generator Guidance"
        section right BELOW this summary with the explicit Prefer First /
        Deprioritize / Reject / Open lists; do NOT duplicate those lists here.

        Your job is to give the narrative around them: the mechanism story,
        the trade-offs, and the confidence story. A reader should finish this
        summary knowing WHY the prefer list looks the way it does and what
        could falsify it, not just WHAT is on the list.

        Required structure (use these exact section headers, in this order):

          ### Bottom line
          One paragraph (3-5 sentences). State the bottleneck, the one
          highest-leverage change, the mechanism by which it wins, and the
          single most decisive kill criterion. No lists.

          ### Why this strategy and not the obvious alternatives
          A short narrative (2-4 sentences or a 3-bullet list) explaining the
          mechanism the prefer-first plan is exploiting and why the most
          tempting alternatives (named explicitly) are weaker on this exact
          kernel/profile/hardware. Reference the local source pack files by
          path when a code-level detail is load-bearing.

          ### Where confidence is high vs low
          A short bulleted breakdown: which prefer-first items are mechanism-
          locked (high confidence) vs which depend on a measurement we have
          not yet made (low confidence, name the measurement).

          ### Risks the implementer must watch for
          2-4 bullets. Each one names a specific failure mode (e.g. "VGPR
          spill if K=2 register blocking pushes past ~50 VGPRs"), the symptom
          the implementer would see, and the fallback.

        Hard constraints:
          - Keep total length under ~700 words. The structured guidance below
            does the heavy lifting; this summary is the narrative on top of it.
          - DO NOT restate the prefer-first / reject / open lists. Reference
            them by name where useful ("the LDS-tiling bet"), do not enumerate.
          - Be specific to the local kernel: cite source files by path, cite
            facts (B/N/M shapes, hardware target, bottleneck) explicitly.
          - Output Markdown only. NO JSON, NO surrounding code fence, no
            preamble like "Here is the summary:", no closing remarks.

        ## Facts
        ```json
        {facts_json}
        ```

        ## Structured guidance (already resolved; this is the playbook the
        ## artifact will render below your summary - do not restate it)
        ```json
        {guidance_json}
        ```

        ## All answers (first + second pass) - use for mechanism / evidence
        ```json
        {answers_json}
        ```

        ## Blindspots - use to populate the "Risks" and "low confidence" sections
        ```json
        {blindspots_json}
        ```
        """
    )
)


# ---------------------------------------------------------------------------
# Experimental directions (separate pass, runs after deep_search)
# ---------------------------------------------------------------------------

EXPERIMENTAL_PROMPT = (
    _HEADER
    + textwrap.dedent(
        """\

        ## Task
        Propose orthogonal, deliberately-different optimization directions that
        challenge the dominant thesis represented by `deep_search` below.

        Orthogonal here means at least one of:
          - a different strategy family
          - a different target file or dependency
          - a different backend choice
          - a different optimization granularity
          - a different assumption about what is truly limiting performance

        Each direction MUST include `assumption_challenged` and `kill_criteria`.
        These are the load-bearing fields that keep this branch disciplined
        instead of random.

        Return between 2 and 5 directions.

        ## Facts
        ```json
        {facts_json}
        ```

        ## Deep-search summary (the thesis to challenge)
        ```json
        {deep_search_summary_json}
        ```

        Return compact JSON with key `directions`. Each direction should include:
        direction_id, thesis, why_orthogonal, assumption_challenged,
        strategy_family, target_files_or_functions, expected_upside,
        implementation_cost, kill_criteria, notes_for_taskgen.
        """
    )
)
