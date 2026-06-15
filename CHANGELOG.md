# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Multi-language graph analysis — JavaScript/TypeScript.** JS/TS agents are now
  recovered into the same Workflow IR as Python and run through the *same* graph
  detectors (recursive/loop cycles via Tarjan SCC, context explosion, fan-out),
  token prediction, simulation, and scoring — not just the advisory textual lint.
  Two real shapes are recovered, deterministically and stdlib-only (no Node, no
  tree-sitter): **LangGraph.js / `StateGraph`** builders (`addNode` / `addEdge` /
  `addConditionalEdges` / `setEntryPoint`, with back-edges classified as unbounded
  cycles) and **imperative agents** (an infinite `while`/`for` loop around an LLM
  SDK call). A comment/string-aware blanker means builder calls inside comments or
  strings are never mistaken for real edges. Files it can't recover fall back to the
  textual lint (honest failure — never a fabricated graph). New parser
  `parsers/javascript.py`; `.js/.jsx/.ts/.tsx/.mjs/.cjs` join discovery; cap/output
  lint is merged into JS workflows without double-counting loops. A general
  tree-sitter backend for arbitrary JS/TS control flow remains a future option.
- **PR-delta gating (`--baseline`).** Gate on the *change*, not the whole repo: a
  fresh run is diffed against a baseline report and only **new or worsened**
  findings can fail the build — pre-existing risk in untouched code is reported as
  `unchanged` and never blocks the PR (resolved findings show as `fixed`). Finding
  identity is line-number-independent (category + file + node + digit-normalized
  message) and occurrence-counted, so unrelated edits don't look "new" while a real
  new occurrence still counts. The delta is the headline of the markdown/terminal/
  HTML reports and is serialized under `baseline_diff` in JSON. The GitHub Action
  gains `pr-delta: true` (auto-builds the baseline from the PR's base commit) and a
  `baseline:` input. CLI: `tollgate analyze … --baseline report.json`. Works
  identically across every language layer (graph, Python AST lint, textual lint).
- **Language-agnostic agentic lint.** Beyond Python's AST checks, the linter now
  runs a deterministic textual pass on any other language (JS/TS, Go, Java, Ruby,
  …) that flags the two universal risks — an infinite loop wrapping an LLM call,
  and an LLM call with no output-token cap. Advisory; never claims a recovered graph.
- **Language-agnostic prompt mining (`prompt_scan`).** Finds LLM prompts hidden in
  source as string constants / heredocs / config values across any language
  (Python, JS/TS, Go, Ruby, YAML, …) via a heuristic literal scanner with
  negative-signal filtering (SQL/HTML/logs/code are ignored). Advisory only —
  surfaced in all reporters, never affects the gate. Config: `prompt_scan`,
  `prompt_scan_min_score`; CLI: `--no-prompt-scan`.
- `requests_per_week` / `requests_per_day` scenario keys in `.tollgate.yml`.
- Self-healing outputs: report `fingerprint` + `tollgate verify`.

## [0.1.0] — 2026-06-13

Initial public release.

### Added
- **Parsers → Workflow IR:** LangGraph, CrewAI (`Agent`/`Task`/`Crew`, delegation
  loops), AutoGPT block-graph exports, hand-rolled imperative agents (loop around a
  recognized LLM SDK call, framework-agnostic), a native YAML/JSON DSL, and raw
  prompt templates.
- **Prediction & simulation:** per-node token distributions (p50/p95/p99),
  Monte-Carlo over a configurable traffic base (default 10,000 requests/week),
  monthly forecast. Exact tokenization with optional `tiktoken`, deterministic
  heuristic otherwise.
- **Detectors:** recursive/delegation loops (Tarjan SCC + termination-guard),
  context explosion, fan-out, prompt bloat, retry storms, model mismatch.
- **Strict agentic lint:** source-level checks for unbounded loops, missing
  iteration/recursion caps (LangChain / AutoGen / LlamaIndex / smolagents / CrewAI /
  LangGraph), uncapped output tokens (incl. LangChain/LlamaIndex model wrappers), and
  unbounded fan-out. `lint_strictness: strict | balanced | off`.
- **Decision & output:** 0–100 risk score with `pass | warn | block` gate, policy
  engine, cheaper-model right-sizing, remediation plan; reporters for
  terminal/markdown/json/sarif/gitlab/html.
- **Self-healing outputs:** every report carries a tamper-evident fingerprint, and
  `tollgate verify` re-derives the gate to catch edited or stale reports in CI.
- **Traffic controls:** `--traffic-per-week` / `--traffic-per-day` / `--horizon-days`
  on the CLI, the `run_study` harness, and `scripts/scan-github-repo.sh`.
- **CI integrations:** GitHub composite Action, GitLab CI template (Code Quality +
  gate), pre-commit hook; one-shot read-only repo scanners under `scripts/`.
- **Validation suite:** labeled corpus + benchmark harness (precision/recall/F1 vs.
  baselines, train/held-out split), metamorphic / fuzz / mutation / determinism /
  calibration, plus sampled-precision tooling (sample/precision/recall + an
  independent auto-triage oracle).

[Unreleased]: https://github.com/nirupam014/tollgate/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nirupam014/tollgate/releases/tag/v0.1.0
