# Tollgate

**Prevention-first token-risk analysis for AI agents — a strict gate in your CI/CD.**
Tollgate statically analyzes your agent workflows and prompts *before they ship*,
predicts token consumption, detects the structural failure modes that cause runaway
spend (context explosion, recursive/delegation loops, uncapped generation), recommends
cheaper models, and **blocks the pull/merge request** when a change is too risky.

It is a static + simulated analyzer, not a runtime meter — the point of control is the
PR, not the invoice. It is **deterministic, makes no LLM calls, and never executes the
code it scans.**

## Why it's trustworthy

- **Deterministic & offline.** Pure static analysis (Python `ast`) + Monte-Carlo math; no model calls, reproducible output.
- **Never runs your code.** It parses and reasons about source; it does not import or execute it.
- **Honest by construction.** Verifiable structural findings are kept separate from fuzzy cost estimates; figures are reported in **tokens**, not invented dollars; unrecognized inputs are dropped (honest failure) rather than scored as a misleading PASS.
- **Tamper-evident.** Every report carries a fingerprint binding the verdict to its exact inputs; `tollgate verify` re-derives it so an edited or stale gate output is caught in CI.

## What it does

1. **Parses agent workflows & prompts → a normalized IR.** LangGraph, **CrewAI** (`Agent`/`Task`/`Crew`, flags hierarchical/delegation loops), AutoGPT block-graph exports, **any hand-rolled imperative agent** (a `while`/`for` loop around a recognized LLM SDK call — framework-agnostic), a native YAML/JSON DSL, and raw prompt templates.
2. **Predicts token consumption** — per-node input/output distributions (p50/p95/p99), exact with `tiktoken` or a deterministic heuristic otherwise.
3. **Simulates cost under traffic** — Monte-Carlo over a configurable base (default **10,000 requests/week**; override per run with `--traffic-per-week` / `--traffic-per-day`).
4. **Detects context explosion** — history/retrieval growth inside loops vs. the model's context limit.
5. **Detects recursive/delegation loops** — Tarjan SCC + termination-guard analysis; unbounded cycles are critical.
6. **Strict agentic lint** — a source-level reviewer for *agentic-specific* risks only: unbounded loops, **missing iteration/recursion caps** (LangChain `AgentExecutor`, AutoGen `GroupChat`/teams, LlamaIndex `ReActAgent`, smolagents `CodeAgent`, CrewAI `max_iter`, LangGraph `recursion_limit`), **uncapped output tokens** (including LangChain/LlamaIndex model wrappers), and **unbounded fan-out**. Python gets high-fidelity AST checks; **any other language** (JS/TS, Go, Java, Ruby, …) gets a deterministic, advisory textual pass that flags the two universal risks — an infinite loop wrapping an LLM call, and an LLM call with no output cap. It stays silent on non-agentic code. Tunable: `lint_strictness: strict | balanced | off`.
7. **Recommends cheaper models** — re-prices each node under safe alternatives (a cost lever; same token volume).
8. **Generates a deployment risk score** — 0–100 with a `pass | warn | block` gate and driver breakdown.
9. **Integrates with GitHub & GitLab** — a GitHub Action (check-run + PR comment + SARIF), a GitLab CI template (Code Quality + pipeline gate), and a pre-commit hook.
10. **Executive forecasts + engineering remediation** — projected monthly tokens with drivers, plus ranked, copy-pasteable fixes.
11. **Mines prompts hidden in code (any language)** — finds LLM prompts living as string constants/heredocs/config values (`prompts.py`, `prompts.ts`, YAML, Go raw strings, Ruby heredocs, …) via a heuristic literal scanner, so prompt bloat isn't invisible just because it sits outside a recognized LLM call. Detection is heuristic and **advisory** — it surfaces candidates for review and never drives the gate. Tunable via `prompt_scan` / `--no-prompt-scan`.

Recognized LLM call surfaces include OpenAI and OpenAI-compatible vendors (Azure, Groq,
Together, DeepSeek, Fireworks, OpenRouter, xAI, Perplexity, vLLM, LM Studio, Ollama
`/v1`), Anthropic, Google Gemini/Vertex, Mistral, Amazon Bedrock, Cohere, Replicate,
Ollama, Hugging Face, and LiteLLM — plus LangChain and LlamaIndex model wrappers.

## Install

```bash
pip install tollgate                  # core (deterministic heuristic tokenizer)
pip install "tollgate[tokenizers]"    # + tiktoken for exact OpenAI-family counts
pip install ./tollgate                # from source
```

## Quick start

```bash
tollgate analyze ./agents ./prompts --fail-on block   # scan & gate
tollgate analyze ./agents --traffic-per-week 50000    # set the traffic estimate
tollgate init                                         # write a starter .tollgate.yml
tollgate models                                       # inspect the model catalog
tollgate verify report.json ./agents                  # re-derive & detect a tampered/stale report
```

Exit codes: `0` pass/warn, `1` block (or warn with `--fail-on warn`), `2` usage error.

Try the bundled examples:

```bash
tollgate analyze examples/workflows/runaway_agent.yaml   # -> BLOCK
tollgate analyze examples/agents/crewai_hierarchical.py  # -> BLOCK (delegation loop)
tollgate analyze examples/workflows/safe_pipeline.yaml   # -> PASS
```

## Scan any GitHub repo (one shot, read-only)

```bash
scripts/scan-github-repo.sh https://github.com/org/agents
scripts/scan-github-repo.sh https://github.com/org/private --token "$GITHUB_TOKEN"
scripts/scan-github-repo.sh https://github.com/org/repo --traffic-per-week 50000
```

Shallow-clones into a temp dir, writes reports (md/json/sarif/html) outside the clone,
and always deletes the clone. It never modifies the scanned repo.

## GitHub / GitLab / pre-commit

- **GitHub:** add `.github/workflows/tollgate.yml` (see `ci-templates/github-workflow.yml`); posts a sticky PR comment, uploads SARIF, fails the check on `block`. Make it a required status check to block merges.
- **GitLab:** add `ci-templates/.gitlab-ci.yml`; publishes a Code Quality report and fails the pipeline on `block`.
- **Local:** `ci-templates/.pre-commit-hooks.yaml`.

## Configuration — `.tollgate.yml`

```yaml
default_model: gpt-4o
fail_on: block
lint_strictness: strict          # strict | balanced | off

# Base traffic assumption: 10,000 requests/week. Override per run with
# --traffic-per-week / --traffic-per-day.
scenarios:
  - { name: steady_state, requests_per_week: 10000, horizon_days: 30 }
  # also accepted: requests_per_day: 1500  —  or the raw rps: 0.0165

substitution: { min_capability: 0.75, min_savings_pct: 20 }

policies:
  - name: loops_must_terminate
    type: loop_guard
    enforcement: block
    rule: { require_termination_guard: true, max_depth: 10 }
  - name: prod_token_ceiling
    type: token_ceiling
    enforcement: block
    rule: { max_monthly_tokens: 2000000000, metric: projected_p95 }
```

Policy types: `token_ceiling`, `model_allowlist`, `context_cap`, `loop_guard`,
`gate_threshold`. Ceilings are in **tokens**, not dollars. Point `--models ops/models.yaml`
at your own catalog for accurate pricing (the bundled catalog is illustrative).

## Output formats

`-f terminal | markdown | json | sarif | gitlab | html`, and `-o format=path` to write
files (e.g. `-o markdown=report.md -o sarif=out.sarif -o html=dashboard.html`). Every
report includes a **fingerprint**; re-check it any time with `tollgate verify`.

## Validation & precision

Tollgate separates *proven correctness* from *behavior in the wild*:

- **Correctness (`validation/`).** A hand-labeled corpus with known-correct answers.
  `validation/harness.py` scores discovery, unbounded-loop precision/recall/F1 (against
  trivial baselines it must beat), gate accuracy and recommendation accuracy, and
  `--strict` makes it a CI gate. It supports a **train/held-out split** so detection
  tuning is never evaluated on the cases it was tuned against. The metamorphic, fuzz,
  mutation and determinism suites live here too.
- **Sampled precision.** Over any large finding set you can publish one honest
  correctness number — adjudicated precision with a **Wilson 95% CI** — by labeling a
  random sample: `validation/sample.py` → fill verdicts → `validation/precision.py`
  (and `validation/recall.py` for the miss-rate). `validation/auto_triage.py` is an
  independent oracle that pre-labels the kwarg-decidable findings so you only
  hand-adjudicate what it can't confirm (reported as *agreement*, never as validated
  precision).

```bash
python -m unittest discover -s tests          # unit suite (stdlib only)
python validation/harness.py --strict         # correctness benchmark / CI gate
bash validation/run_all.sh                     # full validation suite
```

## Field study

Tollgate's behavior was measured across a large public population of agent
repositories. The results are published in aggregate and **anonymized**:
[`FIELD-STUDY.md`](FIELD-STUDY.md) (summary) and
[`docs/field-study.html`](docs/field-study.html) (interactive report). The
repositories that make up the population are credited by name in
[`ACKNOWLEDGEMENTS.md`](ACKNOWLEDGEMENTS.md). The study is a behavior measurement, not
a correctness proof — the latter lives in `validation/`.

## Repository layout

| Path | What it is |
|---|---|
| `src/tollgate/` | The analyzer library + `tollgate` CLI (parsers, prediction, simulation, detectors, lint, scoring, reporters, fingerprint/verify). |
| `validation/` | Correctness benchmark + precision tooling: labeled corpus, `harness.py`, sample/precision/recall, auto-triage oracle, metamorphic / fuzz / mutation / determinism / calibration. |
| `scripts/` | One-shot read-only repo scanners. |
| `ci-templates/` | GitHub workflow, GitLab CI, and pre-commit templates. |
| `action.yml` | GitHub composite Action. |
| `examples/` | Runnable sample workflows and agents. |
| `docs/`, `FIELD-STUDY.md`, `ACKNOWLEDGEMENTS.md` | Published field-study results and credits. |
| `artefacts/` | Larger platform design (`Tollgate-Design.md`) and the product requirement doc. |

## How it works

```
parsers ─► Workflow IR ─► prediction ─► simulation ─┐
                              │                      ├─► risk scorer ─► gate (pass/warn/block)
        agentic lint ────────┤   detectors ─► policy ┘        │
                              └──────────► substitution ─► remediation + forecast
```

This CLI is the pre-deploy control plane of the larger Tollgate platform design
(`artefacts/Tollgate-Design.md`), packaged to run anywhere your CI does.

## Contributing

Issues and PRs welcome. Before submitting: run the unit suite and
`validation/harness.py --strict` (both must pass). New detection behavior should come
with a labeled fixture in `validation/corpus/` so it's covered by the benchmark. Keep
the core deterministic and dependency-light (stdlib + PyYAML; `tiktoken` optional).

## License

Apache-2.0.
