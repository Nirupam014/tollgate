# Tollgate field study (summary)

A one-shot measurement of how Tollgate behaves across a large, public population
of agent repositories. **This is a behavior measurement in the wild, not a
correctness proof** — see the honesty note below. Correctness is proven separately
on a labeled corpus (`validation/`).

The repositories that make up the population are credited by name in
[ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md). Findings here are reported **in
aggregate and anonymized** — never as a named call-out.

Full interactive report: [`docs/field-study.html`](docs/field-study.html).

## Headline numbers (500 repositories)

- Analyzed: **484 ok**, 16 could not be cloned.
- In scope (imported an agent framework / SDK marker): **446**.
- **Discovery rate: 96%** — a repo counts as discovered if it yields a workflow
  graph *or* an embedded prompt. Recovered a workflow graph: **301**; reached only
  via embedded prompts: **127**; honestly declined (neither): **18**.
- Gate split over the 301 repos with a recovered graph: **74 block · 130 warn ·
  97 pass**. (Prompt-only repos are not gated — prompt detection is advisory.)
- Structural findings: **7,341** — `uncapped_output` 6,456, `missing_iteration_cap`
  331, `recursive_loop` 239, `prompt_bloat` 203, `fanout` 112.
- **Prompts mined from code/config: 108,111** across **429** repositories, with
  **~3.06M tokens reclaimable** in total (sum of per-call savings; ~28 tokens/call
  per prompt on average, per the deterministic reviewer).

## What changed since the previous run

- **Prompt mining is new.** Language-agnostic prompt detection surfaced ~108K
  embedded prompts across 429 of 446 in-scope repos — previously invisible. This
  is what lifts the discovery rate from ~67% (workflow-graph only) to **96%**.
- **Structural findings are stable.** The graph + lint finding total is essentially
  flat run-to-run (~7.3K; `uncapped_output` remains ~88% of the mix), i.e. the
  detection engine is reproducible, not drifting.
- The gate is now reported strictly over repos with a recovered graph; prompt
  detection never moves the gate.

## What this can and cannot claim

A thousand random public repos have **no ground truth**, so this study can only
*describe behavior* (the counts above) — it cannot, by itself, prove those
verdicts are correct. The single correctness number the study is entitled to
publish is an **adjudicated precision** with a Wilson 95% confidence interval,
from a hand-labeled random sample (see `validation/sample.py` →
`validation/precision.py`, plus the independent `validation/auto_triage.py`
pre-labeler). That figure is published here only once a sample has been labeled;
until then, treat the counts as descriptive, not validated.

Two specific caveats on the new prompt numbers: detection is a **heuristic** (a
large, SDK-heavy repo can inflate the count with many prompt-like literals), and
the reclaimable-token figures are **advisory** per-call estimates from a rule-based
reviewer, not guarantees.

_Generated from the study run; regenerate the HTML with the (private) study
harness. Counts are descriptive measurements, not validated-correct numbers._
