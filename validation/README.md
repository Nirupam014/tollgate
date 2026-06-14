# Tollgate validation

This directory is the evidence behind the claim that Tollgate works. Tollgate is
a prevention-first analyzer: it gates CI/CD on token-consumption risk, so a wrong
verdict either ships a cost bomb (false pass) or blocks a safe PR (false block).
"It runs" is not enough — the question is *are the verdicts correct, and when the
estimates are wrong, are they wrong in the safe direction?*

Run everything:

```bash
./validation/run_all.sh           # full suite, exhaustive mutation gate
FAST=1 ./validation/run_all.sh    # skip the exhaustive mutation pass
```

Every gate is hermetic: standard-library Python only, no network, no extra
dependencies. The same commands run in `.github/workflows/validation.yml`.

## The honest core: structural verdicts vs. cost magnitudes

Tollgate produces two very different kinds of output, and they are validated
differently because they deserve different levels of confidence:

- **Structural verdicts** — *is there an unbounded loop? does context accumulate
  inside a cycle? what is the gate decision (pass/warn/block)?* These are derived
  from graph structure (cycle detection, termination guards, fan-out) and are
  **exactly correct or they are bugs**. They are pinned hard: boundary unit tests,
  metamorphic invariants, a labeled corpus, and mutation testing.

- **Cost magnitudes** — *how many tokens, how many dollars per month?* Without
  telemetry these are heuristic estimates. We do not pretend they are exact. We
  hold them to an explicit, falsifiable tolerance (order-of-magnitude cold,
  tight once telemetry exists) and we surface their known weaknesses rather than
  hide them.

Keeping these separate is the whole methodology. A mutation that only perturbs a
token *magnitude* legitimately survives the structural gate — it is caught (or
explicitly tolerated) by the calibration harness instead. Conflating the two
would force us either to over-claim on cost or to fake tests on structure.

## The five gates

### 1. Labeled benchmark corpus (`harness.py`)

11 hand-labeled workflows with known-correct discovery, gate decision, and
remediation. Measured: discovery **11/11**, gate decision **8/8**, recommendation
**2/2**. On unbounded-loop detection Tollgate scores **F1 = 1.00** and beats three
trivial baselines that a skeptic would reach for — `always_block` (F1 0.61,
because it false-positives every safe PR), `grep_while_true` (0.57), and
`always_pass` (0.00). `--strict` fails on any hard-expectation violation.

### 2. Metamorphic relations (`metamorphic.py`)

Properties that must hold under semantics-preserving edits: renaming nodes,
reordering edges, or duplicating an unrelated branch must not change the verdict;
adding a real unbounded loop must never *lower* the risk band; telemetry that
reports higher depth must never *reduce* a loop's severity. These catch logic
that happens to pass point examples but violates an invariant.

### 3. Fuzz (`fuzz.py`)

Two safety invariants over thousands of randomly generated graphs:
**I1 — never crash** (a malformed workflow yields an honest finding, not a
traceback) and **I2 — never silently pass** (a graph containing a planted
unbounded loop is never returned as clean). Default 2000 iterations, fixed seed,
reproducible.

### 4. Determinism (`test_determinism.py`)

The same input produces byte-identical reports across repeated runs. A gate that
flickers is not a gate; this pins ordering and formatting so diffs in CI are
real signal.

### 5. Cost calibration (`calibration.py`)

Replays recorded traces and compares predictions to measured token usage, in two
regimes with **documented, gated thresholds** (each maps to a sentence we are
willing to be held to):

| regime | claim | gate | measured |
| --- | --- | --- | --- |
| **cold** (no history) | point predictions are order-of-magnitude | ≥60% within **3×** | 83% within 3× |
| **warm** (telemetry) | once Tollgate has seen your traffic, predictions are tight | ≥80% within **2×** and MAPE ≤ 0.25 | 100% within 2×, MAPE 0.073 |

The p95 *envelope* (predicted p95 ≥ measured p95) is reported but **not** gated:
a sample p95 is exceeded ~5% of the time by definition, so over a handful of
nodes an unbiased estimate sits above the holdout only ~half the time — gating it
would be a vanity metric. We surface it so users can see the cold heuristic's
real weakness (it under-states the input tail when runtime content dwarfs a short
static prompt) — exactly the gap telemetry closes.

## Mutation testing (`mutation.py`) — and why the gate is not a kill rate

Mutation testing perturbs the source (swap `>=`→`>`, `and`→`or`, flip a boolean
constant) and checks whether the test suite notices. It answers "are the tests
actually load-bearing, or do they pass vacuously?"

Standard mutation tools (mutmut) aren't installable in this hermetic environment,
so `mutation.py` is a dependency-free AST mutator. It targets four logic modules —
`detectors.py`, `scoring.py`, `graphutil.py`, `prediction.py` — with four gated
operator classes: comparison swaps, binary-op swaps, boolean-connective swaps,
and boolean-constant flips. Each mutant is run against the structural kill slice
(unit boundaries, logic-boundary tests, metamorphic, determinism, and the strict
harness). Runs are crash-safe and resumable (per-mutant JSONL, source-tree
restore on SIGINT/SIGTERM), so the exhaustive pass survives interruption.

**The raw kill rate is a diagnostic, not the gate.** Currently **60/89 ≈ 67%** of
gated mutants are killed. A naive reading calls that a failing grade. It is not,
and inflating it would mean writing tests that assert on cost magnitudes we have
explicitly chosen *not* to pin structurally.

**The actual gate is: every surviving mutant is a documented equivalent.** Each
survivor is listed in an `EQUIVALENT_ALLOWLIST` with a specific justification, and
`--report --strict` fails the moment a survivor appears that is *not* on the list
(a real coverage hole) — it also warns when an allow-listed entry is now killed so
the list can be pruned. Survivors fall into three honest buckets:

- **True equivalents** — the mutation cannot change behavior on any well-formed
  input (e.g. `block_on_policy_violation=True`→`False` is redundant because a
  policy violation already forces `block` via the score path; an `and`→`or` on an
  adjacency guard that only differs for dangling edges that can't occur).
- **Cost-magnitude only** — the mutation shifts a token/dollar estimate but no
  structural verdict (most `prediction.py` and `expected_executions` arithmetic).
  These are covered by the calibration tolerance above, not the structural slice.
- **Explanatory output only** — the mutation changes a `drivers` ordering, a
  `reasons` string, or a finding `message`, never `score`/`band`/`gate`.

When a survivor was a *genuine* coverage hole we fixed the tests, not the
allow-list. The most recent example: a `frac >= 1.0`→`> 1.0` mutant in the
context-explosion severity cutoff (critical vs. high — a real gate difference)
survived, so we added boundary tests that bracket `frac == 1.0` exactly; the
mutant now dies. The allow-list is reserved for mutations that genuinely cannot
change a verdict.

## What "working correctly" means here

We do **not** claim perfect dollar prediction with no history — we claim, and
gate on, order-of-magnitude cold and tight-with-telemetry. We **do** claim the
structural verdicts are correct, and we back that with a labeled corpus, boundary
tests, metamorphic invariants, fuzzing, determinism, and a mutation gate that
forces every untested line to be either covered or explicitly justified as
equivalent. Where the heuristics are weak we name the weakness in the output
rather than paper over it. That is the standard this suite is built to hold.
