#!/usr/bin/env bash
# Tollgate validation suite — single entry point.
#
# Runs every gate that backs the project's correctness claims, in fail-fast
# order (cheapest / most fundamental first). Any non-zero exit fails the suite.
#
#   ./validation/run_all.sh            # full suite (mutation runs ALL gated mutants)
#   FAST=1 ./validation/run_all.sh     # skip the exhaustive mutation pass (sampled instead)
#
# Designed to run identically on a laptop and in CI. No network, no extra deps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

MUT_RESULTS="${MUT_RESULTS:-$(mktemp -t tollgate_mut.XXXXXX.jsonl)}"
FUZZ_ITERS="${FUZZ_ITERS:-2000}"

step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

step "Unit tests (analyzer + logic boundaries)"
python -m unittest -q tests.test_analyzer tests.test_logic_boundaries

step "Determinism (byte-identical reports across repeated runs)"
python -m validation.test_determinism

step "Metamorphic relations (invariants under semantics-preserving edits)"
python -m validation.metamorphic

step "Benchmark harness — labeled corpus (STRICT)"
python validation/harness.py --strict

step "Fuzz (never-crash + never-silently-pass, ${FUZZ_ITERS} iters)"
python validation/fuzz.py --iterations "$FUZZ_ITERS"

step "Cost calibration — cold regime (STRICT)"
python validation/calibration.py --strict

step "Cost calibration — warm regime / telemetry (STRICT)"
python validation/calibration.py --warm --strict

if [ "${FAST:-0}" = "1" ]; then
  step "Mutation testing — SAMPLED (FAST=1; not a gate)"
  python validation/mutation.py --logic-per-module 6 --per-module 0 \
    --results "$MUT_RESULTS"
  python validation/mutation.py --report "$MUT_RESULTS" || true
else
  step "Mutation testing — ALL gated logic mutants"
  python validation/mutation.py --all-logic --per-module 0 \
    --results "$MUT_RESULTS"
  step "Mutation gate — no unexpected survivor (STRICT)"
  python validation/mutation.py --report "$MUT_RESULTS" --strict
fi

printf '\n\033[1;32mALL VALIDATION GATES PASSED\033[0m\n'
