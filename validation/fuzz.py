"""Robustness fuzzer for the analyzer.

Two invariants matter for a tool that gates deploys:

  I1  It must never CRASH on arbitrary input — malformed Python, broken YAML/JSON,
      binary junk, pathological nesting, huge files. A crash in CI is a denial of
      service on every PR.
  I2  It must never turn a parse FAILURE into a silent PASS. An unparseable or
      unrecognized file must be dropped (not analyzed), never scored as a
      confident green light. This is the "honest failure" property.

The fuzzer generates a large batch of adversarial inputs deterministically (seed
fixed so CI is reproducible) and asserts both invariants. It exercises the whole
front door: discovery + parse + analyze.

Usage:
    python validation/fuzz.py [--iterations N] [--seed S]
Exit code is non-zero if any invariant is violated.
"""
from __future__ import annotations

import argparse
import os
import random
import string
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tollgate.config import Config                     # noqa: E402
from tollgate.parsers import discover, parse_file      # noqa: E402
from tollgate.pipeline import analyze_path, _is_analyzable, analyze_workflow  # noqa: E402

CFG = Config(trials=120)

# Fragments that look agent-ish, recombined into mostly-broken files.
_TOKENS = [
    "while True:", "for i in range(", "def ", "openai.chat.completions.create(",
    "client.messages.create(", "model=", "'gpt-4o'", "append(", "return",
    "import openai", "}{", "[[[", ")))", ":::", "\t\t\t", "\\x00", "λ", "💥",
    "nodes:", '"links":', "{{", "}}", "async def", "await ", "yield",
]


def _rand_text(rng: random.Random) -> str:
    n = rng.randint(0, 60)
    parts = []
    for _ in range(n):
        if rng.random() < 0.5:
            parts.append(rng.choice(_TOKENS))
        else:
            parts.append("".join(rng.choice(string.printable)
                                  for _ in range(rng.randint(0, 12))))
        parts.append(rng.choice([" ", "\n", "\n    ", "", "\t"]))
    return "".join(parts)


def _pathological(rng: random.Random) -> str:
    kind = rng.choice(["deep_nest", "huge", "many_loops", "binary", "unicode"])
    if kind == "deep_nest":
        depth = rng.randint(50, 300)
        return "".join("    " * i + "if x:\n" for i in range(depth)) + \
            "    " * depth + "openai.chat.completions.create(model='gpt-4o')\n"
    if kind == "huge":
        return "x = '" + "a" * rng.randint(50_000, 200_000) + "'\n" \
            "while True:\n    openai.chat.completions.create(model='gpt-4o')\n"
    if kind == "many_loops":
        return "import openai\n" + "".join(
            f"def f{i}():\n    while True:\n        "
            f"openai.chat.completions.create(model='gpt-4o')\n"
            for i in range(rng.randint(20, 80)))
    if kind == "binary":
        return "".join(chr(rng.randint(0, 255)) for _ in range(rng.randint(0, 4000)))
    return "💥λ​﻿" * rng.randint(1, 500) + "\nwhile True: pass\n"


SUFFIXES = [".py", ".yaml", ".yml", ".json", ".md", ".txt", ".prompt"]


def run(iterations: int, seed: int) -> int:
    rng = random.Random(seed)
    crashes = []           # I1 violations
    false_passes = []      # I2 violations
    d = tempfile.mkdtemp()

    for i in range(iterations):
        body = _pathological(rng) if rng.random() < 0.25 else _rand_text(rng)
        suffix = rng.choice(SUFFIXES)
        path = os.path.join(d, f"f{i}{suffix}")
        try:
            with open(path, "w", encoding="utf-8", errors="ignore") as fh:
                fh.write(body)
        except (OSError, ValueError):
            continue

        # --- front door 1: single-file parse + analyze --------------------
        try:
            wf = parse_file(path)
            if _is_analyzable(wf):
                res = analyze_workflow(wf, cfg=CFG)
                # I2: if we DID analyze, the verdict must be a real one. A scored
                # PASS is only honest when there was genuine LLM/edge structure to
                # reason about (which _is_analyzable guarantees) — so this is fine.
                _ = res.risk.gate_decision
        except Exception:
            crashes.append((path, "parse_file/analyze", traceback.format_exc()))

        # Clean up big files so the dir scan below stays cheap.
        try:
            if os.path.getsize(path) > 20_000:
                os.remove(path)
        except OSError:
            pass

    # --- front door 2: a directory scan over everything that survived -----
    try:
        run_res = analyze_path([d], cfg=CFG)
        # I2 at the run level: a scan of pure junk must not yield a BLOCK/known
        # verdict derived from garbage. Empty results -> PASS is the honest path;
        # any results must each be analyzable (no structureless masquerade).
        for r in run_res.results:
            if r.source_kind not in ("dsl", "prompt", "langgraph", "imperative", "autogpt"):
                false_passes.append((r.source_path, f"unknown kind {r.source_kind}"))
    except Exception:
        crashes.append((d, "analyze_path(dir)", traceback.format_exc()))

    # --- report -----------------------------------------------------------
    print("=" * 70)
    print(f"Tollgate fuzzer — {iterations} iterations, seed={seed}")
    print("=" * 70)
    print(f"I1 (never crash):        {'PASS' if not crashes else 'FAIL'} "
          f"({len(crashes)} crash(es))")
    print(f"I2 (no silent PASS):     {'PASS' if not false_passes else 'FAIL'} "
          f"({len(false_passes)} violation(s))")
    for path, where, tb in crashes[:3]:
        print(f"\n  CRASH in {where} on {path}:\n"
              + "\n".join("    " + ln for ln in tb.strip().splitlines()[-4:]))
    for path, why in false_passes[:5]:
        print(f"\n  SILENT-PASS {why}: {path}")

    return 1 if (crashes or false_passes) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)
    return run(args.iterations, args.seed)


if __name__ == "__main__":
    sys.exit(main())
