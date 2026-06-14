#!/usr/bin/env python3
"""Compute the miss-rate / recall (with a Wilson 95% CI) from a labeled audit sheet.

Reads the worksheet produced by audit_misses.py after you've filled in `verdict`
for each repo in the "Tollgate found nothing" stratum. This is the recall-side
counterpart to precision.py.

verdict values (set by the human adjudicator):
  miss   - a real, gateable workflow Tollgate failed to recover -> false negative
  tn     - true negative: genuinely no gateable structure (one-shot call,
           embeddings only, out-of-scope language, marker only in a comment,
           dynamic graph, ...). The `reason` field says which.
  unsure - excluded from the denominator (counted separately, not hidden)
  (blank = not yet adjudicated)

  miss_rate = miss / (miss + tn)      over the audited sample
  recall (workflow-presence) = 1 - miss_rate

Both are reported with a Wilson score interval — correct for a proportion on a
small sample, and it doesn't degenerate at p near 0 or 1. The estimate is exactly
as strong as the number of repos you actually labeled; label more to tighten it.

IMPORTANT scope note: this estimates how often Tollgate MISSES a workflow that is
present, over the audited stratum. It is not a global accuracy claim, and it says
nothing about cost accuracy — only about structural workflow recovery.

Usage:
  python validation/recall.py --labeled study/audit.todo.jsonl [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_rows(path: str):
    meta, rows = None, []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r.get("_meta"):
                meta = r
                continue
            rows.append(r)
    return meta, rows


def tally(rows, key=None):
    c = defaultdict(int)
    for r in rows:
        if key and r.get("marker_tier") != key:
            continue
        v = (r.get("verdict") or "").strip().lower()
        bucket = {"miss": "miss", "tn": "tn", "unsure": "unsure"}.get(v, "unlabeled")
        c[bucket] += 1
    return c


def rate_line(name, c) -> str:
    miss, tn = c.get("miss", 0), c.get("tn", 0)
    n = miss + tn
    if n == 0:
        return (f"  {name:<14} no adjudicated miss/tn "
                f"(unsure={c.get('unsure',0)}, unlabeled={c.get('unlabeled',0)})")
    mr = miss / n
    lo, hi = wilson(miss, n)
    rlo, rhi = 1 - hi, 1 - lo
    return (f"  {name:<14} miss_rate={mr:6.1%} [95% CI {lo:.1%}-{hi:.1%}]  "
            f"recall={1-mr:6.1%} [95% CI {rlo:.1%}-{rhi:.1%}]  "
            f"(miss={miss} tn={tn} n={n})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)

    meta, rows = load_rows(args.labeled)
    overall = tally(rows)

    print("=" * 70)
    print("Tollgate field study - adjudicated miss-rate / recall (wf-presence)")
    print("=" * 70)
    if meta:
        print(f"stratum: {meta.get('stratum')}   "
              f"stratum_size={meta.get('stratum_size')}   "
              f"source_oracle={meta.get('source_oracle')}")
    unlabeled = overall.get("unlabeled", 0)
    if unlabeled:
        print(f"WARNING: {unlabeled} sampled repo(s) still have no verdict - "
              f"label them or the estimate is partial.\n")
    print(rate_line("OVERALL", overall))
    print(f"  excluded: unsure={overall.get('unsure',0)} "
          f"unlabeled={overall.get('unlabeled',0)}")

    print("\nby marker tier:")
    for tier in ("strong", "medium", "weak"):
        print(rate_line(tier, tally(rows, key=tier)))

    # Reason breakdown — why repos were judged true-negative vs missed. Makes the
    # "no workflow is usually correct" claim auditable, and points at what to fix.
    miss_reasons = defaultdict(int)
    tn_reasons = defaultdict(int)
    for r in rows:
        v = (r.get("verdict") or "").strip().lower()
        reason = (r.get("reason") or "?").strip() or "?"
        if v == "miss":
            miss_reasons[reason] += 1
        elif v == "tn":
            tn_reasons[reason] += 1
    if miss_reasons:
        print("\nmiss reasons (what Tollgate failed to parse):")
        for k in sorted(miss_reasons, key=lambda x: -miss_reasons[x]):
            print(f"  {k:<24} {miss_reasons[k]}")
    if tn_reasons:
        print("\ntrue-negative reasons (correctly found nothing):")
        for k in sorted(tn_reasons, key=lambda x: -tn_reasons[x]):
            print(f"  {k:<24} {tn_reasons[k]}")

    if args.json_out:
        def pack(c):
            miss, tn = c.get("miss", 0), c.get("tn", 0)
            n = miss + tn
            lo, hi = wilson(miss, n)
            return {"miss": miss, "tn": tn, "n": n,
                    "miss_rate": (miss / n) if n else None,
                    "miss_rate_ci95": [lo, hi] if n else None,
                    "recall": (1 - miss / n) if n else None,
                    "recall_ci95": [1 - hi, 1 - lo] if n else None,
                    "unsure": c.get("unsure", 0),
                    "unlabeled": c.get("unlabeled", 0)}
        out = {"overall": pack(overall),
               "by_tier": {t: pack(tally(rows, key=t))
                           for t in ("strong", "medium", "weak")},
               "miss_reasons": dict(miss_reasons),
               "tn_reasons": dict(tn_reasons)}
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)

    return 0 if (overall.get("miss", 0) + overall.get("tn", 0)) else 1


if __name__ == "__main__":
    sys.exit(main())
