#!/usr/bin/env python3
"""Compute precision (with a Wilson 95% CI) from a hand-labeled sample.

Reads the labeling sheet produced by sample.py after you've filled in `verdict`
for each finding. Reports overall precision and a per-category breakdown, each
with a Wilson score interval — the right interval for a proportion on a small
sample (it doesn't fall apart at p near 0 or 1 the way the normal approximation
does).

precision = tp / (tp + fp). `unsure`/blank rows are excluded from the
denominator and counted separately so the exclusion is visible, not hidden.

This is the ONLY correctness number the field study is entitled to publish, and
it is exactly as strong as the number of findings you actually labeled. Label
more to tighten the interval.

Usage:
  python validation/precision.py --labeled validation/sample.todo.jsonl [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict


def wilson(tp: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = tp / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_labels(path: str):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r.get("_meta"):
                continue
            rows.append(r)
    return rows


def tally(rows):
    """Return overall and per-category {tp, fp, unsure, unlabeled}."""
    overall = defaultdict(int)
    per_cat = defaultdict(lambda: defaultdict(int))
    for r in rows:
        v = (r.get("verdict") or "").strip().lower()
        cat = r.get("category", "?")
        bucket = {"tp": "tp", "fp": "fp", "unsure": "unsure"}.get(v, "unlabeled")
        overall[bucket] += 1
        per_cat[cat][bucket] += 1
    return overall, per_cat


def precision_line(name, c) -> str:
    tp, fp = c.get("tp", 0), c.get("fp", 0)
    n = tp + fp
    if n == 0:
        return f"  {name:<22} no adjudicated tp/fp (unsure={c.get('unsure',0)}, " \
               f"unlabeled={c.get('unlabeled',0)})"
    p = tp / n
    lo, hi = wilson(tp, n)
    return (f"  {name:<22} precision={p:6.1%}  [95% CI {lo:.1%}–{hi:.1%}]  "
            f"(tp={tp} fp={fp} n={n})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)

    rows = load_labels(args.labeled)
    overall, per_cat = tally(rows)

    print("=" * 70)
    print("Tollgate field study — adjudicated precision")
    print("=" * 70)
    unlabeled = overall.get("unlabeled", 0)
    if unlabeled:
        print(f"WARNING: {unlabeled} sampled finding(s) still have no verdict — "
              f"label them or the estimate is partial.\n")
    print(precision_line("OVERALL", overall))
    print(f"  excluded: unsure={overall.get('unsure',0)} "
          f"unlabeled={overall.get('unlabeled',0)}")
    print("\nper category:")
    for cat in sorted(per_cat):
        print(precision_line(cat, per_cat[cat]))

    if args.json_out:
        def pack(c):
            tp, fp = c.get("tp", 0), c.get("fp", 0)
            n = tp + fp
            lo, hi = wilson(tp, n)
            return {"tp": tp, "fp": fp, "n": n,
                    "precision": (tp / n) if n else None,
                    "ci95": [lo, hi] if n else None,
                    "unsure": c.get("unsure", 0),
                    "unlabeled": c.get("unlabeled", 0)}
        out = {"overall": pack(overall),
               "by_category": {k: pack(v) for k, v in per_cat.items()}}
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)

    # Exit non-zero if nothing was labeled, so this can sit in a pipeline.
    return 0 if (overall.get("tp", 0) + overall.get("fp", 0)) else 1


if __name__ == "__main__":
    sys.exit(main())
