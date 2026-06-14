#!/usr/bin/env python3
"""Draw a reproducible random sample of findings for manual adjudication.

This is the bridge from "we found N things" to a defensible precision claim.
You cannot eyeball 1000 repos; you CAN hand-label a random sample and report
precision with a confidence interval. This script writes that sample as a
JSONL "labeling sheet" with a blank `verdict` field for each finding.

Fill each `verdict` with:
  tp   - true positive: the finding is real and correctly characterized
  fp   - false positive: the finding is wrong, spurious, or misclassified
  unsure - exclude from the precision denominator (use sparingly; explain in notes)
(leave blank = not yet adjudicated)

Stratified mode (--per-category K) samples up to K findings per category so rare
categories get enough labels for a per-category precision, not just the overall.

Usage:
  python validation/sample.py --results study/results.jsonl --n 150 --seed 7 \
      --out validation/sample.todo.jsonl
  python validation/sample.py --results study/results.jsonl --per-category 25 \
      --seed 7 --out validation/sample.todo.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict


def load_findings(path: str):
    pop = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r.get("status") != "ok":
                continue
            for i, f in enumerate(r.get("findings", [])):
                pop.append({
                    "repo": r["repo"],
                    "sha": r.get("sha", "unknown"),
                    "file": f.get("file"),
                    "workflow_id": f.get("workflow_id"),
                    "category": f["category"],
                    "severity": f["severity"],
                    "node_id": f.get("node_id"),
                    "message": f.get("message", ""),
                    "finding_index": i,
                })
    return pop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150,
                    help="overall sample size (ignored if --per-category set)")
    ap.add_argument("--per-category", type=int, default=0,
                    help="stratified: up to K findings per category")
    ap.add_argument("--seed", type=int, default=7, help="fixed for reproducibility")
    args = ap.parse_args(argv)

    pop = load_findings(args.results)
    if not pop:
        print("no findings in results", file=sys.stderr)
        return 1
    rng = random.Random(args.seed)

    if args.per_category:
        buckets = defaultdict(list)
        for f in pop:
            buckets[f["category"]].append(f)
        sample = []
        for cat in sorted(buckets):
            items = buckets[cat][:]
            rng.shuffle(items)
            sample.extend(items[: args.per_category])
        rng.shuffle(sample)
    else:
        k = min(args.n, len(pop))
        sample = rng.sample(pop, k)

    with open(args.out, "w", encoding="utf-8") as fh:
        # Provenance header line so the sheet documents how it was drawn.
        fh.write(json.dumps({
            "_meta": True,
            "results": args.results,
            "population": len(pop),
            "sampled": len(sample),
            "seed": args.seed,
            "mode": "per_category" if args.per_category else "uniform",
        }) + "\n")
        for f in sample:
            f["verdict"] = ""   # <- fill: tp | fp | unsure
            f["notes"] = ""
            fh.write(json.dumps(f) + "\n")

    print(f"population={len(pop)} sampled={len(sample)} seed={args.seed} "
          f"-> {args.out}\nFill the 'verdict' field (tp/fp/unsure), then run "
          f"precision.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
