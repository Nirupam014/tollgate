"""Benchmark harness: score Tollgate against a labeled corpus.

Runs the analyzer over every case in ``corpus/labels.yaml``, derives predicted
labels, and reports:

  * Discovery accuracy   - does a scan surface real agents and skip non-agents?
  * Unbounded-loop detection - precision / recall / F1 vs. ground truth, with a
                               confusion matrix, AND the same metrics for three
                               trivial baselines (it must beat them).
  * Gate-decision accuracy - pass/warn/block vs. expected, where labeled.
  * Recommendation recall - did a cheaper-model rec fire where expected?

Exit code is non-zero if any *hard expectation* is violated (a labeled property
predicted wrong), so this doubles as a CI gate on the analyzer's own behavior.

Usage:
    python validation/harness.py [--labels PATH] [--json OUT.json] [--strict]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tollgate.config import Config                       # noqa: E402
from tollgate.parsers import discover, parse_file        # noqa: E402
from tollgate.pipeline import analyze_workflow, _is_analyzable  # noqa: E402

from baselines import BASELINES                          # noqa: E402

CFG = Config(trials=400)


# --------------------------------------------------------------------------
# Run Tollgate on one case and extract the predicted labels.
# --------------------------------------------------------------------------
def predict_case(abs_path: str) -> dict:
    """Return predicted labels for a single file.

    discovered  - would a directory scan surface this file?
    analyzable  - parsed into something with LLM/edges to analyze?
    loop        - worst recursive_loop severity: critical | medium | none
    gate        - pass | warn | block (None if not analyzable)
    recommend   - did a model-substitution recommendation fire?
    """
    out = {"discovered": False, "analyzable": False,
           "loop": "none", "gate": None, "recommend": False}

    # Discovery is judged as a *scan* of the containing directory would: a file
    # is "discovered" only if the candidate filter picks it out of its folder.
    folder = os.path.dirname(abs_path)
    out["discovered"] = abs_path in set(discover([folder]))

    try:
        wf = parse_file(abs_path)
    except Exception:
        return out
    if not _is_analyzable(wf):
        return out
    out["analyzable"] = True

    res = analyze_workflow(wf, cfg=CFG)
    sev_rank = {"none": 0, "low": 0, "medium": 1, "high": 2, "critical": 3}
    worst = "none"
    for f in res.findings:
        if f.category == "recursive_loop" and sev_rank.get(f.severity, 0) > sev_rank[worst]:
            worst = f.severity
    out["loop"] = worst if worst in ("medium", "high", "critical") else "none"
    out["gate"] = res.risk.gate_decision
    out["recommend"] = len(res.recommendations) > 0
    return out


# --------------------------------------------------------------------------
# Metric helpers.
# --------------------------------------------------------------------------
def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3)}


def bar(label: str, m: dict, width: int = 22) -> str:
    return (f"  {label:<{width}} P={m['precision']:.2f}  R={m['recall']:.2f}  "
            f"F1={m['f1']:.2f}   (tp={m['tp']} fp={m['fp']} fn={m['fn']})")


def fold_of(case: dict, seed: int, holdout_pct: int) -> str:
    """Deterministically assign a case to 'train' or 'holdout'.

    An explicit `fold:` in the case wins (so you can pin a case). Otherwise a
    stable hash of the path (salted by --seed) buckets it — same seed always
    yields the same split, and no case ever silently moves between folds. With
    holdout_pct == 0 every case is 'train' (i.e. the whole corpus, the default).
    """
    explicit = (case.get("fold") or "").strip().lower()
    if explicit in ("train", "holdout"):
        return explicit
    if holdout_pct <= 0:
        return "train"
    h = hashlib.sha1(f"{seed}:{case['path']}".encode()).hexdigest()
    return "holdout" if (int(h[:8], 16) % 100) < holdout_pct else "train"


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "corpus", "labels.yaml"))
    ap.add_argument("--json", help="write machine-readable results here")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on any violated hard expectation")
    ap.add_argument("--holdout-pct", type=int, default=0,
                    help="percent of cases held out for evaluation (0=use all). "
                    "Tune against --split train; report/gate on --split holdout.")
    ap.add_argument("--seed", type=int, default=7, help="fold-assignment seed")
    ap.add_argument("--split", choices=["all", "train", "holdout"], default="all",
                    help="which fold to score (default all)")
    args = ap.parse_args(argv)

    with open(args.labels) as fh:
        spec = yaml.safe_load(fh)
    cases = spec["cases"]
    # Restrict to the requested fold so tuning (train) never sees the held-out
    # cases the published precision/recall is reported on (holdout).
    if args.split != "all":
        cases = [c for c in cases if fold_of(c, args.seed, args.holdout_pct) == args.split]
        if not cases:
            print(f"no cases in split '{args.split}' "
                  f"(holdout_pct={args.holdout_pct}, seed={args.seed}).")
            return 0

    rows = []
    violations = []

    # Tollgate loop-detection counters + baseline counters.
    tg = {"tp": 0, "fp": 0, "fn": 0}
    bl = {name: {"tp": 0, "fp": 0, "fn": 0} for name in BASELINES}
    disc_ok = disc_total = 0
    gate_ok = gate_total = 0
    rec_ok = rec_total = 0

    for c in cases:
        rel = c["path"]
        abs_path = os.path.join(ROOT, rel)
        pred = predict_case(abs_path)
        row = {"path": rel, "expected": {}, "predicted": pred, "violations": []}

        # --- discovery -----------------------------------------------------
        exp_disc = bool(c.get("discovered"))
        row["expected"]["discovered"] = exp_disc
        disc_total += 1
        if pred["discovered"] == exp_disc:
            disc_ok += 1
        else:
            v = f"discovered: expected {exp_disc}, got {pred['discovered']}"
            row["violations"].append(v)
            violations.append((rel, v))

        # Ground-truth "is this an unbounded loop?" for the detection task.
        # Only meaningful for discovered/analyzable files.
        exp_loop = c.get("loop")
        truth_unbounded = (exp_loop == "critical")
        pred_unbounded = (pred["loop"] == "critical")
        if exp_disc:  # score detection only on files that are agents
            if truth_unbounded and pred_unbounded:
                tg["tp"] += 1
            elif not truth_unbounded and pred_unbounded:
                tg["fp"] += 1
            elif truth_unbounded and not pred_unbounded:
                tg["fn"] += 1
            for name, fn in BASELINES.items():
                bpred = fn(abs_path)
                if truth_unbounded and bpred:
                    bl[name]["tp"] += 1
                elif not truth_unbounded and bpred:
                    bl[name]["fp"] += 1
                elif truth_unbounded and not bpred:
                    bl[name]["fn"] += 1

        # --- exact loop severity (when labeled) ---------------------------
        if exp_loop is not None and exp_disc:
            row["expected"]["loop"] = exp_loop
            if pred["loop"] != exp_loop:
                v = f"loop severity: expected {exp_loop}, got {pred['loop']}"
                row["violations"].append(v)
                violations.append((rel, v))

        # --- gate ----------------------------------------------------------
        exp_gate = c.get("gate")
        if exp_gate is not None:
            row["expected"]["gate"] = exp_gate
            gate_total += 1
            if pred["gate"] == exp_gate:
                gate_ok += 1
            else:
                v = f"gate: expected {exp_gate}, got {pred['gate']}"
                row["violations"].append(v)
                violations.append((rel, v))

        # --- recommendation ------------------------------------------------
        exp_rec = c.get("recommend")
        if exp_rec is not None:
            row["expected"]["recommend"] = exp_rec
            rec_total += 1
            if pred["recommend"] == exp_rec:
                rec_ok += 1
            else:
                v = f"recommend: expected {exp_rec}, got {pred['recommend']}"
                row["violations"].append(v)
                violations.append((rel, v))

        rows.append(row)

    tg_m = prf(**tg)
    bl_m = {name: prf(**counts) for name, counts in bl.items()}

    # ---- report -----------------------------------------------------------
    print("=" * 70)
    print("Tollgate benchmark — labeled corpus")
    print("=" * 70)
    split_note = (f" [split={args.split}, holdout_pct={args.holdout_pct}, seed={args.seed}]"
                  if args.split != "all" else "")
    print(f"\nCases: {len(cases)}{split_note}\n")
    print(f"Discovery accuracy:       {disc_ok}/{disc_total} "
          f"({disc_ok / disc_total:.0%})")
    if gate_total:
        print(f"Gate-decision accuracy:   {gate_ok}/{gate_total} "
              f"({gate_ok / gate_total:.0%})")
    if rec_total:
        print(f"Recommendation accuracy:  {rec_ok}/{rec_total} "
              f"({rec_ok / rec_total:.0%})")

    print("\nUnbounded-loop detection (Tollgate vs. trivial baselines):")
    print(bar("tollgate", tg_m))
    for name, m in sorted(bl_m.items(), key=lambda kv: -kv[1]["f1"]):
        print(bar(name, m))

    beats = [n for n, m in bl_m.items() if tg_m["f1"] > m["f1"]]
    ties = [n for n, m in bl_m.items() if abs(tg_m["f1"] - m["f1"]) < 1e-9]
    print(f"\nTollgate F1={tg_m['f1']:.2f} beats {len(beats)}/"
          f"{len(bl_m)} baselines"
          + (f"; ties {ties}" if ties else ""))

    if violations:
        print(f"\n{len(violations)} hard-expectation violation(s):")
        for rel, v in violations:
            print(f"  - {rel}: {v}")
    else:
        print("\nNo hard-expectation violations.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({
                "cases": rows,
                "metrics": {
                    "discovery": {"ok": disc_ok, "total": disc_total},
                    "gate": {"ok": gate_ok, "total": gate_total},
                    "recommend": {"ok": rec_ok, "total": rec_total},
                    "loop_detection": {"tollgate": tg_m, "baselines": bl_m},
                },
                "violations": [{"path": r, "detail": v} for r, v in violations],
            }, fh, indent=2)
        print(f"\nWrote {args.json}")

    # Must beat every baseline that isn't a degenerate tie, and have no
    # violations, to pass in strict mode.
    failed = bool(violations) or tg_m["f1"] < max(m["f1"] for m in bl_m.values())
    if args.strict and failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
