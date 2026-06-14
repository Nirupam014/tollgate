"""Cost / token calibration harness.

The prediction engine emits a token DISTRIBUTION (p50/p95/p99) per node. The
honest question for an open-source release is: *how close are those numbers to
reality, and when they are wrong, are they wrong in the safe direction?*

This harness answers that against a **trace** — a workflow plus per-node measured
token counts from a real run (an LLM-proxy log, or OpenTelemetry GenAI spans:
``gen_ai.usage.input_tokens`` / ``output_tokens``). See
``validation/traces/sample_trace.json`` for the schema; drop in an export of your
own traffic to calibrate on your workload.

Two regimes are scored, because they back two very different claims:

  COLD  (default)  — predict with NO telemetry (pure heuristic/static). This is
                     the "first PR, no history yet" case. Expect order-of-magnitude
                     accuracy, not precision. We report it as such.
  WARM  (--warm)   — split each node's observations, feed the first half back as
                     telemetry, validate on the second half. This is the steady
                     state once Tollgate has seen your traffic.

Metrics (per node, for input and output token p50, and for the p95 tail):

  ratio          predicted_p50 / measured_p50        (1.0 == perfect)
  APE            |pred - meas| / meas                 (-> MAPE when aggregated)
  within-2x/3x   fraction of node-metrics with 1/K <= ratio <= K
  p95 envelope   fraction of nodes where predicted p95 >= measured p95
                 (the FAIL-SAFE property: a budget guard must rarely under-call
                 the tail. We care more about this than about point accuracy.)

Usage:
    python validation/calibration.py [--trace FILE] [--warm] [--json OUT] [--strict]

``--strict`` exits non-zero if calibration falls below the documented thresholds,
so CI can gate on "the predictor still behaves the way we claim it does".
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tollgate.catalog import ModelCatalog          # noqa: E402
from tollgate.ir import IREdge, IRNode, Workflow    # noqa: E402
from tollgate.prediction import PredictionEngine     # noqa: E402
from tollgate.tokenizer import count_tokens          # noqa: E402

# --- documented strictness thresholds -------------------------------------
# These encode the claims we are willing to put in the README. Each gate maps to
# a sentence a user could hold us to.
#
#   COLD  "with no history, point predictions are order-of-magnitude": within-3x.
#   WARM  "once Tollgate has seen your traffic, predictions are tight": within-2x
#         and bounded MAPE.
#
# The p95 *envelope* (predicted p95 >= measured p95) is reported but NOT gated.
# A sample p95 is, by definition, exceeded ~5% of the time, so an unbiased p95
# estimate sits above the holdout p95 only about half the time — over a handful
# of nodes that is sampling noise, not a defect. Gating on it would be a vanity
# metric. We surface it so users can SEE the cold heuristic's known weakness:
# it under-states the INPUT tail when runtime content (the user message, the
# accumulated history) dwarfs a short static prompt. That is exactly the gap
# telemetry closes, and we say so rather than hide it.
COLD_MIN_WITHIN_3X = 0.60     # >=60% of point predictions within 3x with no history
WARM_MIN_WITHIN_2X = 0.80     # with telemetry, >=80% within 2x
WARM_MAX_MAPE = 0.25          # with telemetry, mean abs pct error on p50 <= 25%


# --- percentile (dependency-free, linear interpolation, matches numpy default) --
def percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[int(pos)])
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def _build_workflow(spec: dict) -> Workflow:
    nodes = []
    for raw in spec.get("nodes", []):
        n = IRNode(
            node_id=str(raw["id"]),
            kind=raw.get("kind", "llm_call"),
            intended_model=raw.get("model"),
            prompt_template=raw.get("prompt"),
            task_class=raw.get("task_class"),
            appends_history=bool(raw.get("appends_history", False)),
            retrieves_context=bool(raw.get("retrieves_context", False)),
            retrieved_context_cap=raw.get("retrieved_context_cap"),
            max_output_tokens=raw.get("max_output_tokens"),
        )
        if n.prompt_template and n.static_input_tokens == 0:
            n.static_input_tokens = count_tokens(n.prompt_template)
        nodes.append(n)
    edges = [
        IREdge(str(e["from"]), str(e["to"]), edge_type=e.get("type", "sequence"))
        for e in spec.get("edges", [])
    ]
    return Workflow(
        workflow_id=spec.get("workflow_id", "trace"),
        source_kind=spec.get("source_kind", "dsl"),
        nodes=nodes,
        edges=edges,
        entry=spec.get("entry"),
    )


def _measured(obs: List[dict], key: str) -> List[float]:
    return [float(o[key]) for o in obs if key in o]


def _telemetry_from(obs_by_node: Dict[str, List[dict]]) -> Dict[str, dict]:
    """Use the FIRST half of each node's observations as the 'known' history."""
    tel = {}
    for nid, obs in obs_by_node.items():
        half = obs[: max(1, len(obs) // 2)]
        ins = _measured(half, "input_tokens")
        outs = _measured(half, "output_tokens")
        tel[nid] = {
            "input_p50": percentile(ins, 50), "input_p95": percentile(ins, 95),
            "output_p50": percentile(outs, 50), "output_p95": percentile(outs, 95),
        }
    return tel


def _holdout(obs_by_node: Dict[str, List[dict]], warm: bool) -> Dict[str, List[dict]]:
    """In WARM mode, validate on the SECOND half (disjoint from the telemetry)."""
    if not warm:
        return obs_by_node
    return {nid: obs[max(1, len(obs) // 2):] or obs for nid, obs in obs_by_node.items()}


def calibrate(trace: dict, warm: bool) -> dict:
    wf = _build_workflow(trace["workflow"])
    obs_by_node: Dict[str, List[dict]] = trace["observations"]

    telemetry = _telemetry_from(obs_by_node) if warm else None
    engine = PredictionEngine(ModelCatalog.load(), telemetry=telemetry)
    pred = engine.predict(wf)
    pred_by_node = {n.node_id: n for n in pred.nodes}

    eval_obs = _holdout(obs_by_node, warm)

    rows: List[dict] = []
    apes: List[float] = []
    within2 = within3 = total_pts = 0
    envelope_ok = envelope_total = 0

    for node in wf.llm_nodes():
        nid = node.node_id
        p = pred_by_node.get(nid)
        obs = eval_obs.get(nid, [])
        if p is None or not obs:
            continue
        for kind, dist in (("input", p.input_tokens), ("output", p.output_tokens)):
            meas = _measured(obs, f"{kind}_tokens")
            if not meas:
                continue
            m_p50 = percentile(meas, 50)
            m_p95 = percentile(meas, 95)
            ratio = (dist.p50 / m_p50) if m_p50 else float("inf")
            ape = abs(dist.p50 - m_p50) / m_p50 if m_p50 else float("inf")
            apes.append(ape)
            total_pts += 1
            if m_p50 and 0.5 <= ratio <= 2.0:
                within2 += 1
            if m_p50 and (1 / 3) <= ratio <= 3.0:
                within3 += 1
            envelope_total += 1
            covered = dist.p95 >= m_p95
            if covered:
                envelope_ok += 1
            rows.append({
                "node": nid, "metric": kind,
                "pred_p50": round(dist.p50, 1), "meas_p50": round(m_p50, 1),
                "ratio": round(ratio, 2), "ape": round(ape, 3),
                "pred_p95": round(dist.p95, 1), "meas_p95": round(m_p95, 1),
                "p95_envelope_ok": covered,
            })

    mape = sum(apes) / len(apes) if apes else 0.0
    summary = {
        "regime": "warm" if warm else "cold",
        "basis": pred.basis,
        "n_node_metrics": total_pts,
        "mape_p50": round(mape, 3),
        "within_2x": round(within2 / total_pts, 3) if total_pts else 0.0,
        "within_3x": round(within3 / total_pts, 3) if total_pts else 0.0,
        "p95_envelope_coverage": round(envelope_ok / envelope_total, 3) if envelope_total else 0.0,
    }
    return {"summary": summary, "rows": rows}


def _check_strict(summary: dict, warm: bool) -> List[str]:
    fails = []
    if warm:
        if summary["within_2x"] < WARM_MIN_WITHIN_2X:
            fails.append(f"warm within_2x {summary['within_2x']} < {WARM_MIN_WITHIN_2X}")
        if summary["mape_p50"] > WARM_MAX_MAPE:
            fails.append(f"warm mape_p50 {summary['mape_p50']} > {WARM_MAX_MAPE}")
    else:
        if summary["within_3x"] < COLD_MIN_WITHIN_3X:
            fails.append(f"cold within_3x {summary['within_3x']} < {COLD_MIN_WITHIN_3X}")
    return fails


def _print(report: dict) -> None:
    s = report["summary"]
    print("=" * 78)
    print(f"Tollgate cost calibration — regime={s['regime']} basis={s['basis']}")
    print("=" * 78)
    print(f"{'node':<18}{'metric':<8}{'pred p50':>10}{'meas p50':>10}"
          f"{'ratio':>8}{'APE':>8}{'p95 env':>9}")
    print("-" * 78)
    for r in report["rows"]:
        print(f"{r['node']:<18}{r['metric']:<8}{r['pred_p50']:>10}{r['meas_p50']:>10}"
              f"{r['ratio']:>8}{r['ape']:>8}{'ok' if r['p95_envelope_ok'] else 'UNDER':>9}")
    print("-" * 78)
    print(f"node-metrics={s['n_node_metrics']}  MAPE(p50)={s['mape_p50']}  "
          f"within2x={s['within_2x']}  within3x={s['within_3x']}  "
          f"p95-envelope={s['p95_envelope_coverage']} (diagnostic, not gated)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=os.path.join(os.path.dirname(__file__), "traces", "sample_trace.json"))
    ap.add_argument("--warm", action="store_true", help="feed first half of obs as telemetry, validate on second half")
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)

    with open(args.trace, "r", encoding="utf-8") as fh:
        trace = json.load(fh)

    report = calibrate(trace, warm=args.warm)
    _print(report)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.json_out}")

    if args.strict:
        fails = _check_strict(report["summary"], warm=args.warm)
        if fails:
            print("\nSTRICT FAIL:")
            for f in fails:
                print(f"  - {f}")
            return 1
        print("\nSTRICT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
