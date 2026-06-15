"""PR-delta / baseline gating.

The whole-repo gate answers "is this codebase risky?". In a pull request that is
the wrong question: a team should not be blocked by pre-existing issues they did
not touch — that is exactly how CI gates get switched off. The right question is
"does *this change* make things worse?".

This module diffs a fresh analysis against a baseline report (a prior
`tollgate analyze ... -o json=baseline.json`, typically produced on the default
branch) and computes a **delta gate** scored on *new and worsened* findings only.
Pre-existing findings are reported as `unchanged` and never drive the gate;
findings that disappeared are reported as `fixed`.

Design notes (honesty + determinism, consistent with the rest of Tollgate):

- Finding *identity* is deliberately line-number-independent so that unrelated
  edits above a finding don't make it look "new". Identity is
  (category, file basename, node id, normalized message), where the message is
  normalized by collapsing whitespace and replacing digit runs with ``N`` (so a
  changed token-count in the text doesn't fork one issue into two).
- We also track per-identity *counts*. If a file gains a second uncapped LLM call
  that normalizes to the same identity, the occurrence count rises and the extra
  occurrence is reported as new — so multi-site regressions can't hide behind a
  line-insensitive key.
- The delta gate is conservative in the safe direction: when in doubt it treats a
  finding as new (which can over-report, never under-report new risk).

It is pure data-over-data, so it works identically for every language layer
(graph findings, Python AST lint, the language-agnostic textual lint).
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .findings import Finding, severity_rank
from .scoring import RiskScorer

Identity = Tuple[str, str, str, str]

_DIGITS = re.compile(r"\d+")


def _norm_message(msg: str) -> str:
    """Collapse whitespace and digit runs so cosmetic/numeric drift in a message
    does not fork a single issue into 'fixed' + 'new'."""
    one_line = " ".join((msg or "").split())
    return _DIGITS.sub("N", one_line).lower()


def _basename(path: Optional[str]) -> str:
    return os.path.basename(path) if path else ""


def _identity(category: str, source_path: Optional[str], node_id: Optional[str],
              message: str) -> Identity:
    return (category or "", _basename(source_path), node_id or "", _norm_message(message))


def _iter_finding_dicts(report: Dict[str, Any]):
    """Yield (identity, finding_dict) for every gate-relevant finding in a report
    dict (the structure produced by ``RunResult.to_dict``). Pulls workflow
    findings, policy violations, and agentic-lint findings, falling back to the
    parent's source_path when a finding doesn't carry its own."""
    for r in report.get("results", []) or []:
        parent = r.get("source_path")
        for f in (r.get("findings", []) or []) + (r.get("policy_violations", []) or []):
            sp = f.get("source_path") or parent
            yield _identity(f.get("category", ""), sp, f.get("node_id"),
                            f.get("message", "")), f
    for lr in report.get("lint_results", []) or []:
        parent = lr.get("source_path")
        for f in lr.get("findings", []) or []:
            sp = f.get("source_path") or parent
            yield _identity(f.get("category", ""), sp, f.get("node_id"),
                            f.get("message", "")), f


def _index(report: Dict[str, Any]):
    """Build {identity: count}, {identity: representative finding}, and
    {identity: worst-severity-rank} for a report."""
    counts: Counter = Counter()
    rep: Dict[Identity, Dict[str, Any]] = {}
    worst: Dict[Identity, int] = {}
    for ident, f in _iter_finding_dicts(report):
        counts[ident] += 1
        rank = severity_rank(f.get("severity", "low"))
        if ident not in worst or rank > worst[ident]:
            worst[ident] = rank
            rep[ident] = f          # representative = highest-severity instance
        elif ident not in rep:
            rep[ident] = f
    return counts, rep, worst


_RANK_SEV = {v: k for k, v in
             {"low": 0, "medium": 1, "high": 2, "critical": 3}.items()}


def diff_reports(current: Dict[str, Any], baseline: Dict[str, Any],
                 block_at_score: int = 75, warn_at_score: int = 50) -> Dict[str, Any]:
    """Diff a fresh report against a baseline report and compute the delta gate.

    Returns a JSON-serializable dict with ``new`` / ``worsened`` / ``fixed``
    lists, an ``unchanged`` count, the ``delta_gate`` (scored on new+worsened
    only), and the baseline/current fingerprints for traceability.
    """
    cur_counts, cur_rep, cur_worst = _index(current)
    base_counts, base_rep, base_worst = _index(baseline)

    new: List[Dict[str, Any]] = []
    worsened: List[Dict[str, Any]] = []
    fixed: List[Dict[str, Any]] = []
    unchanged = 0

    delta_findings: List[Finding] = []

    for ident, ccount in cur_counts.items():
        bcount = base_counts.get(ident, 0)
        added = ccount - bcount
        f = cur_rep[ident]
        if bcount == 0:
            # Brand-new identity: every occurrence is new.
            entry = dict(f)
            if added > 1:
                entry["occurrences"] = added
            new.append(entry)
            delta_findings += _as_findings(f, added)
        else:
            if added > 0:
                # Same issue, but more occurrences than the baseline had.
                entry = dict(f)
                entry["occurrences"] = added
                entry["note"] = f"{added} new occurrence(s) beyond baseline"
                new.append(entry)
                delta_findings += _as_findings(f, added)
            cur_rank = cur_worst[ident]
            base_rank = base_worst.get(ident, -1)
            if cur_rank > base_rank:
                entry = dict(f)
                entry["from_severity"] = _RANK_SEV.get(base_rank, "low")
                entry["to_severity"] = _RANK_SEV.get(cur_rank, f.get("severity", "low"))
                worsened.append(entry)
                delta_findings += _as_findings(f, 1)
            if added <= 0:
                unchanged += min(ccount, bcount)
            else:
                unchanged += bcount

    for ident, bcount in base_counts.items():
        ccount = cur_counts.get(ident, 0)
        removed = bcount - ccount
        if removed > 0:
            entry = dict(base_rep[ident])
            if removed > 1:
                entry["occurrences"] = removed
            fixed.append(entry)

    delta_gate = _score_delta(delta_findings, block_at_score, warn_at_score)

    new.sort(key=lambda d: severity_rank(d.get("severity", "low")), reverse=True)
    worsened.sort(key=lambda d: severity_rank(d.get("to_severity", "low")), reverse=True)
    fixed.sort(key=lambda d: severity_rank(d.get("severity", "low")), reverse=True)

    return {
        "delta_gate": delta_gate,
        "full_gate": current.get("gate_decision"),
        "baseline_gate": baseline.get("gate_decision"),
        "baseline_fingerprint": baseline.get("fingerprint"),
        "current_fingerprint": current.get("fingerprint"),
        "counts": {
            "new": sum(int(d.get("occurrences", 1)) for d in new),
            "worsened": len(worsened),
            "fixed": sum(int(d.get("occurrences", 1)) for d in fixed),
            "unchanged": unchanged,
        },
        "new": new,
        "worsened": worsened,
        "fixed": fixed,
    }


def _as_findings(f: Dict[str, Any], n: int) -> List[Finding]:
    n = max(1, int(n))
    return [Finding(
        finding_id=f.get("finding_id", "delta"),
        category=f.get("category", "unknown"),
        severity=f.get("severity", "low"),
        message=f.get("message", ""),
        node_id=f.get("node_id"),
        source_path=f.get("source_path"),
        line=f.get("line"),
    ) for _ in range(n)]


def _score_delta(delta_findings: List[Finding], block_at_score: int,
                 warn_at_score: int) -> str:
    """Gate decision over the delta only. New policy violations are routed through
    the scorer's policy channel so a newly-introduced policy breach blocks."""
    if not delta_findings:
        return "pass"
    policy = [f for f in delta_findings if f.category == "policy_violation"]
    structural = [f for f in delta_findings if f.category != "policy_violation"]
    scorer = RiskScorer(block_at_score=block_at_score, warn_at_score=warn_at_score)
    return scorer.score(structural, policy_violations=policy).gate_decision


def load_baseline(path: str) -> Dict[str, Any]:
    """Read a baseline report.json. Raises OSError/ValueError on read/parse error.
    A valid JSON object missing the finding arrays is treated as an empty baseline
    (everything in the current run counts as new) — which is the correct behavior
    for the very first run before a baseline exists."""
    import json
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("baseline is not a Tollgate report object")
    return data
