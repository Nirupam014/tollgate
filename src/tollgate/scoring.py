"""Risk Scorer & gate decision (capability 7).

Combines structural findings + policy posture into a 0-100 deployment risk score
and a pass|warn|block gate. Severity contributions saturate (diminishing returns
past the first critical) so one extra low finding can't dominate.

The score is purely structural — it reflects token-waste risk (unbounded loops,
context explosion, fan-out, prompt bloat, retry storms), not a dollar budget.
Projected token consumption is reported alongside the score for context.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .findings import Finding
SEVERITY_WEIGHT = {'critical': 40, 'high': 24, 'medium': 10, 'low': 3}
BANDS = [(0, 'low'), (25, 'medium'), (50, 'high'), (75, 'critical')]

@dataclass
class RiskScore:
    score: int
    band: str
    gate_decision: str
    drivers: List[Dict]
    projected_monthly_tokens: Dict[str, float]
    reasons: List[str] = field(default_factory=list)

    def to_dict(self):
        return {'score': self.score, 'band': self.band, 'gate_decision': self.gate_decision, 'drivers': self.drivers, 'projected_monthly_tokens': self.projected_monthly_tokens, 'reasons': self.reasons}

def _band(score: float) -> str:
    band = 'low'
    for (threshold, name) in BANDS:
        if score >= threshold:
            band = name
    return band

class RiskScorer:

    def __init__(self, block_at_score: int=75, warn_at_score: int=50, block_on_policy_violation: bool=True):
        self.block_at_score = block_at_score
        self.warn_at_score = warn_at_score
        self.block_on_policy_violation = block_on_policy_violation

    def score(self, findings: List[Finding], projected_monthly_tokens: Optional[Dict[str, float]]=None, policy_violations: Optional[List[Finding]]=None) -> RiskScore:
        projected_monthly_tokens = projected_monthly_tokens or {'p50': 0.0, 'p95': 0.0}
        policy_violations = policy_violations or []
        drivers: Dict[str, float] = {}
        by_cat: Dict[str, List[Finding]] = {}
        for f in findings:
            by_cat.setdefault(f.category, []).append(f)
        total = 0.0
        for (cat, items) in by_cat.items():
            items_sorted = sorted(items, key=lambda f: SEVERITY_WEIGHT.get(f.severity, 0), reverse=False)
            cat_contrib = 0.0
            for (i, f) in enumerate(items_sorted):
                w = SEVERITY_WEIGHT.get(f.severity, 0)
                cat_contrib += w * 0.5 ** i
            drivers[cat] = drivers.get(cat, 0.0) + cat_contrib
            total += cat_contrib
        reasons: List[str] = []
        if policy_violations:
            pol = 100.0
            drivers['policy_violation'] = pol
            total += pol
            for pv in policy_violations:
                reasons.append(pv.message)
        score = int(max(0, min(100, round(total))))
        band = _band(score)
        has_block_policy = bool(policy_violations) and self.block_on_policy_violation
        has_critical = any((f.severity == 'critical' for f in findings))
        if has_block_policy or score >= self.block_at_score or has_critical:
            gate = 'block'
        elif score >= self.warn_at_score:
            gate = 'warn'
        else:
            gate = 'pass'
        if has_critical and 'critical findings present' not in reasons:
            crit = [f for f in findings if f.severity == 'critical']
            reasons.append(f'{len(crit)} critical finding(s): ' + '; '.join(sorted({f.category for f in crit})))
        drivers_list = sorted(({'category': k, 'contribution': round(v, 1)} for (k, v) in drivers.items()), key=lambda d: d['contribution'], reverse=True)
        return RiskScore(score, band, gate, drivers_list, projected_monthly_tokens, reasons)