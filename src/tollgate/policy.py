"""Policy evaluation (pre-deploy half of the real-time policy plane).

One rule language evaluated against a prediction/simulation result. Policy types:
token_ceiling, model_allowlist, context_cap, loop_guard, gate_threshold.
Violations are emitted as Findings so they flow into scoring and reporting
uniformly. Limits are expressed in TOKENS, not dollars.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .findings import Finding
from .ir import Workflow
from .prediction import WorkflowPrediction
from .graphutil import component_iterations, find_cycles


@dataclass
class Policy:
    name: str
    type: str
    rule: Dict[str, Any]
    enforcement: str = "block"   # warn | block
    enabled: bool = True


def load_policies(raw: Optional[List[Dict[str, Any]]]) -> List[Policy]:
    out = []
    for p in (raw or []):
        out.append(Policy(
            name=p.get("name", p.get("type", "policy")),
            type=p["type"],
            rule=p.get("rule", {}),
            enforcement=p.get("enforcement", "block"),
            enabled=p.get("enabled", True),
        ))
    return out


class PolicyEngine:
    def __init__(self, policies: List[Policy]):
        self.policies = [p for p in policies if p.enabled]

    def evaluate(self, wf: Workflow, prediction: WorkflowPrediction,
                 monthly_tokens: Dict[str, float]) -> List[Finding]:
        violations: List[Finding] = []
        for i, p in enumerate(self.policies):
            handler = getattr(self, f"_check_{p.type}", None)
            if handler is None:
                continue
            v = handler(p, wf, prediction, monthly_tokens)
            for f in v:
                f.evidence["enforcement"] = p.enforcement
                f.evidence["policy"] = p.name
                violations.append(f)
        return violations

    # --- individual policy types ----------------------------------------------
    def _check_token_ceiling(self, p, wf, prediction, monthly_tokens) -> List[Finding]:
        # Per-request token ceiling.
        per_req = float(p.rule.get("max_tokens_per_request", 0))
        if per_req:
            val = prediction.request_tokens.p95
            if val > per_req:
                return [Finding(
                    finding_id=f"pol_tokceil_{p.name}",
                    category="token_ceiling_exceeded",
                    severity="critical" if p.enforcement == "block" else "high",
                    message=(f"Token ceiling '{p.name}': projected p95 {int(val):,} tokens/request "
                             f"exceeds limit {int(per_req):,}."),
                    node_id=None, source_path=wf.source_path,
                    evidence={"limit_tokens_per_request": int(per_req),
                              "projected_p95_tokens_per_request": int(val)},
                )]
        # Monthly token ceiling.
        monthly_limit = float(p.rule.get("max_monthly_tokens", 0))
        if monthly_limit:
            metric = p.rule.get("metric", "projected_p95")
            val = monthly_tokens.get("p95" if "p95" in metric else "p50", 0.0)
            if val > monthly_limit:
                return [Finding(
                    finding_id=f"pol_tokceil_{p.name}",
                    category="token_ceiling_exceeded",
                    severity="critical" if p.enforcement == "block" else "high",
                    message=(f"Token ceiling '{p.name}': projected {metric} {int(val):,} tokens/mo "
                             f"exceeds limit {int(monthly_limit):,}/mo."),
                    node_id=None, source_path=wf.source_path,
                    evidence={"limit_monthly_tokens": int(monthly_limit),
                              "projected_monthly_tokens": int(val), "metric": metric},
                )]
        return []

    def _check_model_allowlist(self, p, wf, prediction, monthly_tokens) -> List[Finding]:
        allow = set(p.rule.get("allow", []))
        deny_tier_above = p.rule.get("deny_tier_above")
        out = []
        for n in prediction.nodes:
            if allow and n.model not in allow:
                out.append(Finding(
                    finding_id=f"pol_allow_{n.node_id}",
                    category="policy_violation",
                    severity="critical" if p.enforcement == "block" else "high",
                    message=f"Model '{n.model}' on node '{n.node_id}' is not in the allowlist.",
                    node_id=n.node_id, source_path=wf.source_path,
                    evidence={"allow": sorted(allow), "found": n.model},
                ))
        return out

    def _check_context_cap(self, p, wf, prediction, monthly_tokens) -> List[Finding]:
        cap = int(p.rule.get("max_context_tokens", 0))
        out = []
        if not cap:
            return out
        for n in prediction.nodes:
            if n.input_tokens.p95 > cap:
                out.append(Finding(
                    finding_id=f"pol_ctx_{n.node_id}",
                    category="policy_violation",
                    severity="high",
                    message=(f"Node '{n.node_id}' p95 input {int(n.input_tokens.p95):,} tokens "
                             f"exceeds context cap {cap:,}."),
                    node_id=n.node_id, source_path=wf.source_path,
                    evidence={"cap": cap, "p95_input_tokens": int(n.input_tokens.p95)},
                ))
        return out

    def _check_loop_guard(self, p, wf, prediction, monthly_tokens) -> List[Finding]:
        require = bool(p.rule.get("require_termination_guard", True))
        max_depth = p.rule.get("max_depth")
        out = []
        for comp in find_cycles(wf):
            iters, bounded = component_iterations(wf, comp)
            if require and not bounded:
                out.append(Finding(
                    finding_id=f"pol_loop_{comp[0]}",
                    category="policy_violation",
                    severity="critical",
                    message=f"Loop {('/'.join(comp))} violates loop_guard policy (no termination guard).",
                    node_id=comp[0], source_path=wf.source_path,
                    evidence={"cycle": comp, "bounded": bounded},
                ))
            elif max_depth and bounded and iters > max_depth:
                out.append(Finding(
                    finding_id=f"pol_loop_{comp[0]}",
                    category="policy_violation",
                    severity="high",
                    message=(f"Loop {('/'.join(comp))} bound ~{iters} exceeds policy max_depth {max_depth}."),
                    node_id=comp[0], source_path=wf.source_path,
                    evidence={"cycle": comp, "iterations": iters, "max_depth": max_depth},
                ))
        return out

    def _check_gate_threshold(self, p, wf, prediction, monthly_tokens) -> List[Finding]:
        # Gate thresholds are applied in scoring; nothing to emit here.
        return []
