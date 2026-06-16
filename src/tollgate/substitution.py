"""Model Substitution Recommender (capability 6) — dynamic, requirement-driven.

For each llm_call node, this derives a *requirement profile from the workflow*
(the task it performs, the context window its predicted tokens need, whether it
calls tools, the provider allowlist) and then searches the **whole catalog** for
the cheapest model that satisfies those requirements. It does NOT depend on a
hand-maintained substitution graph: any catalog model is a candidate, and the one
that wins is simply the cheapest that still supports the workflow.

What "supports the workflow" means here is deliberately the part we can verify
deterministically from the recovered IR and the catalog:
  * capability floor   — a minimum quality tier implied by the node's task class
                         (a router/classifier can run on a small model; reasoning
                         needs a frontier one). For an unrecognized task we stay
                         conservative.
  * context capacity   — the model's context window must hold the node's predicted
                         p95 input + output, and its max-output must cover the cap.
  * tool calling       — if the node (or workflow) uses tools, the model must too.
  * provider allowlist — honored if configured.

Quality beyond the tier floor cannot be verified without evaluation, so the
recommendation stays advisory. Where a maintainer curated a capability score for a
specific swap, we surface it as added confidence; otherwise we report a coarse
tier-based estimate and say so. No LLM calls; fully deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .catalog import Model, ModelCatalog
from .ir import Workflow
from .prediction import WorkflowPrediction

# Minimum quality tier (1=smallest .. 5=frontier) a task class can safely run on.
# Unknown/None tasks stay mid-tier so we never quietly recommend a tiny model for
# work we couldn't classify.
_TASK_MIN_TIER: Dict[Optional[str], int] = {
    "routing": 1,
    "classification": 1,
    "extraction": 2,
    "tool": 2,
    "generation": 3,
    "reasoning": 4,
    None: 3,
}
_DEFAULT_MIN_TIER = 3
_CTX_HEADROOM = 1.2   # require 20% slack over predicted p95 context use


@dataclass
class Recommendation:
    node_id: str
    from_model: str
    to_model: str
    capability_score: float
    current_call_usd: float
    new_call_usd: float
    savings_per_call_usd: float
    savings_pct: float
    expected_calls: float
    notes: str = ""

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "from_model": self.from_model,
            "to_model": self.to_model,
            "capability_score": round(self.capability_score, 3),
            "current_call_usd": round(self.current_call_usd, 6),
            "new_call_usd": round(self.new_call_usd, 6),
            "savings_per_call_usd": round(self.savings_per_call_usd, 6),
            "savings_pct": round(self.savings_pct, 1),
            "expected_calls": round(self.expected_calls, 3),
            "notes": self.notes,
        }


@dataclass
class _Requirements:
    min_tier: int
    needs_tools: bool
    min_context: int
    min_max_output: int


class SubstitutionEngine:
    def __init__(self, catalog: ModelCatalog, min_capability: float = 0.72,
                 min_savings_pct: float = 15.0, default_model: str = "gpt-4o",
                 task_min_tier: Optional[Dict[Optional[str], int]] = None):
        self.catalog = catalog
        self.min_capability = min_capability
        self.min_savings_pct = min_savings_pct
        self.default_model = default_model
        self.task_min_tier = task_min_tier or _TASK_MIN_TIER

    # --- requirement derivation ------------------------------------------------
    def _requirements(self, wf: Workflow, node, np_) -> _Requirements:
        min_tier = self.task_min_tier.get(node.task_class, _DEFAULT_MIN_TIER)
        wf_uses_tools = any(n.kind == "tool" for n in wf.nodes)
        needs_tools = node.task_class == "tool" or wf_uses_tools
        in95 = getattr(np_.input_tokens, "p95", np_.input_tokens.p50)
        out95 = getattr(np_.output_tokens, "p95", np_.output_tokens.p50)
        min_context = int((in95 + out95) * _CTX_HEADROOM)
        min_max_output = int(max(out95, node.max_output_tokens or 0))
        return _Requirements(min_tier, needs_tools, min_context, min_max_output)

    def _supports(self, cand: Model, req: _Requirements) -> bool:
        if cand.quality_tier < req.min_tier:
            return False
        if req.needs_tools and not cand.supports_tools:
            return False
        if cand.context_limit < req.min_context:
            return False
        if cand.max_output < req.min_max_output:
            return False
        return True

    def _capability(self, from_id: str, cand: Model, current: Model) -> float:
        """Curated score if a maintainer listed this exact swap, else a coarse
        tier-ratio estimate (a cheaper model at an equal/higher tier scores 1.0)."""
        curated = self.catalog.capability_hint(from_id, cand.id)
        if curated is not None:
            return curated
        if current.quality_tier <= 0:
            return 1.0
        return round(min(1.0, cand.quality_tier / current.quality_tier), 3)

    # --- recommendation --------------------------------------------------------
    def recommend(self, wf: Workflow, prediction: WorkflowPrediction,
                  allowlist: Optional[List[str]] = None) -> List[Recommendation]:
        recs: List[Recommendation] = []
        pred_by_node = {n.node_id: n for n in prediction.nodes}

        for node in wf.llm_nodes():
            np_ = pred_by_node.get(node.node_id)
            if np_ is None:
                continue
            current_id = node.intended_model or self.default_model
            current = self.catalog.get(current_id)
            if current is None:
                continue
            in50, out50 = np_.input_tokens.p50, np_.output_tokens.p50
            current_cost = current.cost(in50, out50)
            if current_cost <= 0:
                continue

            req = self._requirements(wf, node, np_)
            best: Optional[Recommendation] = None
            best_cost = current_cost

            # Search the WHOLE catalog, not a pre-wired edge list.
            for cand in self.catalog.all():
                if cand.id == current.id:
                    continue
                if allowlist and cand.id not in allowlist:
                    continue
                if not self._supports(cand, req):
                    continue
                new_cost = cand.cost(in50, out50)
                if new_cost >= current_cost:
                    continue
                pct = 100.0 * (current_cost - new_cost) / current_cost
                if pct < self.min_savings_pct:
                    continue
                cap = self._capability(current.id, cand, current)
                # A curated swap below the configured capability floor is skipped;
                # tier-qualified candidates without a curated score are governed by
                # the tier floor (the requirement profile already vetted them).
                if self.catalog.capability_hint(current.id, cand.id) is not None \
                        and cap < self.min_capability:
                    continue
                # Cheapest wins; ties broken toward the higher-tier (safer) model.
                if new_cost < best_cost or (
                        best is not None and abs(new_cost - best_cost) < 1e-12
                        and cand.quality_tier > self.catalog.get(best.to_model).quality_tier):
                    notes = self._notes(req, cand)
                    best = Recommendation(
                        node_id=node.node_id,
                        from_model=current.id,
                        to_model=cand.id,
                        capability_score=cap,
                        current_call_usd=current_cost,
                        new_call_usd=new_cost,
                        savings_per_call_usd=current_cost - new_cost,
                        savings_pct=pct,
                        expected_calls=np_.expected_calls,
                        notes=notes,
                    )
                    best_cost = new_cost
            if best:
                recs.append(best)
        return recs

    @staticmethod
    def _notes(req: _Requirements, cand: Model) -> str:
        parts = [f"meets tier>={req.min_tier}",
                 f"ctx>={req.min_context:,}"]
        if req.needs_tools:
            parts.append("tool-calling")
        basis = "cheapest model that supports this node (" + ", ".join(parts) + ")"
        if cand.self_hosted:
            basis += "; self-hosted candidate, savings assume amortized infra"
        return basis
