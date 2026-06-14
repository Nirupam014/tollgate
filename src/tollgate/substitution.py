"""Model Substitution Recommender (capability 6).

For each llm_call node, finds cheaper catalog models that are safe enough for the
node's task class, then quantifies savings by re-pricing the node's predicted
token profile under each candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .catalog import ModelCatalog
from .ir import Workflow
from .prediction import WorkflowPrediction


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


class SubstitutionEngine:
    def __init__(self, catalog: ModelCatalog, min_capability: float = 0.72,
                 min_savings_pct: float = 15.0, default_model: str = "gpt-4o"):
        self.catalog = catalog
        self.min_capability = min_capability
        self.min_savings_pct = min_savings_pct
        self.default_model = default_model

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
            current_cost = current.cost(np_.input_tokens.p50, np_.output_tokens.p50)
            if current_cost <= 0:
                continue

            best: Optional[Recommendation] = None
            for sub in self.catalog.cheaper_alternatives(current_id, self.min_capability):
                cand = self.catalog.get(sub.to_model)
                if cand is None:
                    continue
                if allowlist and cand.id not in allowlist:
                    continue
                # Respect tool support and context capacity.
                if node.task_class == "tool" and not cand.supports_tools:
                    continue
                if cand.context_limit < current.context_limit * 0.25:
                    continue
                new_cost = cand.cost(np_.input_tokens.p50, np_.output_tokens.p50)
                if new_cost >= current_cost:
                    continue
                savings = current_cost - new_cost
                pct = 100.0 * savings / current_cost
                if pct < self.min_savings_pct:
                    continue
                rec = Recommendation(
                    node_id=node.node_id,
                    from_model=current.id,
                    to_model=cand.id,
                    capability_score=sub.capability_score,
                    current_call_usd=current_cost,
                    new_call_usd=new_cost,
                    savings_per_call_usd=savings,
                    savings_pct=pct,
                    expected_calls=np_.expected_calls,
                    notes=("self-hosted candidate; savings assume amortized infra"
                           if cand.self_hosted else ""),
                )
                # Prefer the highest capability_score among qualifying candidates.
                if best is None or rec.capability_score > best.capability_score:
                    best = rec
            if best:
                recs.append(best)
        return recs
