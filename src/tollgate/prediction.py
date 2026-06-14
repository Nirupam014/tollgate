"""Prediction Engine (capability 2).

Predicts a token DISTRIBUTION (p50/p95/p99) of input and output tokens per node,
then composes per-request cost across the IR graph. Three bases, in priority:

  1. telemetry      - empirical per-node history (when provided)
  2. heuristic      - regression on static prompt size + task-class priors
  3. tokenizer_static - exact static prompt count; output from family priors

Without telemetry the engine runs on heuristic/static and reports lower
confidence with wider bands (fail-safe, never silently optimistic).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .catalog import ModelCatalog
from .ir import IRNode, Workflow
from .graphutil import expected_executions
_OUTPUT_PRIOR = {'classification': 30, 'routing': 20, 'extraction': 200, 'tool': 120, 'generation': 600, 'reasoning': 900, None: 400}
_DEFAULT_RETRIEVAL_TOKENS = 4000

@dataclass
class Dist:
    p50: float
    p95: float
    p99: float

    @staticmethod
    def from_p50(p50: float, spread: float=1.6) -> 'Dist':
        return Dist(p50=p50, p95=p50 * spread, p99=p50 * spread * 2.35)

    def scale(self, k: float) -> 'Dist':
        return Dist(self.p50 * k, self.p95 * k, self.p99 * k)

    def add(self, other: 'Dist') -> 'Dist':
        return Dist(self.p50 + other.p50, self.p95 + other.p95, self.p99 + other.p99)

    def to_dict(self) -> Dict[str, float]:
        return {'p50': round(self.p50, 4), 'p95': round(self.p95, 4), 'p99': round(self.p99, 4)}

@dataclass
class NodePrediction:
    node_id: str
    model: Optional[str]
    input_tokens: Dist
    output_tokens: Dist
    expected_calls: float
    basis: str
    confidence: float
    cost_usd: Dist = field(default=None)

    def call_tokens(self) -> Dist:
        """Total tokens (input + output) for a single call of this node."""
        return self.input_tokens.add(self.output_tokens)

    def request_tokens(self) -> Dist:
        """Total tokens this node contributes per request (x expected calls)."""
        return self.call_tokens().scale(self.expected_calls)

@dataclass
class WorkflowPrediction:
    workflow_id: str
    nodes: List[NodePrediction]
    request_tokens: Dist
    request_cost_usd: Dist
    confidence: float
    basis: str

    def to_dict(self):
        return {'workflow_id': self.workflow_id, 'request_tokens': self.request_tokens.to_dict(), 'confidence': round(self.confidence, 3), 'basis': self.basis, 'nodes': [{'node_id': n.node_id, 'model': n.model, 'expected_calls': round(n.expected_calls, 3), 'input_tokens': n.input_tokens.to_dict(), 'output_tokens': n.output_tokens.to_dict(), 'request_tokens': n.request_tokens().to_dict(), 'basis': n.basis, 'confidence': round(n.confidence, 3)} for n in self.nodes]}

class PredictionEngine:

    def __init__(self, catalog: ModelCatalog, telemetry: Optional[Dict[str, dict]]=None, default_model: str='gpt-4o'):
        self.catalog = catalog
        self.telemetry = telemetry or {}
        self.default_model = default_model

    def predict(self, wf: Workflow) -> WorkflowPrediction:
        execs = expected_executions(wf)
        node_preds: List[NodePrediction] = []
        total_tokens = Dist(0, 0, 0)
        total_cost = Dist(0, 0, 0)
        confidences: List[float] = []
        bases = set()
        for node in wf.nodes:
            if node.kind != 'llm_call':
                continue
            np_ = self._predict_node(node, execs.get(node.node_id, 1.0))
            model = self.catalog.get(np_.model or self.default_model)
            if model is None:
                model = self.catalog.get(self.default_model)
            call_cost = Dist(model.cost(np_.input_tokens.p50, np_.output_tokens.p50), model.cost(np_.input_tokens.p95, np_.output_tokens.p95), model.cost(np_.input_tokens.p99, np_.output_tokens.p99))
            np_.cost_usd = call_cost
            total_cost = total_cost.add(call_cost.scale(np_.expected_calls))
            total_tokens = total_tokens.add(np_.request_tokens())
            node_preds.append(np_)
            confidences.append(np_.confidence)
            bases.add(np_.basis)
        confidence = min(confidences) if confidences else 0.4
        basis = 'telemetry' if 'telemetry' in bases else 'heuristic' if 'heuristic' in bases else 'tokenizer_static'
        return WorkflowPrediction(wf.workflow_id, node_preds, total_tokens, total_cost, confidence, basis)

    def _predict_node(self, node: IRNode, expected_calls: float) -> NodePrediction:
        model_id = node.intended_model or self.default_model
        tel = self.telemetry.get(node.node_id)
        if tel:
            inp = Dist(tel['input_p50'], tel['input_p95'], tel.get('input_p99', tel['input_p95'] * 1.3))
            out = Dist(tel['output_p50'], tel['output_p95'], tel.get('output_p99', tel['output_p95'] * 1.3))
            return NodePrediction(node.node_id, model_id, inp, out, expected_calls, 'telemetry', 0.9)
        static = node.static_input_tokens or 0
        input_p50 = float(static)
        basis = 'tokenizer_static' if static else 'heuristic'
        confidence = 0.6 if static else 0.45
        if node.retrieves_context:
            cap = node.retrieved_context_cap or _DEFAULT_RETRIEVAL_TOKENS
            input_p50 += cap
            confidence = min(confidence, 0.5)
        if node.appends_history:
            input_p50 += max(500.0, static * 0.5)
        if input_p50 <= 0:
            input_p50 = 800.0
            basis = 'heuristic'
            confidence = 0.35
        inp = Dist.from_p50(input_p50, spread=1.7 if node.retrieves_context or node.appends_history else 1.4)
        out_p50 = float(_OUTPUT_PRIOR.get(node.task_class, _OUTPUT_PRIOR[None]))
        if node.max_output_tokens:
            out_p50 = min(out_p50, node.max_output_tokens)
        out = Dist(out_p50, min(out_p50 * 2.2, node.max_output_tokens or out_p50 * 2.2), min(out_p50 * 3.0, node.max_output_tokens or out_p50 * 3.0))
        return NodePrediction(node.node_id, model_id, inp, out, expected_calls, basis, confidence)