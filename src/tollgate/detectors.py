"""Risk Detectors (capabilities 4 & 5, plus fanout/prompt-bloat/retry/model-fit).

Static + simulated structural analysis over the IR. Each detector emits Findings
with machine-readable evidence. These are deterministic given the IR.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from .catalog import ModelCatalog
from .findings import Finding
from .graphutil import component_iterations, find_cycles, loop_edges_within
from .ir import IRNode, Workflow
from .prediction import WorkflowPrediction
DEFAULT_THRESHOLDS = {'prompt_bloat_tokens': 8000, 'context_explosion_fraction': 0.6, 'context_projection_iters': 20, 'fanout_warn_factor': 25, 'model_mismatch_min_tier': 4}
_CHEAP_TASK_CLASSES = {'classification', 'routing', 'extraction', 'tool'}

class DetectorEngine:

    def __init__(self, catalog: ModelCatalog, thresholds: Optional[Dict]=None, default_model: str='gpt-4o'):
        self.catalog = catalog
        self.t = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            self.t.update(thresholds)
        self.default_model = default_model

    def run(self, wf: Workflow, prediction: WorkflowPrediction, telemetry_depths: Optional[Dict[str, int]]=None) -> List[Finding]:
        findings: List[Finding] = []
        findings += self._recursive_loops(wf, telemetry_depths or {})
        findings += self._context_explosion(wf)
        findings += self._prompt_bloat(wf, prediction)
        findings += self._fanout(wf)
        findings += self._retry_storms(wf)
        findings += self._model_mismatch(wf)
        return findings

    def _recursive_loops(self, wf: Workflow, tel_depths: Dict[str, int]) -> List[Finding]:
        out = []
        for (i, comp) in enumerate(find_cycles(wf)):
            (iters, bounded) = component_iterations(wf, comp)
            edges = loop_edges_within(wf, comp)
            guards = [e.guard for e in edges if e.guard and e.guard.is_bounded]
            cycle_path = _format_cycle(wf, comp)
            observed = max((tel_depths.get(n, 0) for n in comp), default=0)
            if not bounded:
                severity = 'critical'
                msg = f'Cycle {cycle_path} has no bounded termination guard; cost is unbounded under adverse inputs.'
            elif observed and any((g.max_depth and observed > g.max_depth for g in guards)):
                severity = 'high'
                msg = f'Cycle {cycle_path} has a guard but production telemetry observed depth {observed} exceeding it.'
            else:
                severity = 'medium'
                msg = f'Cycle {cycle_path} is guarded (~{iters} iterations); verify the bound matches intent.'
            out.append(Finding(finding_id=f'loop_{i + 1}', category='recursive_loop', severity=severity, node_id=comp[0], source_path=wf.source_path, message=msg, evidence={'cycle': comp, 'termination_guard': 'bounded' if bounded else 'none_detected', 'estimated_iterations': iters, 'observed_max_depth': observed or None}))
        return out

    def _context_explosion(self, wf: Workflow) -> List[Finding]:
        out = []
        cyclic_nodes = {n for comp in find_cycles(wf) for n in comp}
        idx = 0
        for node in wf.llm_nodes():
            if not node.appends_history:
                continue
            model = self.catalog.get(node.intended_model or self.default_model) or self.catalog.get(self.default_model)
            limit = model.context_limit
            per_iter = max(node.static_input_tokens, 400)
            if node.retrieves_context:
                per_iter += node.retrieved_context_cap or 4000
            in_loop = node.node_id in cyclic_nodes
            horizon = self.t['context_projection_iters']
            projected = per_iter * (horizon if in_loop else 1)
            frac = projected / float(limit)
            if in_loop and frac >= self.t['context_explosion_fraction']:
                severity = 'critical' if frac >= 1.0 else 'high'
                idx += 1
                out.append(Finding(finding_id=f'ctx_{idx}', category='context_explosion', severity=severity, node_id=node.node_id, source_path=wf.source_path, message=f"Node '{node.node_id}' appends history inside a loop without truncation; context grows ~{per_iter} tokens/iteration and reaches ~{int(projected)} tokens by iteration {horizon} ({int(frac * 100)}% of the {limit:,}-token limit).", evidence={'growth_pattern': 'linear_accumulation', 'per_iteration_token_delta': per_iter, 'projected_tokens': int(projected), 'model_context_limit': limit, 'fraction_of_limit': round(frac, 3), 'in_loop': True, 'unbounded': True}))
            elif node.retrieves_context and node.retrieved_context_cap is None:
                idx += 1
                out.append(Finding(finding_id=f'ctx_{idx}', category='context_explosion', severity='medium', node_id=node.node_id, source_path=wf.source_path, message=f"Node '{node.node_id}' injects retrieved context with no cap; a large retrieval can spike input tokens unpredictably.", evidence={'growth_pattern': 'uncapped_retrieval', 'in_loop': in_loop}))
        return out

    def _prompt_bloat(self, wf: Workflow, prediction: WorkflowPrediction) -> List[Finding]:
        out = []
        idx = 0
        for node in wf.llm_nodes():
            if node.static_input_tokens and node.static_input_tokens >= self.t['prompt_bloat_tokens']:
                idx += 1
                out.append(Finding(finding_id=f'bloat_{idx}', category='prompt_bloat', severity='medium', node_id=node.node_id, source_path=wf.source_path, message=f"Static prompt for '{node.node_id}' is {node.static_input_tokens:,} tokens; consider trimming or caching the system prompt.", evidence={'static_input_tokens': node.static_input_tokens, 'threshold': self.t['prompt_bloat_tokens']}))
        return out

    def _fanout(self, wf: Workflow) -> List[Finding]:
        out = []
        idx = 0
        for node in wf.nodes:
            is_map = node.kind == 'map' or any((e.edge_type == 'fanout' for e in wf.out_edges(node.node_id)))
            if not is_map:
                continue
            if node.fanout_factor is None:
                idx += 1
                out.append(Finding(finding_id=f'fanout_{idx}', category='fanout', severity='high', node_id=node.node_id, source_path=wf.source_path, message=f"Map/parallel node '{node.node_id}' has input-driven (uncapped) fan-out; cost scales linearly with input size.", evidence={'fanout_factor': 'input_driven_unbounded'}))
            elif node.fanout_factor >= self.t['fanout_warn_factor']:
                idx += 1
                out.append(Finding(finding_id=f'fanout_{idx}', category='fanout', severity='medium', node_id=node.node_id, source_path=wf.source_path, message=f"Node '{node.node_id}' fans out to {node.fanout_factor} parallel LLM calls per request.", evidence={'fanout_factor': node.fanout_factor}))
        return out

    def _retry_storms(self, wf: Workflow) -> List[Finding]:
        out = []
        idx = 0
        for node in wf.llm_nodes():
            r = node.retry
            if r is None:
                continue
            unbounded = r.max_attempts is None or r.max_attempts > 5
            no_backoff = not r.backoff
            if unbounded or no_backoff:
                idx += 1
                out.append(Finding(finding_id=f'retry_{idx}', category='retry_storm', severity='high' if unbounded and no_backoff else 'medium', node_id=node.node_id, source_path=wf.source_path, message=f"Node '{node.node_id}' retries {('without an attempt cap' if unbounded else 'with a high cap')}{(' and no backoff' if no_backoff else '')}; transient errors can multiply token spend.", evidence={'max_attempts': r.max_attempts, 'backoff': r.backoff}))
        return out

    def _model_mismatch(self, wf: Workflow) -> List[Finding]:
        out = []
        idx = 0
        for node in wf.llm_nodes():
            model = self.catalog.get(node.intended_model or self.default_model)
            if model is None:
                continue
            if node.task_class in _CHEAP_TASK_CLASSES and model.quality_tier >= self.t['model_mismatch_min_tier']:
                idx += 1
                out.append(Finding(finding_id=f'mismatch_{idx}', category='model_mismatch', severity='medium', node_id=node.node_id, source_path=wf.source_path, message=f"Node '{node.node_id}' runs a tier-{model.quality_tier} model ('{model.id}') on a '{node.task_class}' task that a cheaper model typically handles.", evidence={'model': model.id, 'quality_tier': model.quality_tier, 'task_class': node.task_class}))
        return out

_UUID_RE = __import__('re').compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', __import__('re').I)


def _short_node(n: str) -> str:
    """Readable label for a node id: UUIDs collapse to an 8-char stub, very long
    ids are truncated. Keeps already-human ids (e.g. 'planner') untouched."""
    s = str(n)
    if _UUID_RE.match(s):
        return s[:8]
    return s if len(s) <= 32 else s[:29] + '…'


def _format_cycle(wf: Workflow, comp: List[str], max_nodes: int = 4) -> str:
    """Human-friendly one-line cycle path. Shortens opaque (UUID) node ids and
    collapses the middle of long cycles so the message reads as a sentence, not a
    wall of identifiers."""
    order = [_short_node(n) for n in comp]
    if not order:
        return 'a cycle'
    if len(order) > max_nodes:
        head = ' → '.join(order[:max_nodes])
        return f'{head} → … → {order[0]} ({len(order)} nodes)'
    return ' → '.join(order + [order[0]])