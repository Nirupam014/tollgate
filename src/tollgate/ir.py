"""Workflow Intermediate Representation (IR).

Every supported agent/workflow source (DSL, raw prompt, LangGraph, ...) is
normalized into this provider-agnostic directed graph. All downstream engines
(prediction, simulation, detectors, scoring) operate on the IR only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Node kinds in the IR DAG.
NODE_KINDS = {"llm_call", "tool", "router", "map", "reduce", "human", "start", "end"}

# Control-flow edge types.
EDGE_TYPES = {"sequence", "conditional", "loop", "fanout"}

# Coarse task class drives output-token priors and model-fit checks.
TASK_CLASSES = {"classification", "extraction", "generation", "reasoning", "tool", "routing"}


@dataclass
class Retry:
    max_attempts: Optional[int] = None
    backoff: Optional[str] = None  # "exponential", "linear", None


@dataclass
class Guard:
    """Termination guard on a loop/recursive edge."""
    max_depth: Optional[int] = None
    counter: bool = False
    stop_condition: Optional[str] = None

    @property
    def is_bounded(self) -> bool:
        return bool(self.max_depth) or self.counter or bool(self.stop_condition)


@dataclass
class IRNode:
    node_id: str
    kind: str = "llm_call"
    intended_model: Optional[str] = None
    prompt_template: Optional[str] = None
    # Token accounting (filled by parser/prediction).
    static_input_tokens: int = 0
    max_output_tokens: Optional[int] = None
    task_class: Optional[str] = None
    # Structural risk signals.
    appends_history: bool = False          # seed of context explosion
    retrieves_context: bool = False
    retrieved_context_cap: Optional[int] = None  # None == uncapped
    fanout_factor: Optional[int] = None    # for map/parallel nodes; None == input-driven (unbounded)
    retry: Optional[Retry] = None
    # Branch annotations.
    branch_probability: float = 1.0        # prob this node executes given its parent ran

    def __post_init__(self):
        if self.kind not in NODE_KINDS:
            raise ValueError(f"invalid node kind: {self.kind!r}")
        if self.task_class and self.task_class not in TASK_CLASSES:
            raise ValueError(f"invalid task_class: {self.task_class!r}")


@dataclass
class IREdge:
    from_node: str
    to_node: str
    edge_type: str = "sequence"
    condition: Optional[str] = None
    probability: Optional[float] = None    # for conditional edges; default split applied if None
    guard: Optional[Guard] = None

    def __post_init__(self):
        if self.edge_type not in EDGE_TYPES:
            raise ValueError(f"invalid edge type: {self.edge_type!r}")


@dataclass
class Workflow:
    workflow_id: str
    source_kind: str
    nodes: List[IRNode] = field(default_factory=list)
    edges: List[IREdge] = field(default_factory=list)
    entry: Optional[str] = None
    content_hash: Optional[str] = None
    source_path: Optional[str] = None

    # --- convenience accessors -------------------------------------------------
    def node(self, node_id: str) -> IRNode:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        raise KeyError(node_id)

    @property
    def node_ids(self) -> List[str]:
        return [n.node_id for n in self.nodes]

    def out_edges(self, node_id: str) -> List[IREdge]:
        return [e for e in self.edges if e.from_node == node_id]

    def in_edges(self, node_id: str) -> List[IREdge]:
        return [e for e in self.edges if e.to_node == node_id]

    def adjacency(self) -> Dict[str, List[str]]:
        adj: Dict[str, List[str]] = {n.node_id: [] for n in self.nodes}
        for e in self.edges:
            adj.setdefault(e.from_node, []).append(e.to_node)
            adj.setdefault(e.to_node, adj.get(e.to_node, []))
        return adj

    def llm_nodes(self) -> List[IRNode]:
        return [n for n in self.nodes if n.kind == "llm_call"]

    def resolved_entry(self) -> Optional[str]:
        if self.entry:
            return self.entry
        # Heuristic: a node with no incoming edges.
        targets = {e.to_node for e in self.edges}
        for n in self.nodes:
            if n.node_id not in targets:
                return n.node_id
        return self.nodes[0].node_id if self.nodes else None
