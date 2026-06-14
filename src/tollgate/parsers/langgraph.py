"""Best-effort LangGraph / LangChain source parser.

We statically analyze a Python source file with the `ast` module and recover the
graph structure from common builder calls:

    g = StateGraph(State)
    g.add_node("plan", plan_fn)
    g.add_edge("plan", "act")
    g.add_edge("act", "plan")                 # loop edge
    g.add_conditional_edges("act", route, {...})
    g.set_entry_point("plan")

Model binding is recovered heuristically by scanning for known model id strings
in the source. This parser is intentionally tolerant: anything it cannot resolve
is left unset so downstream detectors treat it conservatively (e.g. an
unannotated loop edge with no guard is flagged as unbounded).
"""
from __future__ import annotations

import ast
import hashlib
import os
from typing import List, Optional

from ..ir import IREdge, IRNode, Workflow

_MODEL_HINTS = [
    "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1",
    "claude-opus-4", "claude-sonnet-4", "claude-haiku-3.5",
    "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash",
    "llama-3.3-70b", "llama-3.1-8b", "mixtral-8x7b",
]

_LOOP_BUILDER_METHODS = {"add_edge", "add_conditional_edges"}


def _scan_models(source: str) -> Optional[str]:
    for hint in _MODEL_HINTS:
        if hint in source:
            return hint
    return None


def parse_langgraph(path: str) -> Workflow:
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()

    default_model = _scan_models(source)
    node_ids: List[str] = []
    edges: List[IREdge] = []
    entry: Optional[str] = None
    appends_history = _detects_history(source)

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        # Unparseable (bad syntax, null bytes, or pathological nesting): degrade
        # to a single opaque node so the file still scores without crashing.
        return _fallback(path, source, default_model)

    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        method = _attr_name(call.func)
        if method is None:
            continue
        str_args = [a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]

        if method == "add_node" and str_args:
            if str_args[0] not in node_ids:
                node_ids.append(str_args[0])
        elif method == "add_edge" and len(str_args) >= 2:
            edges.append(IREdge(str_args[0], str_args[1], edge_type="sequence"))
        elif method == "add_conditional_edges" and str_args:
            src = str_args[0]
            # Targets often live in a dict literal arg; collect its string values.
            for a in call.args:
                if isinstance(a, ast.Dict):
                    for v in a.values:
                        if isinstance(v, ast.Constant) and isinstance(v.value, str):
                            edges.append(IREdge(src, v.value, edge_type="conditional"))
        elif method in ("set_entry_point", "set_finish_point") and str_args:
            if method == "set_entry_point":
                entry = str_args[0]

    # Ensure all edge endpoints exist as nodes.
    for e in edges:
        for nid in (e.from_node, e.to_node):
            if nid not in node_ids and nid not in ("__start__", "__end__", "END", "START"):
                node_ids.append(nid)

    # Reclassify back-edges (target appears earlier in declared order) as loops.
    order = {nid: i for i, nid in enumerate(node_ids)}
    for e in edges:
        if e.edge_type == "sequence" and order.get(e.to_node, 1 << 30) <= order.get(e.from_node, -1):
            e.edge_type = "loop"  # guard unknown -> unbounded -> will be flagged

    nodes = [
        IRNode(
            node_id=nid,
            kind="llm_call",
            intended_model=default_model,
            appends_history=appends_history,
        )
        for nid in node_ids
        if nid not in ("__start__", "__end__", "END", "START")
    ]

    wf = Workflow(
        workflow_id=os.path.splitext(os.path.basename(path))[0],
        source_kind="langgraph",
        nodes=nodes,
        edges=[e for e in edges if e.from_node in {n.node_id for n in nodes} and e.to_node in {n.node_id for n in nodes}],
        entry=entry,
        source_path=path,
    )
    wf.content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return wf


def _detects_history(source: str) -> bool:
    markers = ["messages +", "messages.append", "add_messages", "history.append", "state['messages']", 'state["messages"]']
    return any(m in source for m in markers)


def _attr_name(func) -> Optional[str]:
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _fallback(path: str, source: str, model: Optional[str]) -> Workflow:
    node = IRNode(node_id="graph", kind="llm_call", intended_model=model, appends_history=_detects_history(source))
    wf = Workflow(
        workflow_id=os.path.splitext(os.path.basename(path))[0],
        source_kind="langgraph",
        nodes=[node],
        edges=[],
        entry="graph",
        source_path=path,
    )
    wf.content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return wf
