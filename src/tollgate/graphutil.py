"""Graph utilities shared by prediction, simulation and detectors.

- Tarjan strongly-connected-components for cycle detection.
- Expected-execution propagation over the condensation DAG, accounting for
  conditional branch probabilities and loop iteration counts.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from .ir import IREdge, Workflow
DEFAULT_UNBOUNDED_ITERS = 8
DEFAULT_BOUNDED_ITERS = 3

def tarjan_scc(wf: Workflow) -> List[List[str]]:
    """Return strongly connected components (each a list of node ids)."""
    index_counter = [0]
    stack: List[str] = []
    on_stack: Dict[str, bool] = {}
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    result: List[List[str]] = []
    adj: Dict[str, List[str]] = {n.node_id: [] for n in wf.nodes}
    for e in wf.edges:
        if e.from_node in adj and e.to_node in adj:
            adj[e.from_node].append(e.to_node)

    def strongconnect(v: str):
        indices[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in adj.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w):
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            result.append(comp)
    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 10000))
    try:
        for n in wf.nodes:
            if n.node_id not in indices:
                strongconnect(n.node_id)
    finally:
        sys.setrecursionlimit(old_limit)
    return result

def find_cycles(wf: Workflow) -> List[List[str]]:
    """Return node sets that form cycles (SCCs of size>1, or self-loops)."""
    cycles = []
    selfloop = {e.from_node for e in wf.edges if e.from_node == e.to_node}
    for comp in tarjan_scc(wf):
        if len(comp) > 1 or (len(comp) == 1 and comp[0] in selfloop):
            cycles.append(comp)
    return cycles

def loop_edges_within(wf: Workflow, component: List[str]) -> List[IREdge]:
    cset = set(component)
    return [e for e in wf.edges if e.from_node in cset and e.to_node in cset]

def component_iterations(wf: Workflow, component: List[str]) -> Tuple[int, bool]:
    """Return (estimated_iterations, is_bounded) for a cyclic component."""
    edges = loop_edges_within(wf, component)
    bounded_depths = []
    any_guard = False
    for e in edges:
        if e.guard is not None and e.guard.is_bounded:
            any_guard = True
            if e.guard.max_depth:
                bounded_depths.append(e.guard.max_depth)
    if bounded_depths:
        return (max(bounded_depths), True)
    if any_guard:
        return (DEFAULT_BOUNDED_ITERS, True)
    return (DEFAULT_UNBOUNDED_ITERS, False)

def expected_executions(wf: Workflow) -> Dict[str, float]:
    """Expected number of executions per node for a single request.

    Conditional out-edges scale by their probability (even split if unspecified);
    cyclic components multiply their members by the estimated iteration count.
    """
    if not wf.nodes:
        return {}
    entry = wf.resolved_entry()
    nodes = wf.node_ids
    comps = tarjan_scc(wf)
    comp_of: Dict[str, int] = {}
    for (i, comp) in enumerate(comps):
        for n in comp:
            comp_of[n] = i
    iters: Dict[int, Tuple[int, bool]] = {}
    selfloop = {e.from_node for e in wf.edges if e.from_node == e.to_node}
    for (i, comp) in enumerate(comps):
        if len(comp) > 1 or (len(comp) == 1 and comp[0] in selfloop):
            iters[i] = component_iterations(wf, comp)
        else:
            iters[i] = (1, True)
    inter: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(comps))}
    cond_counts: Dict[str, int] = {}
    for e in wf.edges:
        if e.edge_type == 'conditional':
            cond_counts[e.from_node] = cond_counts.get(e.from_node, 0) + 1
    for e in wf.edges:
        (cu, cv) = (comp_of[e.from_node], comp_of[e.to_node])
        if cu == cv:
            continue
        if e.edge_type == 'conditional':
            prob = e.probability if e.probability is not None else 1.0 / max(1, cond_counts.get(e.from_node, 1))
        else:
            prob = 1.0
        inter[cu].append((cv, prob))
    indeg = {i: 0 for i in range(len(comps))}
    for (u, outs) in inter.items():
        for (v, _) in outs:
            indeg[v] += 1
    from collections import deque
    q = deque([i for i in range(len(comps)) if indeg[i] == 0])
    topo: List[int] = []
    while q:
        u = q.popleft()
        topo.append(u)
        for (v, _) in inter[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(topo) < len(comps):
        topo = list(range(len(comps)))
    entry_comp = comp_of.get(entry, topo[0] if topo else 0)
    comp_visits: Dict[int, float] = {i: 0.0 for i in range(len(comps))}
    comp_visits[entry_comp] = 1.0
    for u in topo:
        base = comp_visits[u]
        if base <= 0:
            continue
        for (v, prob) in inter[u]:
            comp_visits[v] += base * prob
    result: Dict[str, float] = {}
    for n in nodes:
        ci = comp_of[n]
        (it, _) = iters[ci]
        result[n] = comp_visits.get(ci, 0.0) * it
    return result