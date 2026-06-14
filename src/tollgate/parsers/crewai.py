"""CrewAI source parser (static AST).

Why this exists
---------------
A CrewAI agent file has neither a LangGraph builder call nor a recognizable raw
LLM SDK call — the model invocation is hidden inside the framework, and the
control loop *is* the framework's delegation loop. So the LangGraph parser and the
imperative (loop-around-an-SDK-call) parser both see nothing, and the file was
silently dropped (workflow_count == 0). The field-study miss audit surfaced this:
real CrewAI repos scored as "no workflow found". This parser closes that gap.

What it recovers
----------------
    from crewai import Agent, Task, Crew, Process
    researcher = Agent(role="...", llm="gpt-4o", allow_delegation=True)
    writer     = Agent(role="...", llm="gpt-4o-mini")
    t1 = Task(description="research ...", agent=researcher)
    t2 = Task(description="write ...",    agent=writer)
    crew = Crew(agents=[researcher, writer], tasks=[t1, t2],
                process=Process.hierarchical, memory=True)
    crew.kickoff()

Mapping to the Workflow IR:
  * each Task               -> an `llm_call` node (the unit of LLM work), prompt =
                               the task description, model = the executing agent's
                               llm (falling back to a model string seen in source).
  * sequential process      -> `sequence` edges in task order (Crew(tasks=[...]) if
                               resolvable, else declaration order).
  * hierarchical process,
    a manager_llm/manager,
    or any agent with
    allow_delegation=True   -> a delegation **loop** edge (last task -> first). CrewAI
                               has no built-in delegation-depth cap, so the edge is
                               left UNGUARDED on purpose — the recursive-loop
                               detector then flags it as an unbounded cycle to
                               verify. (Erring toward flagging is the safe direction
                               for a prevention tool; a known-terminating crew can be
                               whitelisted via policy.)
  * crew memory=True or a
    delegation loop         -> `appends_history` on nodes, so context-explosion
                               analysis applies across delegation rounds.

Honest-failure posture (matches the rest of Tollgate): the file is never imported
or executed, only AST-parsed. Anything we cannot resolve (a dynamically built
task list, an agent llm we can't read) is left unset so downstream detectors stay
conservative. If we cannot recover a single task or agent, we return an empty
workflow, which the pipeline drops rather than scoring as a misleading PASS.
"""
from __future__ import annotations

import ast
import hashlib
import os
from typing import Dict, List, Optional

from ..ir import Guard, IREdge, IRNode, Workflow

# Model id fragments we can lift straight out of source as a last resort.
_MODEL_HINTS = [
    "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-4-turbo", "gpt-4",
    "o3-mini", "o3", "o1-mini", "o1", "gpt-3.5-turbo",
    "claude-opus-4", "claude-sonnet-4", "claude-3-7-sonnet", "claude-3-5-sonnet",
    "claude-3-haiku", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash",
    "llama-3.3-70b", "llama-3.1-70b", "llama-3.1-8b", "mixtral-8x7b",
    "mistral-large", "deepseek-chat", "deepseek-reasoner", "qwen2.5",
]

_CTORS = {"Agent", "Task", "Crew"}


def looks_like_crewai(source: str) -> bool:
    """True when a file is a CrewAI workflow we can structurally read.

    Gated on an actual crewai import plus a Crew(...) construction so we don't
    fire on unrelated code that happens to define a class called Agent or Task.
    """
    if "crewai" not in source.lower():
        return False
    return "Crew(" in source


# --- small AST helpers -------------------------------------------------------

def _call_name(func) -> Optional[str]:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _kwargs(call: ast.Call) -> Dict[str, ast.AST]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg}


def _str(node) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # f-strings / concatenations: take the literal parts so a description still
    # contributes static prompt tokens.
    if isinstance(node, ast.JoinedStr):
        parts = [v.value for v in node.values
                 if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        return "".join(parts) if parts else None
    return None


def _bool(node) -> Optional[bool]:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _ref_name(node) -> Optional[str]:
    """Resolve `agent=researcher` or `agent=self.researcher` to a binding name."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):  # agent=build_researcher()
        return _call_name(node.func)
    return None


def _list_refs(node) -> List[str]:
    out: List[str] = []
    if isinstance(node, (ast.List, ast.Tuple)):
        for el in node.elts:
            nm = _ref_name(el)
            if nm:
                out.append(nm)
    return out


def _process(node) -> Optional[str]:
    if isinstance(node, ast.Attribute):       # Process.hierarchical
        return node.attr.lower()
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.lower()
    return None


def _model_of(kw: Dict[str, ast.AST]) -> Optional[str]:
    """Pull a model id from an agent's llm=/model= kwarg, if statically present."""
    for key in ("llm", "model", "model_name"):
        v = kw.get(key)
        if v is None:
            continue
        s = _str(v)
        if s:
            return s
        if isinstance(v, ast.Call):           # llm=ChatOpenAI(model="gpt-4o")
            inner = _kwargs(v)
            for ik in ("model", "model_name", "model_id"):
                s = _str(inner.get(ik))
                if s:
                    return s
    return None


def _scan_models(source: str) -> Optional[str]:
    for hint in _MODEL_HINTS:
        if hint in source:
            return hint
    return None


def _binding_map(tree: ast.AST) -> Dict[int, str]:
    """Map id(Call node) -> the variable/function name it is bound to.

    Handles both the imperative style (`x = Agent(...)`) and CrewAI's decorator
    style (`@agent def researcher(self): return Agent(...)`).
    """
    bind: Dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                bind[id(node.value)] = node.targets[0].id
        elif isinstance(node, ast.FunctionDef):
            for sub in node.body:
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Call):
                    bind[id(sub.value)] = node.name
    return bind


# --- main parse --------------------------------------------------------------

def parse_crewai(path: str) -> Workflow:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        source = fh.read()

    default_model = _scan_models(source)
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return _empty(path, source)

    bind = _binding_map(tree)
    agents: Dict[str, dict] = {}
    tasks: List[dict] = []
    crew: dict = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        cname = _call_name(node.func)
        if cname not in _CTORS:
            continue
        kw = _kwargs(node)
        name = bind.get(id(node))
        if cname == "Agent":
            key = name or f"agent_{len(agents)}"
            agents[key] = {
                "role": _str(kw.get("role")),
                "model": _model_of(kw),
                "allow_delegation": _bool(kw.get("allow_delegation")),
                "max_iter": _int(kw.get("max_iter")),
            }
        elif cname == "Task":
            tasks.append({
                "var": name or f"task_{len(tasks)}",
                "description": _str(kw.get("description")) or _str(kw.get("prompt")),
                "agent": _ref_name(kw.get("agent")),
            })
        elif cname == "Crew":
            crew = {
                "process": _process(kw.get("process")),
                "memory": _bool(kw.get("memory")),
                "has_manager": ("manager_llm" in kw or "manager_agent" in kw),
                "task_order": _list_refs(kw.get("tasks")),
            }

    nodes, order = _build_nodes(tasks, agents, crew, default_model)
    if not nodes:
        return _empty(path, source)

    delegates = (
        (crew.get("process") == "hierarchical")
        or crew.get("has_manager", False)
        or any(a.get("allow_delegation") for a in agents.values())
    )
    if delegates:
        for n in nodes:
            n.appends_history = True  # context accumulates across delegation rounds

    edges = _build_edges(order, delegates)

    wf = Workflow(
        workflow_id=os.path.splitext(os.path.basename(path))[0],
        source_kind="crewai",
        nodes=nodes,
        edges=edges,
        entry=order[0] if order else None,
        source_path=path,
    )
    wf.content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return wf


def _build_nodes(tasks, agents, crew, default_model):
    """Return (nodes, ordered_node_ids). One llm_call node per task (or per agent
    if a crew defines agents but no explicit Task)."""
    from ..tokenizer import count_tokens

    def model_for(agent_ref):
        a = agents.get(agent_ref or "")
        if a and a.get("model"):
            return a["model"]
        return default_model

    nodes: List[IRNode] = []
    order: List[str] = []

    if tasks:
        # Prefer the order declared in Crew(tasks=[...]) when we could resolve it.
        ordered = crew.get("task_order") or []
        by_var = {t["var"]: t for t in tasks}
        seq = [by_var[v] for v in ordered if v in by_var] or tasks
        seen = set()
        for i, t in enumerate(seq):
            nid = t["var"] if t["var"] not in seen else f"{t['var']}_{i}"
            seen.add(nid)
            node = IRNode(
                node_id=nid,
                kind="llm_call",
                intended_model=model_for(t.get("agent")),
                prompt_template=t.get("description"),
                task_class="reasoning",
            )
            if t.get("description"):
                node.static_input_tokens = count_tokens(t["description"])
            nodes.append(node)
            order.append(nid)
    else:
        # No tasks declared — fall back to one node per agent so the crew is still
        # visible (and a delegation loop among agents can still be flagged).
        for i, (key, a) in enumerate(agents.items()):
            node = IRNode(node_id=key, kind="llm_call",
                          intended_model=a.get("model") or default_model,
                          task_class="reasoning")
            nodes.append(node)
            order.append(key)
    return nodes, order


def _build_edges(order: List[str], delegates: bool) -> List[IREdge]:
    edges: List[IREdge] = []
    for a, b in zip(order, order[1:]):
        edges.append(IREdge(a, b, edge_type="sequence"))
    if delegates and order:
        # Delegation/hierarchical re-dispatch: model as an unguarded back-edge so
        # the loop detector flags it as unbounded (CrewAI has no delegation cap).
        edges.append(IREdge(order[-1], order[0], edge_type="loop", guard=None))
    return edges


def _int(node) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def _empty(path: str, source: str) -> Workflow:
    """No recoverable crew structure -> empty workflow (pipeline drops it)."""
    wf = Workflow(
        workflow_id=os.path.splitext(os.path.basename(path))[0],
        source_kind="crewai",
        nodes=[],
        edges=[],
        source_path=path,
    )
    wf.content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return wf
