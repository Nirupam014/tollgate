"""Tree-sitter graph recovery for Go, Java, and Ruby → Workflow IR.

Brings these languages to parity with the Python/JS parsers: real graph recovery
(cycles, context, fan-out, cost projection) instead of only the advisory textual
lint. It runs only when the optional `multilang` extra is installed; otherwise the
caller honest-fails to the textual lint (no fabricated graph).

Recovered shapes (deterministic, mapped from the actual grammar node types):
  * Java — an unbounded `while (true)` loop around an LLM SDK call; and the
    LangGraph4j builder (`new StateGraph(...).addNode/.addEdge/.addConditionalEdges`).
  * Go   — an unbounded `for {}` loop around an LLM SDK call.
  * Ruby — an unbounded `loop do ... end` (or `while true`) around an LLM SDK call.

Bounded loops (Java `for`, Go `for i := ...`, Ruby `N.times`) and loops with a
break/return are recovered with a bounded guard (or left to the lint), never as a
critical unbounded cycle. Anything not cleanly recoverable yields an empty
workflow → dropped → textual lint.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import List, Optional, Tuple

from ..ir import Guard, IREdge, IRNode, Workflow
from .langgraph import _MODEL_HINTS as MODEL_HINTS
from . import treesitter_backend as tsb

TS_EXTS = {".go": "go", ".java": "java", ".rb": "ruby"}

_SENTINELS = {"__start__", "__end__", "START", "END", "start", "end"}

# Cheap discovery sniff (no parse): does this file look like a recoverable agent?
_INF_LOOP_TEXT = re.compile(
    r"(?im)(?:\bwhile\s*\(\s*true\s*\)|\bfor\s*\{|\bloop\s+do\b|\bloop\s*\{|"
    r"\bwhile\s+true\b)")
_SDK_TEXT_MARKERS = (
    "CreateChatCompletion", "CreateCompletion", "GenerateContent", "Messages.New",
    "completions().create", "messages().create", "createChatCompletion",
    ".chat(", "chat.completions.create", "messages.create", "generate_content",
)
_BUILDER_MARKERS = ("StateGraph", "MessageGraph", "addNode", "addEdge",
                    "addConditionalEdges")
_HISTORY_TEXT = ("messages.add", "messages.push", "msgs <<", "messages <<",
                 "<<", "...messages", "addMessages", ".append(")


def looks_like_ts_workflow(text: str) -> bool:
    if any(m in text for m in _BUILDER_MARKERS):
        return True
    return bool(_INF_LOOP_TEXT.search(text)) and any(m in text for m in _SDK_TEXT_MARKERS)


# --- tiny node helpers --------------------------------------------------------
def _text(node) -> str:
    return tsb.node_text(node)


def _field(node, name):
    try:
        return node.child_by_field_name(name)
    except Exception:
        return None


def _named_of_type(node, *types):
    return [c for c in node.named_children if c.type in types]


def _descendants(node, *types):
    return [n for n in tsb.walk(node) if n.type in types]


def _str_value(node) -> Optional[str]:
    """Literal value of a string node across grammars (Java string_literal →
    string_fragment, Ruby string → string_content, etc.)."""
    for c in node.named_children:
        if c.type in ("string_fragment", "string_content",
                      "interpreted_string_literal_content"):
            return _text(c)
    t = _text(node)
    if len(t) >= 2 and t[0] in "\"'`" and t[-1] in "\"'`":
        return t[1:-1]
    return t or None


def _scan_model(source: str) -> Optional[str]:
    for hint in MODEL_HINTS:
        if hint in source:
            return hint
    return None


def _has_exit(node) -> bool:
    """A break/return/throw anywhere in the loop body subtree (→ can terminate)."""
    return bool(_descendants(node, "break_statement", "return_statement",
                             "throw_statement", "break", "return", "next"))


# --- public entry -------------------------------------------------------------
def parse_treesitter(path: str) -> Workflow:
    lang = TS_EXTS.get(os.path.splitext(path)[1].lower())
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        source = fh.read()
    wf = Workflow(workflow_id=os.path.splitext(os.path.basename(path))[0],
                  source_kind=lang or "treesitter", nodes=[], edges=[], source_path=path)
    wf.content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    if not lang or not tsb.available():
        return wf  # honest failure → textual lint
    tree = tsb.parse(source, lang)
    if tree is None:
        return wf
    root = tree.root_node
    model = _scan_model(source)
    if lang == "java":
        nodes, edges, entry, kind = _parse_java(root, model)
    elif lang == "go":
        nodes, edges, entry, kind = _parse_go(root, model)
    elif lang == "ruby":
        nodes, edges, entry, kind = _parse_ruby(root, model)
    else:
        nodes, edges, entry, kind = [], [], None, lang
    if not nodes:
        return wf
    wf.nodes, wf.edges, wf.entry, wf.source_kind = nodes, edges, entry, kind
    return wf


# --- shared: build a self-loop workflow from infinite-loop bodies -------------
def _self_loops(loops: List[Tuple[object, int]], model, history_text: str, kind: str):
    """loops: list of (body_node, line). Each becomes one llm_call node with a
    self-loop edge (unbounded unless the body can break/return)."""
    nodes: List[IRNode] = []
    edges: List[IREdge] = []
    entry: Optional[str] = None
    used = set()
    for body, line in loops:
        nid = f"llm_call_{line}"
        while nid in used:
            nid += "_b"
        used.add(nid)
        grows = any(h in _text(body) for h in _HISTORY_TEXT)
        nodes.append(IRNode(node_id=nid, kind="llm_call", intended_model=model,
                            appends_history=grows))
        bounded = _has_exit(body)
        guard = Guard(counter=True, stop_condition="break") if bounded else None
        edges.append(IREdge(nid, nid, edge_type="loop", guard=guard))
        if entry is None:
            entry = nid
    return nodes, edges, entry, kind


# --- Java ---------------------------------------------------------------------
def _java_call_name(mi) -> Optional[str]:
    n = _field(mi, "name")
    return _text(n) if n else None


def _java_dotted(mi) -> str:
    parts: List[str] = []
    cur = mi
    while cur is not None and cur.type == "method_invocation":
        nm = _field(cur, "name")
        if nm:
            parts.append(_text(nm))
        cur = _field(cur, "object")
    if cur is not None and cur.type in ("identifier", "field_access"):
        parts.append(_text(cur))
    return ".".join(reversed(parts))


def _is_java_sdk(mi) -> bool:
    name = _java_call_name(mi)
    dotted = _java_dotted(mi)
    if name == "create" and ("completions" in dotted or "messages" in dotted):
        return True
    if name in ("createChatCompletion", "generateContent", "generateContentStream"):
        return True
    return False


def _java_while_true(node) -> bool:
    cond = _field(node, "condition")
    if cond is None:
        return False
    if any(c.type == "true" for c in cond.named_children):
        return True
    return _text(cond).strip("() \t\n") == "true"


def _java_arg_nodes(mi):
    al = _field(mi, "arguments")
    return list(al.named_children) if al is not None else []


def _parse_java(root, model):
    # 1) LangGraph4j builder (explicit named graph) takes precedence.
    nodes, edges, entry, kind = _parse_java_builder(root, model)
    if nodes:
        return nodes, edges, entry, kind
    # 2) Imperative: while(true) loops that issue an SDK call.
    loops = []
    for w in _descendants(root, "while_statement"):
        if not _java_while_true(w):
            continue
        body = _field(w, "body") or w
        if any(_is_java_sdk(mi) for mi in _descendants(body, "method_invocation")):
            loops.append((body, body.start_point[0] + 1))
    return _self_loops(loops, model, "", "imperative-java")


def _parse_java_builder(root, model):
    node_ids: List[str] = []
    raw_edges: List[Tuple[str, str, str]] = []
    entry: Optional[str] = None
    for mi in _descendants(root, "method_invocation"):
        name = _java_call_name(mi)
        if name not in ("addNode", "addEdge", "addConditionalEdges",
                        "setEntryPoint", "setFinishPoint"):
            continue
        args = _java_arg_nodes(mi)
        if name == "addNode":
            v = _arg_str(args, 0)
            if v and v not in node_ids:
                node_ids.append(v)
        elif name == "setEntryPoint":
            entry = _arg_str(args, 0) or entry
        elif name == "addEdge":
            a, b = _arg_endpoint(args, 0), _arg_endpoint(args, 1)
            if a and b:
                raw_edges.append((a, b, "sequence"))
        elif name == "addConditionalEdges":
            src = _arg_str(args, 0)
            if not src:
                continue
            for t in _java_conditional_targets(args):
                raw_edges.append((src, t, "conditional"))
    if not node_ids:
        return [], [], None, "langgraph4j"
    edges = _resolve_edges(node_ids, raw_edges)
    nodes = [IRNode(node_id=n, kind="llm_call", intended_model=model) for n in node_ids]
    if not entry:
        targets = {e.to_node for e in edges}
        entry = next((n for n in node_ids if n not in targets), node_ids[0])
    return nodes, edges, entry, "langgraph4j"


def _arg_str(args, i) -> Optional[str]:
    strs = [a for a in args if a.type == "string_literal"]
    return _str_value(strs[i]) if i < len(strs) else None


def _arg_endpoint(args, i) -> Optional[str]:
    """ith positional endpoint: a string literal value, or a bare START/END id."""
    if i < len(args):
        a = args[i]
        if a.type == "string_literal":
            return _str_value(a)
        if a.type == "identifier":
            return _text(a)
    return None


def _java_conditional_targets(args) -> List[str]:
    """Targets in addConditionalEdges: the values of a Map.of(...) path map, i.e.
    the odd-indexed string args of the inner `Map.of` call."""
    out: List[str] = []
    for a in args:
        if a.type == "method_invocation" and _java_call_name(a) == "of":
            inner = _java_arg_nodes(a)
            for idx, node in enumerate(inner):
                if idx % 2 == 1 and node.type == "string_literal":
                    out.append(_str_value(node))
    return [t for t in out if t]


# --- Go -----------------------------------------------------------------------
def _go_call_name(call) -> Optional[str]:
    fn = _field(call, "function")
    if fn is None:
        return None
    if fn.type == "selector_expression":
        f = _field(fn, "field")
        return _text(f) if f else None
    if fn.type == "identifier":
        return _text(fn)
    return None


def _is_go_sdk(call) -> bool:
    name = _go_call_name(call)
    return name in ("CreateChatCompletion", "CreateChatCompletionStream",
                    "CreateCompletion", "GenerateContent", "GenerateContentStream",
                    "New")  # Messages.New (anthropic-sdk-go); 'New' is gated by loop


def _go_for_infinite(node) -> bool:
    kids = node.named_children
    return len(kids) == 1 and kids[0].type == "block"


def _parse_go(root, model):
    loops = []
    for f in _descendants(root, "for_statement"):
        if not _go_for_infinite(f):
            continue
        body = _named_of_type(f, "block")
        body = body[0] if body else f
        if any(_is_go_sdk(c) for c in _descendants(body, "call_expression")):
            loops.append((body, body.start_point[0] + 1))
    return _self_loops(loops, model, "", "imperative-go")


# --- Ruby ---------------------------------------------------------------------
def _ruby_call_method(call) -> Optional[str]:
    m = _field(call, "method")
    if m is not None:
        return _text(m)
    ids = _named_of_type(call, "identifier")
    return _text(ids[-1]) if ids else None


def _ruby_block_of(call):
    for c in call.named_children:
        if c.type in ("do_block", "block"):
            return c
    return None


def _is_ruby_sdk(call) -> bool:
    m = _ruby_call_method(call)
    if m == "chat":
        return "parameters" in _text(call)
    if m in ("create",):
        return "messages" in _text(call)
    if m in ("generate_content",):
        return True
    return False


def _parse_ruby(root, model):
    loops = []
    # loop do ... end  (a `call` whose method is `loop` with a block)
    for call in _descendants(root, "call"):
        if _ruby_call_method(call) != "loop":
            continue
        body = _ruby_block_of(call)
        if body is None:
            continue
        if any(_is_ruby_sdk(c) for c in _descendants(body, "call")):
            loops.append((body, body.start_point[0] + 1))
    # while true / until false
    for w in _descendants(root, "while", "until"):
        cond = _field(w, "condition")
        cond_txt = _text(cond).strip() if cond is not None else ""
        is_inf = (w.type == "while" and cond_txt == "true") or \
                 (w.type == "until" and cond_txt == "false")
        if not is_inf:
            continue
        body = _field(w, "body") or w
        if any(_is_ruby_sdk(c) for c in _descendants(body, "call")):
            loops.append((body, body.start_point[0] + 1))
    return _self_loops(loops, model, "", "imperative-ruby")


# --- shared edge resolution (sentinels + back-edge → loop) --------------------
def _resolve_edges(node_ids: List[str], raw_edges) -> List[IREdge]:
    def real(n):
        return None if n in _SENTINELS else n
    for f, t, _et in raw_edges:
        for nid in (real(f), real(t)):
            if nid and nid not in node_ids:
                node_ids.append(nid)
    order = {n: i for i, n in enumerate(node_ids)}
    edges: List[IREdge] = []
    for f, t, et in raw_edges:
        rf, rt = real(f), real(t)
        if not rf or not rt:
            continue
        if order.get(rt, 1 << 30) <= order.get(rf, -1):
            et = "loop"
        edges.append(IREdge(rf, rt, edge_type=et))
    return edges
