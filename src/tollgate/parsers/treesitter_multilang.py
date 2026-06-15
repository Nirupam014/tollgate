"""Tree-sitter graph recovery for Go, Java, and Ruby → Workflow IR.

Brings these languages to parity with the Python/JS parsers when the optional
`multilang` extra is installed: real graph recovery (cycles, context, fan-out,
cost projection), not just the advisory textual lint. Without the extra the caller
honest-fails to the lint — never a fabricated graph. Tree-sitter parsing is
deterministic, preserving reproducibility.

This is a faithful port of the Python imperative parser (`parsers/imperative.py`)
onto a concrete syntax tree, parameterized by a small per-language adapter:

  1. Find functions/methods; scan each one's *own* body (not nested defs) for a
     direct LLM SDK call, the names it calls, and a local model id.
  2. Mark "LLM functions" — own SDK call, or (transitively) calls another LLM
     function — so a chain of thin wrappers (agent → call_llm → SDK) is recovered.
  3. Split drivers (entry points / loop owners we descend into) from wrappers
     (LLM helpers we *site* the call to and don't descend into, so the SDK call
     inside isn't double-counted).
  4. Collect trigger sites (a direct SDK call, or a call to a wrapper) in source
     order, group them by their enclosing outermost loop, and emit one IR node per
     distinct callee with sequence edges between them. A while-style loop (or a
     for-style loop that grows its own queue) closes the chain with a `loop` edge
     whose guard reflects whether the loop can actually terminate.

Plus the Java **LangGraph4j** builder (`StateGraph.addNode/.addEdge/...`), an
explicit named graph recovered like the Python/JS LangGraph parsers.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from ..ir import Guard, IREdge, IRNode, Workflow
from .langgraph import _MODEL_HINTS as MODEL_HINTS
from . import treesitter_backend as tsb

TS_EXTS = {".go": "go", ".java": "java", ".rb": "ruby"}

_SENTINELS = {"__start__", "__end__", "START", "END", "start", "end"}

# Cheap discovery sniff (no parse).
_INF_LOOP_TEXT = re.compile(
    r"(?im)(?:\bwhile\s*\(|\bfor\s*[\{\(]|\bloop\s+do\b|\bloop\s*\{|"
    r"\bwhile\s+\w|\buntil\s+\w|\.times\b|\.each\b)")
_SDK_TEXT_MARKERS = (
    "CreateChatCompletion", "CreateCompletion", "GenerateContent", "Messages.New",
    "completions().create", "messages().create", "createChatCompletion",
    ".chat(", "chat.completions.create", "messages.create", "generate_content",
)
_BUILDER_MARKERS = ("StateGraph", "MessageGraph", "addNode", "addEdge",
                    "addConditionalEdges")
_HISTORY_TEXT = ("messages.add", "messages.push", "msgs <<", "messages <<", "<<",
                 "...messages", "addMessages", ".append(", ".push(", ".add(")


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


def _line(node) -> int:
    return node.start_point[0] + 1


def _col(node) -> int:
    return node.start_point[1]


def _ancestors(node):
    p = node.parent
    while p is not None:
        yield p
        p = p.parent


def _str_value(node) -> Optional[str]:
    for c in node.named_children:
        if c.type in ("string_fragment", "string_content",
                      "interpreted_string_literal_content"):
            return _text(c)
    t = _text(node)
    if len(t) >= 2 and t[0] in "\"'`" and t[-1] in "\"'`":
        return t[1:-1]
    return t or None


def _scan_model(text: str) -> Optional[str]:
    for hint in MODEL_HINTS:
        if hint in text:
            return hint
    return None


# --- public entry -------------------------------------------------------------
def parse_treesitter(path: str) -> Workflow:
    lang = TS_EXTS.get(os.path.splitext(path)[1].lower())
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        source = fh.read()
    wf = Workflow(workflow_id=os.path.splitext(os.path.basename(path))[0],
                  source_kind=lang or "treesitter", nodes=[], edges=[], source_path=path)
    wf.content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    if not lang or not tsb.available():
        return wf
    tree = tsb.parse(source, lang)
    if tree is None:
        return wf
    root = tree.root_node
    file_model = _scan_model(source)

    if lang == "java":
        nodes, edges, entry, kind = _parse_java_builder(root, file_model)
        if nodes:
            wf.nodes, wf.edges, wf.entry, wf.source_kind = nodes, edges, entry, kind
            return wf

    adapter = _ADAPTERS[lang]()
    nodes, edges, entry = _recover_imperative(root, adapter, file_model)
    if not nodes:
        return wf
    wf.nodes, wf.edges, wf.entry, wf.source_kind = nodes, edges, entry, f"imperative-{lang}"
    return wf


# --- generic imperative recovery (port of parsers/imperative.py) --------------
def _recover_imperative(root, ad: "LangAdapter", file_model):
    funcs = ad.functions(root)
    fid_name = {id(fn): ad.func_name(fn) for fn in funcs}
    local = {id(fn): _local_scan(fn, ad) for fn in funcs}   # (has_sdk, called:set, model)

    is_llm = {}
    for fn in funcs:
        n = fid_name[id(fn)]
        is_llm[n] = is_llm.get(n, False) or local[id(fn)][0]
    changed = True
    while changed:
        changed = False
        for fn in funcs:
            n = fid_name[id(fn)]
            if not is_llm.get(n) and any(is_llm.get(c) for c in local[id(fn)][1]):
                is_llm[n] = True
                changed = True
    llm_names = {n for n, v in is_llm.items() if v}
    func_model = {fid_name[id(fn)]: local[id(fn)][2] for fn in funcs}

    called_by_llm = set()
    for fn in funcs:
        if is_llm.get(fid_name[id(fn)]):
            called_by_llm |= {c for c in local[id(fn)][1] if c != fid_name[id(fn)]}
    entry_llm = {n for n in llm_names if n not in called_by_llm}
    has_loop = {fid_name[id(fn)] for fn in funcs
                if fid_name[id(fn)] in llm_names and _body_has_loop(fn, ad)}
    driver_names = entry_llm | has_loop
    wrapper_ids = {id(fn) for fn in funcs
                   if fid_name[id(fn)] in llm_names and fid_name[id(fn)] not in driver_names}

    # trigger sites
    sites: List[Dict] = []
    for call in (n for n in tsb.walk(root) if ad.is_call(n)):
        nm = ad.call_name(call)
        is_site = ad.is_sdk(call) or (nm in llm_names and nm not in driver_names)
        if not is_site or _inside_ids(call, wrapper_ids):
            continue
        if nm in llm_names and not ad.is_sdk(call):
            name = nm
            model = ad.call_model(call) or func_model.get(nm) or file_model
        else:
            name = f"llm_call_{_line(call)}"
            model = ad.call_model(call) or file_model
        loops = _enclosing_loops(call, ad)
        sites.append({"name": name, "model": model,
                      "outer": loops[-1] if loops else None,
                      "line": _line(call), "col": _col(call)})
    if not sites:
        return [], [], None

    sites.sort(key=lambda s: (s["line"], s["col"]))
    groups: "OrderedDict" = OrderedDict()
    for s in sites:
        key = id(s["outer"]) if s["outer"] is not None else None
        groups.setdefault(key, {"loop": s["outer"], "sites": []})["sites"].append(s)

    used: set = set()

    def uid(base):
        nid, i = base, 2
        while nid in used:
            nid = f"{base}_{i}"
            i += 1
        used.add(nid)
        return nid

    nodes: List[IRNode] = []
    edges: List[IREdge] = []
    entry: Optional[str] = None
    for g in groups.values():
        loop = g["loop"]
        accum = bool(loop is not None and ad.grows_accumulator(loop))
        seen, ordered = set(), []
        for s in g["sites"]:
            if s["name"] in seen:
                continue
            seen.add(s["name"])
            ordered.append(s)
        node_ids = []
        for s in ordered:
            nid = uid(s["name"])
            nodes.append(IRNode(node_id=nid, kind="llm_call",
                                intended_model=s["model"] or file_model,
                                appends_history=accum))
            node_ids.append(nid)
        if entry is None and node_ids:
            entry = node_ids[0]
        for a, b in zip(node_ids, node_ids[1:]):
            edges.append(IREdge(a, b, edge_type="sequence"))
        if node_ids and loop is not None:
            kind = ad.loop_kind(loop)
            make_cycle = kind == "while" or (kind == "for" and ad.grows_accumulator(loop))
            if make_cycle:
                bounded, reason = ad.loop_guard(loop)
                guard = Guard(counter=True, stop_condition=reason) if bounded else None
                edges.append(IREdge(node_ids[-1], node_ids[0], edge_type="loop", guard=guard))
    return nodes, edges, entry


def _local_scan(fn, ad: "LangAdapter"):
    """(has_sdk, called names, model id) from a function's own body, skipping
    nested function/closure scopes."""
    body = ad.func_body(fn) or fn
    has_sdk = False
    called: set = set()

    def visit(node):
        nonlocal has_sdk
        for ch in node.named_children:
            if ch.type in ad.FUNC_TYPES or ch.type in ad.NESTED_FUNC_TYPES:
                continue
            if ad.is_call(ch):
                nm = ad.call_name(ch)
                if nm:
                    called.add(nm)
                if ad.is_sdk(ch):
                    has_sdk = True
            visit(ch)

    visit(body)
    return has_sdk, called, _scan_model(_text(body))


def _body_has_loop(fn, ad: "LangAdapter") -> bool:
    body = ad.func_body(fn) or fn
    found = [False]

    def visit(node):
        for ch in node.named_children:
            if ch.type in ad.FUNC_TYPES or ch.type in ad.NESTED_FUNC_TYPES:
                continue
            if ad.loop_kind(ch):
                found[0] = True
            visit(ch)

    visit(body)
    return found[0]


def _enclosing_loops(node, ad: "LangAdapter"):
    return [a for a in _ancestors(node) if ad.loop_kind(a)]   # innermost first


def _inside_ids(node, ids) -> bool:
    return any(id(a) in ids for a in _ancestors(node))


# --- language adapters --------------------------------------------------------
class LangAdapter:
    FUNC_TYPES: Tuple[str, ...] = ()
    NESTED_FUNC_TYPES: Tuple[str, ...] = ()
    BREAK_TYPES: Tuple[str, ...] = ()
    ACC_MARKERS: Tuple[str, ...] = ()

    def functions(self, root):
        return _descendants(root, *self.FUNC_TYPES)

    def func_name(self, fn) -> Optional[str]:
        n = _field(fn, "name")
        return _text(n) if n else None

    def func_body(self, fn):
        return _field(fn, "body")

    def is_call(self, node) -> bool:
        raise NotImplementedError

    def call_name(self, call) -> Optional[str]:
        raise NotImplementedError

    def is_sdk(self, call) -> bool:
        raise NotImplementedError

    def call_model(self, call) -> Optional[str]:
        return _scan_model(_text(call))

    def loop_kind(self, node) -> Optional[str]:
        raise NotImplementedError

    def loop_body(self, loop):
        return _field(loop, "body")

    def is_infinite(self, loop) -> bool:
        raise NotImplementedError

    def grows_accumulator(self, loop) -> bool:
        body = self.loop_body(loop) or loop
        t = _text(body)
        return any(m in t for m in self.ACC_MARKERS)

    def own_break(self, loop) -> bool:
        body = self.loop_body(loop) or loop
        return self._scan_break(body)

    def _scan_break(self, node) -> bool:
        for ch in node.named_children:
            if ch.type in self.BREAK_TYPES:
                return True
            if self.loop_kind(ch) or ch.type in self.FUNC_TYPES \
                    or ch.type in self.NESTED_FUNC_TYPES:
                continue
            if self._scan_break(ch):
                return True
        return False

    def loop_guard(self, loop) -> Tuple[bool, str]:
        has_break = self.own_break(loop)
        if self.loop_kind(loop) == "while":
            if self.is_infinite(loop):
                return (True, "break") if has_break else (False, "infinite")
            if has_break:
                return (True, "break")
            if self.grows_accumulator(loop):
                return (False, "growing_condition")
            return (True, "condition")
        if has_break:
            return (True, "break")
        return (False, "growing_queue")


class JavaAdapter(LangAdapter):
    FUNC_TYPES = ("method_declaration", "constructor_declaration")
    NESTED_FUNC_TYPES = ("lambda_expression",)
    BREAK_TYPES = ("break_statement", "return_statement")
    ACC_MARKERS = (".add(", ".addAll(", ".put(", ".offer(", ".push(", ".append(", "+=")

    def is_call(self, node):
        return node.type == "method_invocation"

    def call_name(self, call):
        return _java_call_name(call)

    def is_sdk(self, call):
        return _is_java_sdk(call)

    def loop_kind(self, node):
        if node.type in ("while_statement", "do_statement"):
            return "while"
        if node.type in ("for_statement", "enhanced_for_statement"):
            return "for"
        return None

    def is_infinite(self, loop):
        if loop.type == "while_statement":
            return _java_while_true(loop)
        if loop.type == "for_statement":
            return _field(loop, "condition") is None
        return False


class GoAdapter(LangAdapter):
    FUNC_TYPES = ("function_declaration", "method_declaration")
    NESTED_FUNC_TYPES = ("func_literal",)
    BREAK_TYPES = ("break_statement", "return_statement")
    ACC_MARKERS = ("append(", "+=")

    def is_call(self, node):
        return node.type == "call_expression"

    def call_name(self, call):
        return _go_call_name(call)

    def is_sdk(self, call):
        return _is_go_sdk(call)

    def loop_kind(self, node):
        if node.type != "for_statement":
            return None
        if _named_of_type(node, "for_clause", "range_clause"):
            return "for"
        return "while"   # `for {}` or `for cond {}` are while-style

    def is_infinite(self, loop):
        return len(loop.named_children) == 1 and loop.named_children[0].type == "block"


class RubyAdapter(LangAdapter):
    FUNC_TYPES = ("method", "singleton_method")
    NESTED_FUNC_TYPES = ("lambda",)
    BREAK_TYPES = ("break", "return", "next")
    ACC_MARKERS = ("<<", ".push", ".append", ".concat", "+=")
    _CALL_TYPES = ("call", "method_call", "command", "command_call")
    _FOR_METHODS = ("times", "each", "upto", "downto", "step", "each_with_index")

    def is_call(self, node):
        return node.type in self._CALL_TYPES

    def call_name(self, call):
        m = _field(call, "method")
        if m is not None:
            return _text(m)
        ids = _named_of_type(call, "identifier", "constant")
        return _text(ids[-1]) if ids else None

    def is_sdk(self, call):
        return _is_ruby_sdk(call, self.call_name(call))

    def loop_kind(self, node):
        if node.type in ("while", "until"):
            return "while"
        if node.type in self._CALL_TYPES and _ruby_block(node) is not None:
            m = self.call_name(node)
            if m == "loop":
                return "while"
            if m in self._FOR_METHODS:
                return "for"
        return None

    def loop_body(self, loop):
        if loop.type in ("while", "until"):
            return _field(loop, "body") or loop
        return _ruby_block(loop) or loop

    def is_infinite(self, loop):
        if loop.type == "while":
            return _text(_field(loop, "condition") or loop).strip() == "true"
        if loop.type == "until":
            return _text(_field(loop, "condition") or loop).strip() == "false"
        return self.call_name(loop) == "loop"   # loop do


_ADAPTERS = {"java": JavaAdapter, "go": GoAdapter, "ruby": RubyAdapter}


# --- Java SDK / builder primitives --------------------------------------------
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
    return name in ("createChatCompletion", "generateContent", "generateContentStream")


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
            if src:
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
    if i < len(args):
        a = args[i]
        if a.type == "string_literal":
            return _str_value(a)
        if a.type == "identifier":
            return _text(a)
    return None


def _java_conditional_targets(args) -> List[str]:
    out: List[str] = []
    for a in args:
        if a.type == "method_invocation" and _java_call_name(a) == "of":
            inner = _java_arg_nodes(a)
            for idx, node in enumerate(inner):
                if idx % 2 == 1 and node.type == "string_literal":
                    out.append(_str_value(node))
    return [t for t in out if t]


# --- Go / Ruby SDK primitives -------------------------------------------------
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
    return _go_call_name(call) in (
        "CreateChatCompletion", "CreateChatCompletionStream", "CreateCompletion",
        "GenerateContent", "GenerateContentStream", "New")


def _ruby_block(call):
    for c in call.named_children:
        if c.type in ("do_block", "block"):
            return c
    return None


def _is_ruby_sdk(call, name=None) -> bool:
    m = name if name is not None else None
    if m is None:
        mm = _field(call, "method")
        m = _text(mm) if mm is not None else None
    if m == "chat":
        return "parameters" in _text(call)
    if m == "create":
        return "messages" in _text(call)
    return m == "generate_content"


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
