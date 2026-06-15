"""JavaScript / TypeScript agent parser → Workflow IR.

This brings the *same* graph analysis Tollgate runs on Python (cycle detection,
context-explosion, fan-out, token prediction, cost/scoring) to the JS/TS agent
ecosystem — not just the advisory textual lint. It recovers two real, common
shapes deterministically with the standard library (no tree-sitter, no Node):

  1. LangGraph.js / `StateGraph` builders — the JS analog of the Python LangGraph
     parser. Explicit, declarative, string-named edges:

         const g = new StateGraph(...)
           .addNode("agent", callModel)
           .addNode("tools", toolNode)
           .addEdge(START, "agent")
           .addConditionalEdges("agent", shouldContinue, { tools: "tools", end: END })
           .addEdge("tools", "agent");          // back-edge → cycle

  2. Hand-rolled imperative agents — an infinite loop (`while (true)`, `for (;;)`)
     whose body issues an LLM SDK call. This is the JS twin of the imperative
     Python parser's unbounded-loop case.

It is deliberately conservative and honest: it only emits a workflow when it can
actually recover named nodes/edges (LangGraph) or a concrete infinite loop around
a real SDK call (imperative). Anything it cannot resolve yields an empty workflow,
which the pipeline drops (honest failure) so the file still reaches the advisory
textual lint instead of being scored as a misleading PASS.

Recovery limits (by design, not by bug): only explicit `StateGraph` builder calls
and clearly-infinite loops are recovered; arbitrary control flow is not reverse
engineered with regex. Broader imperative recovery (bounded loops, arbitrary call
graphs) is the job of a future tree-sitter backend; until then those files get the
textual lint, which never claims a graph it didn't recover.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import List, Optional, Tuple

from ..ir import Guard, IREdge, IRNode, Workflow
from .langgraph import _MODEL_HINTS as MODEL_HINTS

JS_EXTS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}

# LangGraph.js / StateGraph builder surface.
_LANGGRAPH_JS_MARKERS = ("StateGraph", "MessageGraph", "addNode", "addEdge",
                         "addConditionalEdges", "setEntryPoint", "setFinishPoint")
_BUILDER_METHODS = ("addNode", "addEdge", "addConditionalEdges",
                    "setEntryPoint", "setFinishPoint")

# LLM SDK call shapes seen in JS/TS. Includes the OpenAI-compatible surface (also
# used by Azure/Groq/Together/etc.), Anthropic, Google, plus the Vercel AI SDK and
# LangChain JS helpers. Kept call-shaped so a bare import doesn't trigger.
_JS_SDK_MARKERS = (
    "chat.completions.create", "chat.completions.parse", "completions.create",
    "responses.create", "responses.parse",
    "messages.create", "messages.stream",
    "generateContent", "generateContentStream",
    "generateText", "streamText", "generateObject", "streamObject",  # Vercel AI SDK
    "createMessage",
    ".invoke(", ".stream(", ".batch(",                               # LangChain JS runnables
)

_SENTINELS = {"__start__", "__end__", "START", "END", "start", "end"}

_HISTORY_MARKERS = ("addMessages", "state.messages", ".messages.push", "...messages",
                    "messages.concat", "[...state.messages", "...state.messages",
                    "history.push", "...history")

_STR_LITERAL_RE = re.compile(r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'|`([^`$\\]*)`')
_INF_LOOP_RE = re.compile(r'(?:\bwhile\s*\(\s*(?:true|1)\s*\)|\bfor\s*\(\s*;\s*;\s*\))')


# ---------------------------------------------------------------------------
# Comment / string blanking. Produces a same-length copy of the source with the
# *contents* of comments and string/template literals replaced by spaces, so
# bracket matching and call-site scanning operate on real code structure only.
# Literal *values* are read from the original source within a known span.
# ---------------------------------------------------------------------------
def _blank_js(src: str) -> str:
    out = list(src)
    i, n, state = 0, len(src), "code"
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                out[i] = out[i + 1] = " "; i += 2; state = "line"; continue
            if c == "/" and nxt == "*":
                out[i] = out[i + 1] = " "; i += 2; state = "block"; continue
            if c in "'\"`":
                state = {"'": "sq", '"': "dq", "`": "tpl"}[c]; i += 1; continue
            i += 1; continue
        if state == "line":
            if c == "\n":
                state = "code"
            else:
                out[i] = " "
            i += 1; continue
        if state == "block":
            if c == "*" and nxt == "/":
                out[i] = out[i + 1] = " "; i += 2; state = "code"; continue
            if c != "\n":
                out[i] = " "
            i += 1; continue
        # in a string/template literal
        q = {"sq": "'", "dq": '"', "tpl": "`"}[state]
        if c == "\\":
            out[i] = " "
            if i + 1 < n and src[i + 1] != "\n":
                out[i + 1] = " "
            i += 2; continue
        if c == q:
            state = "code"; i += 1; continue
        if c != "\n":
            out[i] = " "
        i += 1
    return "".join(out)


def _match_bracket(blanked: str, open_idx: int) -> int:
    """Index of the bracket matching the one at ``open_idx`` (paren or brace),
    or -1. Operates on blanked source so brackets inside strings are ignored."""
    pairs = {"(": ")", "{": "}", "[": "]"}
    opener = blanked[open_idx]
    closer = pairs[opener]
    depth = 0
    for i in range(open_idx, len(blanked)):
        ch = blanked[i]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _string_literals(span: str) -> List[str]:
    """All simple string/template-literal values in a source span, in order."""
    out: List[str] = []
    for m in _STR_LITERAL_RE.finditer(span):
        val = m.group(1) if m.group(1) is not None else (
            m.group(2) if m.group(2) is not None else m.group(3))
        if val is not None:
            out.append(val)
    return out


def looks_like_langgraph_js(text: str) -> bool:
    has_builder = any(m in text for m in ("addNode", "addEdge", "addConditionalEdges"))
    return has_builder and ("StateGraph" in text or "MessageGraph" in text
                            or "addNode" in text)


def has_imperative_llm_js(text: str) -> bool:
    """An infinite loop AND an SDK call somewhere in the file (discovery sniff)."""
    return bool(_INF_LOOP_RE.search(text)) and any(m in text for m in _JS_SDK_MARKERS)


def is_js_workflow_source(text: str) -> bool:
    return looks_like_langgraph_js(text) or has_imperative_llm_js(text)


def _scan_model(source: str) -> Optional[str]:
    for hint in MODEL_HINTS:
        if hint in source:
            return hint
    return None


def _word_in(blanked_span: str, word: str) -> bool:
    return re.search(r"\b" + re.escape(word) + r"\b", blanked_span) is not None


# ---------------------------------------------------------------------------
def parse_javascript(path: str) -> Workflow:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        source = fh.read()

    wf = Workflow(
        workflow_id=os.path.splitext(os.path.basename(path))[0],
        source_kind="javascript", nodes=[], edges=[], source_path=path,
    )
    wf.content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    blanked = _blank_js(source)
    model = _scan_model(source)
    appends_history = any(m in source for m in _HISTORY_MARKERS)

    nodes, edges, entry, kind = _parse_langgraph_js(source, blanked, model, appends_history)
    if not nodes:
        nodes, edges, entry, kind = _parse_imperative_js(source, blanked, model)
    if not nodes:
        return wf  # honest failure → routed to the textual lint

    wf.nodes, wf.edges, wf.entry, wf.source_kind = nodes, edges, entry, kind
    return wf


# --- LangGraph.js / StateGraph -------------------------------------------------
def _parse_langgraph_js(source: str, blanked: str, model: Optional[str],
                        appends_history: bool):
    node_ids: List[str] = []
    raw_edges: List[Tuple[str, str, str]] = []   # (from, to, edge_type)
    entry: Optional[str] = None

    for m in re.finditer(r"\.(" + "|".join(_BUILDER_METHODS) + r")\s*\(", blanked):
        method = m.group(1)
        open_paren = m.end() - 1
        close = _match_bracket(blanked, open_paren)
        if close == -1:
            continue
        span = source[open_paren + 1:close]
        lits = _string_literals(span)

        if method == "addNode":
            if lits and lits[0] not in node_ids:
                node_ids.append(lits[0])
        elif method == "setEntryPoint":
            if lits:
                entry = lits[0]
        elif method == "setFinishPoint":
            pass
        elif method == "addEdge":
            ends = _edge_endpoints(span, blanked, open_paren, close)
            if len(ends) >= 2:
                raw_edges.append((ends[0], ends[1], "sequence"))
        elif method == "addConditionalEdges":
            if not lits:
                continue
            src = lits[0]
            targets = _conditional_targets(source, blanked, open_paren, close)
            for t in targets:
                raw_edges.append((src, t, "conditional"))

    # Resolve sentinels and ensure edge endpoints exist as nodes.
    def real(n: str) -> Optional[str]:
        return None if n in _SENTINELS else n

    for f, t, _et in raw_edges:
        for nid in (real(f), real(t)):
            if nid and nid not in node_ids:
                node_ids.append(nid)

    if not node_ids:
        return [], [], None, "javascript"

    order = {nid: i for i, nid in enumerate(node_ids)}
    edges: List[IREdge] = []
    for f, t, et in raw_edges:
        rf, rt = real(f), real(t)
        if not rf or not rt:
            continue  # edge touching START/END: bounds the graph, not a real edge
        # Back-edge (target declared at/before source) → loop with unknown guard.
        if et == "sequence" and order.get(rt, 1 << 30) <= order.get(rf, -1):
            et = "loop"
        elif et == "conditional" and order.get(rt, 1 << 30) <= order.get(rf, -1):
            et = "loop"
        edges.append(IREdge(rf, rt, edge_type=et))

    nodes = [IRNode(node_id=nid, kind="llm_call", intended_model=model,
                    appends_history=appends_history) for nid in node_ids]
    kind = "langgraph-js"
    if not entry:
        targets = {e.to_node for e in edges}
        for nid in node_ids:
            if nid not in targets:
                entry = nid
                break
    return nodes, edges, entry, kind


def _edge_endpoints(span: str, blanked: str, open_paren: int, close: int) -> List[str]:
    """addEdge endpoints: string literals, with bare START/END identifiers mapped
    to their sentinel names so they're recognised and dropped as graph bounds."""
    ends = _string_literals(span)
    if len(ends) >= 2:
        return ends[:2]
    # Mixed form like addEdge(START, "agent") — recover bare START/END identifiers
    # positionally from the blanked arg list.
    arglist = blanked[open_paren + 1:close]
    parts = [p.strip() for p in arglist.split(",")]
    out: List[str] = []
    si = 0
    for p in parts:
        if p in ("START", "END"):
            out.append(p)
        elif '"' in p or "'" in p or "`" in p:
            if si < len(ends):
                out.append(ends[si]); si += 1
    return out[:2]


def _conditional_targets(source: str, blanked: str, open_paren: int, close: int) -> List[str]:
    """Targets of addConditionalEdges: the string values of the path-map object
    literal (or array) argument; plus any string literals after the source."""
    targets: List[str] = []
    # Object literal { key: "target", ... } — take its string-literal values.
    brace = blanked.find("{", open_paren + 1, close)
    if brace != -1:
        bclose = _match_bracket(blanked, brace)
        if bclose != -1:
            obj = source[brace + 1:bclose]
            # values appear after a ':'
            for piece in re.split(r",(?![^{\[]*[}\]])", obj):
                if ":" in piece:
                    val = piece.split(":", 1)[1]
                    lits = _string_literals(val)
                    if lits:
                        targets.extend(lits)
                    elif "END" in val:
                        targets.append("END")
            return targets
    arr = blanked.find("[", open_paren + 1, close)
    if arr != -1:
        aclose = _match_bracket(blanked, arr)
        if aclose != -1:
            targets.extend(_string_literals(source[arr + 1:aclose]))
            return targets
    # Fallback: any string literals after the first (the source) in the call.
    lits = _string_literals(source[open_paren + 1:close])
    return lits[1:]


# --- imperative infinite loop around an SDK call ------------------------------
def _parse_imperative_js(source: str, blanked: str, model: Optional[str]):
    nodes: List[IRNode] = []
    edges: List[IREdge] = []
    entry: Optional[str] = None
    used = set()

    for m in _INF_LOOP_RE.finditer(blanked):
        brace = blanked.find("{", m.end())
        if brace == -1 or brace - m.end() > 80:   # must be the loop's own block
            continue
        bclose = _match_bracket(blanked, brace)
        if bclose == -1:
            continue
        body = blanked[brace + 1:bclose]
        if not any(mk in body for mk in _JS_SDK_MARKERS):
            continue
        line = source.count("\n", 0, m.start()) + 1
        nid = f"llm_call_{line}"
        while nid in used:
            nid += "_b"
        used.add(nid)
        grows = any(h in source[brace:bclose + 1] for h in _HISTORY_MARKERS)
        nodes.append(IRNode(node_id=nid, kind="llm_call", intended_model=model,
                            appends_history=grows))
        # Bounded only if the loop can actually leave: break / return / throw.
        bounded = (_word_in(body, "break") or _word_in(body, "return")
                   or _word_in(body, "throw"))
        guard = Guard(counter=True, stop_condition="break") if bounded else None
        edges.append(IREdge(nid, nid, edge_type="loop", guard=guard))
        if entry is None:
            entry = nid

    return nodes, edges, entry, "javascript"
