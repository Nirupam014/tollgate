"""Imperative-Python agent parser.

Most real-world agents are not framework graphs — they are hand-rolled Python: a
``while``/``for`` loop that calls an LLM SDK (directly, or through a helper
function) once per turn, growing a task list or message history. BabyAGI, the
original AutoGPT script, and countless internal agents are written this way, and
they are exactly the unbounded-loop risk Tollgate exists to flag.

This parser statically recovers that structure with the ``ast`` module:

  1. Find user functions whose body issues an LLM SDK call ("LLM functions").
  2. Walk the executable code and collect *trigger sites* — a direct SDK call, or
     a call to an LLM function — in source order.
  3. Group sites by the loop that encloses them and emit an IR node per site.
     For a ``while`` loop (or a ``for`` loop that grows its own queue) we close
     the cycle with a ``loop`` edge whose guard reflects whether the loop can
     actually terminate (a ``break``, or a non-constant ``while`` condition).

A ``while True:`` with no ``break`` becomes an unguarded cycle, which the
recursive-loop detector reports as a critical, unbounded-cost finding. If the
file has no recoverable LLM activity the parser returns an empty workflow, which
the pipeline drops (honest failure) rather than scoring a misleading PASS.
"""
from __future__ import annotations

import ast
import hashlib
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from ..catalog import ModelCatalog
from ..ir import Guard, IREdge, IRNode, Workflow
from .autogpt import _normalize_model
from .langgraph import _MODEL_HINTS as MODEL_HINTS

# Textual markers of an LLM SDK invocation. Used both by discovery (cheap content
# sniff) and as the AST match below. Kept deliberately call-shaped (a dotted path
# ending in the SDK's request method) so a bare ``import openai`` or a mention in
# a comment does not by itself mark a file as an agent.
#
# Breadth note: the OpenAI request shapes below are not "just OpenAI". Every
# OpenAI-compatible provider exposes the identical surface, so a single marker
# covers a long tail of vendors and local servers: Azure OpenAI, Groq, Together,
# DeepSeek, Fireworks, OpenRouter, xAI (Grok), Perplexity, Mistral (compat
# endpoint), Nvidia NIM, Anyscale, Databricks, vLLM, LM Studio, llama.cpp server,
# and Ollama's /v1 endpoint. The non-OpenAI-shaped SDKs are enumerated explicitly.
SDK_CALL_MARKERS = (
    # OpenAI + all OpenAI-compatible providers (see breadth note above).
    "chat.completions.create",   # openai >=1.x and every compatible vendor
    "chat.completions.parse",    # openai structured-output helper
    "completions.create",        # azure / legacy completions
    "ChatCompletion.create",     # openai <1.x
    "responses.create",          # openai responses API
    "responses.parse",           # openai responses structured output
    # Anthropic
    "messages.create",           # anthropic
    "messages.stream",           # anthropic streaming
    # Google Gemini / Vertex AI
    "generate_content",          # google genai / gemini / vertex
    "generate_content_async",
    # Mistral (native SDK)
    "chat.complete",             # mistralai >=1.x
    "chat.complete_async",
    "chat.stream",
    # Amazon Bedrock (boto3 runtime client)
    "converse",                  # bedrock converse API
    "converse_stream",
    "invoke_model",
    "invoke_model_with_response_stream",
    # Cohere
    "chat_stream",               # cohere streaming (co.chat is too generic to match)
    # Replicate
    "replicate.run",
    "replicate.async_run",
    # Ollama (native python client)
    "ollama.chat",
    "ollama.generate",
    # Hugging Face Inference API / huggingface_hub InferenceClient
    "chat_completion",
    "text_generation",
    # LiteLLM (proxies 100+ providers behind one call)
    "litellm.completion",
    "litellm.acompletion",
)

# AST: the final attribute(s) of the call's dotted func path that confirm an SDK
# call. Mirrors SDK_CALL_MARKERS; matched as an exact dotted tail (``a.b.c`` or
# ``....a.b.c``) so generic single verbs do not over-trigger.
_SDK_TAILS = (
    "chat.completions.create",
    "chat.completions.parse",
    "completions.create",
    "ChatCompletion.create",
    "responses.create",
    "responses.parse",
    "messages.create",
    "messages.stream",
    "generate_content",
    "generate_content_async",
    "chat.complete",
    "chat.complete_async",
    "chat.stream",
    "converse",
    "converse_stream",
    "invoke_model",
    "invoke_model_with_response_stream",
    "chat_stream",
    "replicate.run",
    "replicate.async_run",
    "ollama.chat",
    "ollama.generate",
    "chat_completion",
    "text_generation",
    "litellm.completion",
    "litellm.acompletion",
)

# Keyword names that carry the model id across SDKs. Bedrock uses camelCase
# ``modelId``; most others use ``model``; Azure deployments use ``deployment_id``.
_MODEL_KWARGS = ("model", "model_name", "model_id", "modelId", "deployment_id")
_ACCUMULATOR_METHODS = ("append", "extend", "insert", "add", "put", "appendleft")


def has_imperative_llm(text: str) -> bool:
    """True if the source text contains an LLM SDK call shape (discovery sniff)."""
    return any(m in text for m in SDK_CALL_MARKERS)


# --- AST helpers --------------------------------------------------------------
def _dotted(func: ast.AST) -> str:
    """Render a call's func as a dotted path, e.g. 'client.chat.completions.create'."""
    parts: List[str] = []
    node = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_sdk_call(call: ast.Call) -> bool:
    d = _dotted(call.func)
    return any(d == t or d.endswith("." + t) for t in _SDK_TAILS)


def _called_name(call: ast.Call) -> Optional[str]:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _local_scan(fn: ast.AST) -> Tuple[bool, set, Optional[str]]:
    """Scan a function's *own* body (not nested defs): does it call an SDK
    directly, what functions does it call, and what model id appears locally?"""
    has_sdk = False
    called: set = set()
    model: Optional[str] = None

    def visit(node: ast.AST) -> None:
        nonlocal has_sdk, model
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # nested def is its own function
            if isinstance(child, ast.Call):
                nm = _called_name(child)
                if nm:
                    called.add(nm)
                if _is_sdk_call(child):
                    has_sdk = True
                # Capture a model id from the SDK call or from a model= passed to a
                # wrapper (agents commonly do `openai_call(prompt, model="gpt-4")`).
                if model is None:
                    model = _call_model(child)
            visit(child)

    visit(fn)
    return has_sdk, called, model


def _norm_model(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    # Reuse the AutoGPT model normalizer: it maps real SDK ids (gpt-4-turbo,
    # claude-3-opus-..., gemini-1.5-flash, perplexity/sonar-pro, ...) onto catalog
    # ids, so costing and cheaper-model substitution actually fire. Fall back to a
    # hint substring, then the raw id (catalog lookup uses default_model if unknown).
    norm = _normalize_model(raw)
    if norm and ModelCatalog.load().get(norm):
        return norm
    for hint in MODEL_HINTS:
        if hint in raw:
            return hint
    return norm or raw


def _call_model(call: ast.Call) -> Optional[str]:
    for kw in call.keywords:
        if kw.arg in _MODEL_KWARGS and isinstance(kw.value, ast.Constant) \
                and isinstance(kw.value.value, str):
            return _norm_model(kw.value.value)
    return None


def _build_parents(tree: ast.AST) -> Dict[int, ast.AST]:
    parents: Dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _enclosing_loops(node: ast.AST, parents: Dict[int, ast.AST]) -> List[ast.AST]:
    """Loops lexically enclosing ``node``, innermost first."""
    loops: List[ast.AST] = []
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, (ast.While, ast.For, ast.AsyncFor)):
            loops.append(cur)
        cur = parents.get(id(cur))
    return loops


def _inside_any(node: ast.AST, parents: Dict[int, ast.AST], targets: set) -> bool:
    cur = parents.get(id(node))
    while cur is not None:
        if id(cur) in targets:
            return True
        cur = parents.get(id(cur))
    return False


def _contains_own_break(loop: ast.AST) -> bool:
    """A ``break`` that belongs to this loop (not to a nested loop or function)."""
    def visit(node: ast.AST) -> bool:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Break):
                return True
            if isinstance(child, (ast.For, ast.While, ast.AsyncFor,
                                  ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # a break in there does not exit *this* loop
            if visit(child):
                return True
        return False
    return any(visit(stmt) or isinstance(stmt, ast.Break) for stmt in loop.body)


def _own_body_has_loop(fn: ast.AST) -> bool:
    """True if the function's own body contains a loop (not in a nested def)."""
    found = False

    def visit(node: ast.AST) -> None:
        nonlocal found
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(child, (ast.While, ast.For, ast.AsyncFor)):
                found = True
            visit(child)

    for stmt in getattr(fn, "body", []):
        visit(stmt)
    return found


def _grows_accumulator(loop: ast.AST) -> bool:
    """The loop body grows a list/queue (append/extend/insert or ``+=``)."""
    for node in ast.walk(loop):
        if isinstance(node, ast.Call):
            if _called_name(node) in _ACCUMULATOR_METHODS:
                return True
        if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
            return True
    return False


def _while_is_true(loop: ast.While) -> bool:
    t = loop.test
    return isinstance(t, ast.Constant) and bool(t.value)


def _loop_guard(loop: ast.AST) -> Tuple[bool, str]:
    """Return (is_bounded, reason) for a cyclic loop."""
    has_break = _contains_own_break(loop)
    if isinstance(loop, ast.While):
        if _while_is_true(loop):
            return (True, "break") if has_break else (False, "infinite")
        if has_break:
            return (True, "break")
        if _grows_accumulator(loop):
            return (False, "growing_condition")
        return (True, "condition")
    # for / async-for that reaches here only because it grows its own iterable
    if has_break:
        return (True, "break")
    return (False, "growing_queue")


# --- parser -------------------------------------------------------------------
def parse_imperative(path: str) -> Workflow:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        source = fh.read()

    wf = Workflow(
        workflow_id=os.path.splitext(os.path.basename(path))[0],
        source_kind="imperative",
        nodes=[], edges=[], source_path=path,
    )
    wf.content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        # SyntaxError: not valid Python. ValueError: source has null bytes.
        # RecursionError: pathologically deep nesting. All -> empty -> dropped.
        return wf  # honest failure, never a crash on a CI scan

    parents = _build_parents(tree)
    file_model = _scan_hint(source)

    # 1. LLM functions. A function is "LLM" if its own body issues an SDK call,
    #    OR it calls another LLM function — propagated transitively, so a chain of
    #    thin wrappers (agent -> openai_call -> SDK) is fully recovered.
    defs = [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    local: Dict[str, Tuple[bool, set, Optional[str]]] = {}
    for fn in defs:
        local[fn.name] = _local_scan(fn)

    is_llm: Dict[str, bool] = {name: hs for name, (hs, _c, _m) in local.items()}
    changed = True
    while changed:
        changed = False
        for name, (_hs, calls, _m) in local.items():
            if not is_llm.get(name) and any(is_llm.get(c) for c in calls):
                is_llm[name] = True
                changed = True
    llm_func_names = {n for n, v in is_llm.items() if v}
    func_model = {name: m for name, (_hs, _c, m) in local.items()}

    # Drivers vs wrappers. A *driver* is an LLM function we descend into: either an
    # entry point (no other LLM function calls it) or one that owns a loop. A
    # *wrapper* is an LLM function invoked from a driver (agent -> openai_call ->
    # SDK); we site the *call* to it and exclude its body so the SDK call inside is
    # not double-counted. This keeps the loop that lives inside a driver (e.g. a
    # ``while True`` in ``main``) visible instead of swallowing its call sites.
    called_by_llm: set = set()
    for name, (_hs, calls, _m) in local.items():
        if is_llm.get(name):
            called_by_llm |= {c for c in calls if c != name}
    entry_llm = {n for n in llm_func_names if n not in called_by_llm}
    has_loop = {fn.name for fn in defs
                if fn.name in llm_func_names and _own_body_has_loop(fn)}
    driver_names = entry_llm | has_loop
    wrapper_ids = {id(fn) for fn in defs
                   if fn.name in llm_func_names and fn.name not in driver_names}

    # 2. Trigger sites: a direct SDK call, or a call to a *wrapper* LLM function —
    #    taken outside any wrapper body (which is the wrapper's own implementation).
    sites: List[Dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        nm = _called_name(node)
        is_site = _is_sdk_call(node) or (nm in llm_func_names and nm not in driver_names)
        if not is_site or _inside_any(node, parents, wrapper_ids):
            continue
        if nm in llm_func_names:
            name = nm
            model = _call_model(node) or func_model.get(nm) or file_model
        else:
            name = f"llm_call_{getattr(node, 'lineno', 0)}"
            model = _call_model(node) or file_model
        sites.append({"name": name, "model": model, "node": node})

    if not sites:
        return wf  # no recoverable LLM activity
    for s in sites:
        loops = _enclosing_loops(s["node"], parents)
        s["outer"] = loops[-1] if loops else None
        s["line"] = getattr(s["node"], "lineno", 0)
        s["col"] = getattr(s["node"], "col_offset", 0)

    sites.sort(key=lambda s: (s["line"], s["col"]))

    # 3. Group by enclosing outermost loop (None = straight-line code).
    groups: "OrderedDict[Optional[int], Dict]" = OrderedDict()
    for s in sites:
        key = id(s["outer"]) if s["outer"] is not None else None
        groups.setdefault(key, {"loop": s["outer"], "sites": []})["sites"].append(s)

    used: set = set()

    def uid(base: str) -> str:
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
        accum = bool(loop is not None and _grows_accumulator(loop))
        # Distinct callees in first-seen order within this group.
        seen, ordered = set(), []
        for s in g["sites"]:
            if s["name"] in seen:
                continue
            seen.add(s["name"])
            ordered.append(s)

        node_ids: List[str] = []
        for s in ordered:
            nid = uid(s["name"])
            nodes.append(IRNode(
                node_id=nid, kind="llm_call",
                intended_model=s["model"] or file_model,
                appends_history=accum,
            ))
            node_ids.append(nid)
        if entry is None and node_ids:
            entry = node_ids[0]

        for a, b in zip(node_ids, node_ids[1:]):
            edges.append(IREdge(a, b, edge_type="sequence"))

        # Close a cycle only for genuine recursion shapes: any while-loop, or a
        # for-loop that grows its own queue. Plain bounded for-loops stay acyclic.
        make_cycle = node_ids and (
            isinstance(loop, ast.While)
            or (isinstance(loop, (ast.For, ast.AsyncFor)) and _grows_accumulator(loop))
        )
        if make_cycle:
            bounded, reason = _loop_guard(loop)
            guard = Guard(counter=True, stop_condition=reason) if bounded else None
            edges.append(IREdge(node_ids[-1], node_ids[0], edge_type="loop", guard=guard))

    wf.nodes = nodes
    wf.edges = edges
    wf.entry = entry
    return wf


def _scan_hint(source: str) -> Optional[str]:
    for hint in MODEL_HINTS:
        if hint in source:
            return hint
    return None
