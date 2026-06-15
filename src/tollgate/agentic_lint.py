"""Agentic source linter — the strict "last gate before production" pass.

Where the IR detectors reason about a *recovered graph*, this linter reasons
directly about the *source* with the AST. Its job is to be an exhaustive, strict
code reviewer for the failure modes that are specific to **agentic** code — and
nothing else. It never flags generic code smells; every check is gated on a
positive agentic signal (a known agent framework import, or a recognized LLM SDK
call). If a file shows no agentic signal, the linter is silent.

It complements (does not replace) the graph detectors:
  * graph detectors  -> proven structural risk on a parsed workflow (cycles,
                        context explosion, fan-out) with cost projection.
  * this linter      -> config-absence and source-shape risks that don't need a
                        full graph, so it also reaches the *many* frameworks we
                        can recognize but not (yet) parse into a graph. Findings
                        are STRUCTURAL only — no token/cost numbers are invented.

Checks (all deterministic, all agentic-gated):
  1. unbounded_loop        `while True:` (or equivalent) whose body issues an LLM
                           call with no `break`/`return` and no iteration cap ->
                           genuinely unbounded -> critical.
  2. missing_iteration_cap a known agent constructor/runner invoked without its
                           termination cap (LangChain max_iterations, AutoGen
                           max_round / max_consecutive_auto_reply, LlamaIndex
                           max_iterations, smolagents max_steps, CrewAI max_iter,
                           LangGraph recursion_limit). These frameworks have
                           defaults, so this is a WARN ("relying on the default
                           bound; set an explicit one"), not a hard block.
  3. uncapped_output       a recognized LLM SDK call with no max_tokens /
                           max_output_tokens / max_completion_tokens /
                           max_new_tokens -> unbounded generation per call -> warn.
  4. fanout (unbounded)    parallel agent/LLM fan-out (asyncio.gather over a
                           comprehension, or a comprehension of LLM calls) with no
                           concurrency bound (no Semaphore / no slice) -> warn.

Strictness (set by config): "strict" surfaces all of the above and escalates any
present lint finding to at least WARN (an unbounded_loop or any critical -> BLOCK).
"balanced" keeps findings visible but only escalates the genuinely-unbounded ones.
"off" disables the linter.
"""
from __future__ import annotations

import ast
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from .findings import Finding
from .parsers.imperative import SDK_CALL_MARKERS

# --- framework registry ------------------------------------------------------
# Each agent framework: import substrings that mark its presence, and the agent
# constructors / runners whose *absence of a cap kwarg* is a missing-cap finding.
_FRAMEWORKS: Dict[str, dict] = {
    "langchain": {
        "imports": ("langchain",),
        "agents": {
            "AgentExecutor": ("max_iterations", "max_execution_time"),
            "initialize_agent": ("max_iterations", "max_execution_time"),
        },
    },
    "autogen": {
        "imports": ("autogen", "ag2", "pyautogen", "autogen_agentchat"),
        "agents": {
            # Classic autogen: the loop lives in the group chat / initiate_chat.
            "GroupChat": ("max_round",),
            # Modern autogen_agentchat: teams cap with max_turns or a
            # termination_condition. (Per-agent AssistantAgent is intentionally
            # NOT flagged: it has no such kwarg in the new API, so flagging it
            # would be a false positive.)
            "RoundRobinGroupChat": ("max_turns", "termination_condition"),
            "SelectorGroupChat": ("max_turns", "termination_condition"),
            "Swarm": ("max_turns", "termination_condition"),
        },
        "runners": {"initiate_chat": ("max_turns",)},
    },
    "llama_index": {
        "imports": ("llama_index", "llama-index", "llamaindex"),
        "agents": {
            "ReActAgent": ("max_iterations",),
            "FunctionAgent": ("max_iterations",),
            "AgentRunner": ("max_iterations",),
            "OpenAIAgent": ("max_function_calls",),
        },
    },
    "smolagents": {
        "imports": ("smolagents",),
        "agents": {
            "CodeAgent": ("max_steps",),
            "ToolCallingAgent": ("max_steps",),
        },
    },
    "crewai": {
        "imports": ("crewai",),
        # Crew-level structure is recovered by parsers/crewai.py; here we add the
        # per-Agent internal-loop cap (max_iter) check.
        "agents": {"Agent": ("max_iter",)},
    },
    "langgraph": {
        "imports": ("langgraph",),
        # recursion_limit lives in the .invoke/.stream config dict; checked
        # textually below rather than as a constructor kwarg.
        "agents": {},
    },
}

# Token-cap kwargs that make an LLM generation bounded.
_OUTPUT_CAP_KWARGS = {
    "max_tokens", "max_output_tokens", "max_completion_tokens",
    "max_new_tokens", "maxOutputTokens", "maxTokens", "max_tokens_to_sample",
    # Go struct fields / Java builder setters use Pascal/camelCase.
    "MaxTokens", "MaxCompletionTokens", "MaxOutputTokens",
    "maxCompletionTokens", "maxOutputTokens",
}

# LangChain / LlamaIndex chat-model *constructors*. Code using these frameworks
# calls the model through a wrapper (`ChatOpenAI(...).invoke(...)`), not a vendor
# SDK call — `.invoke()` is far too generic to match safely, so we check the
# constructor for an output cap instead. A chat model built with no max_tokens is
# an uncapped-generation risk regardless of how it is later invoked.
_MODEL_CTORS = {
    "ChatOpenAI", "AzureChatOpenAI", "ChatAnthropic", "ChatAnthropicMessages",
    "ChatGoogleGenerativeAI", "ChatVertexAI", "ChatBedrock", "ChatBedrockConverse",
    "ChatGroq", "ChatMistralAI", "ChatCohere", "ChatFireworks", "ChatOllama",
    "ChatLiteLLM", "ChatTogether", "ChatDeepSeek", "ChatXAI", "ChatPerplexity",
    "ChatNVIDIA", "init_chat_model",
}

# Severities (display + scoring). Gate escalation is separate (see lint_gate).
_SEV = {
    "unbounded_loop": "critical",
    "missing_iteration_cap": "high",
    "uncapped_output": "medium",
    "fanout": "high",
}

_GATE_ORDER = {"pass": 0, "warn": 1, "block": 2}


# --- AST helpers -------------------------------------------------------------

def _dotted(func) -> str:
    """Return the dotted attribute path of a call target, e.g. client.chat.create."""
    parts: List[str] = []
    node = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _call_tail_name(func) -> Optional[str]:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_llm_call(call: ast.Call) -> bool:
    dotted = _dotted(call.func)
    return any(dotted == m or dotted.endswith("." + m) for m in SDK_CALL_MARKERS)


def _kwnames(call: ast.Call) -> Set[str]:
    return {kw.arg for kw in call.keywords if kw.arg}


def _detect_frameworks(source: str) -> Set[str]:
    low = source.lower()
    present = set()
    for fw, spec in _FRAMEWORKS.items():
        if any(imp in low for imp in spec["imports"]):
            present.add(fw)
    return present


# --- the linter --------------------------------------------------------------

def lint_source(path: str, strictness: str = "strict",
                source: Optional[str] = None) -> List[Finding]:
    """Return agentic findings for a source file in any language. Empty if not agentic.

    Python is parsed with the AST for high-fidelity checks. For every other
    language we fall back to a deterministic **textual** pass (regex) that flags
    the two highest-value, language-universal risks — an infinite loop wrapping an
    LLM call, and an LLM call with no output-token cap. The textual pass is
    explicitly advisory/lower-fidelity; it never claims a recovered graph.

    `strictness` only affects *which severities exist*; the gate escalation lives
    in `lint_gate`. Both are pure functions of the findings, so results are
    deterministic.
    """
    if strictness == "off":
        return []
    if source is None:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                source = fh.read()
        except OSError:
            return []

    frameworks = _detect_frameworks(source)
    # Agentic signal: a known framework, a Python/JS dotted SDK tail, or any of the
    # cross-language SDK call shapes (Go / Java / Ruby) the textual pass recognizes.
    has_sdk = (any(m in source for m in SDK_CALL_MARKERS)
               or (not path.endswith(".py") and _has_textual_sdk(source)))
    if not frameworks and not has_sdk:
        return []  # no agentic signal -> stay silent

    out: List[Finding] = []
    if path.endswith(".py"):
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError):
            tree = None
        if tree is not None:
            out += _check_unbounded_loops(tree, path)
            out += _check_missing_caps(tree, path, frameworks, source)
            out += _check_uncapped_output(tree, path)
            out += _check_model_output_cap(tree, path)
            out += _check_fanout(tree, path, source)
        else:
            out += _textual_checks(source, path)
    else:
        # Any non-Python language: deterministic regex pass.
        out += _textual_checks(source, path)
    # Stable order + ids for deterministic output.
    for i, f in enumerate(out):
        f.finding_id = f"lint_{i + 1}"
    return out


# --- language-agnostic textual checks (non-Python) ---------------------------
# An infinite-loop header across C-family / Python / Go / Rust / Ruby syntax. We
# only flag *infinite* loops; a bounded `for x of items` / `N.times` is not one.
#   while(true) / while true   C-family, Python, Ruby (`while true`)
#   for(;;)                     C / Java / JS
#   for {                       Go
#   loop {                      Rust
#   loop do / until false       Ruby
_INF_LOOP_RE = re.compile(
    r'(?im)(?:\bwhile\s*\(\s*(?:true|1)\s*\)|\bwhile\s+true\b|'
    r'\bfor\s*\(\s*;\s*;\s*\)|\bfor\s*\{|\bloop\s*\{|'
    r'\bloop\s+do\b|\buntil\s+false\b)')

# Recognized LLM SDK call shapes across languages. The Python/JS dotted tails come
# from SDK_CALL_MARKERS; the rest cover the idiomatic Go / Java / Ruby client
# calls those markers miss. Longer alternatives are listed first so the regex
# prefers the most specific match at a position. Each is matched followed by `(`.
_TEXTUAL_SDK_FRAGMENTS = [re.escape(m) for m in SDK_CALL_MARKERS] + [
    # Go — go-openai, Google generative-ai-go, anthropic-sdk-go.
    r"CreateChatCompletionStream", r"CreateChatCompletion", r"CreateCompletion",
    r"GenerateContentStream", r"GenerateContent", r"Messages\.New",
    # Java — openai-java, anthropic-java (fluent `.completions().create(...)`),
    # plus the older theokanning client.
    r"completions\(\)\s*\.\s*create", r"messages\(\)\s*\.\s*create",
    r"createChatCompletion",
    # Ruby — ruby-openai `client.chat(parameters: {...})` (guarded below to avoid
    # matching unrelated `.chat(` calls). anthropic/google ruby already covered.
    r"\.chat",
]
_TEXTUAL_SDK_RE = re.compile(r"(?:" + r"|".join(_TEXTUAL_SDK_FRAGMENTS) + r")\s*\(")


def _textual_sdk_calls(source: str):
    """Yield (start, args_start, call_name) for each recognized SDK call in any
    language. The Ruby `.chat(` shape is confirmed by its `parameters:` keyword so
    a generic `.chat(` elsewhere isn't mistaken for an LLM call."""
    for m in _TEXTUAL_SDK_RE.finditer(source):
        name = m.group(0).rstrip("( \t").lstrip(".")
        if name == "chat" and "parameters" not in source[m.end():m.end() + 48]:
            continue
        yield m.start(), m.end(), name


def _has_textual_sdk(text: str) -> bool:
    return next(_textual_sdk_calls(text), None) is not None


def _line_at(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def _blank_code(src: str) -> str:
    """Same-length copy of source with comment and string/char-literal *contents*
    replaced by spaces, across the languages the textual pass targets (C-family
    `//` and `/* */`, shell/Ruby/Python `#`, and ' " ` literals). This keeps real
    code structure (loops, call names, kwarg keys) intact while ensuring a loop
    keyword or the word "break" sitting in a comment or string can't create — or
    suppress — a finding. Line numbers are preserved (newlines kept)."""
    out = list(src)
    i, n, st = 0, len(src), "code"
    while i < n:
        c = src[i]
        nx = src[i + 1] if i + 1 < n else ""
        if st == "code":
            if c == "/" and nx == "/":
                out[i] = out[i + 1] = " "; i += 2; st = "line"; continue
            if c == "#":
                out[i] = " "; i += 1; st = "line"; continue
            if c == "/" and nx == "*":
                out[i] = out[i + 1] = " "; i += 2; st = "block"; continue
            if c in "'\"`":
                st = {"'": "sq", '"': "dq", "`": "tk"}[c]; i += 1; continue
            i += 1; continue
        if st == "line":
            if c == "\n":
                st = "code"
            else:
                out[i] = " "
            i += 1; continue
        if st == "block":
            if c == "*" and nx == "/":
                out[i] = out[i + 1] = " "; i += 2; st = "code"; continue
            if c != "\n":
                out[i] = " "
            i += 1; continue
        q = {"sq": "'", "dq": '"', "tk": "`"}[st]
        if c == "\\":
            out[i] = " "
            if i + 1 < n and src[i + 1] != "\n":
                out[i + 1] = " "
            i += 2; continue
        if c == q:
            st = "code"; i += 1; continue
        if c != "\n":
            out[i] = " "
        i += 1
    return "".join(out)


def _textual_checks(source: str, path: str) -> List[Finding]:
    # Scan over a comment/string-blanked copy so loop keywords or break/return in
    # comments and string literals neither create nor suppress findings.
    blanked = _blank_code(source)
    out: List[Finding] = []
    out += _textual_unbounded_loops(blanked, path)
    out += _textual_uncapped_output(blanked, path)
    return out


def _textual_unbounded_loops(source: str, path: str) -> List[Finding]:
    """Infinite loop whose body (heuristic window) issues an LLM call with no
    break/return. Erring toward under-flagging: a break anywhere in the window
    suppresses the finding."""
    out: List[Finding] = []
    for m in _INF_LOOP_RE.finditer(source):
        window = source[m.end(): m.end() + 1500]
        if not _has_textual_sdk(window):
            continue
        if re.search(r'\b(break|return)\b', window):
            continue
        out.append(Finding(
            finding_id="lint", category="unbounded_loop",
            severity=_SEV["unbounded_loop"],
            message="An infinite loop drives an LLM call with no break/return in "
                    "sight — the agent loop looks unbounded; cost is unbounded under "
                    "adverse inputs. (Heuristic, non-Python source.)",
            source_path=path, line=_line_at(source, m.start()),
            evidence={"check": "unbounded_loop", "engine": "textual"}))
    return out


def _textual_uncapped_output(source: str, path: str) -> List[Finding]:
    """A recognized LLM SDK call with no output-token cap kwarg in its argument
    window. Heuristic (the window is bounded, not brace-matched)."""
    out: List[Finding] = []
    seen: Set[int] = set()
    for start, args_start, call in _textual_sdk_calls(source):
        args = source[args_start: args_start + 500]
        if any(k in args for k in _OUTPUT_CAP_KWARGS):
            continue
        ln = _line_at(source, start)
        if ln in seen:
            continue
        seen.add(ln)
        out.append(Finding(
            finding_id="lint", category="uncapped_output",
            severity=_SEV["uncapped_output"],
            message=f"LLM call `{call}(...)` sets no max output-token cap "
                    f"(max_tokens / maxTokens / ...); a single response can run to "
                    f"the model's full limit. (Heuristic, non-Python source.)",
            source_path=path, line=ln,
            evidence={"check": "uncapped_output", "engine": "textual", "call": call}))
    return out


def _llm_calls_in(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) and _is_llm_call(n) for n in ast.walk(node))


def _check_unbounded_loops(tree: ast.AST, path: str) -> List[Finding]:
    out: List[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.While):
            continue
        # only `while True:` / `while 1:` (a `for` over an iterable is bounded).
        test_true = (isinstance(node.test, ast.Constant) and bool(node.test.value))
        if not test_true:
            continue
        if not _llm_calls_in(node):
            continue
        has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
        has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
        if has_break or has_return:
            continue  # a terminal path exists; the imperative parser models it
        out.append(Finding(
            finding_id="lint",
            category="unbounded_loop",
            severity=_SEV["unbounded_loop"],
            message="`while True:` drives an LLM call with no break/return and no "
                    "iteration cap — the agent loop is unbounded; cost is unbounded "
                    "under adverse inputs.",
            source_path=path,
            line=getattr(node, "lineno", None),
            evidence={"check": "unbounded_loop", "loop_test": "True"},
        ))
    return out


def _check_missing_caps(tree: ast.AST, path: str, frameworks: Set[str],
                        source: str) -> List[Finding]:
    out: List[Finding] = []
    if not frameworks:
        return out
    # Build the set of capped constructors active for this file's frameworks.
    active: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
    for fw in frameworks:
        spec = _FRAMEWORKS[fw]
        for ctor, caps in spec.get("agents", {}).items():
            active[ctor] = (fw, caps)
        for runner, caps in spec.get("runners", {}).items():
            active[runner] = (fw, caps)

    seen: Set[Tuple[str, int]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_tail_name(node.func)
        if name not in active:
            continue
        fw, caps = active[name]
        if _kwnames(node) & set(caps):
            continue  # a cap is set — good
        key = (name, getattr(node, "lineno", 0))
        if key in seen:
            continue
        seen.add(key)
        cap_list = " / ".join(caps)
        out.append(Finding(
            finding_id="lint",
            category="missing_iteration_cap",
            severity=_SEV["missing_iteration_cap"],
            message=f"{fw} `{name}(...)` has no termination cap ({cap_list}); it "
                    f"relies on the framework default. Set an explicit bound so a "
                    f"prompt-injected or confused agent can't loop indefinitely.",
            source_path=path,
            line=getattr(node, "lineno", None),
            evidence={"check": "missing_iteration_cap", "framework": fw,
                      "constructor": name, "expected_kwargs": list(caps)},
        ))

    # LangGraph recursion_limit lives in the run config, not a constructor kwarg.
    if "langgraph" in frameworks:
        runs = (".invoke(" in source or ".stream(" in source
                or ".ainvoke(" in source or ".astream(" in source)
        if runs and "recursion_limit" not in source:
            out.append(Finding(
                finding_id="lint",
                category="missing_iteration_cap",
                severity=_SEV["missing_iteration_cap"],
                message="LangGraph graph is invoked without a `recursion_limit` in "
                        "the run config; it relies on the default. Set an explicit "
                        "recursion_limit so a cyclic graph can't run away.",
                source_path=path,
                line=None,
                evidence={"check": "missing_iteration_cap", "framework": "langgraph",
                          "constructor": "invoke", "expected_kwargs": ["recursion_limit"]},
            ))
    return out


def _check_uncapped_output(tree: ast.AST, path: str) -> List[Finding]:
    out: List[Finding] = []
    seen: Set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_llm_call(node):
            continue
        if _kwnames(node) & _OUTPUT_CAP_KWARGS:
            continue
        ln = getattr(node, "lineno", 0)
        if ln in seen:
            continue
        seen.add(ln)
        out.append(Finding(
            finding_id="lint",
            category="uncapped_output",
            severity=_SEV["uncapped_output"],
            message=f"LLM call `{_dotted(node.func)}(...)` sets no max output-token "
                    f"cap (max_tokens / max_output_tokens / ...); a single response "
                    f"can run to the model's full limit. Set an explicit cap.",
            source_path=path,
            line=getattr(node, "lineno", None),
            evidence={"check": "uncapped_output", "call": _dotted(node.func)},
        ))
    return out


def _llamaindex_llm_aliases(tree: ast.AST) -> Set[str]:
    """Names bound by `from llama_index(.core).llms... import X` (incl. `as` alias).

    LlamaIndex LLM classes collide with vendor SDK client names (`OpenAI`,
    `Anthropic`), so we identify them by import origin to avoid flagging a raw
    `openai.OpenAI()` client (which doesn't generate and takes no max_tokens)."""
    aliases: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            if "llama_index" in mod and "llms" in mod:
                for a in node.names:
                    aliases.add(a.asname or a.name)
    return aliases


def _check_model_output_cap(tree: ast.AST, path: str) -> List[Finding]:
    """Flag a LangChain/LlamaIndex model wrapper built with no output cap.

    These frameworks invoke the model through a wrapper object, so the
    vendor-SDK uncapped check never sees the call; the cap belongs on the
    constructor (`ChatOpenAI(max_tokens=...)`, `OpenAILike(max_tokens=...)`)."""
    out: List[Finding] = []
    seen: Set[int] = set()
    llama = _llamaindex_llm_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_tail_name(node.func)
        is_lc = name in _MODEL_CTORS
        is_li = name in llama
        if not (is_lc or is_li):
            continue
        if _kwnames(node) & _OUTPUT_CAP_KWARGS:
            continue
        ln = getattr(node, "lineno", 0)
        if ln in seen:
            continue
        seen.add(ln)
        fw = "LangChain" if is_lc else "LlamaIndex"
        out.append(Finding(
            finding_id="lint",
            category="uncapped_output",
            severity=_SEV["uncapped_output"],
            message=f"{fw} model `{name}(...)` is built with no max output-token cap "
                    f"(max_tokens); every call through it can run to the model's full "
                    f"limit. Set the cap on the constructor.",
            source_path=path,
            line=getattr(node, "lineno", None),
            evidence={"check": "uncapped_model_ctor", "constructor": name, "framework": fw},
        ))
    return out


def _check_fanout(tree: ast.AST, path: str, source: str) -> List[Finding]:
    out: List[Finding] = []
    bounded_globally = "Semaphore" in source  # a concurrency limiter anywhere
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted(node.func)
        is_gather = dotted.endswith("gather") and ("asyncio" in dotted or dotted == "gather")
        if not is_gather:
            continue
        # gather(*[coro(x) for x in <input>]) — a comprehension argument is the
        # tell of input-driven fan-out.
        comp_arg = any(isinstance(a, (ast.ListComp, ast.GeneratorExp))
                       or (isinstance(a, ast.Starred)
                           and isinstance(a.value, (ast.ListComp, ast.GeneratorExp)))
                       for a in node.args)
        if not comp_arg or bounded_globally:
            continue
        # only flag if there is LLM activity in the file (agentic-gated already)
        out.append(Finding(
            finding_id="lint",
            category="fanout",
            severity=_SEV["fanout"],
            message="`asyncio.gather(...)` fans out over an input-driven "
                    "comprehension with no concurrency bound (no Semaphore); a "
                    "large input spawns unbounded parallel LLM calls. Bound the "
                    "fan-out (Semaphore / batch / cap).",
            source_path=path,
            line=getattr(node, "lineno", None),
            evidence={"check": "unbounded_fanout"},
        ))
    return out


# --- gate escalation ---------------------------------------------------------

def lint_gate(findings: List[Finding], strictness: str = "strict") -> str:
    """Map lint findings to a gate contribution: pass | warn | block.

    Honest calibration: only a *genuinely* unbounded construct (a raw unbounded
    loop, or any critical) blocks. Framework cap-absences have a default bound, so
    in strict mode they raise a WARN ("set an explicit cap"), not a block.
    """
    if strictness == "off" or not findings:
        return "pass"
    cats = {f.category for f in findings}
    if any(f.severity == "critical" for f in findings) or "unbounded_loop" in cats:
        return "block"
    if strictness == "strict":
        return "warn" if findings else "pass"
    # balanced: warn only on high-severity structural risks.
    if any(f.severity in ("high", "critical") for f in findings):
        return "warn"
    return "pass"


def worse_gate(a: str, b: str) -> str:
    return a if _GATE_ORDER.get(a, 0) >= _GATE_ORDER.get(b, 0) else b


# Categories this module introduces, for documentation/reporting.
LINT_CATEGORIES = ("unbounded_loop", "missing_iteration_cap", "uncapped_output", "fanout")
# Subset safe to merge into an already-analyzable result. We exclude only
# `unbounded_loop`, since the IR graph detector already owns loop cycles for a
# parsed workflow (merging it would double-count). The asyncio.gather fan-out
# check is a different mechanism from the IR map-node fan-out detector, so it is
# additive, not a duplicate.
LINT_MERGE_CATEGORIES = ("missing_iteration_cap", "uncapped_output", "fanout")
