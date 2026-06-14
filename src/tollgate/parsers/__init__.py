"""Parsers normalize heterogeneous agent/workflow sources into the Workflow IR."""
from __future__ import annotations

import os
from typing import List, Optional

from ..ir import Workflow
from .dsl import parse_dsl
from .prompt import parse_prompt
from .langgraph import parse_langgraph
from .autogpt import parse_autogpt, looks_like_autogpt
from .crewai import parse_crewai, looks_like_crewai
from .imperative import parse_imperative, has_imperative_llm, SDK_CALL_MARKERS

__all__ = ["parse_file", "discover", "parse_dsl", "parse_prompt",
           "parse_langgraph", "parse_autogpt", "looks_like_autogpt",
           "parse_crewai", "looks_like_crewai",
           "parse_imperative", "has_imperative_llm"]


def parse_file(path: str, source_kind: Optional[str] = None) -> Workflow:
    """Parse a single file into a Workflow, auto-detecting the source kind."""
    kind = source_kind or _detect_kind(path)
    if kind in ("dsl", "yaml", "json"):
        return parse_dsl(path)
    if kind == "prompt":
        return parse_prompt(path)
    if kind == "langgraph":
        return parse_langgraph(path)
    if kind == "crewai":
        return parse_crewai(path)
    if kind == "imperative":
        return parse_imperative(path)
    if kind == "agentic":
        # Recognized agentic framework with no recoverable graph: return an empty
        # workflow (non-analyzable) so the pipeline routes it to the strict linter
        # instead of inventing structure.
        return Workflow(workflow_id=os.path.splitext(os.path.basename(path))[0],
                        source_kind="agentic", nodes=[], edges=[], source_path=path)
    raise ValueError(f"unsupported source kind: {kind!r} for {path}")


def _detect_kind(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml", ".json"):
        # Distinguish a workflow DSL from anything else by content sniff.
        try:
            with open(path, "r", encoding="utf-8") as fh:
                head = fh.read(4096)
            if "nodes:" in head or '"nodes"' in head:
                return "dsl"
        except Exception:
            pass
        return "dsl"
    if ext in (".txt", ".md", ".jinja", ".j2", ".prompt", ".tmpl"):
        return "prompt"
    if ext == ".py":
        # LangGraph (framework graph) takes precedence; otherwise an imperative
        # hand-rolled agent (a loop around LLM SDK calls). Fall back to LangGraph's
        # tolerant single-node parse if neither signal is present.
        head = _read_head(path)
        if any(m in head for m in _LANGGRAPH_MARKERS):
            return "langgraph"
        if looks_like_crewai(head):
            return "crewai"
        if has_imperative_llm(head):
            return "imperative"
        if _has_agentic_framework(head):
            # Recognizably agentic (a known framework) but no recoverable graph —
            # hand it to the strict linter rather than fabricating a graph node.
            return "agentic"
        return "langgraph"
    return "prompt"


# ---------------------------------------------------------------------------
# Noise filtering for directory/repo scans.
#
# When walking a repo we must only surface files that are *actually* agent
# workflow artifacts. Otherwise documentation, project-meta files and editor
# config (README, CONTRIBUTING, SKILL.md, AGENTS.md, docs/*.md, ...) get parsed
# as one-node "prompts" and generate prompt-bloat noise that buries real risk.
#
# An explicit path passed straight to parse_file is always honored; the filter
# below only governs what a *scan* picks up on its own.
# ---------------------------------------------------------------------------

# Files we will consider as workflow artifacts when scanning a directory/diff.
_WORKFLOW_EXTS = (".yaml", ".yml", ".json", ".txt", ".md", ".jinja", ".j2", ".prompt", ".tmpl", ".py")

# Unambiguous prompt-template extensions: always treated as a prompt.
_PROMPT_EXTS = (".prompt", ".jinja", ".j2", ".tmpl")
# Ambiguous text: needs a positive prompt signal (front-matter / template vars /
# a prompts directory) before we treat it as a deployable prompt template.
_AMBIGUOUS_TEXT_EXTS = (".md", ".txt")

# Documentation / project-meta basenames (sans extension) that are never agents.
_DOC_STEMS = {
    "readme", "contributing", "code_of_conduct", "codeofconduct", "license",
    "licence", "changelog", "changes", "history", "security", "notice",
    "authors", "maintainers", "owners", "codeowners", "support", "governance",
    "roadmap", "install", "installation", "upgrading", "migration", "faq",
    "agents", "claude", "gemini", "copilot", "cursorrules", "skill",
    "llms", "llms-full", "llms.txt",
}

# Path segments that never contain deployable agent workflows.
_SKIP_DIR_SEGMENTS = {
    ".git", ".github", ".gitlab", ".claude", ".cursor", ".vscode", ".idea",
    "node_modules", "__pycache__", ".venv", "venv", "site-packages",
    "dist", "build", "docs", "doc", "tests", "test", "__tests__", "testing",
}

# Recognized prompt front-matter keys and LangGraph builder markers.
_PROMPT_META_KEYS = ("model", "task_class", "appends_history", "retrieves_context",
                     "prompt", "system", "node_id", "max_output_tokens")
_LANGGRAPH_MARKERS = ("StateGraph", "MessageGraph", "add_node", "add_edge",
                      "add_conditional_edges", "set_entry_point", ".compile(")

# Agent-framework import markers used only to *widen discovery* so framework files
# with no recoverable graph still reach the strict agentic linter. (The linter
# itself stays silent unless it finds a concrete agentic risk, so non-agentic
# uses of these libraries produce nothing.) Kept here, not imported from
# agentic_lint, to avoid a parsers<->agentic_lint import cycle.
_AGENTIC_IMPORT_MARKERS = (
    "langchain", "crewai", "autogen", "pyautogen", "autogen_agentchat",
    "llama_index", "llama-index", "llamaindex", "smolagents", "langgraph",
)


def _has_agentic_framework(head: str) -> bool:
    low = head.lower()
    return any(m in low for m in _AGENTIC_IMPORT_MARKERS)

_MAX_SNIFF_BYTES = 262144  # 256 KB is plenty to classify a source file.


def _read_head(path: str, limit: int = _MAX_SNIFF_BYTES) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read(limit)
    except OSError:
        return ""


def _looks_like_prompt(path: str) -> bool:
    """True only when an ambiguous .md/.txt is genuinely a prompt template."""
    norm = path.replace(os.sep, "/").lower()
    if any(seg in norm.split("/") for seg in ("prompts", "prompt", "templates")):
        return True
    text = _read_head(path)
    if not text:
        return False
    # Jinja/Handlebars template variables are the hallmark of a real template.
    if "{{" in text and "}}" in text:
        return True
    # YAML front-matter that declares known prompt fields.
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            front = text[3:end].lower()
            if any(k in front for k in _PROMPT_META_KEYS):
                return True
    return False


def _is_workflow_candidate(path: str) -> bool:
    """Decide whether a discovered file is worth parsing as a workflow."""
    base = os.path.basename(path)
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    if ext not in _WORKFLOW_EXTS:
        return False
    if stem.lower() in _DOC_STEMS:
        return False
    if ext in (".yaml", ".yml", ".json"):
        head = _read_head(path, 8192)
        return "nodes:" in head or '"nodes"' in head
    if ext == ".py":
        head = _read_head(path)
        # A LangGraph graph, a CrewAI crew, or a hand-rolled imperative agent
        # (loop around an LLM SDK call) — all are deployable workflows worth
        # analyzing.
        return (any(m in head for m in _LANGGRAPH_MARKERS)
                or looks_like_crewai(head)
                or any(m in head for m in SDK_CALL_MARKERS)
                or _has_agentic_framework(head))
    if ext in _PROMPT_EXTS:
        return True
    if ext in _AMBIGUOUS_TEXT_EXTS:
        return _looks_like_prompt(path)
    return False


def discover(paths: List[str]) -> List[str]:
    """Expand a list of files/dirs into candidate workflow artifact paths.

    Directories are filtered to real agent artifacts (see module notes). A file
    named explicitly is always returned — the caller asked for it by name.
    """
    found: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                # Prune non-deployable directories in-place so we don't descend.
                dirs[:] = [d for d in dirs if d.lower() not in _SKIP_DIR_SEGMENTS]
                for f in files:
                    fpath = os.path.join(root, f)
                    if _is_workflow_candidate(fpath):
                        found.append(fpath)
        elif os.path.isfile(p):
            found.append(p)
    return found
