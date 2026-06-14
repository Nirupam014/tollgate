"""Raw prompt-template parser.

A single prompt file becomes a one-node workflow. Front-matter (optional, YAML
between leading `---` fences) can declare the intended model and signals:

    ---
    model: gpt-4o
    task_class: generation
    appends_history: true
    retrieves_context: true
    ---
    You are a helpful assistant. {{history}} {{context}}

Template variables named history/messages/conversation imply history append;
context/documents/retrieved imply retrieval. This lets us flag context-growth
risk even for plain prompt files.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, Tuple

import yaml

from ..ir import IRNode, Workflow
from ..tokenizer import count_tokens

_FRONT_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.S)
_VAR_RE = re.compile(r"\{\{?\s*([a-zA-Z0-9_\.]+)\s*\}?\}")

_HISTORY_VARS = {"history", "messages", "conversation", "chat_history", "memory"}
_CONTEXT_VARS = {"context", "documents", "docs", "retrieved", "knowledge", "rag"}


def _split_front_matter(text: str) -> Tuple[Dict[str, Any], str]:
    m = _FRONT_RE.match(text)
    if not m:
        return {}, text
    meta = yaml.safe_load(m.group(1)) or {}
    return meta, m.group(2)


def parse_prompt(path: str) -> Workflow:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    meta, body = _split_front_matter(text)

    var_names = {v.split(".")[0].lower() for v in _VAR_RE.findall(body)}
    appends_history = bool(meta.get("appends_history", bool(var_names & _HISTORY_VARS)))
    retrieves_context = bool(meta.get("retrieves_context", bool(var_names & _CONTEXT_VARS)))

    node = IRNode(
        node_id=meta.get("node_id", "prompt"),
        kind="llm_call",
        intended_model=meta.get("model"),
        prompt_template=body,
        task_class=meta.get("task_class"),
        appends_history=appends_history,
        retrieves_context=retrieves_context,
        retrieved_context_cap=meta.get("retrieved_context_cap"),
        max_output_tokens=meta.get("max_output_tokens"),
    )
    node.static_input_tokens = count_tokens(body)

    wf = Workflow(
        workflow_id=meta.get("workflow") or os.path.splitext(os.path.basename(path))[0],
        source_kind="prompt",
        nodes=[node],
        edges=[],
        entry=node.node_id,
        source_path=path,
    )
    wf.content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return wf
