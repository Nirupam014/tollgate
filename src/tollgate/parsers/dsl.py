"""DSL parser: the native Tollgate workflow format (YAML or JSON).

Schema (all node/edge fields optional except ids):

    workflow: my_agent            # id (defaults to filename)
    source_kind: dsl
    entry: plan                   # optional; inferred if omitted
    nodes:
      - id: plan
        kind: llm_call            # llm_call|tool|router|map|reduce|human
        model: gpt-4o
        task_class: reasoning
        prompt: "..."             # inline static prompt
        prompt_file: prompts/plan.txt
        appends_history: true
        retrieves_context: true
        retrieved_context_cap: 4000   # omit/null == uncapped
        max_output_tokens: 1024
        retry: { max_attempts: 3, backoff: exponential }
        fanout_factor: 10         # for map nodes; omit == input-driven
    edges:
      - from: plan
        to: act
        type: sequence            # sequence|conditional|loop|fanout
        condition: "needs_tool"
        probability: 0.5          # for conditional
        guard: { max_depth: 10 }  # for loop edges
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict

import yaml

from ..ir import Guard, IREdge, IRNode, Retry, Workflow
from ..tokenizer import count_tokens
from .autogpt import looks_like_autogpt, parse_autogpt


def _load(path: str) -> Dict[str, Any]:
    """Load a DSL file, failing gracefully.

    Malformed YAML/JSON, unreadable bytes, or a non-mapping top level (a bare
    list/scalar) all resolve to an empty dict — which yields an empty Workflow
    that the pipeline drops (honest failure), never a crash on a CI scan.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
    except OSError:
        return {}
    try:
        if path.lower().endswith(".json"):
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError, ValueError, RecursionError):
        return {}
    return data if isinstance(data, dict) else {}


def parse_dsl(path: str) -> Workflow:
    data = _load(path)
    # A JSON file that is actually an AutoGPT graph export is handled by its own
    # adapter; the native-DSL field names below would silently mis-read it.
    if looks_like_autogpt(data):
        return parse_autogpt(data, path)
    wf_id = str(data.get("workflow") or data.get("id") or os.path.splitext(os.path.basename(path))[0])
    base_dir = os.path.dirname(os.path.abspath(path))

    nodes = []
    nodes_raw = data.get("nodes")
    if not isinstance(nodes_raw, list):
        nodes_raw = []
    for raw in nodes_raw:
        if not isinstance(raw, dict) or raw.get("id") is None:
            continue  # skip malformed node entries rather than crashing
        prompt = raw.get("prompt")
        if not prompt and raw.get("prompt_file"):
            pf = os.path.join(base_dir, raw["prompt_file"])
            try:
                with open(pf, "r", encoding="utf-8") as fh:
                    prompt = fh.read()
            except OSError:
                prompt = None

        retry = None
        if raw.get("retry"):
            r = raw["retry"]
            retry = Retry(max_attempts=r.get("max_attempts"), backoff=r.get("backoff"))

        node = IRNode(
            node_id=str(raw["id"]),
            kind=raw.get("kind", "llm_call"),
            intended_model=raw.get("model"),
            prompt_template=prompt,
            task_class=raw.get("task_class"),
            appends_history=bool(raw.get("appends_history", False)),
            retrieves_context=bool(raw.get("retrieves_context", False)),
            retrieved_context_cap=raw.get("retrieved_context_cap"),
            max_output_tokens=raw.get("max_output_tokens"),
            fanout_factor=raw.get("fanout_factor"),
            retry=retry,
            branch_probability=float(raw.get("branch_probability", 1.0)),
        )
        if prompt and node.static_input_tokens == 0:
            node.static_input_tokens = count_tokens(prompt)
        nodes.append(node)

    edges = []
    edges_raw = data.get("edges")
    if not isinstance(edges_raw, list):
        edges_raw = []
    for raw in edges_raw:
        if not isinstance(raw, dict) or raw.get("from") is None or raw.get("to") is None:
            continue  # skip malformed edge entries rather than crashing
        guard = None
        if raw.get("guard"):
            g = raw["guard"]
            guard = Guard(
                max_depth=g.get("max_depth"),
                counter=bool(g.get("counter", False)),
                stop_condition=g.get("stop_condition"),
            )
        edges.append(
            IREdge(
                from_node=str(raw["from"]),
                to_node=str(raw["to"]),
                edge_type=raw.get("type", "sequence"),
                condition=raw.get("condition"),
                probability=raw.get("probability"),
                guard=guard,
            )
        )

    wf = Workflow(
        workflow_id=wf_id,
        source_kind=data.get("source_kind", "dsl"),
        nodes=nodes,
        edges=edges,
        entry=data.get("entry"),
        source_path=path,
    )
    wf.content_hash = _hash(path)
    return wf


def _hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()[:16]
