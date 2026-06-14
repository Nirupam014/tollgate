"""AutoGPT block-graph adapter.

AutoGPT (autogpt_platform) exports agents as a graph JSON whose schema differs
from Tollgate's native DSL: nodes carry a ``block_id`` and an ``input_default``
bag instead of ``kind``/``model``/``prompt``, and control flow lives under
top-level ``links`` (``source_id``/``sink_id``) instead of ``edges``.

The native DSL parser silently mis-reads this format — every node defaults to
``llm_call`` on a fabricated default model and *all* links are dropped, so the
graph loses its structure and the analyzer emits a confident-but-meaningless
PASS. This adapter maps the AutoGPT shape into the IR faithfully:

* an LLM block (its input bag carries a ``model`` and/or prompt signals) becomes
  a ``llm_call`` node with the real model; every other block becomes a ``tool``;
* ``links`` are translated to edges, so cycle / context-explosion detection works.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional

from ..ir import IREdge, IRNode, Workflow
from ..tokenizer import count_tokens

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def looks_like_autogpt(data: Any) -> bool:
    """True for an AutoGPT graph export (vs. native Tollgate DSL or other JSON)."""
    if not isinstance(data, dict):
        return False
    nodes = data.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return False
    # Native DSL never uses block_id/links; AutoGPT always uses at least one.
    if isinstance(data.get("links"), list):
        return True
    return any(isinstance(n, dict) and "block_id" in n for n in nodes)


# ---------------------------------------------------------------------------
# Model-name normalization: AutoGPT LlmModel enum values -> catalog ids.
#
# We match by family + size/tier rather than enumerating every dated snapshot,
# so new ``...-2024xxxx`` suffixes still resolve. Unmapped strings are returned
# as-is (prediction then falls back to the configured default model).
# ---------------------------------------------------------------------------

def _normalize_model(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    m = str(raw).strip().lower()
    # Drop a provider prefix like "perplexity/sonar-pro" or "openai/gpt-4o".
    if "/" in m:
        m = m.rsplit("/", 1)[1]

    # OpenAI
    if "gpt-5" in m or "gpt5" in m:
        if "nano" in m:
            return "gpt-5-nano"
        if "mini" in m:
            return "gpt-5-mini"
        return "gpt-5"
    if m.startswith(("o1", "o3", "o4")):
        return "gpt-4.1-mini" if "mini" in m else "gpt-4.1"
    if "gpt-4.1" in m or "gpt-4-1" in m:
        return "gpt-4.1-mini" if ("mini" in m or "nano" in m) else "gpt-4.1"
    if "gpt-4o" in m or "gpt4o" in m:
        return "gpt-4o-mini" if "mini" in m else "gpt-4o"
    if "gpt-4" in m:  # gpt-4, gpt-4-turbo
        return "gpt-4o"
    if "gpt-3.5" in m or "gpt-35" in m:
        return "gpt-4o-mini"

    # Anthropic
    if "claude" in m:
        if "opus" in m:
            return "claude-opus-4"
        if "haiku" in m:
            return "claude-haiku-3.5"
        return "claude-sonnet-4"  # sonnet (and unspecified claude)

    # Google
    if "gemini" in m:
        if "2.0" in m or "2-0" in m:
            return "gemini-2.0-flash"
        if "flash" in m:
            return "gemini-1.5-flash"
        return "gemini-1.5-pro"

    # Perplexity Sonar (search-grounded)
    if "sonar" in m:
        if "deep" in m or "research" in m:
            return "sonar-deep-research"
        if "pro" in m:
            return "sonar-pro"
        return "sonar"

    # Open-source (Groq / Ollama / Together names)
    if "mixtral" in m:
        return "mixtral-8x7b"
    if "llama" in m:
        # any 8b-class -> 8b; everything larger -> 70b
        if "8b" in m or "7b" in m or "instant" in m:
            return "llama-3.1-8b"
        return "llama-3.3-70b"

    return raw  # unknown: let prediction apply the default model


# Input-bag keys that signal a node is really an LLM call.
_MODEL_KEYS = ("model", "llm_model", "model_name")
_PROMPT_KEYS = ("prompt", "sys_prompt", "system_prompt", "user_prompt", "messages")
_LLM_HINT_KEYS = ("max_tokens", "temperature", "top_p", "sys_prompt", "system_prompt")

# block_id substrings for known AutoGPT AI blocks (covers nodes whose model is
# wired in via a link rather than hardcoded, so no ``model`` key is present).
_AI_BLOCK_HINTS = ("aitextgenerator", "aistructured", "aiconversation",
                   "smartdecision", "llmcall", "ai_", "_ai", "summariz")

# Non-text-generation model blocks (image / audio / video / speech). These carry
# a ``model`` field but are NOT priced per token, so they must not be counted as
# LLM calls — they become tools and are excluded from token costing.
_NONTEXT_BLOCK_HINTS = ("image", "imagegen", "dalle", "dall_e", "flux",
                        "stablediffusion", "stable_diffusion", "sdxl",
                        "ideogram", "midjourney", "texttospeech", "tts",
                        "speech", "whisper", "transcri", "audio", "video",
                        "musicgen", "voice")
_NONTEXT_MODEL_HINTS = ("flux", "dall-e", "dall_e", "dalle", "stable-diffusion",
                        "stable diffusion", "sdxl", "midjourney", "ideogram",
                        "whisper", "tts-", "elevenlabs", "kling", "veo", "sora",
                        "runway", "musicgen", "playht", "recraft")


def _is_nontext_model_node(raw: Dict[str, Any], bag: Dict[str, Any]) -> bool:
    """True for image/audio/video model blocks (not token-priced LLMs)."""
    block_id = str(raw.get("block_id", "")).lower()
    if any(h in block_id for h in _NONTEXT_BLOCK_HINTS):
        return True
    model = _first(bag, _MODEL_KEYS)
    if model is not None:
        ml = str(model).strip().lower()
        if any(h in ml for h in _NONTEXT_MODEL_HINTS):
            return True
    return False


def _inputs(raw: Dict[str, Any]) -> Dict[str, Any]:
    bag = raw.get("input_default")
    if not isinstance(bag, dict):
        bag = raw.get("hardcoded_values")
    return bag if isinstance(bag, dict) else {}


def _first(bag: Dict[str, Any], keys) -> Any:
    for k in keys:
        if k in bag and bag[k] not in (None, "", []):
            return bag[k]
    return None


def _prompt_text(bag: Dict[str, Any]) -> Optional[str]:
    val = _first(bag, _PROMPT_KEYS)
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, list):  # chat-style messages
        parts = []
        for msg in val:
            if isinstance(msg, dict):
                c = msg.get("content")
                if isinstance(c, str):
                    parts.append(c)
            elif isinstance(msg, str):
                parts.append(msg)
        return "\n".join(parts) if parts else None
    return str(val)


def _is_llm_node(raw: Dict[str, Any], bag: Dict[str, Any]) -> bool:
    if _first(bag, _MODEL_KEYS) is not None:
        return True
    block_id = str(raw.get("block_id", "")).lower()
    if any(h in block_id for h in _AI_BLOCK_HINTS):
        return True
    # A prompt plus generation knobs is a strong LLM signal even without a model.
    if _first(bag, _PROMPT_KEYS) is not None and _first(bag, _LLM_HINT_KEYS) is not None:
        return True
    return False


def _as_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_autogpt(data: Dict[str, Any], path: str) -> Workflow:
    wf_id = str(data.get("name") or data.get("id")
                or os.path.splitext(os.path.basename(path))[0])

    nodes: List[IRNode] = []
    for raw in data.get("nodes", []):
        if not isinstance(raw, dict) or "id" not in raw:
            continue
        bag = _inputs(raw)
        # Image/audio/video model blocks are tools, not token-priced LLM calls.
        is_llm = _is_llm_node(raw, bag) and not _is_nontext_model_node(raw, bag)
        prompt = _prompt_text(bag) if is_llm else None
        node = IRNode(
            node_id=str(raw["id"]),
            kind="llm_call" if is_llm else "tool",
            intended_model=_normalize_model(_first(bag, _MODEL_KEYS)) if is_llm else None,
            prompt_template=prompt,
            max_output_tokens=_as_int(bag.get("max_tokens")) if is_llm else None,
        )
        if prompt and node.static_input_tokens == 0:
            node.static_input_tokens = count_tokens(prompt)
        nodes.append(node)

    known = {n.node_id for n in nodes}
    edges: List[IREdge] = []
    for ln in data.get("links", []):
        if not isinstance(ln, dict):
            continue
        src = ln.get("source_id")
        dst = ln.get("sink_id")
        if src is None or dst is None:
            continue
        src, dst = str(src), str(dst)
        if src not in known or dst not in known:
            continue
        edges.append(IREdge(from_node=src, to_node=dst, edge_type="sequence"))

    wf = Workflow(
        workflow_id=wf_id,
        source_kind="autogpt",
        nodes=nodes,
        edges=edges,
        entry=None,
        source_path=path,
    )
    wf.content_hash = _hash(path)
    return wf


def _hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()[:16]
