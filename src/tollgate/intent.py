"""Deterministic, built-in intent classifier for LLM nodes.

Substitution needs to know *what kind of task* a node performs — a code task wants
a code-strong model, a classifier can run on a tiny one, an image task shouldn't be
swapped to a text model at all. A single `quality_tier` can't express that.

This classifies a node's intent into a (domain, modality) using a lexical
**embedding**: the text (prompt template + node id) is turned into a token-frequency
vector and compared by cosine similarity to per-category prototype vectors built
from seed lexicons. It is fully deterministic, stdlib-only, makes no model calls,
and reads nothing at runtime — so it is safe to run always-on and it never affects
the gate or the tamper-evident fingerprint (it only informs advisory
recommendations).

It is intentionally NOT a neural model: a neural embedding would be more accurate
but would break the deterministic/offline/zero-LLM guarantees the gate relies on,
so that remains a future *optional* backend behind the `classify` seam, never a
hard dependency. Confidence is reported so callers can ignore low-confidence
guesses and fall back to their defaults.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# domain -> (keywords, phrases, capability tier floor, modality)
# tier: minimum quality tier this kind of task should run on (None = no opinion).
# modality: "text" tasks are substitutable; non-text means "don't swap to a text model".
_CATEGORIES: Dict[str, dict] = {
    "code": {
        "kw": ["code", "function", "python", "javascript", "typescript", "java",
               "golang", "rust", "sql", "query", "refactor", "debug", "compile",
               "api", "schema", "regex", "script", "programming", "bug", "stacktrace",
               "repository", "git", "unittest", "lint", "snippet", "syntax"],
        "ph": ["write code", "unit test", "code review", "sql query", "write a function",
               "fix the bug", "refactor this", "generate code"],
        "tier": 4, "modality": "text"},
    "reasoning": {
        "kw": ["reason", "reasoning", "analyze", "analysis", "plan", "strategy",
               "deduce", "infer", "logic", "evaluate", "solve", "prove", "decide"],
        "ph": ["step by step", "think step", "chain of thought", "reason about",
               "work through"],
        "tier": 4, "modality": "text"},
    "math": {
        "kw": ["math", "calculate", "equation", "integral", "derivative", "theorem",
               "arithmetic", "compute", "probability", "algebra", "calculus"],
        "ph": ["solve for", "math problem", "compute the"],
        "tier": 4, "modality": "text"},
    "creative": {
        "kw": ["story", "poem", "creative", "marketing", "slogan", "copy", "blog",
               "novel", "screenplay", "lyrics", "narrative"],
        "ph": ["write a story", "blog post", "marketing copy"],
        "tier": 3, "modality": "text"},
    "extraction": {
        "kw": ["extract", "parse", "entities", "ner", "structured", "fields", "field"],
        "ph": ["extract the", "pull out", "structured output"],
        "tier": 2, "modality": "text"},
    "summarization": {
        "kw": ["summarize", "summary", "tldr", "abstract", "condense", "brief", "digest"],
        "ph": ["summarize the", "in summary", "give a summary"],
        "tier": 2, "modality": "text"},
    "translation": {
        "kw": ["translate", "translation", "localize", "localization", "bilingual"],
        "ph": ["translate to", "translate the", "into english"],
        "tier": 2, "modality": "text"},
    "classification": {
        "kw": ["classify", "categorize", "label", "sentiment", "category", "tag",
               "spam", "moderation", "detect"],
        "ph": ["classify the", "is this", "categorize the"],
        "tier": 1, "modality": "text"},
    "routing": {
        "kw": ["route", "router", "dispatch", "select", "choose", "triage"],
        "ph": ["which tool", "route to", "decide which"],
        "tier": 1, "modality": "text"},
    "general": {
        "kw": ["help", "helpful", "assistant", "question", "answer", "explain",
               "chat", "conversation", "support"],
        "ph": ["you are a helpful", "answer the question"],
        "tier": 2, "modality": "text"},
    "image": {
        "kw": ["image", "picture", "photo", "draw", "render", "dalle", "diffusion",
               "png", "jpg", "illustration", "logo", "artwork"],
        "ph": ["generate an image", "an image of", "create a picture", "text to image"],
        "tier": None, "modality": "image"},
    "audio": {
        "kw": ["audio", "speech", "voice", "transcribe", "tts", "whisper", "sound"],
        "ph": ["text to speech", "transcribe the", "speech to text"],
        "tier": None, "modality": "audio"},
    "embedding": {
        "kw": ["embedding", "embed", "vector", "similarity", "retrieval", "rerank"],
        "ph": ["embed the", "vector representation", "semantic search"],
        "tier": None, "modality": "embedding"},
}

# Tunables (deterministic): how sure we must be to act on a guess.
_DOMAIN_MIN_SCORE = 2.0      # below this we say "unknown" (don't raise the floor)
_MODALITY_MIN_SCORE = 3.0    # abstaining from a swap is a strong action -> stricter
_PHRASE_WEIGHT = 2.0


@dataclass
class IntentResult:
    domain: str            # e.g. "code", "classification", "image", or "unknown"
    modality: str          # "text" | "image" | "audio" | "embedding"
    tier_floor: Optional[int]   # capability tier this task implies, or None
    confidence: float      # 0..1, relative margin over the runner-up

    def to_dict(self):
        return {"domain": self.domain, "modality": self.modality,
                "tier_floor": self.tier_floor, "confidence": round(self.confidence, 3)}


def _prototype_vectors():
    vecs: Dict[str, Counter] = {}
    for name, spec in _CATEGORIES.items():
        c: Counter = Counter()
        for kw in spec["kw"]:
            c[kw] += 1.0
        vecs[name] = c
    return vecs


_PROTOS = _prototype_vectors()


def _score(text: str) -> Dict[str, float]:
    raw = (text or "").lower()
    tokens = _TOKEN_RE.findall(raw)
    if not tokens:
        return {}
    tf = Counter(tokens)
    norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
    scores: Dict[str, float] = {}
    for name, spec in _CATEGORIES.items():
        proto = _PROTOS[name]
        pnorm = math.sqrt(sum(v * v for v in proto.values())) or 1.0
        dot = sum(tf[t] * w for t, w in proto.items())
        cosine = dot / (norm * pnorm)
        phrase_hits = sum(1 for ph in spec["ph"] if ph in raw)
        # absolute keyword hits + phrase bonus drive the decision; cosine breaks ties.
        scores[name] = dot + _PHRASE_WEIGHT * phrase_hits + cosine
    return scores


def classify(text: str) -> IntentResult:
    """Classify free text (a prompt and/or node id) into an intent."""
    scores = _score(text)
    if not scores:
        return IntentResult("unknown", "text", None, 0.0)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = (top_score - second_score) / (top_score + 1e-9) if top_score > 0 else 0.0

    spec = _CATEGORIES[top]
    # Non-text modality is a hard guard, but only when the signal is strong enough.
    if spec["modality"] != "text":
        if top_score >= _MODALITY_MIN_SCORE:
            return IntentResult(top, spec["modality"], None, confidence)
        return IntentResult("unknown", "text", None, confidence)
    if top_score >= _DOMAIN_MIN_SCORE:
        return IntentResult(top, "text", spec["tier"], confidence)
    return IntentResult("unknown", "text", None, confidence)


def classify_node(node) -> IntentResult:
    """Classify an IR node from its prompt template and node id."""
    parts: List[str] = []
    pt = getattr(node, "prompt_template", None)
    if pt:
        parts.append(str(pt))
    nid = getattr(node, "node_id", None)
    if nid:
        # node ids like "execution_agent" / "code_writer" carry intent; split snake/camel.
        parts.append(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(nid)).replace("_", " "))
    return classify(" ".join(parts))
