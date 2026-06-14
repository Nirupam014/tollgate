"""Token counting.

Uses tiktoken when available for OpenAI-family accuracy; otherwise falls back
to a deterministic heuristic that is good enough for pre-deploy estimation.
The heuristic is intentionally simple and stable so CI results are reproducible.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

_TIKTOKEN = None
try:  # optional dependency
    import tiktoken  # type: ignore

    _TIKTOKEN = tiktoken
except Exception:  # pragma: no cover - environment dependent
    _TIKTOKEN = None

# Map catalog tokenizer names to tiktoken encodings (best effort).
_ENCODING_FOR = {
    "o200k_base": "o200k_base",
    "cl100k_base": "cl100k_base",
}

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=64)
def _get_encoding(name: str):
    if _TIKTOKEN is None:
        return None
    enc_name = _ENCODING_FOR.get(name)
    if enc_name is None:
        return None
    try:
        return _TIKTOKEN.get_encoding(enc_name)
    except Exception:  # pragma: no cover
        return None


def heuristic_tokens(text: str) -> int:
    """Deterministic estimate: blends a chars/4 rule with a word/punct count.

    Empirically tracks real tokenizers within ~10-15% for English prose and is
    fully reproducible (no model download, no network).
    """
    if not text:
        return 0
    char_est = len(text) / 4.0
    word_est = len(_WORD_RE.findall(text)) * 1.3
    return max(1, round((char_est + word_est) / 2.0))


def count_tokens(text: str, tokenizer: str = "heuristic") -> int:
    """Count tokens for `text` under the named tokenizer family."""
    if not text:
        return 0
    enc = _get_encoding(tokenizer)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:  # pragma: no cover
            pass
    return heuristic_tokens(text)


def using_exact_tokenizer(tokenizer: str = "heuristic") -> bool:
    return _get_encoding(tokenizer) is not None
