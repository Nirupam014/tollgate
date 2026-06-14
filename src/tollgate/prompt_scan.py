"""Language-agnostic prompt miner.

Finds LLM **prompts hidden in source code** — string literals, constants,
heredocs, and config values that are actually model instructions — in *any*
language, without a per-language parser. It lexes string literals with tolerant
regexes (the quoting forms are nearly universal) and scores each by how
prompt-like it is.

Why this is its own thing, and why it's honest:
  * The graph parsers/linter only see prompts that sit at a recognized LLM call.
    A huge triple-quoted SYSTEM_PROMPT constant in a prompts.py / prompts.ts / a
    YAML config is invisible to them — yet that's exactly where prompt bloat and
    injection risk live.
  * Deciding "is this string a prompt?" is inherently fuzzy, so detection is
    **heuristic and advisory**: every detected prompt is labeled with its
    likelihood + the reasons, it never hard-blocks a gate, and it stays silent on
    code that isn't prompt-like (SQL, HTML, logs, URLs, plain code).
  * Deterministic: pure regex + scoring, no LLM, never executes anything.

Public API:
  scan_text(path, text, min_score=...) -> List[DetectedPrompt]
  scan_file(path, min_score=...)       -> List[DetectedPrompt]
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .tokenizer import count_tokens

# Files worth lexing for embedded prompts. Broad (any language) but excludes
# binaries and lockfiles. Prompt-template extensions are handled by the prompt
# parser already, so they're not re-scanned here.
SCANNABLE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".go", ".rb", ".php",
    ".java", ".kt", ".kts", ".scala", ".cs", ".swift", ".rs", ".c", ".h",
    ".cpp", ".cc", ".hpp", ".m", ".mm", ".sh", ".bash", ".pl", ".lua", ".dart",
    ".ex", ".exs", ".clj", ".yaml", ".yml", ".json", ".toml", ".env",
}
_SKIP_NAME_SUBSTR = ("package-lock", "yarn.lock", "poetry.lock", ".min.")
_MAX_BYTES = 600_000


@dataclass
class DetectedPrompt:
    source_path: str
    name: Optional[str]      # the identifier/key the literal was assigned to, if any
    line: int
    text: str
    est_tokens: int
    score: int
    reasons: List[str] = field(default_factory=list)
    review: object = None     # an efficiency PromptReview, attached by the pipeline

    def to_dict(self):
        snippet = " ".join(self.text.split())
        if len(snippet) > 200:
            snippet = snippet[:199] + "…"
        return {
            "source_path": self.source_path,
            "name": self.name,
            "line": self.line,
            "est_tokens": self.est_tokens,
            "score": self.score,
            "reasons": self.reasons,
            "snippet": snippet,
            "review": self.review.to_dict() if self.review is not None else None,
        }


# --- string-literal extraction (language-agnostic) ---------------------------
# Each pattern captures a `body` group. We run the multi-line forms first and
# blank out their spans so the simple quoted form doesn't double-match inside.
_TRIPLE = re.compile(r'(?s)("""|\'\'\')(?P<body>.*?)\1')
_BACKTICK = re.compile(r'(?s)`(?P<body>[^`]*)`')
# Heredocs/nowdocs: Ruby/Bash `<<~`/`<<-`/`<<`, PHP `<<<` (three angle brackets).
_HEREDOC = re.compile(r'(?s)<<[<~-]{0,2}["\']?(?P<tag>[A-Za-z_]\w*)["\']?\r?\n'
                      r'(?P<body>.*?)\r?\n[ \t]*(?P=tag)\b')
# YAML block scalars:  key: |   /   key: >   then an indented block.
_YAML_BLOCK = re.compile(r'(?m)^[ \t]*(?P<name>[\w.\-]+):[ \t]*[|>][+\-]?[ \t]*\r?\n'
                         r'(?P<body>(?:[ \t]+\S.*\r?\n?)+)')
_DQ = re.compile(r'"(?P<body>(?:[^"\\\n]|\\.){40,})"')
_SQ = re.compile(r"'(?P<body>(?:[^'\\\n]|\\.){40,})'")

# Identifier/key immediately preceding a literal (look-back), across languages:
#   SYSTEM_PROMPT = ...   const sys = ...   val p = ...   "prompt": ...   p = f"""
# Tolerates a leading quote (JSON keys), a closing quote after the name, and a
# string prefix between the operator and the quote (f / r / b / u / @ / $ / r#).
_NAME_BEFORE = re.compile(
    r'(?:^|[\s({\[,:"\'])'
    r'(?:const|let|var|val|final|static|public|private|String|string|str|prompt|template)?\s*'
    r'(?P<name>[A-Za-z_][A-Za-z0-9_.]*)["\']?\s*'
    r'(?:[:=]|=>)\s*'
    r'[A-Za-z@$#]{0,3}\s*$')


def _unescape(s: str) -> str:
    return (s.replace('\\n', '\n').replace('\\t', '\t')
             .replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\'))


def _name_before(text: str, start: int) -> Optional[str]:
    window = text[max(0, start - 80):start]
    m = _NAME_BEFORE.search(window)
    if m:
        nm = m.group("name")
        if nm.lower() not in ("string", "str", "f", "r", "b", "rb", "u"):
            return nm
    return None


def _line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def _iter_literals(text: str):
    """Yield (body, name, line) for every string literal, longest forms first."""
    spans = []  # (start, end) of consumed regions to avoid double-matching

    def consumed(a, b):
        return any(a < e and s < b for s, e in spans)

    # YAML block scalars carry their own key as the name.
    for m in _YAML_BLOCK.finditer(text):
        s, e = m.span()
        if consumed(s, e):
            continue
        spans.append((s, e))
        yield m.group("body"), m.group("name"), _line_of(text, s)

    for rx in (_TRIPLE, _HEREDOC, _BACKTICK):
        for m in rx.finditer(text):
            s, e = m.span()
            if consumed(s, e):
                continue
            spans.append((s, e))
            body = m.group("body")
            yield body, _name_before(text, s), _line_of(text, s)

    # Simple quoted strings on the remaining (non-consumed) regions.
    for rx in (_DQ, _SQ):
        for m in rx.finditer(text):
            s, e = m.span()
            if consumed(s, e):
                continue
            yield _unescape(m.group("body")), _name_before(text, s), _line_of(text, s)


# --- prompt-likelihood heuristic ---------------------------------------------
_NAME_RE = re.compile(r'(?i)(system|user|assistant|prompt|instruction|persona|'
                      r'template|preamble|guideline|directive|role|context)')
_INSTR_RE = re.compile(r'(?i)\b(you are|your task|your job|you should|you must|you will|'
                       r'do not|don\'t|respond (with|in|only)|answer (the|in|with|only)|'
                       r'as an? (assistant|ai|expert|agent)|the user|step[- ]by[- ]step|'
                       r'format your|follow these|return (a|the|only|json)|'
                       r'output (a|the|only|json)|act as|reply (with|in))\b')
_ROLE_RE = re.compile(r'(?i)\b(system|assistant|user)\b\s*[:"]')
# Classic system-prompt openers — a very strong, low-false-positive signal.
_OPENER_RE = re.compile(r'(?i)^["\']?\s*(you are\b|act as\b|your (?:task|job|role|goal) is\b|'
                        r'as an? (?:assistant|ai|expert|agent)\b|i want you to\b)')
_PLACEHOLDER_RE = re.compile(r'(\{\{?\s*\w+\s*\}?\}|\$\{\s*\w+\s*\}|%\(\w+\)s|<\w+>)')

# Negative signals — strings that look like something other than a prompt.
_SQL_RE = re.compile(r'(?i)\b(select\s+.+\s+from|insert\s+into|update\s+\w+\s+set|'
                     r'create\s+(table|index)|delete\s+from|alter\s+table)\b')
_HTML_RE = re.compile(r'</?[a-zA-Z][^>]*>')
_URL_RE = re.compile(r'https?://\S+')
_CODEY_RE = re.compile(r'(?i)\b(function|def|return|import|class|public static|'
                       r'console\.log|println|printf)\b|=>|;\s')


def prompt_likelihood(name: Optional[str], body: str):
    """Return (score, reasons). Higher = more prompt-like."""
    text = body.strip()
    reasons: List[str] = []
    if len(text) < 40:
        return 0, ["too short"]
    words = text.split()
    n_words = len(words)
    score = 0

    if name and _NAME_RE.search(name):
        score += 3
        reasons.append(f"prompt-like name `{name}`")
    if _OPENER_RE.match(text):
        score += 2
        reasons.append("system-prompt opener")
    instr = len(_INSTR_RE.findall(text))
    if instr:
        score += 2 + (1 if instr >= 3 else 0)
        reasons.append(f"instruction language ×{instr}")
    if _ROLE_RE.search(text):
        score += 1
        reasons.append("role marker")
    if _PLACEHOLDER_RE.search(text):
        score += 1
        reasons.append("template placeholder")
    if n_words >= 50:
        score += 2
        reasons.append("long prose")
    elif n_words >= 20:
        score += 1
        reasons.append("multi-sentence")
    # prose-ish: mostly letters/spaces, low punctuation density
    letters = sum(c.isalpha() or c.isspace() for c in text)
    if letters / max(1, len(text)) > 0.75 and n_words >= 12:
        score += 1
        reasons.append("natural-language ratio")

    # --- negatives -----------------------------------------------------------
    if _SQL_RE.search(text):
        score -= 4
        reasons.append("looks like SQL")
    if len(_HTML_RE.findall(text)) >= 3:
        score -= 3
        reasons.append("looks like HTML/markup")
    urls = _URL_RE.findall(text)
    if urls and sum(len(u) for u in urls) > 0.4 * len(text):
        score -= 3
        reasons.append("URL-dominated")
    if len(_CODEY_RE.findall(text)) >= 3:
        score -= 2
        reasons.append("looks like code")
    punct = sum(not (c.isalnum() or c.isspace()) for c in text)
    if punct / max(1, len(text)) > 0.35:
        score -= 2
        reasons.append("high punctuation (not prose)")

    return score, reasons


# --- public scan -------------------------------------------------------------
def scan_text(path: str, text: str, min_score: int = 4) -> List[DetectedPrompt]:
    out: List[DetectedPrompt] = []
    seen = set()
    for body, name, line in _iter_literals(text):
        stripped = body.strip()
        if len(stripped) < 40:
            continue
        key = stripped[:120]
        if key in seen:
            continue
        seen.add(key)
        score, reasons = prompt_likelihood(name, body)
        if score >= min_score:
            out.append(DetectedPrompt(
                source_path=path, name=name, line=line, text=stripped,
                est_tokens=count_tokens(stripped), score=score, reasons=reasons))
    out.sort(key=lambda d: (-d.score, -d.est_tokens))
    return out


def scan_file(path: str, min_score: int = 4) -> List[DetectedPrompt]:
    base = os.path.basename(path).lower()
    if os.path.splitext(path)[1].lower() not in SCANNABLE_EXTS:
        return []
    if any(s in base for s in _SKIP_NAME_SUBSTR):
        return []
    try:
        if os.path.getsize(path) > _MAX_BYTES:
            return []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
    except OSError:
        return []
    return scan_text(path, text, min_score=min_score)
