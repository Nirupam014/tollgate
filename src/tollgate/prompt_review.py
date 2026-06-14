"""Rule-based prompt efficiency reviewer.

Tollgate is a *static* analyzer: it never calls an LLM at analysis time, so this
reviewer is deterministic and explainable, not a model rewrite. It reads the real
prompt text the parsers captured (``IRNode.prompt_template``), applies a fixed set
of token-waste heuristics, and emits — per prompt — three things the report needs:

    1. the current prompt,
    2. a plain-English recommendation (what's wasteful and why), and
    3. a concrete *example* rewrite of that same prompt with the waste removed.

The example is produced by applying the same deterministic substitutions, so the
"after" column is always a real, reproducible rewrite of the "before" — never an
invented one. Token savings are estimated with the project tokenizer.

Two issue kinds:
  * substitution issues  — a wasteful pattern we can also auto-remove (filler,
    pleasantries, wordy phrases, duplicate lines, whitespace).
  * advisory issues      — a structural problem we flag but do not auto-edit
    (e.g. no explicit output-length cap), because a safe fix needs human intent.

Everything here is stdlib + the in-repo tokenizer; no new dependencies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .ir import Workflow
from .tokenizer import count_tokens


# --- Substitution rules --------------------------------------------------------
# Each rule: (code, compiled pattern, replacement, recommendation).
# Applied in order; if a rule's pattern matches we record the issue AND rewrite.
# Patterns are written to be conservative — they target stock filler, not domain
# wording — so the example rewrite stays faithful to the prompt's intent.
@dataclass(frozen=True)
class _Rule:
    code: str
    pattern: "re.Pattern[str]"
    replacement: str
    recommendation: str


# NOTE: inter-word gaps are written as \s+ (not a literal space) so a phrase that
# wraps across a line break — common in prompt files — is still matched.
_RULES: List[_Rule] = [
    _Rule("ai_self_reference",
          re.compile(r"(?i)\bas\s+an\s+ai(?:\s+language)?(?:\s+model)?\b,?\s*"),
          "",
          "Remove 'as an AI model' self-reference — it spends tokens and biases tone "
          "without changing the task."),
    _Rule("role_preamble",
          re.compile(r"(?i)\byour\s+(?:job|task|role|goal)\s+is\s+to\b\s*"),
          "",
          "Drop the 'your job is to …' preamble and state the instruction directly "
          "as an imperative."),
    _Rule("i_want_you_to",
          re.compile(r"(?i)\bi\s+(?:would\s+like|want|need)\s+(?:you\s+)?to\b\s*"),
          "",
          "Replace 'I would like you to X' with the direct command 'X'."),
    _Rule("in_this_task",
          re.compile(r"(?i)\bin\s+this\s+(?:task|prompt|exercise|conversation),?\s*"),
          "",
          "Remove scene-setting like 'in this task,' — the instruction itself is the "
          "context."),
    _Rule("politeness",
          re.compile(r"(?i)\b(?:please|kindly)\b\s*"),
          "",
          "Drop politeness fillers (please/kindly); they don't change model behavior."),
    _Rule("courtesy_close",
          re.compile(r"(?i)\b(?:thank\s+you|thanks)(?:\s+(?:so|very)\s+much)?"
                     r"(?:\s+for\s+your\s+(?:help|assistance|time|cooperation))?[.!]*\s*"),
          "",
          "Remove closing pleasantries ('thank you for your help')."),
    _Rule("instruction_padding",
          re.compile(r"(?i)\b(?:please\s+)?(?:make\s+sure\s+(?:that\s+)?(?:you\s+)?|"
                     r"be\s+sure\s+to\s+|it\s+is\s+important\s+(?:that\s+you|to)\s+|"
                     r"ensure\s+that\s+you\s+|note\s+that\s+|keep\s+in\s+mind\s+that\s+|"
                     r"remember\s+to\s+)"),
          "",
          "Trim instruction padding ('make sure that you', 'it is important that you', "
          "'note that') — keep the bare instruction."),
    _Rule("in_order_to",
          re.compile(r"(?i)\bin\s+order\s+to\b"),
          "to",
          "Shorten 'in order to' → 'to'."),
    _Rule("due_to_the_fact",
          re.compile(r"(?i)\bdue\s+to\s+the\s+fact\s+that\b"),
          "because",
          "Shorten 'due to the fact that' → 'because'."),
    _Rule("at_this_point_in_time",
          re.compile(r"(?i)\bat\s+this\s+(?:point|moment)\s+in\s+time\b"),
          "now",
          "Shorten 'at this point in time' → 'now'."),
    _Rule("for_the_purpose_of",
          re.compile(r"(?i)\bfor\s+the\s+purpose\s+of\b"),
          "for",
          "Shorten 'for the purpose of' → 'for'."),
    _Rule("a_number_of",
          re.compile(r"(?i)\ba\s+number\s+of\b"),
          "several",
          "Shorten 'a number of' → 'several'."),
    _Rule("intensifier_filler",
          re.compile(r"(?i)\b(?:very|really|quite|basically|actually|simply|literally)\b\s*"),
          "",
          "Remove empty intensifiers (very/really/basically/actually/simply)."),
]

# Signals that a prompt already bounds its output length, so we shouldn't nag.
_LENGTH_CAP_RE = re.compile(
    r"(?i)\b(?:no more than|at most|fewer than|less than|up to|limit(?:ed)? to|"
    r"(?:in|with(?:in)?|reply with|respond with|use|return)\s+(?:only\s+)?"
    r"(?:a\s+|one\s+|two\s+|three\s+|a single\s+)?"
    r"(?:\d+\s+)?(?:word|sentence|paragraph|character|bullet|item|line|token)s?|"
    r"\d+\s*(?:word|sentence|paragraph|character|bullet|item|line|token)s?|"
    r"concise|brief(?:ly)?|terse|short answer)\b")

# A prompt with template variables that grow unbounded (history/context) but no
# cap is a separate, structural waste worth calling out at the text level too.
_GROWTH_VAR_RE = re.compile(r"\{\{?\s*(history|messages|conversation|context|"
                            r"documents|retrieved|memory)\b")

_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_MULTIBLANK_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")


@dataclass
class PromptIssue:
    code: str
    message: str
    kind: str = "substitution"   # "substitution" | "advisory"

    def to_dict(self):
        return {"code": self.code, "message": self.message, "kind": self.kind}


@dataclass
class PromptReview:
    node_id: str
    original: str
    rewritten: str
    issues: List[PromptIssue] = field(default_factory=list)
    original_tokens: int = 0
    rewritten_tokens: int = 0
    source_path: Optional[str] = None

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.rewritten_tokens)

    @property
    def savings_pct(self) -> float:
        if self.original_tokens <= 0:
            return 0.0
        return round(100.0 * self.tokens_saved / self.original_tokens, 1)

    @property
    def recommendation(self) -> str:
        """One combined, de-duplicated recommendation string for the report cell."""
        seen, msgs = set(), []
        for it in self.issues:
            if it.message not in seen:
                seen.add(it.message)
                msgs.append(it.message)
        return " ".join(msgs)

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "original": self.original,
            "rewritten": self.rewritten,
            "issues": [i.to_dict() for i in self.issues],
            "recommendation": self.recommendation,
            "original_tokens": self.original_tokens,
            "rewritten_tokens": self.rewritten_tokens,
            "tokens_saved": self.tokens_saved,
            "savings_pct": self.savings_pct,
        }


def _dedupe_lines(text: str) -> "tuple[str, bool]":
    """Remove exact duplicate non-empty lines (keep first). Returns (text, changed)."""
    seen = set()
    out = []
    changed = False
    for ln in text.split("\n"):
        key = ln.strip()
        if key and key in seen:
            changed = True
            continue
        if key:
            seen.add(key)
        out.append(ln)
    return "\n".join(out), changed


def _normalize_ws(text: str) -> str:
    text = _TRAILING_WS_RE.sub("\n", text)
    text = _MULTISPACE_RE.sub(" ", text)
    text = _MULTIBLANK_RE.sub("\n\n", text)
    # Tidy spaces left where a filler word was excised mid-sentence.
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    text = re.sub(r"(?m)^[ \t]+", "", text)
    return text.strip()


# Template *machinery* (Jinja/Nunjucks/Liquid control flow, Handlebars blocks).
# The efficiency rewriter trims prose; on control logic its whitespace/line-dedup
# rules are meaningless and actively corrupt the template, so we never review it.
_STMT_TAG_RE = re.compile(
    r'\{%-?\s*(?:if|elif|else|endif|for|endfor|set|macro|endmacro|block|endblock|'
    r'include|extends|with|endwith|call|endcall|filter|endfilter|raw|endraw)\b')
_HB_BLOCK_RE = re.compile(r'\{\{[#/]')


def _is_template_machinery(text: str) -> bool:
    """True for chat/message templates that are control flow, not a prose prompt."""
    if len(_STMT_TAG_RE.findall(text)) >= 2:
        return True
    if len(_HB_BLOCK_RE.findall(text)) >= 2:
        return True
    return False


def review_text(text: str, node_id: str = "prompt",
                max_output_tokens: Optional[int] = None,
                source_path: Optional[str] = None) -> Optional[PromptReview]:
    """Review a single prompt string. Returns a PromptReview if anything was found,
    else None. Deterministic: same input always yields the same rewrite."""
    if not text or not text.strip():
        return None
    # Don't "optimise" template machinery — only natural-language prompts.
    if _is_template_machinery(text):
        return None

    rewritten = text
    issues: List[PromptIssue] = []

    for rule in _RULES:
        if rule.pattern.search(rewritten):
            issues.append(PromptIssue(rule.code, rule.recommendation, "substitution"))
            rewritten = rule.pattern.sub(rule.replacement, rewritten)

    rewritten, deduped = _dedupe_lines(rewritten)
    if deduped:
        issues.append(PromptIssue(
            "duplicate_lines",
            "Remove duplicated instruction lines — repetition wastes tokens and can "
            "confuse instruction priority.", "substitution"))

    normalized = _normalize_ws(rewritten)
    if normalized != _normalize_ws(text) or normalized != rewritten:
        # Whitespace was meaningfully compressible only if it changed beyond the
        # edits we already counted.
        if _MULTISPACE_RE.search(text) or _MULTIBLANK_RE.search(text) or _TRAILING_WS_RE.search(text):
            issues.append(PromptIssue(
                "whitespace",
                "Collapse runs of spaces/blank lines.", "substitution"))
    rewritten = normalized

    # Advisory: no explicit output-length cap. Bounding output is the single
    # biggest lever on output-token cost, so flag it even though we can't safely
    # auto-insert a number.
    if max_output_tokens is None and not _LENGTH_CAP_RE.search(text):
        issues.append(PromptIssue(
            "no_output_cap",
            "Add an explicit output-length cap (e.g. 'Answer in <=100 words' or set "
            "max_output_tokens) — unbounded output is the top driver of output-token "
            "cost.", "advisory"))

    # Advisory: unbounded growth variables with no cap nearby.
    if _GROWTH_VAR_RE.search(text):
        issues.append(PromptIssue(
            "unbounded_context",
            "This prompt interpolates growing history/context; cap or summarize it so "
            "input tokens don't grow every turn.", "advisory"))

    if not issues:
        return None

    return PromptReview(
        node_id=node_id,
        original=text,
        rewritten=rewritten,
        issues=issues,
        original_tokens=count_tokens(text),
        rewritten_tokens=count_tokens(rewritten),
        source_path=source_path,
    )


def review_workflow(wf: Workflow) -> List[PromptReview]:
    """Review every LLM node in a workflow that carries prompt text."""
    out: List[PromptReview] = []
    for node in wf.llm_nodes():
        if not node.prompt_template:
            continue
        rev = review_text(
            node.prompt_template,
            node_id=node.node_id,
            max_output_tokens=node.max_output_tokens,
            source_path=wf.source_path,
        )
        if rev is not None:
            out.append(rev)
    return out
