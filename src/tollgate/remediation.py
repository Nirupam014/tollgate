"""Engineering remediation plan (capability 10, engineering half).

Turns structural findings into ranked, concrete fixes that REDUCE TOKEN
CONSUMPTION, each with an estimated token saving (per request and projected per
month) where the math is clean, a suggested change, and an effort estimate.

Token savings are estimates for human review, not guarantees. Where a saving
cannot be bounded (e.g. an unbounded loop or input-driven fan-out) we report the
per-unit token cost and mark the saving as "uncapped" rather than inventing a
number. Model right-sizing is reported separately as a cost lever — it does not
change how many tokens a workflow emits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .findings import Finding
from .prediction import NodePrediction, WorkflowPrediction
from .substitution import Recommendation

# Seconds/month for converting per-request token savings into monthly figures.
_SECONDS_PER_MONTH = 30 * 86400

# Targets used when estimating realizable savings.
_PROMPT_FLOOR_TOKENS = 1500      # assume a trimmed/cached system prompt floor
_PROMPT_REALIZABLE = 0.6         # fraction of the excess that's realistically removable
_CONTEXT_CAP_TOKENS = 4000       # sliding-window / summarization target
_FANOUT_TARGET = 10              # reasonable bounded fan-out
_DEFAULT_CALL_TOKENS = 800       # fallback per-call tokens when a node isn't predicted

_SEVERITY_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}


@dataclass
class RemediationItem:
    title: str
    category: str
    severity: str
    effort: str                # low | medium | high
    est_tokens_saved_per_request: Optional[float]
    est_monthly_tokens_saved: Optional[float]
    node_id: Optional[str]
    detail: str
    suggested_change: Optional[str] = None
    lever: str = "tokens"      # tokens | cost

    def to_dict(self):
        return {
            "title": self.title,
            "category": self.category,
            "severity": self.severity,
            "effort": self.effort,
            "lever": self.lever,
            "est_tokens_saved_per_request": (round(self.est_tokens_saved_per_request)
                                             if self.est_tokens_saved_per_request is not None else None),
            "est_monthly_tokens_saved": (round(self.est_monthly_tokens_saved)
                                         if self.est_monthly_tokens_saved is not None else None),
            "node_id": self.node_id,
            "detail": self.detail,
            "suggested_change": self.suggested_change,
        }


@dataclass
class RemediationPlan:
    items: List[RemediationItem] = field(default_factory=list)
    est_monthly_tokens_saved: float = 0.0

    def to_dict(self):
        return {
            "est_monthly_tokens_saved": round(self.est_monthly_tokens_saved),
            "items": [i.to_dict() for i in self.items],
        }


def build_plan(findings: List[Finding], prediction: WorkflowPrediction,
               recommendations: Optional[List[Recommendation]] = None,
               assumed_rps: float = 5.0) -> RemediationPlan:
    pred_by_node: Dict[str, NodePrediction] = {n.node_id: n for n in prediction.nodes}
    items: List[RemediationItem] = []
    total_saved = 0.0

    seen_sigs = set()
    for f in findings:
        item = _finding_to_item(f, pred_by_node, assumed_rps)
        if item is None:
            continue
        # Collapse identical fixes (e.g. several agents that all need the same
        # cap) into one plan entry so the remediation list doesn't repeat itself.
        sig = (item.category, item.title, item.suggested_change, item.node_id)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        if item.est_monthly_tokens_saved:
            total_saved += item.est_monthly_tokens_saved
        items.append(item)

    # Model right-sizing: a cost lever, listed after the token reducers.
    for r in (recommendations or []):
        items.append(RemediationItem(
            title=f"Right-size model on '{r.node_id}': {r.from_model} -> {r.to_model}",
            category="model_substitution", severity="low", effort="low",
            est_tokens_saved_per_request=None, est_monthly_tokens_saved=None,
            node_id=r.node_id, lever="cost",
            detail=(f"Capability {r.capability_score:.2f}. Reduces cost, not tokens — "
                    f"the workflow emits the same token count on a cheaper model."),
            suggested_change=f"model: {r.to_model}   # was {r.from_model}",
        ))

    items.sort(key=lambda i: (_SEVERITY_RANK.get(i.severity, 0),
                              i.est_monthly_tokens_saved or 0.0), reverse=True)
    return RemediationPlan(items=items, est_monthly_tokens_saved=total_saved)


def _monthly(saved_per_request: float, rps: float) -> float:
    return saved_per_request * rps * _SECONDS_PER_MONTH


def _call_tokens(np_: Optional[NodePrediction]) -> float:
    return np_.call_tokens().p50 if np_ else _DEFAULT_CALL_TOKENS


def _finding_to_item(f: Finding, pred: Dict[str, NodePrediction],
                     rps: float) -> Optional[RemediationItem]:
    ev = f.evidence or {}
    np_ = pred.get(f.node_id) if f.node_id else None

    if f.category == "recursive_loop":
        cycle = ev.get("cycle", [])
        bounded = ev.get("termination_guard") == "bounded"
        loop_label = ("/".join(cycle)) or f.node_id
        per_iter = _call_tokens(np_)
        if bounded:
            return RemediationItem(
                title=f"Verify the iteration bound on loop {loop_label}",
                category=f.category, severity=f.severity, effort="low",
                est_tokens_saved_per_request=None, est_monthly_tokens_saved=None,
                node_id=f.node_id,
                detail=(f"Loop is guarded (~{ev.get('estimated_iterations')} iterations, "
                        f"~{int(per_iter):,} tokens each). Confirm the bound matches intent; "
                        f"each iteration you remove saves ~{int(per_iter):,} tokens/request."),
                suggested_change="guard: { max_depth: <tighten_if_possible> }",
            )
        return RemediationItem(
            title=f"Add a termination guard to loop {loop_label}",
            category=f.category, severity=f.severity, effort="medium",
            est_tokens_saved_per_request=None, est_monthly_tokens_saved=None,
            node_id=f.node_id,
            detail=(f"UNCAPPED: each iteration adds ~{int(per_iter):,} tokens/request and the "
                    f"cycle can recurse without limit. A bound caps an otherwise unbounded token "
                    f"series — the single highest-leverage fix here."),
            suggested_change="guard: { max_depth: 10 }   # on the loop edge",
        )

    if f.category == "context_explosion":
        if ev.get("growth_pattern") == "linear_accumulation":
            projected = float(ev.get("projected_tokens", 0))
            saved_pr = max(0.0, projected - _CONTEXT_CAP_TOKENS)
            mo = _monthly(saved_pr, rps) if saved_pr else None
            return RemediationItem(
                title=f"Cap/summarize history before '{f.node_id}'",
                category=f.category, severity=f.severity, effort="medium",
                est_tokens_saved_per_request=saved_pr or None,
                est_monthly_tokens_saved=mo, node_id=f.node_id,
                detail=(f"History accumulates ~{int(ev.get('per_iteration_token_delta',0)):,} "
                        f"tokens/iteration to ~{int(projected):,} tokens. A sliding window or "
                        f"per-turn summary holds input near ~{_CONTEXT_CAP_TOKENS:,} tokens."),
                suggested_change="retrieved_context_cap: 4000   # and summarize history each turn",
            )
        # Uncapped retrieval (not in a loop): real but hard to bound without the corpus.
        return RemediationItem(
            title=f"Cap retrieved context for '{f.node_id}'",
            category=f.category, severity=f.severity, effort="low",
            est_tokens_saved_per_request=None, est_monthly_tokens_saved=None,
            node_id=f.node_id,
            detail=("Retrieval is uncapped; a large hit can spike input tokens. Set an explicit "
                    "cap so a single request can't balloon the prompt."),
            suggested_change="retrieved_context_cap: 4000",
        )

    if f.category == "prompt_bloat":
        static = float(ev.get("static_input_tokens", 0))
        calls = np_.expected_calls if np_ else 1.0
        saved_per_call = max(0.0, static - _PROMPT_FLOOR_TOKENS) * _PROMPT_REALIZABLE
        saved_pr = saved_per_call * calls
        mo = _monthly(saved_pr, rps) if saved_pr else None
        return RemediationItem(
            title=f"Trim/cache the system prompt for '{f.node_id}'",
            category=f.category, severity=f.severity, effort="low",
            est_tokens_saved_per_request=saved_pr or None,
            est_monthly_tokens_saved=mo, node_id=f.node_id,
            detail=(f"Static prompt is ~{int(static):,} tokens. Trimming redundant boilerplate "
                    f"and moving stable instructions to a cached prefix can remove most of the "
                    f"excess above a ~{_PROMPT_FLOOR_TOKENS:,}-token floor (x{calls:.1f} calls/request)."),
            suggested_change=None,
        )

    if f.category == "missing_iteration_cap":
        fw = ev.get("framework", "the framework")
        ctor = ev.get("constructor", "agent")
        kwargs = ev.get("expected_kwargs") or ["max_iterations"]
        kw = kwargs[0]
        default_val = {
            "max_iter": 15, "max_iterations": 10, "max_round": 12,
            "max_consecutive_auto_reply": 10, "max_turns": 12, "max_steps": 12,
            "max_function_calls": 10, "max_execution_time": 120, "recursion_limit": 25,
        }.get(kw, 10)
        if kw == "recursion_limit":
            suggested = 'result = app.invoke(inputs, {"recursion_limit": 25})'
        else:
            suggested = f"{ctor}(..., {kw}={default_val})"
        return RemediationItem(
            title=f"Set an explicit {kw} on {fw} {ctor}(...)",
            category=f.category, severity=f.severity, effort="low",
            est_tokens_saved_per_request=None, est_monthly_tokens_saved=None,
            node_id=f.node_id,
            detail=(f"{fw} {ctor} runs on the framework's default step bound. An explicit "
                    f"{kw} caps how many reasoning/tool steps the agent can take, so a "
                    f"confused or prompt-injected agent can't loop and burn tokens "
                    f"indefinitely. Pick a value just above the longest legitimate run."),
            suggested_change=suggested,
        )

    if f.category == "uncapped_output":
        if ev.get("check") == "uncapped_model_ctor":
            ctor = ev.get("constructor", "ChatOpenAI")
            return RemediationItem(
                title=f"Cap output tokens on `{ctor}(...)`",
                category=f.category, severity=f.severity, effort="low",
                est_tokens_saved_per_request=None, est_monthly_tokens_saved=None,
                node_id=f.node_id,
                detail=(f"The {ctor} chat model is built with no output-token limit, so "
                        f"every call through it can run to the model's full context "
                        f"window. Set the cap on the constructor; size it to the longest "
                        f"reply you actually need."),
                suggested_change=f"{ctor}(..., max_tokens=512)",
            )
        call = ev.get("call", "client.chat.completions.create")
        return RemediationItem(
            title="Cap output tokens on the LLM call",
            category=f.category, severity=f.severity, effort="low",
            est_tokens_saved_per_request=None, est_monthly_tokens_saved=None,
            node_id=f.node_id,
            detail=(f"`{call}(...)` sets no output-token limit, so one response can run to "
                    f"the model's full context window. An explicit cap bounds worst-case "
                    f"output cost per call; size it to the longest reply you actually need."),
            suggested_change=f"{call}(..., max_tokens=512)",
        )

    if f.category == "unbounded_loop":
        return RemediationItem(
            title="Bound the agent loop",
            category=f.category, severity=f.severity, effort="medium",
            est_tokens_saved_per_request=None, est_monthly_tokens_saved=None,
            node_id=f.node_id,
            detail=("A `while True:` drives an LLM call with no break or return — the loop "
                    "is unbounded. Add a hard step ceiling and a terminal condition so the "
                    "agent can't spin (and bill) forever under adverse inputs."),
            suggested_change=("for step in range(MAX_STEPS):      # was: while True\n"
                              "    result = call_llm(...)\n"
                              "    if is_done(result):\n"
                              "        break"),
        )

    if f.category == "fanout":
        if ev.get("check") == "unbounded_fanout":
            return RemediationItem(
                title="Bound parallel agent fan-out",
                category=f.category, severity=f.severity, effort="medium",
                est_tokens_saved_per_request=None, est_monthly_tokens_saved=None,
                node_id=f.node_id,
                detail=("asyncio.gather() launches one LLM call per input item with no "
                        "concurrency limit, so a large input spawns unbounded parallel "
                        "calls. Gate concurrency with a Semaphore (or batch the inputs)."),
                suggested_change=("sem = asyncio.Semaphore(10)\n"
                                  "async def bound(x):\n"
                                  "    async with sem:\n"
                                  "        return await one(x)\n"
                                  "await asyncio.gather(*[bound(x) for x in items])"),
            )
        factor = ev.get("fanout_factor")
        per_call = _call_tokens(np_)
        if isinstance(factor, int) and factor > _FANOUT_TARGET:
            saved_pr = (factor - _FANOUT_TARGET) * per_call
            mo = _monthly(saved_pr, rps) if saved_pr else None
            return RemediationItem(
                title=f"Bound the fan-out of '{f.node_id}'",
                category=f.category, severity=f.severity, effort="medium",
                est_tokens_saved_per_request=saved_pr or None,
                est_monthly_tokens_saved=mo, node_id=f.node_id,
                detail=(f"Fans out to {factor} parallel calls (~{int(per_call):,} tokens each). "
                        f"Capping/batching toward ~{_FANOUT_TARGET} cuts the parallel token load."),
                suggested_change=f"fanout_factor: {_FANOUT_TARGET}",
            )
        return RemediationItem(
            title=f"Bound the fan-out of '{f.node_id}'",
            category=f.category, severity=f.severity, effort="medium",
            est_tokens_saved_per_request=None, est_monthly_tokens_saved=None,
            node_id=f.node_id,
            detail=(f"UNCAPPED: fan-out is input-driven (~{int(per_call):,} tokens per parallel "
                    f"call), so token load scales with input size. Cap the parallel width or batch."),
            suggested_change="fanout_factor: 10",
        )

    if f.category == "retry_storm":
        per_call = _call_tokens(np_)
        return RemediationItem(
            title=f"Add bounded retry + backoff to '{f.node_id}'",
            category=f.category, severity=f.severity, effort="low",
            est_tokens_saved_per_request=None, est_monthly_tokens_saved=None,
            node_id=f.node_id,
            detail=(f"Each retried attempt re-spends ~{int(per_call):,} tokens. On error spikes an "
                    f"uncapped retry multiplies token use; cap attempts and back off exponentially."),
            suggested_change="retry: { max_attempts: 3, backoff: exponential }",
        )

    if f.category == "model_mismatch":
        return RemediationItem(
            title=f"Right-size the model on '{f.node_id}'",
            category=f.category, severity=f.severity, effort="low",
            est_tokens_saved_per_request=None, est_monthly_tokens_saved=None,
            node_id=f.node_id, lever="cost",
            detail=("A frontier model on a simple task class. This is a cost lever (a smaller "
                    "model emits the same tokens); switch it to cut spend, not token volume."),
            suggested_change=None,
        )

    # Generic passthrough (e.g. policy violations).
    return RemediationItem(
        title=f.message, category=f.category, severity=f.severity, effort="medium",
        est_tokens_saved_per_request=None, est_monthly_tokens_saved=None,
        node_id=f.node_id, detail=f.message,
    )
