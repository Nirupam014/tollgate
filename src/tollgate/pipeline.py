"""Analysis pipeline orchestrator.

Wires parsers -> prediction -> simulation -> detectors -> policy -> substitution
-> scoring -> remediation -> forecast into a single AnalysisResult per workflow,
and aggregates many workflows (e.g. a whole repo or a PR diff) into a run.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import version as _version
from .agentic_lint import (LINT_MERGE_CATEGORIES, lint_gate, lint_source,
                           worse_gate)
from .catalog import ModelCatalog
from .config import Config
from .detectors import DetectorEngine
from .findings import Finding, severity_rank
from .forecast import build_forecast, Forecast
from .ir import Workflow
from .parsers import discover, parse_file
from .parsers import _SKIP_DIR_SEGMENTS  # noqa: reuse the scan dir-prune set
from .policy import PolicyEngine, load_policies
from .prompt_scan import DetectedPrompt, SCANNABLE_EXTS, scan_file
from .prediction import PredictionEngine, WorkflowPrediction
from .prompt_review import PromptReview, review_text, review_workflow
from .remediation import build_plan, RemediationPlan
from .scoring import RiskScore, RiskScorer
from .simulation import DEFAULT_SCENARIOS, SimulationEngine, SimulationOutput, TrafficScenario
from .substitution import Recommendation, SubstitutionEngine


@dataclass
class AnalysisResult:
    workflow_id: str
    source_path: Optional[str]
    source_kind: str
    prediction: WorkflowPrediction
    simulation: SimulationOutput
    findings: List[Finding]
    policy_violations: List[Finding]
    recommendations: List[Recommendation]
    risk: RiskScore
    remediation: RemediationPlan
    forecast: Forecast
    prompt_reviews: List[PromptReview] = field(default_factory=list)

    def to_dict(self):
        return {
            "workflow_id": self.workflow_id,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "risk": self.risk.to_dict(),
            "prediction": self.prediction.to_dict(),
            "simulation": self.simulation.to_dict(),
            "forecast": self.forecast.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "policy_violations": [f.to_dict() for f in self.policy_violations],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "remediation": self.remediation.to_dict(),
            "prompt_reviews": [pr.to_dict() for pr in self.prompt_reviews],
        }


@dataclass
class LintResult:
    """A strict agentic-lint result for a file we recognized as agentic but did
    not parse into a full workflow graph. Carries STRUCTURAL findings only — no
    token/cost projection is invented for these (honest partial analysis)."""
    source_path: str
    source_kind: str
    findings: List[Finding]
    gate_decision: str
    score: int = 0

    def to_dict(self):
        return {
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "gate_decision": self.gate_decision,
            "score": self.score,
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class RunResult:
    results: List[AnalysisResult] = field(default_factory=list)
    lint_results: List["LintResult"] = field(default_factory=list)
    detected_prompts: List["DetectedPrompt"] = field(default_factory=list)  # advisory
    fingerprint: Optional[str] = None     # tamper-evident digest of inputs + verdict

    @property
    def gate_decision(self) -> str:
        order = {"pass": 0, "warn": 1, "block": 2}
        worst = "pass"
        for r in self.results:
            if order[r.risk.gate_decision] > order[worst]:
                worst = r.risk.gate_decision
        for lr in self.lint_results:
            if order.get(lr.gate_decision, 0) > order[worst]:
                worst = lr.gate_decision
        return worst

    @property
    def max_score(self) -> int:
        return max([r.risk.score for r in self.results]
                   + [lr.score for lr in self.lint_results], default=0)

    def to_dict(self):
        return {
            "gate_decision": self.gate_decision,
            "max_score": self.max_score,
            "workflow_count": len(self.results),
            "lint_count": len(self.lint_results),
            "detected_prompt_count": len(self.detected_prompts),
            "fingerprint": self.fingerprint,
            "results": [r.to_dict() for r in self.results],
            "lint_results": [lr.to_dict() for lr in self.lint_results],
            "detected_prompts": [p.to_dict() for p in self.detected_prompts],
        }

    def verdict_digest(self) -> str:
        """A canonical, order-stable digest of just the *verdict* (gate, scores,
        finding categories/severities) — the part a tamper would target. Excludes
        token projections (which depend on sampling) so it stays reproducible."""
        items = []
        for r in self.results:
            cats = sorted((f.category, f.severity) for f in r.findings)
            items.append(("wf", r.source_kind, r.risk.gate_decision, r.risk.score, cats))
        for lr in self.lint_results:
            cats = sorted((f.category, f.severity) for f in lr.findings)
            items.append(("lint", lr.source_kind, lr.gate_decision, lr.score, cats))
        items.sort(key=lambda x: json.dumps(x, sort_keys=True))
        payload = {"gate": self.gate_decision, "max_score": self.max_score, "items": items}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _rps_from_scenario(s: dict) -> float:
    """Resolve a scenario's request rate. Humans can specify the volume the easy
    way — `requests_per_week` or `requests_per_day` — and we convert to req/s.
    An explicit `rps` still wins if given (back-compat)."""
    if s.get("rps") is not None:
        return float(s["rps"])
    if s.get("requests_per_day") is not None:
        return float(s["requests_per_day"]) / 86400.0
    if s.get("requests_per_week") is not None:
        return float(s["requests_per_week"]) / (7 * 86400.0)
    return 10000.0 / (7 * 86400.0)   # default base: 10,000 req/week


def _scenarios_from_config(cfg: Config) -> List[TrafficScenario]:
    if not cfg.scenarios:
        return DEFAULT_SCENARIOS
    out = []
    for s in cfg.scenarios:
        out.append(TrafficScenario(
            name=s.get("name", "scenario"),
            rps=_rps_from_scenario(s),
            horizon_days=int(s.get("horizon_days", 30)),
            diurnal_peak_multiplier=float(s.get("diurnal_peak_multiplier", 1.0)),
        ))
    return out


def analyze_workflow(wf: Workflow, cfg: Optional[Config] = None,
                     catalog: Optional[ModelCatalog] = None,
                     telemetry: Optional[Dict[str, dict]] = None) -> AnalysisResult:
    cfg = cfg or Config()
    catalog = catalog or ModelCatalog.load(cfg.models_file)
    scenarios = _scenarios_from_config(cfg)

    predictor = PredictionEngine(catalog, telemetry=telemetry, default_model=cfg.default_model)
    prediction = predictor.predict(wf)

    simulator = SimulationEngine(trials=cfg.trials)
    simulation = simulator.run(wf, prediction, scenarios)

    primary = simulation.scenarios[0]
    monthly_tokens = primary.monthly_tokens

    detectors = DetectorEngine(catalog, thresholds=cfg.thresholds, default_model=cfg.default_model)
    tel_depths = {k: v.get("max_depth", 0) for k, v in (telemetry or {}).items()}
    findings = detectors.run(wf, prediction, telemetry_depths=tel_depths)

    # Strict agentic linter: add the config-absence checks the graph detectors
    # don't cover (missing iteration caps, uncapped output tokens). The graph
    # already owns recursive_loop / fanout, so only the non-overlapping lint
    # categories are merged here to avoid double-counting.
    lint_findings: List[Finding] = []
    if cfg.agentic_lint and wf.source_path and wf.source_path.endswith(".py"):
        lint_findings = [f for f in lint_source(wf.source_path, cfg.lint_strictness)
                         if f.category in LINT_MERGE_CATEGORIES]
        findings = findings + lint_findings

    policy_engine = PolicyEngine(load_policies(cfg.policies))
    policy_violations = policy_engine.evaluate(wf, prediction, monthly_tokens)
    blocking_violations = [f for f in policy_violations
                           if f.evidence.get("enforcement", "block") == "block"]

    sub_cfg = cfg.substitution or {}
    sub_engine = SubstitutionEngine(
        catalog,
        min_capability=float(sub_cfg.get("min_capability", 0.72)),
        min_savings_pct=float(sub_cfg.get("min_savings_pct", 15.0)),
        default_model=cfg.default_model,
    )
    allowlist = _allowlist_from_policies(cfg)
    recommendations = sub_engine.recommend(wf, prediction, allowlist=allowlist)

    scorer = RiskScorer(
        block_at_score=cfg.block_at_score,
        warn_at_score=cfg.warn_at_score,
    )
    risk = scorer.score(findings, monthly_tokens, policy_violations=blocking_violations)
    # Strict mode escalates the gate when cap/token lint findings are present.
    if cfg.agentic_lint and lint_findings:
        risk.gate_decision = worse_gate(
            risk.gate_decision, lint_gate(lint_findings, cfg.lint_strictness))

    remediation = build_plan(findings, prediction, recommendations,
                             assumed_rps=scenarios[0].rps)
    forecast = build_forecast(prediction, primary)

    prompt_reviews = review_workflow(wf) if cfg.prompt_review else []

    return AnalysisResult(
        workflow_id=wf.workflow_id,
        source_path=wf.source_path,
        source_kind=wf.source_kind,
        prediction=prediction,
        simulation=simulation,
        findings=findings,
        policy_violations=policy_violations,
        recommendations=recommendations,
        risk=risk,
        remediation=remediation,
        forecast=forecast,
        prompt_reviews=prompt_reviews,
    )


def analyze_path(paths, cfg: Optional[Config] = None,
                 catalog: Optional[ModelCatalog] = None) -> RunResult:
    if isinstance(paths, str):
        paths = [paths]
    cfg = cfg or Config.load(start_dir=paths[0] if paths else ".")
    catalog = catalog or ModelCatalog.load(cfg.models_file)

    run = RunResult()
    for fpath in discover(paths):
        try:
            wf = parse_file(fpath)
        except Exception:
            wf = None  # unparseable; may still be lintable below
        if wf is not None and _is_analyzable(wf):
            run.results.append(analyze_workflow(wf, cfg=cfg, catalog=catalog))
            continue
        # Not a recoverable graph — but if it's recognizably agentic, the strict
        # linter still gives an honest, cost-free verdict instead of dropping it.
        lr = _lint_only(fpath, cfg)
        if lr is not None:
            run.lint_results.append(lr)
    if cfg.agentic_lint:
        run.lint_results.extend(_lint_non_python(paths, cfg))
    if cfg.prompt_scan:
        run.detected_prompts = _scan_prompts(paths, cfg)
    run.fingerprint = _fingerprint(run, cfg)
    return run


def _lint_non_python(paths, cfg: Config) -> List["LintResult"]:
    """Run the language-agnostic textual linter over non-Python source (.py is
    already covered by discovery + the AST linter). Advisory, like the Python
    lint-only path: a LintResult only when the file is agentic and has findings."""
    out: List[LintResult] = []
    seen = set()
    scorer = RiskScorer(block_at_score=cfg.block_at_score, warn_at_score=cfg.warn_at_score)

    def consider(fp: str):
        if fp in seen:
            return
        seen.add(fp)
        ext = os.path.splitext(fp)[1].lower()
        if ext == ".py" or ext not in SCANNABLE_EXTS:
            return
        findings = lint_source(fp, cfg.lint_strictness)
        if not findings:
            return
        gate = worse_gate(scorer.score(findings).gate_decision,
                          lint_gate(findings, cfg.lint_strictness))
        out.append(LintResult(source_path=fp, source_kind=f"agentic-lint:{ext.lstrip('.')}",
                              findings=findings, gate_decision=gate,
                              score=scorer.score(findings).score))

    for p in paths:
        if os.path.isfile(p):
            consider(p)
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d.lower() not in _SKIP_DIR_SEGMENTS]
                for f in files:
                    consider(os.path.join(root, f))
    return out


def _scan_prompts(paths, cfg: Config) -> List["DetectedPrompt"]:
    """Mine embedded prompts across any-language source under the given paths.

    Heuristic + advisory: this is detection, not a verdict, so it does not feed
    the gate or the fingerprint — it surfaces prompts hidden in code/config so a
    human (and the prompt-efficiency review) can look at them."""
    found: List[DetectedPrompt] = []
    seen_files = set()

    def consider(fp: str):
        if fp in seen_files:
            return
        seen_files.add(fp)
        if os.path.splitext(fp)[1].lower() in SCANNABLE_EXTS:
            found.extend(scan_file(fp, min_score=cfg.prompt_scan_min_score))

    for p in paths:
        if os.path.isfile(p):
            consider(p)
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d.lower() not in _SKIP_DIR_SEGMENTS]
                for f in files:
                    consider(os.path.join(root, f))
    # Run the prompt-efficiency reviewer on each mined prompt so bloat/redundancy
    # in code-embedded prompts shows up in the same "Prompt token optimisation"
    # section as workflow prompts (advisory; never affects the gate).
    if cfg.prompt_review:
        for dp in found:
            dp.review = review_text(dp.text, node_id=(dp.name or os.path.basename(dp.source_path)),
                                    source_path=dp.source_path)
    found.sort(key=lambda d: (-d.score, -d.est_tokens))
    return found


def _file_sha(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def _fingerprint(run: "RunResult", cfg: Config) -> str:
    """Tamper-evident digest binding the verdict to the exact inputs that produced
    it: analyzed file contents + the config knobs that affect the gate + the tool
    version + the verdict itself. Re-running on the same inputs reproduces it, so a
    CI re-derivation overwrites any hand-edited report (self-healing) and any
    mismatch flags drift or tampering. Deterministic — no sampling-dependent fields.
    """
    paths = sorted({r.source_path for r in run.results if r.source_path}
                   | {lr.source_path for lr in run.lint_results if lr.source_path})
    inputs = [[os.path.basename(p), _file_sha(p)] for p in paths]
    knobs = {
        "default_model": cfg.default_model,
        "fail_on": cfg.fail_on,
        "lint_strictness": cfg.lint_strictness,
        "agentic_lint": cfg.agentic_lint,
        "block_at_score": cfg.block_at_score,
        "warn_at_score": cfg.warn_at_score,
        "policies": cfg.policies,
    }
    payload = {
        "tollgate_version": _version.__version__,
        "config": knobs,
        "inputs": inputs,
        "verdict": run.verdict_digest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _lint_only(fpath: str, cfg: Config) -> Optional["LintResult"]:
    """Run the agentic linter on a file with no recoverable graph. Returns a
    LintResult only when the file is agentic *and* has findings; else None."""
    if not (cfg.agentic_lint and fpath.endswith(".py")):
        return None
    findings = lint_source(fpath, cfg.lint_strictness)
    if not findings:
        return None
    scorer = RiskScorer(block_at_score=cfg.block_at_score, warn_at_score=cfg.warn_at_score)
    risk = scorer.score(findings)
    gate = worse_gate(risk.gate_decision, lint_gate(findings, cfg.lint_strictness))
    return LintResult(source_path=fpath, source_kind="agentic-lint",
                      findings=findings, gate_decision=gate, score=risk.score)


def _is_analyzable(wf: Workflow) -> bool:
    """Drop structureless parses so an unknown format never masquerades as PASS.

    A real workflow has at least one LLM call to cost, or some control flow to
    reason about. A parse that yielded nodes but *no* LLM node and *no* edges is
    almost always an unrecognized JSON whose schema we didn't map — surfacing it
    as a confident score-0 PASS would be a false negative, so we skip it.
    """
    if not wf.nodes:
        return False
    if wf.llm_nodes():
        return True
    return bool(wf.edges)


def _allowlist_from_policies(cfg: Config) -> Optional[List[str]]:
    for p in cfg.policies:
        if p.get("type") == "model_allowlist":
            allow = p.get("rule", {}).get("allow")
            if allow:
                return list(allow)
    return None
