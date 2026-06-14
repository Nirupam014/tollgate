"""Configuration loading for `.tollgate.yml`.

Example:

    default_model: gpt-4o
    models_file: ops/models.yaml          # override seed model catalog
    fail_on: block                         # block | warn | never
    scenarios:
      - { name: steady_state, requests_per_week: 10000, horizon_days: 30 }
    thresholds:
      prompt_bloat_tokens: 6000
    substitution:
      min_capability: 0.75
      min_savings_pct: 20
    policies:
      - name: prod_token_ceiling
        type: token_ceiling
        enforcement: block
        rule: { max_monthly_tokens: 2000000000, metric: projected_p95 }
      - name: loops_must_terminate
        type: loop_guard
        enforcement: block
        rule: { require_termination_guard: true, max_depth: 10 }
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

CONFIG_NAMES = [".tollgate.yml", ".tollgate.yaml", "tollgate.yml"]


@dataclass
class Config:
    default_model: str = "gpt-4o"
    models_file: Optional[str] = None
    fail_on: str = "block"                  # block | warn | never
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    substitution: Dict[str, Any] = field(default_factory=dict)
    policies: List[Dict[str, Any]] = field(default_factory=list)
    trials: int = 4000
    block_at_score: int = 75
    warn_at_score: int = 50
    prompt_review: bool = True               # prompt efficiency reviewer (on by default)
    agentic_lint: bool = True                # strict source-level agentic linter (on)
    lint_strictness: str = "strict"          # strict | balanced | off
    prompt_scan: bool = True                  # mine embedded prompts in any-language source
    prompt_scan_min_score: int = 4            # detection threshold (higher = stricter)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Optional[str] = None, start_dir: str = ".") -> "Config":
        cfg_path = path or _discover(start_dir)
        if not cfg_path or not os.path.isfile(cfg_path):
            return cls()
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        c = cls(
            default_model=data.get("default_model", "gpt-4o"),
            models_file=data.get("models_file"),
            fail_on=data.get("fail_on", "block"),
            scenarios=data.get("scenarios", []),
            thresholds=data.get("thresholds", {}),
            substitution=data.get("substitution", {}),
            policies=data.get("policies", []),
            trials=int(data.get("trials", 4000)),
            block_at_score=int(data.get("block_at_score", 75)),
            warn_at_score=int(data.get("warn_at_score", 50)),
            prompt_review=bool(data.get("prompt_review", True)),
            agentic_lint=bool(data.get("agentic_lint", True)),
            lint_strictness=str(data.get("lint_strictness", "strict")),
            prompt_scan=bool(data.get("prompt_scan", True)),
            prompt_scan_min_score=int(data.get("prompt_scan_min_score", 4)),
            raw=data,
        )
        # Resolve models_file relative to the config location.
        if c.models_file and not os.path.isabs(c.models_file):
            c.models_file = os.path.join(os.path.dirname(os.path.abspath(cfg_path)), c.models_file)
        return c


def _discover(start_dir: str) -> Optional[str]:
    d = os.path.abspath(start_dir)
    for _ in range(6):
        for name in CONFIG_NAMES:
            candidate = os.path.join(d, name)
            if os.path.isfile(candidate):
                return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None
