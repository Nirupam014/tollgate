"""Token Simulation Engine (capability 3).

Monte Carlo over traffic scenarios. For each trial we sample per-node token
counts from distributions derived from the prediction p50/p95, walk the graph,
and sum TOKENS (input + output). Aggregating trials yields a per-request token
distribution which we scale by RPS x horizon x diurnal profile to a
tokens-over-horizon distribution.

This engine measures token consumption, not cost — model choice does not change
how many tokens a workflow emits, so no pricing is involved here.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List

from .ir import Workflow
from .prediction import WorkflowPrediction


@dataclass
class TrafficScenario:
    name: str
    rps: float
    horizon_days: int = 30
    diurnal_peak_multiplier: float = 1.0  # average load already reflected in rps

    @property
    def requests_over_horizon(self) -> float:
        return self.rps * 86400.0 * self.horizon_days


# Base traffic assumption: 10,000 requests/week.
DEFAULT_REQUESTS_PER_WEEK = 10_000
_SECONDS_PER_WEEK = 7 * 86400.0
_SECONDS_PER_DAY = 86400.0


def scenario_from_volume(requests: float, per: str = "week",
                         name: str = None, horizon_days: int = 30) -> "TrafficScenario":
    """Build a single steady-state scenario from a request volume per day/week."""
    per = (per or "week").lower()
    if per == "week":
        rps = requests / _SECONDS_PER_WEEK
        label = name or f"{int(requests):,}/week"
    elif per == "day":
        rps = requests / _SECONDS_PER_DAY
        label = name or f"{int(requests):,}/day"
    else:
        raise ValueError("per must be 'week' or 'day'")
    return TrafficScenario(label, rps=rps, horizon_days=horizon_days)


# Single steady-state base scenario (no peak/viral multipliers).
DEFAULT_SCENARIOS = [
    scenario_from_volume(DEFAULT_REQUESTS_PER_WEEK, "week", name="steady_state"),
]


@dataclass
class ScenarioResult:
    name: str
    rps: float
    horizon_days: int
    per_request_tokens: Dict[str, float]
    horizon_tokens: Dict[str, float]
    monthly_tokens: Dict[str, float]

    def to_dict(self):
        return {
            "name": self.name,
            "rps": self.rps,
            "horizon_days": self.horizon_days,
            "per_request_tokens": self.per_request_tokens,
            "horizon_tokens": self.horizon_tokens,
            "monthly_tokens": self.monthly_tokens,
        }


@dataclass
class SimulationOutput:
    scenarios: List[ScenarioResult] = field(default_factory=list)

    def to_dict(self):
        return {"scenarios": [s.to_dict() for s in self.scenarios]}


def _lognormal_sample(rng: random.Random, p50: float, p95: float) -> float:
    """Sample from a lognormal matched to the given p50 (median) and p95."""
    if p50 <= 0:
        return 0.0
    mu = math.log(p50)
    # p95/p50 = exp(1.645 * sigma)
    ratio = max(1.01, p95 / p50)
    sigma = math.log(ratio) / 1.645
    return math.exp(rng.gauss(mu, sigma))


class SimulationEngine:
    def __init__(self, trials: int = 4000, seed: int = 1337):
        self.trials = trials
        self.seed = seed

    def run(self, wf: Workflow, prediction: WorkflowPrediction,
            scenarios: List[TrafficScenario]) -> SimulationOutput:
        rng = random.Random(self.seed)
        node_preds = list(prediction.nodes)

        # Sample per-request TOKEN distribution once; scenarios reuse it.
        per_request_tokens: List[float] = []
        for _ in range(self.trials):
            total = 0.0
            for np_ in node_preds:
                calls = _sample_calls(rng, np_.expected_calls)
                if calls <= 0:
                    continue
                for _c in range(calls):
                    inp = _lognormal_sample(rng, np_.input_tokens.p50, np_.input_tokens.p95)
                    out = _lognormal_sample(rng, np_.output_tokens.p50, np_.output_tokens.p95)
                    total += inp + out
            per_request_tokens.append(total)

        per_request_tokens.sort()
        pr = _percentiles(per_request_tokens)

        results = []
        for sc in scenarios:
            reqs = sc.requests_over_horizon
            # Apply diurnal peak only to the tail percentiles (peaks drive p95/p99).
            horizon = {
                "p50": pr["p50"] * reqs,
                "p95": pr["p95"] * reqs * sc.diurnal_peak_multiplier,
                "p99": pr["p99"] * reqs * sc.diurnal_peak_multiplier,
            }
            month_scale = 30.0 / max(1, sc.horizon_days)
            monthly = {k: v * month_scale for k, v in horizon.items()}
            results.append(
                ScenarioResult(
                    name=sc.name, rps=sc.rps, horizon_days=sc.horizon_days,
                    per_request_tokens={k: round(v, 1) for k, v in pr.items()},
                    horizon_tokens={k: round(v) for k, v in horizon.items()},
                    monthly_tokens={k: round(v) for k, v in monthly.items()},
                )
            )
        return SimulationOutput(scenarios=results)


def _sample_calls(rng: random.Random, expected: float) -> int:
    """Sample an integer call count around the expected value."""
    if expected <= 0:
        return 0
    base = int(math.floor(expected))
    frac = expected - base
    return base + (1 if rng.random() < frac else 0)


def _percentiles(sorted_vals: List[float]) -> Dict[str, float]:
    if not sorted_vals:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    def pct(p):
        idx = min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1))))
        return sorted_vals[idx]

    return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99)}
