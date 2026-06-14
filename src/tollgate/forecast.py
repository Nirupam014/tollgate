"""Executive token forecast (capability 10, exec half).

For pre-deploy use we don't yet have telemetry history, so the forecast is the
projected steady-state TOKEN consumption of the analyzed change, decomposed by
node drivers (where the tokens go). When a historical rollup is supplied, an
EWMA baseline is blended in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .prediction import WorkflowPrediction
from .simulation import ScenarioResult


@dataclass
class ForecastDriver:
    name: str
    monthly_tokens: float
    share: float

    def to_dict(self):
        return {"name": self.name, "monthly_tokens": round(self.monthly_tokens),
                "share": round(self.share, 3)}


@dataclass
class Forecast:
    horizon_days: int
    total_monthly_tokens: Dict[str, float]
    drivers: List[ForecastDriver] = field(default_factory=list)
    method: str = "projection"

    def to_dict(self):
        return {
            "method": self.method,
            "horizon_days": self.horizon_days,
            "total_monthly_tokens": self.total_monthly_tokens,
            "drivers": [d.to_dict() for d in self.drivers],
        }


def build_forecast(prediction: WorkflowPrediction,
                   primary_scenario: ScenarioResult,
                   ewma_baseline_monthly: Optional[float] = None) -> Forecast:
    monthly = dict(primary_scenario.monthly_tokens)
    if ewma_baseline_monthly is not None:
        # Blend: projected change on top of existing baseline.
        monthly = {k: round(v + ewma_baseline_monthly) for k, v in monthly.items()}

    # Driver decomposition by node TOKEN share (p50 tokens/call x expected calls).
    node_tokens = [(n.node_id, n.request_tokens().p50) for n in prediction.nodes]
    total_pr = sum(t for _, t in node_tokens) or 1.0
    drivers = []
    for nid, t in sorted(node_tokens, key=lambda x: x[1], reverse=True):
        share = t / total_pr
        drivers.append(ForecastDriver(name=nid, monthly_tokens=monthly["p50"] * share, share=share))

    return Forecast(
        horizon_days=primary_scenario.horizon_days,
        total_monthly_tokens=monthly,
        drivers=drivers,
        method="projection" if ewma_baseline_monthly is None else "projection+ewma",
    )
