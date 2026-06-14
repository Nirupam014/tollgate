"""Model Catalog: source of truth for models, pricing, and the substitution graph.

Ships a seed catalog (data/models.yaml). Prices are illustrative defaults and
should be overridden with a live catalog for accurate forecasts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_DEFAULT_CATALOG = os.path.join(_DATA_DIR, "models.yaml")


@dataclass
class Model:
    id: str
    provider: str
    family: str
    context_limit: int
    max_output: int
    quality_tier: int
    supports_tools: bool
    tokenizer: str
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: Optional[float] = None
    self_hosted: bool = False

    def cost(self, input_tokens: float, output_tokens: float, cached_tokens: float = 0.0) -> float:
        """USD cost for a single call."""
        fresh_input = max(0.0, input_tokens - cached_tokens)
        c = (fresh_input / 1_000_000.0) * self.input_per_mtok
        if cached_tokens and self.cached_input_per_mtok is not None:
            c += (cached_tokens / 1_000_000.0) * self.cached_input_per_mtok
        elif cached_tokens:
            c += (cached_tokens / 1_000_000.0) * self.input_per_mtok
        c += (output_tokens / 1_000_000.0) * self.output_per_mtok
        return c


@dataclass
class Substitution:
    from_model: str
    to_model: str
    capability_score: float


class ModelCatalog:
    def __init__(self, models: List[Model], substitutions: List[Substitution]):
        self._models: Dict[str, Model] = {m.id: m for m in models}
        self._subs = substitutions

    # --- loading ---------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str] = None) -> "ModelCatalog":
        path = path or _DEFAULT_CATALOG
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        models = [
            Model(
                id=m["id"],
                provider=m["provider"],
                family=m.get("family", m["provider"]),
                context_limit=int(m["context_limit"]),
                max_output=int(m["max_output"]),
                quality_tier=int(m["quality_tier"]),
                supports_tools=bool(m.get("supports_tools", True)),
                tokenizer=m.get("tokenizer", "heuristic"),
                input_per_mtok=float(m["input_per_mtok"]),
                output_per_mtok=float(m["output_per_mtok"]),
                cached_input_per_mtok=(
                    float(m["cached_input_per_mtok"]) if m.get("cached_input_per_mtok") is not None else None
                ),
                self_hosted=bool(m.get("self_hosted", False)),
            )
            for m in raw.get("models", [])
        ]
        subs = [
            Substitution(s["from"], s["to"], float(s["capability_score"]))
            for s in raw.get("substitutions", [])
        ]
        return cls(models, subs)

    # --- queries ---------------------------------------------------------------
    def get(self, model_id: str) -> Optional[Model]:
        return self._models.get(model_id)

    def require(self, model_id: str) -> Model:
        m = self._models.get(model_id)
        if m is None:
            raise KeyError(f"unknown model: {model_id!r} (not in catalog)")
        return m

    def all(self) -> List[Model]:
        return list(self._models.values())

    def substitutes(self, model_id: str) -> List[Substitution]:
        return [s for s in self._subs if s.from_model == model_id]

    def cheaper_alternatives(self, model_id: str, min_capability: float = 0.7) -> List[Substitution]:
        """Substitutes that are both safe enough and cheaper on a balanced workload."""
        base = self.get(model_id)
        if base is None:
            return []
        out = []
        for s in self.substitutes(model_id):
            cand = self.get(s.to_model)
            if cand is None or s.capability_score < min_capability:
                continue
            # Approximate cost ratio on a 3:1 input:output blend.
            base_c = base.cost(3000, 1000)
            cand_c = cand.cost(3000, 1000)
            if base_c <= 0:
                continue
            if cand_c < base_c:
                out.append(s)
        return sorted(out, key=lambda s: s.capability_score, reverse=True)
