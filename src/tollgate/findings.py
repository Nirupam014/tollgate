"""Shared finding/severity types used by detectors, scoring and reporting."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

CATEGORIES = {
    "context_explosion",
    "recursive_loop",
    "fanout",
    "prompt_bloat",
    "retry_storm",
    "model_mismatch",
    "token_ceiling_exceeded",
    "policy_violation",
}


@dataclass
class Finding:
    finding_id: str
    category: str
    severity: str
    message: str
    node_id: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None
    line: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def severity_rank(sev: str) -> int:
    return SEVERITY_ORDER.get(sev, 0)
