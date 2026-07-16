"""Aggregate attack results into a single forgetting score.

Per PROJECT.md: the model fails the audit if *any* attack recovers information, so the
aggregate residual-retention score is the MAX over attacks (a battery is only as forgetful
as its most revealing probe). The headline forgetting score is the complement:

    forgetting_score = 1 - retention_score        (both in [0,1], higher forgetting = better)

The calibration seam decides whether that retention is *significant*; v0 leaves it a stub.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..attacks.base import AttackResult
from .calibration import CalibrationResult, Calibrator


@dataclass
class AuditReport:
    retention_score: float           # aggregate residual retention (higher = worse forgetting)
    forgetting_score: float          # 1 - retention (higher = better forgetting)
    attacks: Sequence[AttackResult]
    calibration: CalibrationResult
    aggregate: str
    per_attack: dict = field(default_factory=dict)


def aggregate_retention(results: Sequence[AttackResult], method: str = "max") -> float:
    if not results:
        return 0.0
    scores = [r.retention_score for r in results]
    if method == "max":
        return max(scores)
    if method == "mean":
        return sum(scores) / len(scores)
    raise ValueError(f"unknown aggregate method '{method}'")


def audit(results: Sequence[AttackResult], *, aggregate: str = "max",
          target_far: float = 0.05) -> AuditReport:
    retention = aggregate_retention(results, aggregate)
    calib = Calibrator(target_far=target_far).calibrate(retention, null_scores=None)
    return AuditReport(
        retention_score=retention,
        forgetting_score=1.0 - retention,
        attacks=list(results),
        calibration=calib,
        aggregate=aggregate,
        per_attack={r.name: r.retention_score for r in results},
    )
