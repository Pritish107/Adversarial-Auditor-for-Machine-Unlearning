"""Calibration: turn a raw retention score into a decision with a false-alarm rate.

Method (first-cut, v0):
  1. Build a NULL distribution of retention scores from a case known to have truly
     forgotten — the gold retrain — by bootstrapping its member/non-member losses.
  2. Threshold = the (1 - target_far) quantile of that null.
  3. FAR = fraction of the null at or above the threshold.
  4. Decision for a query score = score >= threshold.

CAVEAT (see PLAN.md): bootstrapping ONE gold model captures sampling variance but NOT
model/seed variance, so this FAR is optimistic — a lower bound on the true false-alarm
rate. The honest version uses multiple gold retrains at different seeds; that is M3 work.
The v0 number is labeled first-cut accordingly.

If no null is supplied we fall back to an explicit uncalibrated stub (far=None) rather than
silently pretending to be calibrated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class CalibrationResult:
    calibrated: bool
    threshold: float
    false_alarm_rate: Optional[float]   # None until a real null distribution is supplied
    note: str


class Calibrator:
    def __init__(self, target_far: float = 0.05):
        self.target_far = target_far

    def calibrate(
        self,
        retention_score: float,
        null_scores: Optional[Sequence[float]] = None,
    ) -> CalibrationResult:
        if null_scores is None or len(null_scores) == 0:
            return CalibrationResult(
                calibrated=False,
                threshold=0.5,           # naive placeholder on the [0,1] retention scale
                false_alarm_rate=None,
                note="uncalibrated - no null distribution supplied; threshold is a placeholder",
            )
        null = np.asarray(null_scores, dtype=float)
        threshold = float(np.quantile(null, 1.0 - self.target_far))
        far = float(np.mean(null > threshold))     # strict >, matches the decision rule (C3)
        return CalibrationResult(
            calibrated=True,
            threshold=threshold,
            false_alarm_rate=far,
            note=(f"first-cut: bootstrap over ONE gold model (n={null.size}); sampling "
                  f"variance only, not seed variance -> FAR is a lower bound"),
        )
