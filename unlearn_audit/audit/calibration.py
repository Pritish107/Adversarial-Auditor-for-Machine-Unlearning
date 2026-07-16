"""Calibration seam — STUB for v0.

The real job (Milestone 3+): turn a raw retention score into a *calibrated* decision with
a stated false-alarm rate. The intended method:

  1. Build a NULL distribution of retention scores on cases known to have truly forgotten
     (e.g. the gold retrain reference, and other clean-forgetting controls).
  2. Pick the decision threshold as the (1 - target_far) quantile of that null.
  3. Report the forgetting score together with the empirical FAR at that threshold.

v0 deliberately does NOT implement this. It returns an *uncalibrated* result with a naive
threshold and far=None, but through the exact interface the real calibrator will use — so
wiring it in later is a drop-in, not a refactor. This is the clean seam PROJECT.md asks for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


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
        """Return a decision threshold + FAR for `retention_score`.

        Real path (null_scores given) is not implemented in v0 — we assert the seam is
        reachable and fall through to the uncalibrated stub, so nothing silently pretends
        to be calibrated.
        """
        if null_scores is not None:
            # TODO(Milestone 3): threshold = quantile(null_scores, 1 - target_far);
            #                    far = empirical exceedance of null above threshold.
            raise NotImplementedError(
                "Real calibration from a null distribution lands in Milestone 3."
            )
        return CalibrationResult(
            calibrated=False,
            threshold=0.5,               # naive placeholder on the [0,1] retention scale
            false_alarm_rate=None,
            note="uncalibrated stub - no null distribution; threshold is a placeholder",
        )
