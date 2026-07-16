"""Readable text report for a v0 audit run.

Renders one ground-truth-labeled case per model (control baseline, each unlearning method,
and the gold retrain reference): accuracy on forget/retain/test, the auditor's MIA finding
(AUC + retention), and the calibrated decision. The closing block shows the ordering that
makes or breaks the demonstration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .audit.score import AuditReport
from .models import EvalResult


@dataclass
class ModelSummary:
    label: str
    ground_truth: str                 # what this case actually is (control/fake/low-quality/reference)
    forget_eval: EvalResult
    retain_eval: EvalResult
    test_eval: EvalResult
    audit: AuditReport


def _decision_str(a: AuditReport) -> str:
    if a.retention_detected is None:
        return "UNCALIBRATED"
    return "RETENTION DETECTED" if a.retention_detected else "forgotten (clean)"


def _fmt_evals(s: ModelSummary) -> str:
    return (f"    acc  forget={s.forget_eval.accuracy:.3f}  "
            f"retain={s.retain_eval.accuracy:.3f}  test={s.test_eval.accuracy:.3f}\n"
            f"    loss forget={s.forget_eval.loss:.3f}  "
            f"retain={s.retain_eval.loss:.3f}  test={s.test_eval.loss:.3f}")


def _fmt_audit(a: AuditReport) -> str:
    auc = a.attacks[0].detail.get("auc") if a.attacks else None
    auc_s = f"MIA-AUC={auc:.3f}  " if auc is not None else ""
    lines = [f"    {auc_s}retention={a.retention_score:.3f}  ->  {_decision_str(a)}"]
    return "\n".join(lines)


def render(summaries: list[ModelSummary], *, header: Optional[str] = None) -> str:
    out: list[str] = []
    out.append("=" * 72)
    out.append(header or "ADVERSARIAL UNLEARNING AUDIT - v0 (CIFAR-10, loss-MIA)")
    out.append("=" * 72)

    # Calibration banner (same null/threshold across all cases).
    calib = summaries[0].audit.calibration if summaries else None
    if calib is not None:
        far = "n/a" if calib.false_alarm_rate is None else f"{calib.false_alarm_rate:.3f}"
        out.append(f"calibration: calibrated={calib.calibrated}  "
                   f"threshold={calib.threshold:.3f}  FAR={far}")
        out.append(f"  [{calib.note}]")

    for s in summaries:
        out.append(f"\n[{s.label}]  (ground truth: {s.ground_truth})")
        out.append(_fmt_evals(s))
        out.append(_fmt_audit(s.audit))

    # Ordering block — the headline of the demonstration.
    by = {s.label: s for s in summaries}
    if {"gold (retrain)", "guardrail"} <= set(by):
        out.append("\n" + "-" * 72)
        out.append("ORDERING (residual retention, want: gold ~0 < gradient_ascent < guardrail):")
        for lbl in ("gold (retrain)", "gradient_ascent", "guardrail"):
            if lbl in by:
                a = by[lbl].audit
                auc = a.attacks[0].detail.get("auc") if a.attacks else None
                auc_s = f"AUC={auc:.3f}  " if auc is not None else ""
                out.append(f"    {lbl:<16} retention={a.retention_score:.3f}  "
                           f"{auc_s}-> {_decision_str(a)}")
        out.append("-" * 72)
    return "\n".join(out)
