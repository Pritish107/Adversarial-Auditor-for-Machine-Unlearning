"""Readable text report for a v0 audit run.

Renders the before/after story: model accuracy on forget/retain/test, then the auditor's
per-attack findings and the aggregate forgetting score, for the pre-unlearning model, the
unlearned model, and (optionally) the gold retrain reference. The gold row is the visual
anchor for "what truly-forgotten looks like".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .audit.score import AuditReport
from .models import EvalResult


@dataclass
class ModelSummary:
    label: str
    forget_eval: EvalResult
    retain_eval: EvalResult
    test_eval: EvalResult
    audit: AuditReport


def _fmt_evals(s: ModelSummary) -> str:
    return (f"    acc  forget={s.forget_eval.accuracy:.3f}  "
            f"retain={s.retain_eval.accuracy:.3f}  test={s.test_eval.accuracy:.3f}\n"
            f"    loss forget={s.forget_eval.loss:.3f}  "
            f"retain={s.retain_eval.loss:.3f}  test={s.test_eval.loss:.3f}")


def _fmt_audit(a: AuditReport) -> str:
    lines = [f"    forgetting score = {a.forgetting_score:.3f}   "
             f"(residual retention = {a.retention_score:.3f}, aggregate={a.aggregate})"]
    for r in a.attacks:
        auc = r.detail.get("auc")
        auc_s = f", auc={auc:.3f}" if auc is not None else ""
        lines.append(f"      - {r.name}: retention={r.retention_score:.3f}{auc_s}")
    c = a.calibration
    far = "n/a" if c.false_alarm_rate is None else f"{c.false_alarm_rate:.3f}"
    lines.append(f"    calibration: calibrated={c.calibrated}, "
                 f"threshold={c.threshold:.3f}, FAR={far}  [{c.note}]")
    return "\n".join(lines)


def render(summaries: list[ModelSummary], *, header: Optional[str] = None) -> str:
    out: list[str] = []
    out.append("=" * 70)
    out.append(header or "ADVERSARIAL UNLEARNING AUDIT - v0 report")
    out.append("=" * 70)
    for s in summaries:
        out.append(f"\n[{s.label}]")
        out.append(_fmt_evals(s))
        out.append(_fmt_audit(s.audit))

    # Headline before/after, if we have both baseline and unlearned.
    by_label = {s.label: s for s in summaries}
    if "baseline (pre-unlearn)" in by_label and "unlearned" in by_label:
        b = by_label["baseline (pre-unlearn)"].audit.retention_score
        u = by_label["unlearned"].audit.retention_score
        out.append("\n" + "-" * 70)
        out.append(f"BEFORE/AFTER residual retention:  {b:.3f}  ->  {u:.3f}   "
                   f"(drop = {b - u:+.3f})")
        if "gold (retrain)" in by_label:
            g = by_label["gold (retrain)"].audit.retention_score
            out.append(f"gold (truly-forgotten) reference: {g:.3f}   "
                       f"<- what a clean audit looks like")
        out.append("-" * 70)
    return "\n".join(out)
