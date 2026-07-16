"""Smoke test: the whole pipeline runs on a tiny MNIST subset and produces sane numbers.

Downloads MNIST on first run (small). Marked slow-ish but still seconds on CPU.
"""
from __future__ import annotations

from unlearn_audit.pipeline import run
from unlearn_audit.utils import load_config


def test_pipeline_runs_end_to_end():
    cfg = load_config("configs/test_tiny.yaml")
    text, summaries = run(cfg)

    # We get a report with the control, both methods, and the gold reference.
    assert "AUDIT" in text
    labels = {s.label for s in summaries}
    assert {"baseline (pre-unlearn)", "gradient_ascent", "guardrail", "gold (retrain)"} <= labels

    for s in summaries:
        # scores are well-formed and in range
        assert 0.0 <= s.audit.retention_score <= 1.0
        assert 0.0 <= s.audit.forgetting_score <= 1.0
        assert abs(s.audit.retention_score + s.audit.forgetting_score - 1.0) < 1e-9
        # calibration ran off the gold null -> a real FAR and a decision exist
        assert s.audit.calibration.calibrated is True
        assert s.audit.calibration.false_alarm_rate is not None
        assert s.audit.retention_detected in (True, False)


def test_guardrail_hides_forget_accuracy_but_not_membership():
    """The guardrail should collapse forget ACCURACY while keeping the loss/MIA trace."""
    cfg = load_config("configs/test_tiny.yaml")
    _, summaries = run(cfg)
    by = {s.label: s for s in summaries}
    guardrail = by["guardrail"]
    baseline = by["baseline (pre-unlearn)"]
    # forget accuracy is corrupted well below the baseline's forget accuracy
    assert guardrail.forget_eval.accuracy < baseline.forget_eval.accuracy
    # but retain accuracy is untouched (filter only fires on forget-set inputs)
    assert abs(guardrail.retain_eval.accuracy - baseline.retain_eval.accuracy) < 1e-6
